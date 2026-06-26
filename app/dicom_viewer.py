"""Local Streamlit DICOM MRI viewer and metadata audit.

Run from the repository root:
    streamlit run app/dicom_viewer.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dicom_tools.discovery import (
    DicomFile,
    SeriesGroup,
    StudyGroup,
    clinical_series_rows,
    discover_dicom_files,
    group_by_study_series,
    mri_quality_rows,
    series_display_name,
    sorted_series,
    study_summary,
    study_summary_rows,
)
from dicom_tools.metadata import basic_metadata, dataset_to_records, read_metadata, records_as_dicts
from dicom_tools.viewer_utils import read_pixel_array


st.set_page_config(page_title="Local DICOM MRI Viewer", layout="wide")


@st.cache_data(show_spinner=False)
def cached_discovery(root: str) -> list[DicomFile]:
    return discover_dicom_files(root)


def _series_label(series: SeriesGroup, *, technical: bool) -> str:
    first = series.first
    base = (
        f"{series_display_name(series)} | {series.orientation} | "
        f"{series.instance_count} slices | {first.slice_thickness or '?'} mm"
    )
    if technical:
        return f"{base} | SeriesNumber={first.series_number or '?'} | UID={series.series_instance_uid}"
    return base


def _study_label(study: StudyGroup) -> str:
    first = study.first
    description = first.study_description or "No StudyDescription"
    date = first.study_date or "No StudyDate"
    return f"{date} | {description} | {study.series_count} series | {study.file_count} files"


def _render_summary_metrics(summary: dict[str, object]) -> None:
    cols = st.columns(6)
    items = list(summary.items())
    for col, (label, value) in zip(cols, items):
        col.metric(label, value if value not in ("", None) else "UNKNOWN")


def _download_buttons(df: pd.DataFrame, selected_path: Path) -> None:
    stem = selected_path.name.replace(".", "_")
    st.download_button(
        "Download audit CSV (local)",
        df.to_csv(index=False).encode("utf-8"),
        file_name=f"{stem}_metadata_audit.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download audit JSON (local)",
        json.dumps(df.to_dict(orient="records"), indent=2).encode("utf-8"),
        file_name=f"{stem}_metadata_audit.json",
        mime="application/json",
    )


st.title("Local DICOM MRI Viewer")
st.warning(
    "This app is for local viewing and metadata audit only. PatientID can be PHI. "
    "Do not upload reports containing real identifiers outside your controlled environment."
)

root = st.text_input("Root folder or single DICOM file", value=str(PROJECT_ROOT / "data_raw"))
if not root:
    st.stop()

root_path = Path(root).expanduser()
if not root_path.exists():
    st.info("Enter an existing local path to a DICOM file or folder.")
    st.stop()

with st.spinner("Scanning local DICOM headers without pixel data..."):
    files = cached_discovery(str(root_path))

st.caption(f"Found {len(files)} DICOM file(s). Files without extensions are included.")
if not files:
    st.stop()

studies = group_by_study_series(files)
mode = st.radio("View mode", ["Clinical View", "Technical View"], horizontal=True)
technical_mode = mode == "Technical View"

st.subheader("Studies")
studies_df = pd.DataFrame(study_summary_rows(studies))
st.dataframe(studies_df, use_container_width=True, hide_index=True)

study_keys = list(studies.keys())
study_key = st.selectbox(
    "Study",
    options=study_keys,
    format_func=lambda key: _study_label(studies[key]),
)
study = studies[study_key]

with st.expander("Technical metadata", expanded=technical_mode):
    st.write(
        {
            "PatientID (may be PHI)": study.patient_id,
            "StudyInstanceUID": study.study_instance_uid,
            "SeriesInstanceUIDs": list(study.series.keys()),
        }
    )

st.subheader("Study Summary")
_render_summary_metrics(study_summary(study))

st.subheader("MRI Series")
series_table = pd.DataFrame(clinical_series_rows(study))
if technical_mode:
    technical_series_table = pd.DataFrame(mri_quality_rows(study, include_uids=True))
    st.dataframe(technical_series_table, use_container_width=True, hide_index=True)
else:
    st.dataframe(series_table[["Series", "Orientation", "Slices"]], use_container_width=True, hide_index=True)

st.subheader("MRI Quality Summary")
st.dataframe(
    pd.DataFrame(mri_quality_rows(study, include_uids=technical_mode)),
    use_container_width=True,
    hide_index=True,
)

series_options = sorted_series(study)
selected_series = st.selectbox(
    "Series",
    options=series_options,
    format_func=lambda item: _series_label(item, technical=technical_mode),
)
series_files = selected_series.files

instance_numbers = [
    file.instance_number if file.instance_number is not None else index + 1
    for index, file in enumerate(series_files)
]
if len(series_files) == 1:
    selected_index = 0
else:
    selected_index = st.select_slider(
        "InstanceNumber",
        options=list(range(len(series_files))),
        value=0,
        format_func=lambda index: str(instance_numbers[index]),
    )

selected_file = series_files[selected_index]
if technical_mode:
    st.caption(f"Selected local file: {selected_file.path} | InstanceNumber: {instance_numbers[selected_index]}")
else:
    st.caption(f"InstanceNumber: {instance_numbers[selected_index]}")

left, right = st.columns([1.2, 1])

with left:
    st.subheader("Slice")
    pixel_result = read_pixel_array(selected_file.path)
    if pixel_result.image is None:
        st.error(pixel_result.error or "Pixel data could not be displayed.")
    else:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(pixel_result.image, cmap="gray")
        ax.axis("off")
        st.pyplot(fig, clear_figure=True)
        st.caption(f"PhotometricInterpretation: {pixel_result.photometric_interpretation or 'unknown'}")

with right:
    st.subheader("Basic Metadata")
    ds = read_metadata(selected_file.path, stop_before_pixels=True)
    metadata_df = pd.DataFrame(
        [{"Field": key, "Value": value} for key, value in basic_metadata(ds).items()]
    )
    st.dataframe(metadata_df, use_container_width=True, hide_index=True)
    burned_in = str(getattr(ds, "BurnedInAnnotation", "")).upper()
    if burned_in != "NO":
        st.warning("BurnedInAnnotation is YES or unknown. Manual pixel review is required before de-identification.")

st.subheader("Metadata Audit")
records = dataset_to_records(ds)
audit_df = pd.DataFrame(records_as_dicts(records))

phi_df = audit_df[audit_df["phi_risk"]]
private_df = audit_df[audit_df["is_private"]]

tab_all, tab_phi, tab_private = st.tabs(["All tags", "PHI-risk tags", "Private tags"])
with tab_all:
    st.dataframe(audit_df, use_container_width=True, hide_index=True)
with tab_phi:
    st.dataframe(phi_df, use_container_width=True, hide_index=True)
with tab_private:
    st.dataframe(private_df, use_container_width=True, hide_index=True)

_download_buttons(audit_df, selected_file.path)
