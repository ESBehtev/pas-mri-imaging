"""Local DICOM discovery and grouping utilities.

The functions in this module read DICOM headers only by default and never log
patient-identifying values. UI layers may display identifiers with an explicit
PHI warning when the user is working locally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pydicom
from pydicom.dataset import Dataset
from pydicom.errors import InvalidDicomError


@dataclass(frozen=True)
class DicomFile:
    """A DICOM file discovered under a root directory."""

    path: Path
    patient_id: str
    study_instance_uid: str
    series_instance_uid: str
    sop_instance_uid: str
    modality: str
    study_description: str
    series_description: str
    study_date: str
    series_number: str
    instance_number: int | None
    rows: int | None
    columns: int | None
    pixel_spacing: str
    slice_thickness: str
    image_orientation_patient: tuple[float, ...] | None
    image_position_patient: str
    manufacturer: str
    magnetic_field_strength: str
    sequence_name: str
    protocol_name: str


@dataclass
class SeriesGroup:
    """Files that belong to one DICOM Series."""

    patient_id: str
    study_instance_uid: str
    series_instance_uid: str
    files: list[DicomFile] = field(default_factory=list)
    source_file_count: int = 0

    def add_file(self, item: DicomFile) -> None:
        """Add one SOP instance, keeping duplicate SOPInstanceUID files out of UI lists."""

        self.source_file_count += 1
        if item.sop_instance_uid:
            existing = {file.sop_instance_uid for file in self.files if file.sop_instance_uid}
            if item.sop_instance_uid in existing:
                return
        self.files.append(item)

    @property
    def first(self) -> DicomFile:
        return self.files[0]

    @property
    def instance_count(self) -> int:
        return len(self.files)

    @property
    def orientation(self) -> str:
        return infer_orientation(self.first.image_orientation_patient)


@dataclass
class StudyGroup:
    """Series grouped under one DICOM Study."""

    patient_id: str
    study_instance_uid: str
    series: dict[str, SeriesGroup] = field(default_factory=dict)

    @property
    def series_count(self) -> int:
        return len(self.series)

    @property
    def file_count(self) -> int:
        return sum(series.source_file_count for series in self.series.values())

    @property
    def total_slices(self) -> int:
        return sum(series.instance_count for series in self.series.values())

    @property
    def first(self) -> DicomFile:
        return next(iter(self.series.values())).first


def iter_candidate_files(root: str | Path) -> Iterable[Path]:
    """Yield regular files under *root*, including files without extensions."""

    root_path = Path(root).expanduser()
    if root_path.is_file():
        yield root_path
        return
    if not root_path.exists():
        return
    for path in root_path.rglob("*"):
        if path.is_file():
            yield path


def read_dicom_header(path: str | Path, *, force: bool = False) -> Dataset | None:
    """Read a DICOM header without pixels.

    With ``force=False`` pydicom validates the DICOM preamble. Some valid files
    in research exports are missing it, so discovery retries with ``force=True``.
    """

    dicom_path = Path(path)
    try:
        return pydicom.dcmread(
            dicom_path,
            stop_before_pixels=True,
            specific_tags=None,
            force=force,
        )
    except (InvalidDicomError, OSError, EOFError, AttributeError, ValueError):
        return None


def is_probable_dicom(ds: Dataset | None) -> bool:
    """Return True when a header has enough DICOM identity to be useful."""

    if ds is None:
        return False
    return any(
        getattr(ds, attr, None)
        for attr in (
            "SOPClassUID",
            "SOPInstanceUID",
            "StudyInstanceUID",
            "SeriesInstanceUID",
            "Modality",
        )
    )


def _safe_str(value: object, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _safe_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _safe_float_tuple(value: object) -> tuple[float, ...] | None:
    if value is None:
        return None
    try:
        return tuple(float(item) for item in value)  # type: ignore[union-attr]
    except (TypeError, ValueError):
        return None


def dicom_file_from_dataset(path: Path, ds: Dataset) -> DicomFile:
    """Create a lightweight discovery record from a DICOM header."""

    return DicomFile(
        path=path,
        patient_id=_safe_str(getattr(ds, "PatientID", ""), "UNKNOWN"),
        study_instance_uid=_safe_str(getattr(ds, "StudyInstanceUID", ""), "UNKNOWN_STUDY"),
        series_instance_uid=_safe_str(getattr(ds, "SeriesInstanceUID", ""), "UNKNOWN_SERIES"),
        sop_instance_uid=_safe_str(getattr(ds, "SOPInstanceUID", ""), ""),
        modality=_safe_str(getattr(ds, "Modality", ""), ""),
        study_description=_safe_str(getattr(ds, "StudyDescription", ""), ""),
        series_description=_safe_str(getattr(ds, "SeriesDescription", ""), ""),
        study_date=_safe_str(getattr(ds, "StudyDate", ""), ""),
        series_number=_safe_str(getattr(ds, "SeriesNumber", ""), ""),
        instance_number=_safe_int(getattr(ds, "InstanceNumber", None)),
        rows=_safe_int(getattr(ds, "Rows", None)),
        columns=_safe_int(getattr(ds, "Columns", None)),
        pixel_spacing=_safe_str(getattr(ds, "PixelSpacing", ""), ""),
        slice_thickness=_safe_str(getattr(ds, "SliceThickness", ""), ""),
        image_orientation_patient=_safe_float_tuple(getattr(ds, "ImageOrientationPatient", None)),
        image_position_patient=_safe_str(getattr(ds, "ImagePositionPatient", ""), ""),
        manufacturer=_safe_str(getattr(ds, "Manufacturer", ""), ""),
        magnetic_field_strength=_safe_str(getattr(ds, "MagneticFieldStrength", ""), ""),
        sequence_name=_safe_str(getattr(ds, "SequenceName", ""), ""),
        protocol_name=_safe_str(getattr(ds, "ProtocolName", ""), ""),
    )


def discover_dicom_files(root: str | Path) -> list[DicomFile]:
    """Recursively find DICOM files under *root*.

    Files without extensions are included. Pixel data is not read.
    """

    discovered: list[DicomFile] = []
    for path in iter_candidate_files(root):
        ds = read_dicom_header(path, force=False)
        if not is_probable_dicom(ds):
            ds = read_dicom_header(path, force=True)
        if not is_probable_dicom(ds) or ds is None:
            continue
        discovered.append(dicom_file_from_dataset(path, ds))
    return discovered


def group_by_study_series(files: Iterable[DicomFile]) -> dict[tuple[str, str], StudyGroup]:
    """Group discovery records by PatientID and StudyInstanceUID."""

    studies: dict[tuple[str, str], StudyGroup] = {}
    for item in files:
        study_key = (item.patient_id, item.study_instance_uid)
        study = studies.setdefault(
            study_key,
            StudyGroup(patient_id=item.patient_id, study_instance_uid=item.study_instance_uid),
        )
        series = study.series.setdefault(
            item.series_instance_uid,
            SeriesGroup(
                patient_id=item.patient_id,
                study_instance_uid=item.study_instance_uid,
                series_instance_uid=item.series_instance_uid,
            ),
        )
        series.add_file(item)

    for study in studies.values():
        for series in study.series.values():
            series.files.sort(key=lambda f: (f.instance_number is None, f.instance_number or 0, str(f.path)))
    return studies


def infer_orientation(image_orientation_patient: tuple[float, ...] | None) -> str:
    """Infer MRI slice orientation from DICOM ImageOrientationPatient.

    DICOM stores row and column direction cosines. Their cross product is the
    slice normal. The dominant normal axis maps to SAG/COR/AX; mixed normals are
    treated as oblique.
    """

    if image_orientation_patient is None or len(image_orientation_patient) != 6:
        return "UNKNOWN"
    row = np.asarray(image_orientation_patient[:3], dtype=float)
    col = np.asarray(image_orientation_patient[3:], dtype=float)
    if np.linalg.norm(row) == 0 or np.linalg.norm(col) == 0:
        return "UNKNOWN"
    normal = np.cross(row, col)
    norm = np.linalg.norm(normal)
    if norm == 0:
        return "UNKNOWN"
    normal = np.abs(normal / norm)
    dominant_axis = int(np.argmax(normal))
    if normal[dominant_axis] < 0.85:
        return "OBLIQUE"
    return ("SAG", "COR", "AX")[dominant_axis]


def is_localizer_series(series: SeriesGroup) -> bool:
    text = " ".join(
        (
            series.first.series_description,
            series.first.protocol_name,
            series.first.sequence_name,
        )
    ).lower()
    return any(marker in text for marker in ("localizer", "localiser", "scout", "survey", "3-plane", "3 plane"))


def _series_number_value(series: SeriesGroup) -> int:
    try:
        return int(float(series.first.series_number))
    except (TypeError, ValueError):
        return 1_000_000


def mri_series_sort_key(series: SeriesGroup) -> tuple[int, int, str]:
    """Sort MRI series as Localizer/Scout, AX, COR, SAG, then the rest."""

    if is_localizer_series(series):
        group = 0
    else:
        group = {"AX": 1, "COR": 2, "SAG": 3}.get(series.orientation, 4)
    return (group, _series_number_value(series), series.first.series_description.lower())


def sorted_series(study: StudyGroup) -> list[SeriesGroup]:
    return sorted(study.series.values(), key=mri_series_sort_key)


def series_display_name(series: SeriesGroup) -> str:
    first = series.first
    return first.series_description or first.protocol_name or first.sequence_name or "Unnamed series"


def series_summary_rows(files: Iterable[DicomFile]) -> list[dict[str, object]]:
    """Return one metadata row per series for UI display."""

    grouped = group_by_study_series(files)
    rows: list[dict[str, object]] = []
    for study in grouped.values():
        for series in sorted_series(study):
            first = series.first
            rows.append(
                {
                    "PatientID (may be PHI)": study.patient_id,
                    "StudyInstanceUID": study.study_instance_uid,
                    "SeriesInstanceUID": series.series_instance_uid,
                    "Modality": first.modality,
                    "StudyDescription": first.study_description,
                    "SeriesDescription": first.series_description,
                    "StudyDate": first.study_date,
                    "SeriesNumber": first.series_number,
                    "Instances": series.instance_count,
                    "Orientation": series.orientation,
                    "SliceThickness": first.slice_thickness,
                }
            )
    return rows


def study_summary_rows(studies: dict[tuple[str, str], StudyGroup]) -> list[dict[str, object]]:
    """Return one clinical row per study without UIDs."""

    rows: list[dict[str, object]] = []
    for study in studies.values():
        first = study.first
        rows.append(
            {
                "Study Date": first.study_date or "UNKNOWN",
                "Study Description": first.study_description or "No StudyDescription",
                "Series Count": study.series_count,
                "Files": study.file_count,
            }
        )
    return rows


def study_summary(study: StudyGroup) -> dict[str, object]:
    """Return the study-level summary shown above the series list."""

    first = study.first
    manufacturers = sorted({series.first.manufacturer for series in study.series.values() if series.first.manufacturer})
    field_strengths = sorted(
        {series.first.magnetic_field_strength for series in study.series.values() if series.first.magnetic_field_strength}
    )
    return {
        "Study description": first.study_description or "No StudyDescription",
        "Study date": first.study_date or "UNKNOWN",
        "Series count": study.series_count,
        "Total slices": study.total_slices,
        "Manufacturer": ", ".join(manufacturers) if manufacturers else "",
        "Magnetic field strength": ", ".join(field_strengths) if field_strengths else "",
    }


def clinical_series_rows(study: StudyGroup) -> list[dict[str, object]]:
    """Return MRI-first series rows without UIDs."""

    rows: list[dict[str, object]] = []
    for series in sorted_series(study):
        first = series.first
        rows.append(
            {
                "Series": series_display_name(series),
                "Orientation": series.orientation,
                "Slices": series.instance_count,
                "Series Number": first.series_number,
                "Modality": first.modality,
                "Slice Thickness": first.slice_thickness,
            }
        )
    return rows


def mri_quality_rows(study: StudyGroup, *, include_uids: bool = False) -> list[dict[str, object]]:
    """Return per-series geometry and resolution summary."""

    rows: list[dict[str, object]] = []
    for series in sorted_series(study):
        first = series.first
        row: dict[str, object] = {
            "Series": series_display_name(series),
            "Orientation": series.orientation,
            "Rows x Columns": f"{first.rows or '?'} x {first.columns or '?'}",
            "PixelSpacing": first.pixel_spacing,
            "SliceThickness": first.slice_thickness,
            "Slices": series.instance_count,
        }
        if include_uids:
            row["StudyInstanceUID"] = first.study_instance_uid
            row["SeriesInstanceUID"] = first.series_instance_uid
        rows.append(row)
    return rows
