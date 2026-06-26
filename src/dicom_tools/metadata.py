"""DICOM metadata extraction and PHI-risk classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pydicom
from pydicom.datadict import keyword_for_tag
from pydicom.dataset import Dataset
from pydicom.tag import BaseTag, Tag


PHI_RISK_KEYWORDS: set[str] = {
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientBirthTime",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "PatientComments",
    "OtherPatientIDs",
    "OtherPatientNames",
    "OtherPatientIDsSequence",
    "AccessionNumber",
    "InstitutionName",
    "InstitutionAddress",
    "InstitutionalDepartmentName",
    "ReferringPhysicianName",
    "PerformingPhysicianName",
    "OperatorsName",
    "PhysiciansOfRecord",
    "RequestingPhysician",
    "StudyID",
    "StudyDescription",
    "SeriesDescription",
    "ProtocolName",
    "ImageComments",
    "AdditionalPatientHistory",
    "AdmittingDiagnosesDescription",
    "DerivationDescription",
    "AcquisitionComments",
}

COMMENT_OR_FREE_TEXT_KEYWORDS: set[str] = {
    "Comments",
    "ImageComments",
    "PatientComments",
    "StudyComments",
    "VisitComments",
    "ResultsComments",
    "InterpretationText",
    "TextValue",
    "ContentSequence",
}

VIEWER_METADATA_KEYWORDS: tuple[str, ...] = (
    "Modality",
    "StudyDescription",
    "SeriesDescription",
    "StudyDate",
    "SeriesNumber",
    "InstanceNumber",
    "Rows",
    "Columns",
    "PixelSpacing",
    "SliceThickness",
    "ImageOrientationPatient",
    "ImagePositionPatient",
    "Manufacturer",
    "MagneticFieldStrength",
    "SequenceName",
    "ProtocolName",
    "PhotometricInterpretation",
    "BurnedInAnnotation",
)


@dataclass(frozen=True)
class MetadataRecord:
    tag: str
    keyword: str
    name: str
    vr: str
    value: str
    is_private: bool
    phi_risk: bool


def read_metadata(path: str | Path, *, stop_before_pixels: bool = True) -> Dataset:
    """Read a local DICOM dataset."""

    return pydicom.dcmread(Path(path), stop_before_pixels=stop_before_pixels, force=True)


def is_phi_risk_keyword(keyword: str) -> bool:
    """Return True for standard tags that can directly or indirectly carry PHI."""

    return keyword in PHI_RISK_KEYWORDS or any(marker in keyword for marker in COMMENT_OR_FREE_TEXT_KEYWORDS)


def is_phi_risk_element(tag: BaseTag, keyword: str, vr: str) -> bool:
    """Classify one element as PHI-risk without inspecting its actual value."""

    if Tag(tag).is_private:
        return True
    if is_phi_risk_keyword(keyword):
        return True
    if vr == "PN":
        return True
    return False


def _format_tag(tag: BaseTag) -> str:
    parsed = Tag(tag)
    return f"({parsed.group:04X},{parsed.element:04X})"


def _value_to_string(value: Any, *, max_length: int = 240) -> str:
    """Make a UI-safe compact representation of a DICOM value."""

    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    if isinstance(value, (list, tuple)):
        rendered = ", ".join(_value_to_string(v, max_length=60) for v in value[:8])
    else:
        rendered = str(value)
    if len(rendered) > max_length:
        return rendered[: max_length - 3] + "..."
    return rendered


def dataset_to_records(ds: Dataset, *, include_pixel_data: bool = False) -> list[MetadataRecord]:
    """Flatten a DICOM dataset into records for audit tables.

    Sequence items are summarized instead of recursively expanded so exports stay
    manageable and do not accidentally dump large nested payloads.
    """

    rows: list[MetadataRecord] = []
    for elem in ds.iterall():
        if elem.keyword == "PixelData" and not include_pixel_data:
            continue
        keyword = elem.keyword or keyword_for_tag(elem.tag) or ""
        if elem.VR == "SQ":
            value = f"<Sequence with {len(elem.value)} item(s)>"
        else:
            value = _value_to_string(elem.value)
        rows.append(
            MetadataRecord(
                tag=_format_tag(elem.tag),
                keyword=keyword,
                name=elem.name,
                vr=elem.VR,
                value=value,
                is_private=Tag(elem.tag).is_private,
                phi_risk=is_phi_risk_element(elem.tag, keyword, elem.VR),
            )
        )
    return rows


def records_as_dicts(records: list[MetadataRecord]) -> list[dict[str, object]]:
    return [record.__dict__.copy() for record in records]


def basic_metadata(ds: Dataset) -> dict[str, str]:
    """Return the MRI viewer's core metadata panel."""

    result: dict[str, str] = {}
    for keyword in VIEWER_METADATA_KEYWORDS:
        result[keyword] = _value_to_string(getattr(ds, keyword, ""))
    return result
