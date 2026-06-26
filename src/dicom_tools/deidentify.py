"""DICOM de-identification utilities.

This module implements a conservative local-only pydicom workflow inspired by
DICOM PS3.15 Basic Application Level Confidentiality Profile. It does not
inspect or redact burned-in pixel annotations; series with possible burned-in
text are marked for manual review.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pydicom
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from pydicom.tag import Tag
from pydicom.uid import generate_uid

from dicom_tools.discovery import is_probable_dicom, iter_candidate_files
from dicom_tools.metadata import PHI_RISK_KEYWORDS, is_phi_risk_element


LOGGER = logging.getLogger(__name__)

UID_KEYWORDS: tuple[str, ...] = (
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "FrameOfReferenceUID",
)

PRESERVED_CLINICAL_KEYWORDS: set[str] = {
    "ImageOrientationPatient",
    "ImagePositionPatient",
    "PatientPosition",
    "SliceLocation",
    "PixelSpacing",
    "SliceThickness",
    "SpacingBetweenSlices",
    "Rows",
    "Columns",
    "SeriesDescription",
    "StudyDescription",
    "ProtocolName",
    "SequenceName",
    "ScanningSequence",
    "SequenceVariant",
    "ScanOptions",
    "MRAcquisitionType",
}

CLEAR_KEYWORDS: set[str] = {
    "PatientName",
    "PatientBirthDate",
    "PatientBirthTime",
    "PatientAge",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "PatientComments",
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
    "ImageComments",
    "AdditionalPatientHistory",
    "AdmittingDiagnosesDescription",
    "DerivationDescription",
    "AcquisitionComments",
}

DELETE_KEYWORDS: set[str] = {
    "OtherPatientIDs",
    "OtherPatientNames",
    "OtherPatientIDsSequence",
}

TEXT_RISK_MARKERS: tuple[str, ...] = (
    "Comments",
    "Comment",
    "Physician",
    "Operator",
    "Institution",
    "Address",
    "Telephone",
)

SAFE_FILENAMES = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_"


@dataclass
class DeidentifyConfig:
    input_path: Path
    output_dir: Path
    mapping_out: Path
    dry_run: bool = False
    keep_patient_sex: bool = False
    keep_patient_age: bool = False
    remove_private_tags: bool = True


@dataclass
class DeidentifyStats:
    files_found: int = 0
    dicom_read: int = 0
    dicom_failed: int = 0
    files_written: int = 0
    tags_cleared: int = 0
    tags_deleted: int = 0
    uids_replaced: int = 0
    private_tags_removed: int = 0
    manual_review_files: int = 0
    changed_tag_types: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def note(self, tag_type: str) -> None:
        self.changed_tag_types[tag_type] = self.changed_tag_types.get(tag_type, 0) + 1

    def safe_report(self) -> dict[str, Any]:
        return {
            "files_found": self.files_found,
            "dicom_read": self.dicom_read,
            "dicom_failed": self.dicom_failed,
            "files_written": self.files_written,
            "tags_cleared": self.tags_cleared,
            "tags_deleted": self.tags_deleted,
            "uids_replaced": self.uids_replaced,
            "private_tags_removed": self.private_tags_removed,
            "manual_review_files": self.manual_review_files,
            "changed_tag_types": dict(sorted(self.changed_tag_types.items())),
            "warnings": self.warnings,
        }


@dataclass(frozen=True)
class PhiFinding:
    path: str
    tag: str
    keyword: str
    vr: str
    reason: str


def _is_empty(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, tuple, Sequence)):
        return len(value) == 0
    if str(value).strip() == "":
        return True
    return False


def _safe_component(value: str, fallback: str) -> str:
    cleaned = "".join(ch if ch in SAFE_FILENAMES else "_" for ch in value)
    cleaned = cleaned.strip("._")
    return cleaned[:80] or fallback


def _pseudo_patient_id(ds: Dataset) -> str:
    basis = str(
        getattr(ds, "PatientID", "")
        or getattr(ds, "StudyInstanceUID", "")
        or getattr(ds, "SOPInstanceUID", "")
        or "unknown"
    )
    digest = hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16].upper()
    return f"PAT-{digest}"


def _new_uid(old_uid: str, mapping: dict[str, str]) -> str:
    if not old_uid:
        return generate_uid()
    if old_uid not in mapping:
        mapping[old_uid] = generate_uid()
    return mapping[old_uid]


def _clear_element(ds: Dataset, keyword: str, stats: DeidentifyStats) -> None:
    if keyword in ds and not _is_empty(ds.get(keyword)):
        ds.data_element(keyword).value = ""
        stats.tags_cleared += 1
        stats.note(keyword)


def _delete_element(ds: Dataset, keyword: str, stats: DeidentifyStats) -> None:
    if keyword in ds:
        del ds[keyword]
        stats.tags_deleted += 1
        stats.note(keyword)


def _clear_person_names(ds: Dataset, stats: DeidentifyStats) -> None:
    for elem in list(ds.iterall()):
        if elem.VR == "PN" and not _is_empty(elem.value):
            elem.value = ""
            stats.tags_cleared += 1
            stats.note(elem.keyword or elem.name)


def _clear_risky_text(ds: Dataset, stats: DeidentifyStats) -> None:
    preserved_keywords = {"PatientID", "PatientIdentityRemoved", "DeidentificationMethod"}
    for elem in list(ds.iterall()):
        keyword = elem.keyword or ""
        if elem.VR == "SQ":
            continue
        if keyword in preserved_keywords:
            continue
        if keyword in PRESERVED_CLINICAL_KEYWORDS:
            continue
        if keyword in CLEAR_KEYWORDS:
            continue
        if any(marker in keyword for marker in TEXT_RISK_MARKERS) and not _is_empty(elem.value):
            elem.value = ""
            stats.tags_cleared += 1
            stats.note(keyword or elem.name)


def _replace_uids(ds: Dataset, mapping: dict[str, str], stats: DeidentifyStats) -> None:
    for keyword in UID_KEYWORDS:
        if keyword in ds and not _is_empty(ds.get(keyword)):
            old_uid = str(ds.get(keyword))
            ds.data_element(keyword).value = _new_uid(old_uid, mapping)
            stats.uids_replaced += 1
            stats.note(keyword)
    if getattr(ds, "file_meta", None) is not None and "MediaStorageSOPInstanceUID" in ds.file_meta:
        if getattr(ds, "SOPInstanceUID", ""):
            ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        else:
            sop_uid = str(ds.file_meta.MediaStorageSOPInstanceUID)
            ds.file_meta.MediaStorageSOPInstanceUID = _new_uid(sop_uid, mapping)
        stats.uids_replaced += 1
        stats.note("MediaStorageSOPInstanceUID")


def _check_burned_in(ds: Dataset, stats: DeidentifyStats) -> None:
    value = str(getattr(ds, "BurnedInAnnotation", "")).upper()
    if value != "NO":
        stats.manual_review_files += 1


def _save_dataset(ds: Dataset, output_path: Path) -> None:
    """Save with modern pydicom API and fall back for older versions."""

    try:
        ds.save_as(output_path, enforce_file_format=True)
    except TypeError:
        ds.save_as(output_path, write_like_original=False)


def deidentify_dataset(ds: Dataset, mapping: dict[str, str], stats: DeidentifyStats, config: DeidentifyConfig) -> Dataset:
    """Modify a dataset in place and return it."""

    _check_burned_in(ds, stats)

    ds.PatientName = ""
    stats.tags_cleared += 1
    stats.note("PatientName")
    ds.PatientID = _pseudo_patient_id(ds)
    stats.tags_cleared += 1
    stats.note("PatientID")

    for keyword in CLEAR_KEYWORDS:
        if keyword == "PatientAge" and config.keep_patient_age:
            continue
        _clear_element(ds, keyword, stats)
    for keyword in DELETE_KEYWORDS:
        _delete_element(ds, keyword, stats)
    if not config.keep_patient_sex:
        _clear_element(ds, "PatientSex", stats)

    _clear_person_names(ds, stats)
    _clear_risky_text(ds, stats)

    if config.remove_private_tags:
        before = len([elem for elem in ds.iterall() if Tag(elem.tag).is_private])
        ds.remove_private_tags()
        if before:
            stats.private_tags_removed += before
            stats.note("PrivateTags")

    _replace_uids(ds, mapping, stats)
    ds.PatientIdentityRemoved = "YES"
    ds.DeidentificationMethod = "pydicom-local-basic-profile"
    return ds


def output_path_for_dataset(ds: Dataset, source: Path, output_dir: Path, index: int) -> Path:
    """Create a non-PHI output path grouped by de-identified UID hierarchy."""

    study = _safe_component(str(getattr(ds, "StudyInstanceUID", "")), f"study_{index:06d}")
    series = _safe_component(str(getattr(ds, "SeriesInstanceUID", "")), "series")
    instance = _safe_component(str(getattr(ds, "SOPInstanceUID", "")), f"instance_{index:06d}")
    suffix = source.suffix if source.suffix else ".dcm"
    return output_dir / study / series / f"{instance}{suffix}"


def deidentify_path(config: DeidentifyConfig) -> tuple[DeidentifyStats, dict[str, str]]:
    """De-identify a DICOM file or directory tree."""

    stats = DeidentifyStats()
    mapping: dict[str, str] = {}
    candidates = list(iter_candidate_files(config.input_path))
    stats.files_found = len(candidates)

    for index, path in enumerate(candidates, start=1):
        try:
            ds = pydicom.dcmread(path, force=True)
        except Exception:
            stats.dicom_failed += 1
            continue
        if not any(getattr(ds, attr, None) for attr in ("SOPInstanceUID", "StudyInstanceUID", "SeriesInstanceUID")):
            stats.dicom_failed += 1
            continue
        stats.dicom_read += 1
        deidentify_dataset(ds, mapping, stats, config)
        if config.dry_run:
            continue

        out_path = output_path_for_dataset(ds, Path(path), config.output_dir, index)
        if out_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing file: {out_path}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _save_dataset(ds, out_path)
        stats.files_written += 1

    if stats.manual_review_files:
        stats.warnings.append(
            "BurnedInAnnotation was YES or absent/unknown for at least one file; manual pixel review is required."
        )

    if not config.dry_run:
        config.mapping_out.parent.mkdir(parents=True, exist_ok=True)
        if config.mapping_out.exists():
            raise FileExistsError(f"Refusing to overwrite mapping file: {config.mapping_out}")
        config.mapping_out.write_text(json.dumps(mapping, indent=2, sort_keys=True), encoding="utf-8")
    return stats, mapping


def scan_phi_tags(output_dir: str | Path) -> list[PhiFinding]:
    """Scan DICOM files for remaining risky non-empty tags without returning values."""

    findings: list[PhiFinding] = []
    for path in iter_candidate_files(output_dir):
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        except Exception:
            continue
        if not is_probable_dicom(ds):
            continue
        for elem in ds.iterall():
            keyword = elem.keyword or ""
            if keyword in {"PatientIdentityRemoved", "DeidentificationMethod"}:
                continue
            if keyword == "PatientID" and str(elem.value).startswith("PAT-"):
                continue
            if elem.keyword in {"StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "FrameOfReferenceUID"}:
                continue
            if keyword in PRESERVED_CLINICAL_KEYWORDS:
                continue
            if _is_empty(elem.value):
                continue
            if is_phi_risk_element(elem.tag, keyword, elem.VR) or keyword in PHI_RISK_KEYWORDS:
                findings.append(
                    PhiFinding(
                        path=str(Path(path)),
                        tag=f"({Tag(elem.tag).group:04X},{Tag(elem.tag).element:04X})",
                        keyword=keyword,
                        vr=elem.VR,
                        reason="non-empty PHI-risk/private/person-name tag",
                    )
                )
    return findings
