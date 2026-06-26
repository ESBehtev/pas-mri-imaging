from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from dicom_tools.discovery import (
    clinical_series_rows,
    discover_dicom_files,
    group_by_study_series,
    infer_orientation,
    sorted_series,
)


def make_test_dicom(
    path: Path,
    *,
    patient_id: str = "PATIENT-1",
    study_uid: str | None = None,
    series_uid: str | None = None,
    sop_uid: str | None = None,
    instance_number: int = 1,
    series_number: str = "3",
    series_description: str = "T1",
    image_orientation_patient: list[float] | None = None,
    slice_thickness: str = "4",
) -> None:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = MRImageStorage
    file_meta.MediaStorageSOPInstanceUID = sop_uid or generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = MRImageStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = study_uid or generate_uid()
    ds.SeriesInstanceUID = series_uid or generate_uid()
    ds.PatientName = "Doe^Jane"
    ds.PatientID = patient_id
    ds.Modality = "MR"
    ds.StudyDescription = "Brain MRI"
    ds.SeriesDescription = series_description
    ds.StudyDate = "20260101"
    ds.SeriesNumber = series_number
    ds.InstanceNumber = instance_number
    ds.SliceThickness = slice_thickness
    ds.PixelSpacing = [0.7, 0.7]
    ds.ImageOrientationPatient = image_orientation_patient or [1, 0, 0, 0, 1, 0]
    ds.Manufacturer = "Test MRI Vendor"
    ds.MagneticFieldStrength = "1.5"
    ds.Rows = 2
    ds.Columns = 2
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = np.arange(4, dtype=np.uint16).tobytes()
    ds.save_as(path, write_like_original=False)


def test_discover_dicom_files_includes_files_without_extension(tmp_path: Path) -> None:
    nested = tmp_path / "study" / "series"
    nested.mkdir(parents=True)
    dicom_path = nested / "IM0001"
    make_test_dicom(dicom_path)
    (nested / "notes.txt").write_text("not dicom", encoding="utf-8")

    files = discover_dicom_files(tmp_path)

    assert len(files) == 1
    assert files[0].path == dicom_path
    assert files[0].patient_id == "PATIENT-1"
    assert files[0].modality == "MR"


def test_group_by_study_series(tmp_path: Path) -> None:
    dicom_path = tmp_path / "image.dcm"
    make_test_dicom(dicom_path, instance_number=7)
    files = discover_dicom_files(tmp_path)

    grouped = group_by_study_series(files)

    assert len(grouped) == 1
    study = next(iter(grouped.values()))
    assert len(study.series) == 1
    series = next(iter(study.series.values()))
    assert series.files[0].instance_number == 7


def test_grouping_uses_one_series_per_series_uid_and_deduplicates_sop(tmp_path: Path) -> None:
    study_uid = generate_uid()
    series_uid = generate_uid()
    duplicate_sop_uid = generate_uid()
    make_test_dicom(
        tmp_path / "image1.dcm",
        study_uid=study_uid,
        series_uid=series_uid,
        sop_uid=duplicate_sop_uid,
        instance_number=1,
        series_description="AX T2 FSE",
    )
    make_test_dicom(
        tmp_path / "image1_copy.dcm",
        study_uid=study_uid,
        series_uid=series_uid,
        sop_uid=duplicate_sop_uid,
        instance_number=1,
        series_description="AX T2 FSE",
    )
    make_test_dicom(
        tmp_path / "image2.dcm",
        study_uid=study_uid,
        series_uid=series_uid,
        instance_number=2,
        series_description="AX T2 FSE",
    )

    grouped = group_by_study_series(discover_dicom_files(tmp_path))
    study = next(iter(grouped.values()))
    series = next(iter(study.series.values()))

    assert study.series_count == 1
    assert study.file_count == 3
    assert study.total_slices == 2
    assert series.instance_count == 2
    assert clinical_series_rows(study)[0]["Series"] == "AX T2 FSE"
    assert clinical_series_rows(study)[0]["Slices"] == 2


def test_orientation_inference() -> None:
    assert infer_orientation((1, 0, 0, 0, 1, 0)) == "AX"
    assert infer_orientation((1, 0, 0, 0, 0, 1)) == "COR"
    assert infer_orientation((0, 1, 0, 0, 0, 1)) == "SAG"
    assert infer_orientation((1, 0, 0, 0, 0.7, 0.7)) == "OBLIQUE"
    assert infer_orientation(None) == "UNKNOWN"


def test_mri_series_sorting(tmp_path: Path) -> None:
    study_uid = generate_uid()
    make_test_dicom(
        tmp_path / "sag.dcm",
        study_uid=study_uid,
        series_uid=generate_uid(),
        series_number="3",
        series_description="SAG T2",
        image_orientation_patient=[0, 1, 0, 0, 0, 1],
    )
    make_test_dicom(
        tmp_path / "loc.dcm",
        study_uid=study_uid,
        series_uid=generate_uid(),
        series_number="9",
        series_description="Localizer",
    )
    make_test_dicom(
        tmp_path / "ax.dcm",
        study_uid=study_uid,
        series_uid=generate_uid(),
        series_number="2",
        series_description="AX T2",
        image_orientation_patient=[1, 0, 0, 0, 1, 0],
    )
    make_test_dicom(
        tmp_path / "cor.dcm",
        study_uid=study_uid,
        series_uid=generate_uid(),
        series_number="1",
        series_description="COR T2",
        image_orientation_patient=[1, 0, 0, 0, 0, 1],
    )

    study = next(iter(group_by_study_series(discover_dicom_files(tmp_path)).values()))
    labels = [series.first.series_description for series in sorted_series(study)]

    assert labels == ["Localizer", "AX T2", "COR T2", "SAG T2"]
