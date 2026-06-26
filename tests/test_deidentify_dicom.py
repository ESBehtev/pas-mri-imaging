from __future__ import annotations

from pathlib import Path

import pydicom

from dicom_tools.deidentify import DeidentifyConfig, deidentify_path, scan_phi_tags
from test_dicom_discovery import make_test_dicom


def test_deidentify_dry_run_writes_nothing(tmp_path: Path) -> None:
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "anon"
    input_dir.mkdir()
    make_test_dicom(input_dir / "image.dcm")

    stats, mapping = deidentify_path(
        DeidentifyConfig(
            input_path=input_dir,
            output_dir=output_dir,
            mapping_out=output_dir / "mapping.json",
            dry_run=True,
        )
    )

    assert stats.files_found == 1
    assert stats.dicom_read == 1
    assert stats.files_written == 0
    assert mapping
    assert not output_dir.exists()


def test_deidentify_full_run_removes_basic_phi_and_remaps_uids(tmp_path: Path) -> None:
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "anon"
    input_dir.mkdir()
    source = input_dir / "image.dcm"
    make_test_dicom(source)
    ds = pydicom.dcmread(source, force=True)
    ds.PatientBirthDate = "19700101"
    ds.PatientAge = "052Y"
    ds.PatientSex = "F"
    ds.AccessionNumber = "ACC-123"
    ds.InstitutionName = "Hospital Name"
    ds.ReferringPhysicianName = "Doctor^One"
    ds.OperatorsName = "Operator^One"
    ds.StudyID = "STUDY-123"
    ds.OtherPatientNames = "Alias^Jane"
    ds.ImageComments = "Free text with patient info"
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = [10, 20, 30]
    ds.PatientPosition = "HFS"
    ds.SliceLocation = "30"
    ds.PixelSpacing = [0.7, 0.7]
    ds.SliceThickness = "4"
    ds.SpacingBetweenSlices = "4.4"
    ds.SeriesDescription = "AX T2 FSE"
    ds.StudyDescription = "Brain MRI"
    ds.ProtocolName = "T2 Brain"
    ds.SequenceName = "tse"
    ds.ScanningSequence = "SE"
    ds.SequenceVariant = "SK"
    ds.ScanOptions = "FS"
    ds.MRAcquisitionType = "2D"
    ds.FrameOfReferenceUID = pydicom.uid.generate_uid()
    ds.add_new((0x0011, 0x0010), "LO", "Private Creator")
    ds.add_new((0x0011, 0x1001), "LO", "Private Value")
    original_study_uid = ds.StudyInstanceUID
    original_series_uid = ds.SeriesInstanceUID
    original_sop_uid = ds.SOPInstanceUID
    original_frame_uid = ds.FrameOfReferenceUID
    original_orientation = list(ds.ImageOrientationPatient)
    original_position = list(ds.ImagePositionPatient)
    ds.save_as(source, write_like_original=False)

    stats, mapping = deidentify_path(
        DeidentifyConfig(
            input_path=input_dir,
            output_dir=output_dir,
            mapping_out=output_dir / "mapping.json",
            dry_run=False,
        )
    )

    assert stats.files_written == 1
    assert (output_dir / "mapping.json").exists()
    written_files = [p for p in output_dir.rglob("*") if p.is_file() and p.name != "mapping.json"]
    assert len(written_files) == 1

    anon = pydicom.dcmread(written_files[0], force=True)
    assert str(anon.PatientName) == ""
    assert str(anon.PatientID).startswith("PAT-")
    assert str(getattr(anon, "PatientBirthDate", "")) == ""
    assert str(getattr(anon, "PatientAge", "")) == ""
    assert str(getattr(anon, "PatientSex", "")) == ""
    assert str(getattr(anon, "AccessionNumber", "")) == ""
    assert str(getattr(anon, "InstitutionName", "")) == ""
    assert str(getattr(anon, "ReferringPhysicianName", "")) == ""
    assert "OtherPatientNames" not in anon
    assert "ImageComments" not in anon or str(anon.ImageComments) == ""
    assert anon.PatientIdentityRemoved == "YES"
    assert anon.StudyInstanceUID != original_study_uid
    assert anon.SeriesInstanceUID != original_series_uid
    assert anon.SOPInstanceUID != original_sop_uid
    assert anon.FrameOfReferenceUID != original_frame_uid
    assert anon.file_meta.MediaStorageSOPInstanceUID == anon.SOPInstanceUID
    assert list(anon.ImageOrientationPatient) == original_orientation
    assert list(anon.ImagePositionPatient) == original_position
    assert anon.PatientPosition == "HFS"
    assert str(anon.SliceLocation) == "30"
    assert list(anon.PixelSpacing) == [0.7, 0.7]
    assert str(anon.SliceThickness) == "4"
    assert str(anon.SpacingBetweenSlices) == "4.4"
    assert anon.SeriesDescription == "AX T2 FSE"
    assert anon.StudyDescription == "Brain MRI"
    assert anon.ProtocolName == "T2 Brain"
    assert anon.SequenceName == "tse"
    assert anon.ScanningSequence == "SE"
    assert anon.SequenceVariant == "SK"
    assert anon.ScanOptions == "FS"
    assert anon.MRAcquisitionType == "2D"
    assert original_study_uid in mapping
    assert not [elem for elem in anon.iterall() if elem.tag.is_private]
    assert scan_phi_tags(output_dir) == []
