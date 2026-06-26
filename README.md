# pas-mri-imaging

## Local DICOM MRI viewer and metadata audit

Install dependencies from `requirements.txt`, then run the local Streamlit viewer:

```bash
streamlit run app/dicom_viewer.py
```

The viewer accepts a root folder or a single DICOM file, scans recursively
including files without extensions, groups data by `PatientID` / Study / Series,
and shows MRI slices plus metadata audit tables. The main navigation is organized
for MRI review: Study summary, clinically named series, AX/COR/SAG/OBLIQUE
orientation, slice counts, and geometry quality fields for future NIfTI
conversion. Use Clinical View to hide UIDs and Technical View to inspect DICOM
UIDs and detailed metadata. `PatientID` may be PHI; keep all exports local and
do not publish reports containing real identifiers.

## DICOM de-identification

Dry-run without writing output:

```bash
python -m scripts.deidentify_dicom --input data/raw_dicom --output data/anonymized_dicom --mapping-out data/anonymized_dicom/mapping.json --dry-run
```

Full local run:

```bash
python -m scripts.deidentify_dicom --input data/raw_dicom --output data/anonymized_dicom --mapping-out data/anonymized_dicom/mapping.json
```

The de-identification command never overwrites originals. It writes new DICOM
files under the output directory, replaces Study/Series/SOP UIDs while preserving
internal consistency, removes private tags by default, clears `PatientAge` unless
`--keep-patient-age` is used, and prints a JSON report without PHI values. Pixel
data is not OCR-inspected or redacted; files with `BurnedInAnnotation = YES` or
missing/unknown are marked for manual review.
