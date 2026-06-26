"""CLI wrapper for local DICOM de-identification."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dicom_tools.deidentify import DeidentifyConfig, deidentify_path, scan_phi_tags


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="De-identify DICOM files locally without overwriting originals.")
    parser.add_argument("--input", required=True, type=Path, help="Input DICOM file or directory.")
    parser.add_argument("--output", required=True, type=Path, help="Output directory for de-identified DICOM files.")
    parser.add_argument("--mapping-out", required=True, type=Path, help="JSON path for old UID -> new UID mapping.")
    parser.add_argument("--dry-run", action="store_true", help="Analyze and report without writing files.")
    parser.add_argument(
        "--keep-patient-sex",
        action="store_true",
        help="Keep PatientSex for research workflows that explicitly require it.",
    )
    parser.add_argument(
        "--keep-patient-age",
        action="store_true",
        help="Keep PatientAge for research workflows that explicitly require it. Default is to clear it.",
    )
    parser.add_argument(
        "--keep-private-tags",
        action="store_true",
        help="Keep private tags. Default is to remove them.",
    )
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args()
    config = DeidentifyConfig(
        input_path=args.input,
        output_dir=args.output,
        mapping_out=args.mapping_out,
        dry_run=args.dry_run,
        keep_patient_sex=args.keep_patient_sex,
        keep_patient_age=args.keep_patient_age,
        remove_private_tags=not args.keep_private_tags,
    )
    stats, mapping = deidentify_path(config)
    report = stats.safe_report()
    if args.dry_run:
        report["dry_run"] = True
        report["uid_mappings_that_would_be_created"] = len(mapping)
    else:
        findings = scan_phi_tags(args.output)
        report["dry_run"] = False
        report["phi_scan_findings"] = [finding.__dict__ for finding in findings]
        report["phi_scan_findings_count"] = len(findings)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
