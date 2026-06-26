"""Pixel-data helpers for a local DICOM viewer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset
from pydicom.pixels import apply_modality_lut, apply_voi_lut


@dataclass(frozen=True)
class PixelReadResult:
    image: np.ndarray | None
    error: str | None
    photometric_interpretation: str


def read_dataset_with_pixels(path: str | Path) -> Dataset:
    """Read a full local DICOM dataset for pixel inspection."""

    return pydicom.dcmread(Path(path), force=True)


def read_pixel_array(path: str | Path) -> PixelReadResult:
    """Read one DICOM pixel array with graceful failure for unsupported codecs."""

    try:
        ds = read_dataset_with_pixels(path)
    except Exception as exc:  # pydicom raises several concrete parser errors.
        return PixelReadResult(None, f"Could not read DICOM file: {exc}", "")

    photometric = str(getattr(ds, "PhotometricInterpretation", ""))
    try:
        arr = ds.pixel_array
        arr = apply_modality_lut(arr, ds)
        try:
            arr = apply_voi_lut(arr, ds)
        except Exception:
            pass
        image = normalize_for_display(np.asarray(arr), photometric_interpretation=photometric)
        return PixelReadResult(image=image, error=None, photometric_interpretation=photometric)
    except Exception as exc:
        transfer_syntax = getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", "unknown")
        return PixelReadResult(
            image=None,
            error=(
                "Pixel data could not be decoded locally. "
                f"TransferSyntaxUID={transfer_syntax}; error={exc}"
            ),
            photometric_interpretation=photometric,
        )


def normalize_for_display(arr: np.ndarray, *, photometric_interpretation: str = "") -> np.ndarray:
    """Return a uint8 MONOCHROME image suitable for matplotlib/Streamlit."""

    image = np.asarray(arr)
    if image.ndim == 3:
        image = image[0]
    image = image.astype(np.float64, copy=False)
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros(image.shape, dtype=np.uint8)
    lo, hi = np.percentile(image[finite], [1, 99])
    if hi <= lo:
        lo = float(np.min(image[finite]))
        hi = float(np.max(image[finite]))
    if hi <= lo:
        scaled = np.zeros(image.shape, dtype=np.uint8)
    else:
        scaled_float = np.clip((image - lo) / (hi - lo), 0, 1) * 255
        scaled = scaled_float.astype(np.uint8)
    if photometric_interpretation.upper() == "MONOCHROME1":
        scaled = np.uint8(255) - scaled
    return scaled
