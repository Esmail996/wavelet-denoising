"""
Minimal parameters stub.
BOX_DATA_DIR and OUT_DIR are used as CLI defaults in wavelet_choice.py when run
directly. They are not used when wavelet_choice is imported as a library.
"""
from pathlib import Path

_HERE = Path(__file__).resolve().parent

BOX_DATA_DIR = _HERE / "Multifrequenz Dataset" / "Multifrequenz" / "Box"
OUT_DIR = _HERE / "outputs"
