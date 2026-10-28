"""Download the Olist reviews CSV from Kaggle (free; needs KAGGLE_USERNAME/KAGGLE_KEY)."""

import hashlib
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_reviews(dataset: str, filename: str, out_dir: Path) -> Path:
    """Fetch one file of a Kaggle dataset into `out_dir`; no-op if it is already there."""
    target = out_dir / filename
    if target.exists():
        return target
    out_dir.mkdir(parents=True, exist_ok=True)
    import kaggle  # imported lazily: the package authenticates as soon as it is imported

    kaggle.api.dataset_download_file(dataset, filename, path=str(out_dir), quiet=True)
    archive = out_dir / f"{filename}.zip"
    if archive.exists():
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out_dir)
        archive.unlink()
    return target
