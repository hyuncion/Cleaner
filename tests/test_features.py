from pathlib import Path

from PIL import Image

from cleaner.config import FEATURE_NAMES
from cleaner.features import extract_basic_features
from cleaner.utils import scan_photos


def test_extract_basic_features_shape(tmp_path: Path):
    path = tmp_path / "sample.jpg"
    Image.new("RGB", (64, 48), (128, 128, 128)).save(path)

    features = extract_basic_features(str(path))
    assert features.shape == (len(FEATURE_NAMES),)


def test_scan_photos(tmp_path: Path):
    Image.new("RGB", (32, 32)).save(tmp_path / "a.jpg")
    (tmp_path / "notes.txt").write_text("not an image")

    paths = scan_photos(tmp_path)
    assert len(paths) == 1
    assert paths[0].endswith("a.jpg")
