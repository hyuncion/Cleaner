from pathlib import Path

import pandas as pd
from PIL import Image

from cleaner.config import AppConfig
from cleaner.labeling import export_label_template, import_label_csv, record_action
from cleaner.storage import load_labels


def test_record_action_maps_swipes(tmp_path: Path):
    photo = tmp_path / "a.jpg"
    Image.new("RGB", (16, 16)).save(photo)
    cfg = AppConfig(data_dir=tmp_path / "data")

    record_action(cfg, photo, "important")
    labels = load_labels(cfg)

    assert len(labels) == 1
    assert int(labels.iloc[0]["label"]) == 1
    assert labels.iloc[0]["action"] == "important"
    assert float(labels.iloc[0]["weight"]) == 2.0


def test_export_and_import_label_csv(tmp_path: Path):
    for name in ["a.jpg", "b.jpg"]:
        Image.new("RGB", (16, 16)).save(tmp_path / name)
    cfg = AppConfig(data_dir=tmp_path / "data")

    template_path = export_label_template(cfg, tmp_path, max_items=10)
    df = pd.read_csv(template_path)
    df.loc[0, "label"] = 1
    df.loc[1, "label"] = 0
    df.to_csv(template_path, index=False)

    result = import_label_csv(cfg, template_path)

    assert result["imported"] == 2
    labels = load_labels(cfg)
    assert set(labels["label"].astype(int)) == {0, 1}
