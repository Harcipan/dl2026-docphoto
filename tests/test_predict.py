"""predict.py has to give the numbers the training loop gives.

A resize, a JPEG round trip or a standardisation that differs between the two
is invisible in the validation score and costs you on the leaderboard. This
trains one epoch, then predicts twenty validation photographs both ways: the
loop from its pixel table, predict.py one file at a time from the originals.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from src import config, predict
from src import train as trainer
from src.data.dataset import cache_dir, load_table, pixel_table
from src.data.splits import build_splits

pytestmark = pytest.mark.skipif(not cache_dir(768).exists(), reason="needs the data and the 768 px cache")


def test_predict_matches_the_training_loop(monkeypatch) -> None:
    monkeypatch.setenv("WANDB_MODE", "disabled")
    cfg = yaml.safe_load(open(config.ROOT / "configs" / "lab3_mlp.yaml"))
    cfg["name"], cfg["train"]["epochs"] = "test_predict", 1
    df = load_table()
    X = pixel_table(df, cfg["data"]["pixel_size"])
    train_idx, val_idx = build_splits(df, cfg["split"]["group_column"], cfg["split"]["val_fraction"],
                                      cfg["split"]["seed"])
    run = trainer.train_mlp(cfg, X, df, train_idx, val_idx, log=lambda *a: None)
    try:
        model = predict.load_model(run["checkpoint"], cfg)
        rows = np.linspace(0, len(val_idx) - 1, 20).astype(int)
        paths = [Path(df["path"].iloc[val_idx[r]]) for r in rows]
        for path, r in zip(paths, rows, strict=True):
            np.testing.assert_array_equal(predict.preprocess(path, cfg), X[val_idx[r]],
                                          err_msg=f"{path.name}: the pixels differ from the pixel table")
        ours = np.array([predict.predict_one(model, p, cfg) for p in paths])
        np.testing.assert_allclose(ours, run["val_pred"][rows], atol=1e-5)
    finally:
        run["checkpoint"].unlink()
