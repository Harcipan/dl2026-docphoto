#!/usr/bin/env python3
"""The gradient-boosting baseline with the lab's seven features against seven plus edge_sharpness.

    docker compose run --rm dev python scripts/baseline_features.py

Three numbers per feature set, all on the scoreable rows:

* the validation split of the lab (strict unit 4), the number on the board;
* grouped 6-fold: every strict unit held out once, out-of-fold predictions
  for every photograph, so the score rests on the whole dataset;
* a paired bootstrap over pages of those out-of-fold predictions: how often
  the eight-feature model wins when the pages are resampled.

The feature was designed by looking at the worst out-of-fold errors on the
five training units only, so the validation unit had no say in it.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src import config, evaluate
from src.data.dataset import load_table
from src.data.splits import build_splits
from src.features import handcrafted
from src.models import baseline


def out_of_fold(F: pd.DataFrame, df: pd.DataFrame) -> np.ndarray:
    pred = np.full(len(df), np.nan)
    units = df["group_strict"].to_numpy()
    for u in np.unique(units):
        hold = units == u
        model = baseline.fit_baseline(F[~hold], df.loc[~hold, config.TARGET].to_numpy())
        pred[hold] = baseline.predict_baseline(model, F[hold])
    return pred


def main() -> None:
    df = load_table()
    y, ok = df[config.TARGET].to_numpy(), (df["eval_ok"] == 1).to_numpy()
    train_idx, val_idx = build_splits(df, "group_strict", 0.2, config.SPLIT_SEED)
    val_ok = ok[val_idx]

    t0 = time.time()
    F8 = handcrafted.feature_table(df, native=True)
    print(f"features in {time.time() - t0:.0f} s")
    sets = {"7 features": F8[list(handcrafted.FEATURES)], "7 + edge_sharpness": F8}

    rows, oof = {}, {}
    for name, F in sets.items():
        model = baseline.fit_baseline(F.iloc[train_idx], y[train_idx])
        val_pred = baseline.predict_baseline(model, F.iloc[val_idx])
        oof[name] = out_of_fold(F, df)
        rows[name] = {"val RMSE (unit 4)": evaluate.rmse(y[val_idx][val_ok], val_pred[val_ok]),
                      "grouped 6-fold RMSE": evaluate.rmse(y[ok], oof[name][ok])}
    print(pd.DataFrame(rows).T.round(4).to_string())

    d = df[ok].assign(a=oof["7 features"][ok], b=oof["7 + edge_sharpness"][ok])
    print("\ngrouped 6-fold RMSE by family")
    print(pd.DataFrame({"7 features": evaluate.by_slice(d, d["a"].to_numpy(), "family")["rmse"],
                        "7 + edge_sharpness": evaluate.by_slice(d, d["b"].to_numpy(), "family")["rmse"]})
          .round(4).to_string())

    # Photographs of one page are not independent: resample pages, keep their squared errors together.
    sq = (d.assign(a=(d["a"] - d[config.TARGET]) ** 2, b=(d["b"] - d[config.TARGET]) ** 2, n=1)
          .groupby("page_id")[["a", "b", "n"]].sum())
    rng = np.random.default_rng(config.SEED)
    diffs = []
    for _ in range(2000):
        s = sq.iloc[rng.integers(0, len(sq), len(sq))].sum()
        diffs.append(np.sqrt(s["b"] / s["n"]) - np.sqrt(s["a"] / s["n"]))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"\npaired bootstrap over {len(sq)} pages, 2000 resamples: RMSE change {diffs.mean():+.4f} "
          f"(95 % {lo:+.4f} to {hi:+.4f}), edge_sharpness better in {np.mean(diffs < 0):.0%}")


if __name__ == "__main__":
    main()
