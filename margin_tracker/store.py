"""日ごとの全銘柄データを data/margin/daily/YYYYMMDD.csv.gz に保存し、銘柄ごとの推移を読み出す。

全銘柄を保存しておくので、あとから監視銘柄を増やしても過去の推移が見られる。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

DAILY_DIR = Path("data/margin/daily")

# 保存する列（内訳の前日比などは、前の日のデータとの差で出せるので保存しない）
KEEP = [
    "code", "buy", "buy_chg", "buy_ratio", "sell", "sell_chg", "sell_ratio",
    "buy_val", "sell_val", "buy_neg", "buy_std", "sell_neg", "sell_std",
]


def path_for(day: date, root: Path = DAILY_DIR) -> Path:
    return root / f"{day:%Y%m%d}.csv.gz"


def saved_days(root: Path = DAILY_DIR) -> list[date]:
    return sorted(pd.Timestamp(p.name[:8]).date() for p in root.glob("*.csv.gz"))


def save(day: date, df: pd.DataFrame, root: Path = DAILY_DIR) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    out = path_for(day, root)
    compact = df[[c for c in KEEP if c in df.columns]].copy()
    for col in compact.columns:
        if col == "code":
            continue
        if col.endswith("_ratio"):
            compact[col] = compact[col].round(4)
        else:
            compact[col] = compact[col].round().astype("Int64")  # 株数・金額は整数で保存して容量を減らす
    compact.to_csv(out, index=False, compression="gzip")
    return out


def history(codes: list[str], root: Path = DAILY_DIR) -> pd.DataFrame:
    """監視銘柄の推移（列: date, code, ...）。"""
    frames = []
    for day in saved_days(root):
        df = pd.read_csv(path_for(day, root), dtype={"code": str})
        df = df[df["code"].isin(codes)].copy()
        df.insert(0, "date", pd.Timestamp(day))
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["date", "code"])
    return pd.concat(frames, ignore_index=True).sort_values(["code", "date"])
