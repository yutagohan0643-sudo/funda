"""CLI: python -m sride [options]"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from . import details, report, score, universe
from .config import ScreenConfig


def main() -> None:
    cfg = ScreenConfig()
    p = argparse.ArgumentParser(description="日本株 未発掘銘柄スクリーナー")
    p.add_argument("--min-mcap", type=float, default=cfg.min_mcap / 1e8, help="時価総額下限(億円)")
    p.add_argument("--max-mcap", type=float, default=cfg.max_mcap / 1e8, help="時価総額上限(億円)")
    p.add_argument("--max-per", type=float, default=cfg.max_per)
    p.add_argument("--max-pbr", type=float, default=cfg.max_pbr)
    p.add_argument("--min-roe", type=float, default=cfg.min_roe * 100, help="ROE下限(%%)")
    p.add_argument("--mode", choices=["fundamental", "undiscovered"], default=cfg.mode,
                   help="fundamental=ファンダのみ(既定) / undiscovered=未発掘度を加味")
    p.add_argument("--max-analysts", type=int, default=None, help="アナリスト数上限 (undiscoveredの既定は2)")
    p.add_argument("--min-turnover", type=float, default=cfg.min_turnover / 1e6, help="売買代金下限(百万円/日)")
    p.add_argument("--markets", default=",".join(cfg.markets), help="プライム,スタンダード,グロース")
    p.add_argument("--workers", type=int, default=cfg.workers)
    p.add_argument("--top", type=int, default=cfg.top)
    p.add_argument("--cache", type=Path, default=Path("cache"))
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()

    cfg = ScreenConfig(
        min_mcap=a.min_mcap * 1e8, max_mcap=a.max_mcap * 1e8, max_per=a.max_per,
        max_pbr=a.max_pbr, min_roe=a.min_roe / 100, mode=a.mode,
        max_analysts=a.max_analysts if a.max_analysts is not None
        else (2 if a.mode == "undiscovered" else None),
        min_turnover=a.min_turnover * 1e6, markets=tuple(a.markets.split(",")),
        workers=a.workers, top=a.top,
    )
    a.cache.mkdir(parents=True, exist_ok=True)
    a.out = a.out or Path("reports") / f"{date.today():%Y-%m-%d}" / cfg.mode

    print("1/4 JPX 上場銘柄一覧を取得")
    jpx = universe.load_jpx(a.cache)
    jpx = jpx[jpx["市場"].isin(cfg.markets)]

    print("2/4 Yahoo バルクスクリーナーで基本指標を取得")
    q = universe.bulk_quotes(cfg.min_mcap, cfg.max_mcap)
    df = jpx.merge(q, on="symbol", how="inner")
    # 詳細取得前の粗いフィルタ (API 呼び出し数を削減)
    pre = df[df["trailingPE"].between(0.1, cfg.max_per)
             & (df["priceToBook"] <= cfg.max_pbr)
             & (df["averageDailyVolume3Month"] * df["regularMarketPrice"] >= cfg.min_turnover)]
    print(f"   ユニバース {len(df)} → 一次通過 {len(pre)}")

    print("3/4 個別ファンダメンタルズを取得")
    det = details.fetch_details(pre["symbol"].tolist(), a.cache, cfg.workers)
    df = score.add_metrics(pre.merge(det, on="symbol", how="left"))
    df = score.score(score.apply_filters(df, cfg), cfg.mode)
    print(f"   二次通過 {len(df)}")

    print("4/4 レポート出力")
    path = report.write(df, a.out, cfg.top, cfg)
    print(f"   {path}\n   {a.out / 'candidates.csv'}")


if __name__ == "__main__":
    main()
