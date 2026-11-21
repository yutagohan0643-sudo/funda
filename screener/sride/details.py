"""候補銘柄ごとの詳細ファンダメンタルズ (並列取得 + キャッシュ)."""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yfinance as yf

FIELDS = [
    "returnOnEquity", "returnOnAssets", "operatingMargins", "profitMargins",
    "revenueGrowth", "earningsGrowth", "numberOfAnalystOpinions",
    "heldPercentInstitutions", "heldPercentInsiders", "totalCash",
    "totalDebt", "ebitda", "enterpriseValue", "forwardPE",
    "currentRatio", "totalRevenue", "netIncomeToCommon", "trailingEps",
    "longBusinessSummary", "website",
]
CACHE_TTL = 24 * 3600
CACHE_VERSION = 2


def _fetch_one(symbol: str, cache_dir: Path) -> dict:
    path = cache_dir / f"{symbol}.json"
    if path.exists() and time.time() - path.stat().st_mtime < CACHE_TTL:
        cached = json.loads(path.read_text())
        if cached.get("_v") == CACHE_VERSION:
            return cached
    for attempt in range(3):
        try:
            info = yf.Ticker(symbol).info or {}
            row = {k: info.get(k) for k in FIELDS}
            row["symbol"] = symbol
            row["_v"] = CACHE_VERSION
            if row["returnOnEquity"] is None and row["operatingMargins"] is None:
                raise ValueError("empty info (rate limited / invalid crumb)")
            path.write_text(json.dumps(row, ensure_ascii=False))
            return row
        except Exception:  # noqa: BLE001 - レート制限等はリトライ (失敗はキャッシュしない)
            time.sleep(3 * (attempt + 1))
    return {"symbol": symbol}


def fetch_details(symbols: list[str], cache_dir: Path, workers: int = 4,
                  progress: bool = True) -> pd.DataFrame:
    d = cache_dir / "info"
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch_one, s, d): s for s in symbols}
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if progress and i % 50 == 0:
                print(f"  詳細取得 {i}/{len(symbols)}", flush=True)
    return pd.DataFrame(rows)
