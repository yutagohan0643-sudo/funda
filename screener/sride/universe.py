"""銘柄ユニバース: JPX 上場銘柄一覧 + Yahoo バルクスクリーナー."""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from yfinance import EquityQuery as Q

JPX_URL = (
    "https://www.jpx.co.jp/markets/statistics-equities/misc/"
    "tvdivq0000001vg2-att/data_j.xlsx"
)
UA = {"User-Agent": "Mozilla/5.0 (sride screener)"}


def load_jpx(cache_dir: Path) -> pd.DataFrame:
    """東証上場の内国株式一覧 (コード, 銘柄名, 市場, 業種, 規模)."""
    path = cache_dir / "data_j.xlsx"
    if not path.exists():
        r = requests.get(JPX_URL, headers=UA, timeout=60)
        r.raise_for_status()
        path.write_bytes(r.content)
    df = pd.read_excel(io.BytesIO(path.read_bytes()), dtype={"コード": str})
    df = df[df["市場・商品区分"].str.contains("内国株式", na=False)].copy()
    df["市場"] = df["市場・商品区分"].str.replace("（内国株式）", "", regex=False)
    df["symbol"] = df["コード"].str.strip() + ".T"
    return df.rename(columns={"33業種区分": "業種", "規模区分": "規模"})[
        ["symbol", "コード", "銘柄名", "市場", "業種", "規模"]
    ]


def bulk_quotes(min_mcap: float, max_mcap: float) -> pd.DataFrame:
    """Yahoo のスクリーナーで日本株の基本指標を一括取得 (250件/ページ)."""
    q = Q("and", [Q("eq", ["region", "jp"]),
                  Q("btwn", ["intradaymarketcap", min_mcap, max_mcap])])
    rows, offset = [], 0
    while True:
        res = yf.screen(q, size=250, offset=offset,
                        sortField="intradaymarketcap", sortAsc=True)
        quotes = res.get("quotes", [])
        rows.extend(quotes)
        offset += len(quotes)
        if not quotes or offset >= res.get("total", 0):
            break
    keep = ["symbol", "regularMarketPrice", "marketCap", "trailingPE",
            "priceToBook", "dividendYield", "averageDailyVolume3Month",
            "fiftyTwoWeekChangePercent", "fiftyTwoWeekHighChangePercent",
            "twoHundredDayAverageChangePercent"]
    df = pd.DataFrame(rows)
    df = df[[c for c in keep if c in df.columns]]
    return df[df["symbol"].str.endswith(".T")].drop_duplicates("symbol")
