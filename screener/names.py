"""銘柄の日本語名（Yahoo の検索 API を日本語で引く）。

社名はめったに変わらないので data/names.csv に保存して使い回し、載っていない銘柄だけ問い合わせる。
取れなかった銘柄は、日足と一緒に取れる英語名のままにする。
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

SEARCH_URL = "https://query2.finance.yahoo.com/v1/finance/search"
CACHE = Path("data/names.csv")
INTERVAL = 0.3  # 問い合わせの間隔（秒）


def load(path: Path = CACHE) -> dict[str, str]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str).dropna()
    return dict(zip(df["code"], df["name"]))


def save(names: dict[str, str], path: Path = CACHE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sorted(names.items()), columns=["code", "name"]).to_csv(path, index=False)


def fetch_one(code: str, session: requests.Session) -> str | None:
    from backtest.data import to_symbol

    symbol = to_symbol(code)
    r = session.get(SEARCH_URL, params={"q": symbol, "lang": "ja-JP", "region": "JP", "quotesCount": 3, "newsCount": 0},
                    timeout=30)
    r.raise_for_status()
    for q in r.json().get("quotes") or []:
        if q.get("symbol") == symbol:
            return q.get("longname") or q.get("shortname")
    return None


def get(codes: list[str], path: Path = CACHE, log=print) -> dict[str, str]:
    """{銘柄コード: 日本語名}。保存していない銘柄だけ問い合わせて、保存し直す。"""
    names = load(path)
    missing = [c for c in codes if c not in names]
    if missing:
        s = requests.Session()
        s.headers["User-Agent"] = "Mozilla/5.0"
        fails = 0
        for code in missing:
            try:
                name = fetch_one(code, s)
                if name:
                    names[code] = name
                fails = 0
            except Exception as e:
                fails += 1
                log(f"銘柄名の取得に失敗: {code}: {e}")
                if fails >= 5:
                    log("続けて失敗したので、残りの銘柄名は問い合わせない")
                    break
            time.sleep(INTERVAL)
        save(names, path)
    return {c: names[c] for c in codes if c in names}
