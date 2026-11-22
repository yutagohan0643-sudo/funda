"""JPX「銘柄別信用取引残高」（日次・全銘柄・PDF）の取得と読み取り。

公表ページ: https://www.jpx.co.jp/markets/statistics-equities/margin/01.html
毎営業日16時ごろ、前営業日申込み分が掲載される。サイトには直近5営業日分しか置かれない。
"""

from __future__ import annotations

import io
import re
from datetime import date
from urllib.parse import urljoin

import pandas as pd
import requests

PAGE_URL = "https://www.jpx.co.jp/markets/statistics-equities/margin/01.html"
USER_AGENT = "Mozilla/5.0 (personal margin tracker)"

# 1銘柄につき「株数 Shs.」行と「金額 Val.」行がある。
# 例: "... 23340 JP3131380002株数 Shs. 900 ▲ 20,500 0.0% 12,801,300 1,075,400 29.7% ..."
ROW_RE = re.compile(r"(?P<code>[0-9][0-9A-Z]{3})0 (?P<isin>JP[0-9A-Z]{10})(?P<kind>株数 Shs\.|金額 Val\.) (?P<rest>.+)$")
DATE_RE = re.compile(r"(\d{4})/(\d{1,2})/(\d{1,2})申込み現在")
FILE_DATE_RE = re.compile(r"(\d{8})_mtall\.pdf$")

# 1行に並ぶ14項目（「上場比」は金額行では "-"）
FIELDS = [
    "sell", "sell_chg", "sell_ratio",
    "buy", "buy_chg", "buy_ratio",
    "sell_neg", "sell_neg_chg", "sell_std", "sell_std_chg",
    "buy_neg", "buy_neg_chg", "buy_std", "buy_std_chg",
]


def list_files(session: requests.Session | None = None) -> dict[date, str]:
    """公表ページに置かれている PDF の {申込日: URL}。"""
    s = session or _session()
    r = s.get(PAGE_URL, timeout=30)
    r.raise_for_status()
    out = {}
    for href in re.findall(r'href="([^"]+_mtall\.pdf)"', r.text):
        m = FILE_DATE_RE.search(href)
        if m:
            out[pd.Timestamp(m.group(1)).date()] = urljoin(PAGE_URL, href)
    return dict(sorted(out.items()))


def download(url: str, session: requests.Session | None = None) -> bytes:
    s = session or _session()
    r = s.get(url, timeout=120)
    r.raise_for_status()
    return r.content


def pdf_to_text(data: bytes) -> str:
    import pdfplumber

    with pdfplumber.open(io.BytesIO(data)) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def _numbers(rest: str) -> list[float | None]:
    """'900 ▲ 20,500 0.0% ...' → [900, -20500, 0.0, ...]。'*' や '-' は None。"""
    tokens = rest.split()
    values: list[float | None] = []
    negative = False
    for tok in tokens:
        if tok == "▲":
            negative = True
            continue
        if tok in ("*", "-"):
            values.append(None)
        else:
            tok = tok.replace(",", "")
            pct = tok.endswith("%")
            num = float(tok.rstrip("%"))
            if pct:
                num /= 100
            values.append(-num if negative else num)
        negative = False
    return values


def parse_text(text: str) -> tuple[date | None, pd.DataFrame]:
    """PDF のテキストから全銘柄の残高表を作る。

    戻り値の列: code, isin, 株数の14項目, 金額の14項目（接尾辞 _val）
    """
    m = DATE_RE.search(text)
    as_of = date(int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None

    rows: dict[str, dict] = {}
    for line in text.splitlines():
        m = ROW_RE.search(line)
        if not m:
            continue
        values = _numbers(m.group("rest"))
        if len(values) != len(FIELDS):
            continue  # 読み取りが崩れた行（主に名前の長い ETF）は飛ばす
        row = rows.setdefault(m.group("code"), {"code": m.group("code"), "isin": m.group("isin")})
        suffix = "" if m.group("kind").startswith("株数") else "_val"
        for name, v in zip(FIELDS, values):
            if suffix and name.endswith("_ratio"):
                continue
            row[name + suffix] = v

    df = pd.DataFrame(rows.values())
    if not df.empty:
        df = df.dropna(subset=["buy", "sell"]).reset_index(drop=True)
    return as_of, df


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s
