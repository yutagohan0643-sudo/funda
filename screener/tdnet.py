"""TDnet（適時開示）の一覧を読み、銘柄ごとの直近の開示を「悪材料／好材料／決算」に分ける。

一覧ページ: https://www.release.tdnet.info/inbs/I_list_001_YYYYMMDD.html（1ページ100件、約1か月分が残る）
過ぎた日の一覧は変わらないので data/tdnet/YYYYMMDD.csv に保存して使い回す（当日分だけ毎回取り直す）。
開示の中身（PDF）までは読まず、表題の言葉で分ける。「業績予想の修正」は表題だけでは上か下か分からないので、
株価が下げていれば「下方修正の可能性」として扱う。
"""

from __future__ import annotations

import html
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import requests

LIST_URL = "https://www.release.tdnet.info/inbs/I_list_{page:03d}_{day:%Y%m%d}.html"
CACHE_DIR = Path("data/tdnet")
DAYS = 30  # 直近この日数（暦日）の開示を見る

# 表題に含まれていたら悪材料（下げても「然るべき」）
BAD = {
    "下方修正": "下方修正", "減配": "減配", "無配": "無配", "特別損失": "特別損失", "減損": "減損",
    "公募": "増資（公募）", "第三者割当": "第三者割当", "新株予約権": "新株予約権", "転換社債": "転換社債",
    "売出": "株の売出し", "立会外分売": "立会外分売",
    "不適切": "不適切な会計・取引", "不正": "不正", "調査委員会": "調査委員会", "訴訟": "訴訟",
    "行政処分": "行政処分", "業務停止": "業務停止", "監理銘柄": "監理銘柄", "特別注意": "特別注意銘柄",
    "上場廃止": "上場廃止", "継続企業の前提": "継続企業の注記", "債務超過": "債務超過", "延期": "決算の延期など",
    "業績予想の修正": "業績予想の修正",
}
# 表題に含まれていたら好材料（思惑・反発の材料）
GOOD = {
    "上方修正": "上方修正", "増配": "増配", "復配": "復配", "自己株式の取得": "自社株買い", "自己株式取得": "自社株買い",
    "株式分割": "株式分割", "株主優待": "株主優待", "業務提携": "業務提携", "資本提携": "資本提携",
    "受注": "受注", "契約締結": "契約締結", "共同開発": "共同開発", "承認取得": "承認取得", "公開買付け": "TOB関連",
}
EARNINGS = ("決算短信",)
NEUTRAL_HINTS = ("新株予約権の行使状況", "ストックオプション", "ストック・オプション", "譲渡制限付株式", "取得及び消却", "消却")  # 悪材料とみなさない新株予約権など

_ROW = re.compile(
    r'kjTime"[^>]*>(?P<time>[^<]*)</td>\s*<td[^>]*kjCode"[^>]*>(?P<code>[^<]*)</td>\s*'
    r'<td[^>]*kjName"[^>]*>(?P<name>[^<]*)</td>\s*<td[^>]*kjTitle"[^>]*>(?:<a[^>]*>)?(?P<title>[^<]*)', re.S)


def parse(text: str) -> pd.DataFrame:
    rows = [{k: html.unescape(v).strip() for k, v in m.groupdict().items()} for m in _ROW.finditer(text)]
    df = pd.DataFrame(rows, columns=["time", "code", "name", "title"])
    df["code"] = df["code"].str[:4]  # 5桁（末尾0）を4桁に
    return df


def fetch_day(day: pd.Timestamp, session: requests.Session) -> pd.DataFrame:
    frames = []
    for page in range(1, 40):
        r = session.get(LIST_URL.format(page=page, day=day), timeout=30)
        if r.status_code == 404:
            break
        r.raise_for_status()
        df = parse(r.content.decode("utf-8", "replace"))
        if df.empty:
            break
        frames.append(df)
        time.sleep(0.5)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["time", "code", "name", "title"])
    out.insert(0, "date", day.normalize())
    return out


def load(today: pd.Timestamp, days: int = DAYS, root: Path = CACHE_DIR, log=print) -> pd.DataFrame:
    """直近 days 日分の開示一覧（古い日は保存したものを使う）。"""
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0"
    root.mkdir(parents=True, exist_ok=True)
    frames = []
    for day in pd.date_range(today.normalize() - pd.Timedelta(days=days), today.normalize()):
        if day.weekday() >= 5:
            continue
        path = root / f"{day:%Y%m%d}.csv"
        if path.exists() and day.normalize() < today.normalize():
            frames.append(pd.read_csv(path, dtype={"code": str}, parse_dates=["date"]))
            continue
        try:
            df = fetch_day(day, s)
        except Exception as e:
            log(f"TDnet の取得に失敗: {day:%Y-%m-%d}: {e}")
            continue
        if not df.empty:
            df.to_csv(path, index=False)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["date", "time", "code", "name", "title"])


@dataclass
class News:
    bad: list[str] = field(default_factory=list)  # 悪材料（例: "11/14 下方修正"）
    good: list[str] = field(default_factory=list)  # 好材料
    earnings: pd.Timestamp | None = None  # 直近の決算発表日
    revision: pd.Timestamp | None = None  # 「業績予想の修正」（向き不明）の日


def classify(df: pd.DataFrame, code: str) -> News:
    n = News()
    rows = df[df["code"] == code].sort_values("date")
    for _, r in rows.iterrows():
        title, day = str(r["title"]), pd.Timestamp(r["date"])
        tag = f"{day:%m/%d}"
        if any(w in title for w in EARNINGS):
            n.earnings = day
        good = [v for k, v in GOOD.items() if k in title]
        bad = [] if any(h in title for h in NEUTRAL_HINTS) else [v for k, v in BAD.items() if k in title]
        if "業績予想の修正" in bad and not ({"下方修正"} & set(bad)):
            bad.remove("業績予想の修正")
            if "上方修正" not in good:
                n.revision = day
        if "株主優待" in good and "廃止" in title:
            good.remove("株主優待")
            bad.append("株主優待の廃止")
        if good and "TOB関連" in good and bad:  # TOB に伴う売出しなどは悪材料にしない
            bad = []
        for v in dict.fromkeys(bad):
            n.bad.append(f"{tag} {v}")
        for v in dict.fromkeys(good):
            n.good.append(f"{tag} {v}")
    return n
