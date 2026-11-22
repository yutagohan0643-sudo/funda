"""ファンダ（PER・PBR・直近決算の評価・懸念）。

データは Yahoo Finance の四半期決算（売上・営業利益・純利益・自己資本・株数・現金・有利子負債）。
小型株は値の誤りや反映の遅れがあり得るので、レポートには出典と決算期を必ず出し、決算短信で確認する前提。
PER・PBR は Yahoo の値ではなく、最新の株価と決算データから計算し直す（株価が動いた日も正しくなるように）。
判定の基準値はこのファイルの上の方にまとめてある。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import requests

URL = "https://query1.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries/{symbol}"
FIELDS = {
    "quarterlyTotalRevenue": "revenue",
    "quarterlyOperatingIncome": "op",
    "quarterlyNetIncome": "net",
    "quarterlyStockholdersEquity": "equity",
    "quarterlyOrdinarySharesNumber": "shares",
    "quarterlyCashAndCashEquivalents": "cash",
    "quarterlyTotalDebt": "debt",
}

# ---- 判定の基準値 ----
OP_DROP = -0.20  # 営業利益が前年同期比 -20% 以下なら懸念
REV_DROP = -0.20  # 売上が前年同期比 -20% 以下なら懸念
MARGIN_DROP = 0.03  # 営業利益率が前年同期より3ポイント以上下がったら懸念
DILUTION = 0.03  # 株数が1年で3%以上増えたら「希薄化（増資・新株予約権など）」
RUNWAY_QUARTERS = 8  # 赤字のとき、現金が営業赤字の8四半期分未満なら「資金調達リスク」
EXPENSIVE_PER, EXPENSIVE_PBR = 50.0, 5.0  # 割高の目安
STALE_DAYS = 135  # 決算期末からこれ以上たっていたら「最新の決算が未反映かも」

# ---- 取得の頻度（Yahoo に断られにくくする） ----
CACHE_DIR = Path("data/fundamentals")  # 取った決算データの保存先（四半期に1回しか変わらないので使い回す）
REFRESH_DAYS = 7  # 保存してからこの日数以内なら問い合わせない
MAX_FETCH_PER_RUN = 3  # 1回の実行で問い合わせるのはこの銘柄数まで（残りは次の回）
FETCH_INTERVAL = 20  # 問い合わせの間隔（秒）


@dataclass
class Fundamentals:
    quarters: pd.DataFrame  # 決算期末ごと（古い順）。列は FIELDS の値
    close: float = math.nan
    per: float = math.nan  # 直近4四半期の純利益から
    pbr: float = math.nan
    market_cap: float = math.nan
    verdict: str = ""  # 直近決算の評価（例: 増収増益）
    rev_yoy: float = math.nan
    op_yoy: float = math.nan
    margin: float = math.nan
    margin_diff: float = math.nan  # 営業利益率の前年同期差
    concerns: list[str] = field(default_factory=list)
    positives: list[str] = field(default_factory=list)

    @property
    def latest(self) -> pd.Series:
        return self.quarters.iloc[-1]

    @property
    def year_ago(self) -> pd.Series | None:
        return self.quarters.iloc[-5] if len(self.quarters) >= 5 else None

    @property
    def period(self) -> str:
        """例: 2026年7-9月期"""
        d = self.quarters.index[-1]
        start = (d - pd.DateOffset(months=2)).month
        return f"{d.year}年{start}-{d.month}月期"


def fetch(code: str, session: requests.Session | None = None) -> pd.DataFrame:
    """四半期決算の表（決算期末ごと、古い順）。"""
    from .prices import to_symbol

    s = session or requests.Session()
    s.headers.setdefault("User-Agent", "Mozilla/5.0")
    start = int((pd.Timestamp.now() - pd.DateOffset(years=3)).timestamp())
    params = {"type": ",".join(FIELDS), "period1": start, "period2": int(time.time()) + 86400}
    for wait in (30, None):  # 「問い合わせが多すぎる」(429) と断られたら、1回だけ間をおいて取り直す
        r = s.get(URL.format(symbol=to_symbol(code)), params=params, timeout=30)
        if r.status_code != 429 or wait is None:
            break
        time.sleep(wait)
    r.raise_for_status()
    cols = {}
    for res in r.json()["timeseries"]["result"] or []:
        t = res["meta"]["type"][0]
        if t not in FIELDS:
            continue
        vals = [v for v in res.get(t) or [] if v]
        cols[FIELDS[t]] = pd.Series({pd.Timestamp(v["asOfDate"]): float(v["reportedValue"]["raw"]) for v in vals})
    if not cols:
        raise ValueError("決算データなし")
    return pd.DataFrame(cols).sort_index()


def cache_path(code: str, root: Path = CACHE_DIR) -> Path:
    return root / f"{code}.csv"


def save_cache(code: str, q: pd.DataFrame, root: Path = CACHE_DIR) -> None:
    root.mkdir(parents=True, exist_ok=True)
    q.to_csv(cache_path(code, root))
    stamp = root / "_fetched.csv"
    stamps = pd.read_csv(stamp, index_col=0, dtype={0: str}) if stamp.exists() else pd.DataFrame(columns=["fetched"])
    stamps.loc[code, "fetched"] = pd.Timestamp.now().isoformat(timespec="seconds")
    stamps.sort_index().to_csv(stamp)


def fetched_at(code: str, root: Path = CACHE_DIR) -> pd.Timestamp | None:
    """最後に取得できた日時（git で取り出すとファイルの更新日時は変わるので、別ファイルに記録しておく）。"""
    stamp = root / "_fetched.csv"
    if not stamp.exists():
        return None
    stamps = pd.read_csv(stamp, index_col=0, dtype={0: str})
    return pd.Timestamp(stamps.loc[code, "fetched"]) if code in stamps.index else None


def get_all(codes: list[str], fetcher=None, root: Path = CACHE_DIR, now: pd.Timestamp | None = None,
            log=print) -> dict[str, pd.DataFrame | None]:
    """保存済みのものを使い、古いものだけを1回に MAX_FETCH_PER_RUN 銘柄まで取り直す。"""
    fetcher = fetcher or fetch
    now = now or pd.Timestamp.now()
    out, tried = {}, 0
    # 一度も取れていない銘柄 → 古い順 に取りに行く
    order = sorted(codes, key=lambda c: fetched_at(c, root) or pd.Timestamp.min)
    for code in order:
        path = cache_path(code, root)
        cached = pd.read_csv(path, index_col=0, parse_dates=True) if path.exists() else None
        when = fetched_at(code, root)
        fresh = cached is not None and when is not None and (now - when).days < REFRESH_DAYS
        if not fresh and tried < MAX_FETCH_PER_RUN:
            if tried:
                time.sleep(FETCH_INTERVAL)
            tried += 1
            try:
                q = fetcher(code)
                save_cache(code, q, root)
                cached = q
            except Exception as e:  # 取れなければ保存済みのもので代用する
                log(f"決算データの取得に失敗: {code}: {e}" + ("（保存済みのデータを使う）" if cached is not None else ""))
        out[code] = cached
    return out


def _yoy(now: float, before: float) -> float:
    if before is None or now is None or math.isnan(before) or math.isnan(now) or before <= 0:
        return math.nan
    return now / before - 1


def evaluate(q: pd.DataFrame, close: float, today: pd.Timestamp | None = None) -> Fundamentals:
    f = Fundamentals(q, close)
    if q.empty or "revenue" not in q or "op" not in q:
        return f
    last = q.iloc[-1]
    rev, op = last.get("revenue", math.nan), last.get("op", math.nan)
    shares = q["shares"].dropna().iloc[-1] if "shares" in q and q["shares"].notna().any() else math.nan

    # PER・PBR・時価総額（最新の株価で計算）
    if not math.isnan(shares) and shares > 0 and not math.isnan(close):
        f.market_cap = close * shares
        net4 = q["net"].dropna().tail(4) if "net" in q else pd.Series(dtype=float)
        if len(net4) == 4 and net4.sum() > 0:
            f.per = f.market_cap / net4.sum()
        if "equity" in q and q["equity"].notna().any() and q["equity"].dropna().iloc[-1] > 0:
            f.pbr = f.market_cap / q["equity"].dropna().iloc[-1]

    # 直近四半期の前年同期比（5四半期前と比べる）
    prev = q.iloc[-5] if len(q) >= 5 else None
    if prev is not None:
        f.rev_yoy = _yoy(rev, prev.get("revenue"))
        f.op_yoy = _yoy(op, prev.get("op"))
        p_rev, p_op = prev.get("revenue", math.nan), prev.get("op", math.nan)
        if rev and not math.isnan(rev) and rev > 0:
            f.margin = op / rev
            # 利益率の比較は、両方の期が営業黒字のときだけ（赤字どうしの比較は意味が薄い）
            if p_rev and not math.isnan(p_rev) and p_rev > 0 and op > 0 and p_op > 0:
                f.margin_diff = f.margin - p_op / p_rev
        rev_up = rev >= p_rev if not (math.isnan(rev) or math.isnan(p_rev)) else None
        if op < 0 and p_op < 0:
            f.verdict = "営業赤字が続く（" + ("縮小" if op > p_op else "拡大") + "）"
        elif op < 0:
            f.verdict = "営業赤字に転落"
        elif p_op <= 0:
            f.verdict = "営業黒字に転換"
        else:
            f.verdict = ("増収" if rev_up else "減収") + ("増益" if op >= p_op else "減益")
    elif rev and rev > 0:
        f.margin = op / rev

    # 良い点
    if f.verdict in ("増収増益", "営業黒字に転換"):
        f.positives.append(f.verdict)
    if f.verdict == "減収増益":
        f.positives.append("減収増益")
    if not math.isnan(f.margin_diff) and f.margin_diff >= MARGIN_DROP:
        f.positives.append(f"利益率 {f.margin_diff * 100:+.1f}pt（前年比）")
    if not math.isnan(f.pbr) and f.pbr < 1 and op > 0:
        f.positives.append(f"PBR {f.pbr:.2f}倍で黒字")

    # 懸念
    if op < 0:
        f.concerns.append("営業赤字")
    if not math.isnan(f.rev_yoy) and f.rev_yoy <= REV_DROP:
        f.concerns.append(f"売上 {f.rev_yoy:+.0%}（前年比）")
    if not math.isnan(f.op_yoy) and f.op_yoy <= OP_DROP:
        f.concerns.append(f"営業利益 {f.op_yoy:+.0%}（前年比）")
    if f.verdict == "増収減益":
        f.concerns.append("増収減益（コスト増）")
    if not math.isnan(f.margin_diff) and f.margin_diff <= -MARGIN_DROP:
        f.concerns.append(f"利益率 {f.margin_diff * 100:+.1f}pt（前年比）")
    if "shares" in q:
        sh = q["shares"].dropna()
        if len(sh) >= 5 and sh.iloc[-5] > 0 and sh.iloc[-1] / sh.iloc[-5] - 1 >= DILUTION:
            f.concerns.append(f"株数 {sh.iloc[-1] / sh.iloc[-5] - 1:+.0%}/年（希薄化）")
    if "equity" in q and q["equity"].notna().any() and q["equity"].dropna().iloc[-1] <= 0:
        f.concerns.append("債務超過")
    elif {"equity", "debt"} <= set(q) and q["debt"].notna().any():
        eq, debt = q["equity"].dropna().iloc[-1], q["debt"].dropna().iloc[-1]
        if eq > 0 and debt > eq:
            f.concerns.append(f"負債が自己資本の{debt / eq:.1f}倍")
    op4 = q["op"].dropna().tail(4)
    if len(op4) == 4 and op4.sum() < 0 and "cash" in q and q["cash"].notna().any():
        burn = -op4.sum() / 4
        runway = q["cash"].dropna().iloc[-1] / burn if burn > 0 else math.inf
        if runway < RUNWAY_QUARTERS:
            f.concerns.append(f"現金 約{runway:.0f}四半期分（増資リスク）")
    if not math.isnan(f.per) and f.per >= EXPENSIVE_PER:
        f.concerns.append(f"PER {f.per:.0f}倍（割高）")
    elif math.isnan(f.per) and len(q.get("net", pd.Series(dtype=float)).dropna().tail(4)) == 4:
        f.concerns.append("直近1年は純損失")
    if not math.isnan(f.pbr) and f.pbr >= EXPENSIVE_PBR:
        f.concerns.append(f"PBR {f.pbr:.1f}倍（割高）")
    today = today or pd.Timestamp.now()
    if (today - q.index[-1]).days > STALE_DAYS:
        f.concerns.append("最新決算が未反映かも")
    return f
