"""タッチした銘柄のランキング：ファンダ・思惑・下げの性質（然るべき下げか、不当な下げか）で点数をつける。

総合 100点 = ファンダ 40点 + 思惑 30点 + 反発 30点
  ファンダ: 直近決算の評価・PER・PBR・利益率の改善・実質無借金。懸念の数だけ引く
  思惑:     タッチした日の出来高の急増・売買の活発さ（売買代金÷時価総額）・直近の好材料の開示
  反発:     下げの性質（不当な下げほど高い）・20日線までの戻り余地・下ヒゲ（安値からの戻り）・RSI の売られすぎ

下げの性質
  然るべき下げ: 直近1か月に悪材料の開示（下方修正・増資・減配・特損・不正など）があるか、
               直近1か月に決算を出して中身が悪かった（減益・赤字転落など）
  不当な下げ:   悪材料の開示がなく、下げの半分以上が地合い（TOPIX・グロース250）で説明できるか、
               材料なしにじりじり下げている（出来高を伴う窓あけの急落ではない）
  要確認:       開示は見当たらないのに、出来高を伴って個別に急落した／決算は良いのに売られた など
判定の基準値はこのファイルの上の方にまとめてある。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import core
from .tdnet import News

WINDOW = 5  # 下げを測る期間（営業日）
BETA_DAYS = 120  # 指数との連動（ベータ）を測る期間
MARKET_SHARE = 0.5  # 下げのこの割合以上を指数で説明できたら「地合いの下げ」
GAP = -0.03  # 前日の終値から3%以上下で寄り付き、
GAP_VOL = 2.0  # 出来高が平均の2倍以上なら「材料での急落」
REACTION = -0.02  # 開示の後2日の値動きが、地合いを除いてこれ以下なら「その開示で売られた」とみなす
HEAVY_MARGIN = 0.10  # 信用買残が上場株数の10%以上なら、戻り売りが重いので減点

GOOD_VERDICT = {"増収増益": 16, "営業黒字に転換": 13, "減収増益": 9, "増収減益": 5, "減収減益": 2, "営業赤字が続く（縮小）": 2}
BAD_VERDICT = ("増収減益", "減収減益", "営業赤字に転落", "営業赤字が続く（拡大）")

JUSTIFIED, UNJUSTIFIED, UNCLEAR = "然るべき下げ", "不当な下げ", "要確認"


@dataclass
class Ranked:
    result: core.Result
    total: int
    fund_pts: int
    hope_pts: int  # 思惑
    rebound_pts: int
    verdict: str  # 然るべき下げ / 不当な下げ / 要確認
    reasons: list[str] = field(default_factory=list)
    good_news: list[str] = field(default_factory=list)
    r5: float = math.nan  # 直近5日の騰落
    market_share: float = math.nan  # 下げのうち指数で説明できる割合


def _ret(c: pd.Series, n: int) -> float:
    return float(c.iloc[-1] / c.iloc[-1 - n] - 1) if len(c) > n else math.nan


def market_part(stock: pd.DataFrame, indices: dict[str, pd.DataFrame], n: int = WINDOW) -> tuple[float, float, str]:
    """(直近 n 日の騰落, そのうち指数で説明できる割合, 使った指数)。ベータは n 日より前の120日で測る。"""
    c = stock["Close"].dropna()
    r = _ret(c, n)
    best = (math.nan, "")
    best_r2 = -1.0
    for name, idx in indices.items():
        if idx is None or idx.empty:
            continue
        both = pd.concat([c.pct_change(), idx["Close"].pct_change()], axis=1, keys=["s", "m"]).dropna()
        past = both.iloc[:-n].tail(BETA_DAYS)
        if len(past) < 40 or past["m"].var() == 0:
            continue
        beta = past["s"].cov(past["m"]) / past["m"].var()
        r2 = past["s"].corr(past["m"]) ** 2
        mc = idx["Close"].dropna()
        mc = mc[mc.index <= c.index[-1]]
        expected = beta * _ret(mc, n)
        if r2 > best_r2:
            best_r2, best = r2, (expected, name)
    expected, name = best
    if math.isnan(r) or r >= 0 or math.isnan(expected):
        return r, math.nan, name
    return r, float(np.clip(expected / r, 0, 1)), name


def reaction(stock: pd.DataFrame, indices: dict[str, pd.DataFrame], day: pd.Timestamp) -> float:
    """開示日 day の前日の終値から、開示の翌営業日の終値までの騰落（指数の動きを引く）。引け後の開示も場中の開示も含められる。"""
    c = stock["Close"].dropna()
    before, after = c[c.index < day], c[c.index > day]
    if before.empty or after.empty:
        return math.nan
    r = float(after.iloc[0] / before.iloc[-1] - 1)
    for idx in indices.values():  # TOPIX の動きを引く（最初の指数）
        if idx is None or idx.empty:
            continue
        m = idx["Close"].dropna()
        mb, ma = m[m.index <= before.index[-1]], m[m.index <= after.index[0]]
        if len(mb) and len(ma):
            r -= float(ma.iloc[-1] / mb.iloc[-1] - 1)
        break
    return r


def event_drop(stock: pd.DataFrame, n: int = WINDOW) -> pd.Timestamp | None:
    """直近 n 日に、窓をあけて出来高を伴って急落した日（材料が出た日の目安）。"""
    df = stock.dropna(subset=["Open", "Close"])
    avg = df["Volume"].shift(1).rolling(core.BB_PERIOD).mean()
    gap = df["Open"] / df["Close"].shift(1) - 1
    hit = df.index[-n:][(gap.iloc[-n:] <= GAP) & (df["Volume"].iloc[-n:] >= GAP_VOL * avg.iloc[-n:])]
    return hit[-1] if len(hit) else None


def judge_drop(r: core.Result, news: News, stock: pd.DataFrame, indices: dict[str, pd.DataFrame],
               today: pd.Timestamp, days: int = 30) -> tuple[str, list[str], float, float]:
    """(判定, 理由, 5日騰落, 指数で説明できる割合)"""
    f = r.fund
    since = today.normalize() - pd.Timedelta(days=days)
    r5, share, idx_name = market_part(stock, indices)
    event = event_drop(stock)
    reasons = list(news.bad)
    if news.revision is not None and news.revision >= since:
        rr = reaction(stock, indices, news.revision)
        if rr == rr and rr <= REACTION:
            reasons.append(f"{news.revision:%m/%d} 業績予想の修正（発表後 {rr:+.0%}・地合い除く。下方の可能性）")
    recent_earn = news.earnings is not None and news.earnings >= since
    if recent_earn and f is not None and (f.verdict in BAD_VERDICT or (f.op_yoy == f.op_yoy and f.op_yoy <= -0.20)):
        rr = reaction(stock, indices, news.earnings)
        if rr == rr and rr <= REACTION:
            reasons.append(f"{news.earnings:%m/%d} 決算（{f.verdict}、発表後 {rr:+.0%}・地合い除く）")
    if reasons:
        return JUSTIFIED, reasons, r5, share
    mkt = share == share and share >= MARKET_SHARE
    if mkt:
        why = [f"地合い（下げの{share:.0%}を{idx_name}で説明）"]
    elif event is None:
        why = ["材料なしの下げ（急落の日がない）"]
    else:
        why = []
    if recent_earn and f is not None and f.verdict in GOOD_VERDICT and event is not None and not mkt:
        return UNCLEAR, [f"{news.earnings:%m/%d} 決算は{f.verdict}なのに急落（会社予想・進捗を確認）"], r5, share
    if why:
        if f is not None and f.verdict:
            why.append(f"決算は{f.verdict}")
        return UNJUSTIFIED, why, r5, share
    return UNCLEAR, [f"{event:%m/%d} 出来高を伴う急落・原因の開示は見当たらない（TDnet 以外のニュースを確認）"], r5, share


def fund_points(r: core.Result) -> int:
    f = r.fund
    if f is None:
        return 0
    pts = GOOD_VERDICT.get(f.verdict, 0)
    if f.per == f.per:
        pts += 8 if f.per < 8 else 6 if f.per < 12 else 3 if f.per < 20 else 0
    if f.pbr == f.pbr:
        pts += 8 if f.pbr < 0.7 else 6 if f.pbr < 1 else 3 if f.pbr < 1.5 else 0
    if f.margin_diff == f.margin_diff and f.margin_diff > 0:
        pts += 4
    q = f.quarters
    if {"cash", "debt"} <= set(q) and q["cash"].notna().any() and q["debt"].notna().any() \
            and q["cash"].dropna().iloc[-1] > q["debt"].dropna().iloc[-1]:
        pts += 4  # 実質無借金
    for c in r.concerns:
        pts -= 1 if c.startswith("最新決算が未反映") else 3
    return int(np.clip(pts, 0, 40))


def hope_points(r: core.Result, news: News) -> int:
    t, f = r.touch, r.fund
    pts = 0
    if t.vol_ratio == t.vol_ratio:
        pts += 10 if t.vol_ratio >= 3 else 7 if t.vol_ratio >= 2 else 4 if t.vol_ratio >= 1.5 else 0
    if f is not None and f.market_cap == f.market_cap and f.market_cap > 0:
        turn = t.turnover / f.market_cap
        pts += 8 if turn >= 0.01 else 6 if turn >= 0.005 else 3 if turn >= 0.002 else 0
    pts += min(12, 6 * len(news.good))
    return int(np.clip(pts, 0, 30))


def rebound_points(r: core.Result, verdict: str, stock: pd.DataFrame) -> int:
    t = r.touch
    pts = {UNJUSTIFIED: 15, UNCLEAR: 6, JUSTIFIED: 0}[verdict]
    ma = stock["Close"].tail(core.BB_PERIOD).mean()
    room = ma / t.close - 1
    pts += 5 if room >= 0.10 else 3 if room >= 0.05 else 0
    pts += 5 if t.rebound >= 0.03 else 3 if t.rebound >= 0.015 else 0
    if t.rsi == t.rsi:
        pts += 5 if t.rsi <= 25 else 3 if t.rsi <= 30 else 0
    return int(np.clip(pts, 0, 30))


def rank(results: list[core.Result], bars: dict[str, pd.DataFrame], news: dict[str, News],
         indices: dict[str, pd.DataFrame], uni: pd.DataFrame, today: pd.Timestamp) -> list[Ranked]:
    """致命的なファンダがない銘柄だけを点数の高い順に並べる。"""
    out = []
    for r in results:
        if not r.passed or r.touch.code not in bars:
            continue
        stock = bars[r.touch.code]
        n = news.get(r.touch.code) or News()
        verdict, reasons, r5, share = judge_drop(r, n, stock, indices, today)
        fp, hp, rp = fund_points(r), hope_points(r, n), rebound_points(r, verdict, stock)
        total = fp + hp + rp
        code = r.touch.code
        if code in uni.index and uni.loc[code, "buy_ratio"] == uni.loc[code, "buy_ratio"] \
                and uni.loc[code, "buy_ratio"] >= HEAVY_MARGIN:
            total -= 5
            reasons = reasons + [f"信用買残が上場株数の{uni.loc[code, 'buy_ratio']:.0%}（戻り売りが重い）"]
        out.append(Ranked(r, total, fp, hp, rp, verdict, reasons, n.good, r5, share))
    out.sort(key=lambda x: (-x.total, x.result.touch.low_sigma))
    return out
