"""株価を動かす力の整理（トレードルールの考え方）。

- 引力（地合い）: 株価を動かす一番大きな力。上下どちらを向いているか
- セクターの流れ: 関連テーマ（米国）の勢い
- 表面の評価: PER・PBR の水準だけで見た割安〜割高（妥当な株価はファンダを見て自分で判断する）
- 上値の重し: 新株予約権・MSワラント・増資・売出し・大株主の異動などの開示
- 新規材料: 直近の適時開示（IR 材料ランキングのデータから）
- 下方向の圧力: 株価・信用買残・出来高のバランスで、下げが助長されているか
- 反発サイン: 1日ごとの反発の印（エネルギーが発揮されたか）
テクニカル（ターニングポイント）と需給は、エネルギーが発揮されるポイントを知らせるためのもの。
基準値はこのファイルの上の方にまとめてある。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# ---- 引力（地合い） ----
GRAVITY_MOVE = 0.02  # グロース250・日経の5日騰落がこれ以上なら上向き、以下なら下向きに1点
VIX_CALM, VIX_FEAR, VIX_PEAKOUT = 20.0, 25.0, 0.85
US_MOVE = 0.005  # 米国（S&P500・NASDAQ）の前夜の騰落
# ---- セクター ----
SECTOR_MOVE = 0.02  # 関連テーマの5日騰落の平均がこれ以上なら追い風、以下なら向かい風
# ---- 下方向の圧力（直近5営業日） ----
PRESSURE_DAYS = 5
PRICE_MOVE = 0.03  # 株価がこれ以上動いたら「下げ」「上げ」とみなす
MARGIN_MOVE = 0.03  # 買残がこれ以上増減したら「増加」「減少」
VOL_UP, VOL_DOWN = 1.2, 0.8  # 出来高（直近5日の平均 ÷ その前20日の平均）
# ---- 反発 ----
REBOUND_DAYS = 3
STRONG_UP = 0.03
UPPER_ZONE = 0.7
BAD_DAY = -0.01  # グロース250が前日比これ以下の日の陽線は「逆行陽線」
# ---- 新規材料 ----
IR_DIR = Path("ir/data")
IR_DAYS = 30
OVERHANG_DAYS = 365  # 新株予約権・MSワラントなどは長く効くので1年分を見る
# 上値を押さえる材料（IR 材料ランキングの分類名）
OVERHANG_KINDS = ("MSワラント・行使価額修正条項付", "新株予約権の大量行使", "公募増資", "売出し（大株主の放出）",
                  "第三者割当・新株予約権", "主要株主・大株主の異動")

UP, FLAT, DOWN = "上向き", "横ばい", "下向き"


# ---------- 引力（地合い） ----------
@dataclass
class Gravity:
    direction: str
    reasons: list[str] = field(default_factory=list)
    turning: str = ""  # 反転の兆し（例: VIX ピークアウト）


def _chg(df: pd.DataFrame | None, n: int) -> float | None:
    if df is None or df.empty or len(df["Close"].dropna()) <= n:
        return None
    c = df["Close"].dropna()
    return float(c.iloc[-1] / c.iloc[-1 - n] - 1)


def _above_ma(df: pd.DataFrame | None, n: int = 25) -> bool | None:
    if df is None or df.empty or len(df["Close"].dropna()) < n:
        return None
    c = df["Close"].dropna()
    return bool(c.iloc[-1] >= c.tail(n).mean())


def gravity(series: dict[str, pd.DataFrame]) -> Gravity:
    """series: 地合いの指標の日足（記号 → DataFrame）。グロース250・日経・VIX・米国から引力の向きを出す。"""
    pts, reasons = 0, []
    for sym, name in (("2516.T", "G250"), ("^N225", "日経")):
        df = series.get(sym)
        above, c5 = _above_ma(df), _chg(df, 5)
        if above is None or c5 is None:
            continue
        pts += 1 if above else -1
        if c5 >= GRAVITY_MOVE:
            pts += 1
        elif c5 <= -GRAVITY_MOVE:
            pts -= 1
        reasons.append(f"{name} 5日{c5:+.0%}")
    for sym, name in (("^GSPC", "S&P"), ("^IXIC", "NASDAQ")):
        c1 = _chg(series.get(sym), 1)
        if c1 is None:
            continue
        if c1 >= US_MOVE:
            pts += 1
        elif c1 <= -US_MOVE:
            pts -= 1
    sp = _chg(series.get("^IXIC"), 1)
    if sp is not None:
        reasons.append(f"NASDAQ {sp:+.1%}")
    turning = ""
    vix = series.get("^VIX")
    if vix is not None and not vix.empty:
        v = vix["Close"].dropna()
        level, peak = float(v.iloc[-1]), float(v.tail(5).max())
        if level < VIX_CALM:
            pts += 1
        elif level >= VIX_FEAR:
            pts -= 1
        if peak >= VIX_FEAR and level <= peak * VIX_PEAKOUT:
            turning = "VIXがピークアウト（底の候補）"
        reasons.append(f"VIX {level:.0f}")
    direction = UP if pts >= 2 else DOWN if pts <= -2 else FLAT
    return Gravity(direction, reasons, turning)


# ---------- セクター ----------
def sector_flow(themes: list[tuple[str, float, float]]) -> str:
    """themes: (名前, 前日比, 5日比)。5日比の平均で追い風／向かい風。"""
    if not themes:
        return ""
    avg5 = sum(c5 for _, _, c5 in themes) / len(themes)
    return "追い風" if avg5 >= SECTOR_MOVE else "向かい風" if avg5 <= -SECTOR_MOVE else "横ばい"


# ---------- 下方向の圧力 ----------
@dataclass
class Pressure:
    level: str  # 例: 助長（強い）
    meaning: str  # 例: 下げながら買残が増える（しこり・投げ予備軍）
    price: float | None = None
    margin: float | None = None
    volume: float | None = None

    @property
    def sentence(self) -> str:
        """ひとこと用の文。"""
        return {
            "助長（強い）": "下げながら買残が増え、下方向の圧力が助長されている",
            "解放中": "出来高を伴って買残が減り、投げで圧力が解放されている",
            "弱まる": "買残・出来高とも減り、下方向の圧力は弱まっている",
            "くすぶる": "買残が減らないまま閑散で、下方向の圧力がくすぶっている",
            "あり": "下げているが、偏りは目立たない",
            "上値に重し": "上げで買残が増え、戻り売りが控えている",
            "軽い": "上げで買残が減り、上に軽い",
            "小さい": "上げで、偏りは目立たない",
            "たまる": "株価は横ばいで、買残だけがたまっている",
            "和らぐ": "株価は横ばいで、買残が減って圧力が和らいでいる",
            "中立": "大きな偏りはない",
        }.get(self.level, self.meaning)

    @property
    def breakdown(self) -> str:
        def pct(v: float) -> str:
            return "±0%" if abs(v) < 0.005 else f"{v:+.0%}"

        parts = []
        if self.price is not None:
            parts.append(f"株価{pct(self.price)}")
        if self.margin is not None:
            parts.append(f"買残{pct(self.margin)}")
        if self.volume is not None:
            parts.append(f"出来高{self.volume:.1f}倍")
        return "・".join(parts)


def pressure(prices: pd.DataFrame | None, margin_hist: pd.DataFrame) -> Pressure | None:
    """直近5営業日の株価・信用買残・出来高のバランスで、下方向の圧力が助長されているかを見る。"""
    if prices is None or len(prices) < 30:
        return None
    c = prices["Close"].dropna()
    price = float(c.iloc[-1] / c.iloc[-1 - PRESSURE_DAYS] - 1)
    vol = prices["Volume"].fillna(0)
    prev = float(vol.iloc[-1 - PRESSURE_DAYS - 20: -PRESSURE_DAYS].mean() or 0)
    volume = float(vol.tail(PRESSURE_DAYS).mean() / prev) if prev > 0 else None
    mh = margin_hist.sort_values("date").tail(PRESSURE_DAYS) if margin_hist is not None and len(margin_hist) else None
    margin = None
    if mh is not None and len(mh) >= 2 and mh["buy"].iloc[0] > 0:
        margin = float(mh["buy"].iloc[-1] / mh["buy"].iloc[0] - 1)
    m_up = margin is not None and margin >= MARGIN_MOVE
    m_down = margin is not None and margin <= -MARGIN_MOVE
    v_up = volume is not None and volume >= VOL_UP
    v_down = volume is not None and volume <= VOL_DOWN
    if price <= -PRICE_MOVE:
        if m_up:
            p = ("助長（強い）", "下げながら買残が増える（しこり増）")
        elif m_down and v_up:
            p = ("解放中", "出来高を伴い買残が減る（投げ）")
        elif m_down:
            p = ("弱まる", "買残も出来高も減る（売り枯れ寄り）")
        elif v_down:
            p = ("くすぶる", "買残が減らず閑散（未精算）")
        else:
            p = ("あり", "買残・出来高に偏りなし")
    elif price >= PRICE_MOVE:
        if m_up:
            p = ("上値に重し", "上げで買残が増える（戻り売り予備軍）")
        elif m_down:
            p = ("軽い", "上げで買残が減る（上に軽い）")
        else:
            p = ("小さい", "上げで買残に変化なし")
    else:
        if m_up:
            p = ("たまる", "横ばいで買残だけ増える")
        elif m_down:
            p = ("和らぐ", "横ばいで買残が減る")
        else:
            p = ("中立", "株価・買残とも変化なし")
    return Pressure(p[0], p[1], price, margin, volume)


# ---------- 表面の評価（PER・PBR） ----------
PER_BANDS = ((10, "割安"), (20, "普通"), (40, "やや割高"))  # これ以上は「割高」
PBR_BANDS = ((1, "割安"), (3, "普通"), (8, "やや割高"))


def _band(v: float, bands) -> str:
    for limit, label in bands:
        if v < limit:
            return label
    return "割高"


def valuation(per: float | None, pbr: float | None) -> str:
    """PER・PBR の表面的な水準だけで出す目安。PER が付けば PER で、赤字なら PBR で決める。
    妥当な株価そのものはファンダを見て自分で判断する前提。"""
    if per is not None and not math.isnan(per):
        return f"{_band(per, PER_BANDS)}（PER {per:.0f}倍）"
    if pbr is not None and not math.isnan(pbr):
        return f"{_band(pbr, PBR_BANDS)}（赤字のため PBR {pbr:.1f}倍）"
    return ""


# ---------- 新規材料 ----------
def materials(code: str, today: pd.Timestamp, root: Path = IR_DIR, days: int = IR_DAYS,
              kinds: tuple[str, ...] | None = None, exclude: tuple[str, ...] = ()) -> list[tuple[pd.Timestamp, str, float, str]]:
    """直近 days 日の適時開示（IR 材料ランキングの保存データ）。→ [(日付, 材料の種類, 材料度, 内容)]（新しい順）
    kinds を渡すとその種類だけ、exclude に渡した種類は除く。"""
    out = []
    if not root.exists():
        return out
    for path in sorted(root.glob("*.csv"), reverse=True):
        try:
            day = pd.Timestamp(path.stem)
        except ValueError:
            continue
        if (today - day).days > days:
            break
        df = pd.read_csv(path, dtype={"code": str})
        for _, r in df[df["code"] == code].iterrows():
            kind = str(r["kind"])
            if (kinds is not None and kind not in kinds) or kind in exclude:
                continue
            facts = "" if pd.isna(r.get("facts")) else str(r.get("facts"))
            out.append((day, kind, float(r["score"]), facts))
    return out


def record_start(root: Path = IR_DIR) -> pd.Timestamp | None:
    """IR 材料ランキングの記録が始まった日。"""
    days = [pd.Timestamp(p.stem) for p in root.glob("*.csv") if p.stem.isdigit()] if root.exists() else []
    return min(days) if days else None


def overhangs(code: str, today: pd.Timestamp, root: Path = IR_DIR) -> list[tuple[pd.Timestamp, str, float, str]]:
    """上値を押さえる材料（新株予約権・MSワラント・増資・売出し・大株主の異動）の直近1年分。"""
    return materials(code, today, root, OVERHANG_DAYS, kinds=OVERHANG_KINDS)


# ---------- 反発サイン ----------
def rebound_mark(prev: pd.Series, day: pd.Series) -> str:
    """1日ごとの反発の印。◎強い反発・○反発・△下げ渋り・×反発なし"""
    rng = day["High"] - day["Low"]
    pos = (day["Close"] - day["Low"]) / rng if rng > 0 else 0.5
    up = day["Close"] > prev["Close"]
    if day["Close"] > prev["High"] or (
            day["Close"] >= day["Open"] and day["Close"] / prev["Close"] - 1 >= STRONG_UP and pos >= UPPER_ZONE):
        return "◎"
    if up:
        return "○"
    return "△" if pos >= 0.5 else "×"


@dataclass
class Rebound:
    marks: list[tuple[pd.Timestamp, str, bool]] = field(default_factory=list)  # (日付, 印, 逆行陽線か)

    @property
    def days_text(self) -> str:
        return " ".join(f"{d:%-m/%-d}{m}{'逆' if a else ''}" for d, m, a in self.marks)

    @property
    def latest(self) -> str:
        if not self.marks:
            return "-"
        _, m, against = self.marks[-1]
        if against:
            return "逆行陽線" + ("（強い反発）" if m == "◎" else "")
        return {"◎": "強い反発", "○": "反発", "△": "下げ渋り", "×": "反発なし"}[m]


def rebound(prices: pd.DataFrame, growth: pd.DataFrame | None = None, days: int = REBOUND_DAYS) -> Rebound:
    """直近 days 日の反発を1日ずつ判定する。グロース250が前日比 BAD_DAY 以下の日の陽線は逆行陽線。"""
    p = prices.dropna(subset=["Close"])
    g_chg = {}
    if growth is not None and not growth.empty:
        gc = growth["Close"].dropna()
        g_chg = (gc / gc.shift(1) - 1).to_dict()
    r = Rebound()
    tail = p.tail(days + 1)
    for k in range(1, len(tail)):
        day = tail.iloc[k]
        bad = g_chg.get(tail.index[k])
        against = bad is not None and bad <= BAD_DAY and day["Close"] >= day["Open"]
        r.marks.append((tail.index[k], rebound_mark(tail.iloc[k - 1], day), against))
    return r


# ---------- 本尊の気配（足跡） ----------
# 本尊（主力）の売買は直接は見えないので、動いたときに残りやすい足跡を拾う。どれも「いるかもしれない」の目安。
FOOT_DAYS = 20  # 足跡を見る期間
UPDOWN_VOL = 1.5  # 上げた日の平均出来高が下げた日のこれ倍以上
RS_DAYS, RS_MARGIN = 10, 0.05  # 10日でグロース250より5%以上強い
AGAINST_COUNT = 2  # 逆行陽線がこの回数以上
FLOOR_BAND, FLOOR_TOUCH = 0.015, 3  # 安値の±1.5%以内で3回以上止まる
SELL_UP = 0.20  # 信用売残が20%以上増える
SELL_MIN = 0.05  # かつ売残が買残の5%以上（信用倍率20倍以下）のときだけ
SPIKE_VOL_X, SPIKE_UP = 3.0, 0.05  # 出来高3倍以上で5%以上上げた日
EXIT_DAYS = 5  # 手じまいの気配を見る日数
EXIT_VOL_X, SHADOW_X, SHADOW_MIN = 2.0, 2.0, 0.03  # 出来高2倍以上で、上ヒゲが実体の2倍以上・株価の3%以上
BREAK_VOL_X = 1.5  # 出来高1.5倍以上で直近20日の安値を割る
ACCUM_DAYS = 180  # 大株主の買い集めを見る期間
FOOT_TOTAL = 7


@dataclass
class Footprints:
    hits: list[tuple[str, str]] = field(default_factory=list)  # (足跡の名前, 中身)
    exits: list[str] = field(default_factory=list)  # 手じまいの気配

    @property
    def level(self) -> str:
        n = len(self.hits)
        base = "あり" if n >= 3 else "少し" if n == 2 else "見えない"
        return f"{base}（{n}/{FOOT_TOTAL}）"


def footprints(prices: pd.DataFrame | None, margin_hist: pd.DataFrame | None, growth: pd.DataFrame | None,
               code: str = "", today: pd.Timestamp | None = None, ir_root: Path = IR_DIR) -> Footprints | None:
    if prices is None or len(prices) < FOOT_DAYS + 25:
        return None
    p = prices.dropna(subset=["Close"])
    c, v = p["Close"], p["Volume"].fillna(0)
    f = Footprints()
    recent = p.tail(FOOT_DAYS)
    rc, rv = recent["Close"], recent["Volume"].fillna(0)
    chg = rc.diff()

    # 1. 出来高の偏り
    up_v, down_v = rv[chg > 0].mean(), rv[chg < 0].mean()
    if up_v and down_v and up_v / down_v >= UPDOWN_VOL:
        f.hits.append(("出来高", f"上げ日の商い 下げ日の{up_v / down_v:.1f}倍"))

    # 2. 株価と出来高のずれ（OBV が上向きなのに株価は横ばい・下げ）
    obv = (v * c.diff().apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)).cumsum()
    price_chg = float(c.iloc[-1] / c.iloc[-1 - FOOT_DAYS] - 1)
    obv_chg = float(obv.iloc[-1] - obv.iloc[-1 - FOOT_DAYS])
    if price_chg <= 0.0 and obv_chg > 0:
        f.hits.append(("OBV", f"株価{price_chg:+.0%}・出来高の累積は上向き"))

    # 3. 地合いへの逆行
    if growth is not None and not growth.empty:
        g = growth["Close"].dropna()
        g_chg = (g / g.shift(1) - 1).to_dict()
        s10 = float(c.iloc[-1] / c.iloc[-1 - RS_DAYS] - 1)
        before = g[g.index <= c.index[-1 - RS_DAYS]]
        g10 = float(g.iloc[-1] / before.iloc[-1] - 1) if not before.empty else None
        tail = p.tail(RS_DAYS)
        against = sum(1 for d, r in tail.iterrows()
                      if g_chg.get(d) is not None and g_chg[d] <= BAD_DAY and r["Close"] >= r["Open"])
        parts = []
        if g10 is not None and s10 - g10 >= RS_MARGIN:
            parts.append(f"10日でG250より{s10 - g10:+.0%}")
        if against >= AGAINST_COUNT:
            parts.append(f"逆行陽線{against}回")
        if parts:
            f.hits.append(("逆行", "・".join(parts)))

    # 4. 下値で止まる（同じ安値付近で何度も止まる）
    lows = recent["Low"]
    floor = float(lows.min())
    touch_days = lows[lows <= floor * (1 + FLOOR_BAND)]
    # 続けて止まった日は1回と数える
    touches = 0
    prev_i = -2
    for i, ok in enumerate(lows <= floor * (1 + FLOOR_BAND)):
        if ok and i != prev_i + 1:
            touches += 1
        if ok:
            prev_i = i
    if touches >= FLOOR_TOUCH and len(touch_days):
        f.hits.append(("下値", f"{floor:,.0f}円付近で{touches}回止まる"))

    # 5. 信用売残の増加
    if margin_hist is not None and len(margin_hist) >= 2 and "sell" in margin_hist:
        mh = margin_hist.sort_values("date").tail(EXIT_DAYS + 1)
        s0, s1 = float(mh["sell"].iloc[0]), float(mh["sell"].iloc[-1])
        b1 = float(mh["buy"].iloc[-1]) if "buy" in mh else 0.0
        # もともと売残が少ないと率が大きく出るので、売残が買残の SELL_MIN 以上あるときだけ数える
        if s0 > 0 and s1 / s0 - 1 >= SELL_UP and b1 > 0 and s1 >= b1 * SELL_MIN:
            f.hits.append(("売残", f"{s1 / s0 - 1:+.0%}（踏み上げの燃料）"))

    # 6. 大株主の買い集め
    if code and today is not None:
        acc = materials(code, today, ir_root, ACCUM_DAYS, kinds=("大株主の買い集め",))
        if acc:
            f.hits.append(("買い集め", f"{acc[0][0]:%-m/%-d} 大株主の買い集めの開示"))

    # 7. 大商いの後の値持ち
    base_v = v.shift(1).rolling(20).mean()
    spikes = [d for d in recent.index
              if base_v.get(d, 0) and v[d] >= base_v[d] * SPIKE_VOL_X and c[d] / c.shift(1)[d] - 1 >= SPIKE_UP]
    if spikes:
        d = spikes[-1]
        after = p[p.index > d]
        if len(after) >= 1 and float(after["Close"].min()) >= float(p.loc[d, "Low"]):
            f.hits.append(("値持ち", f"{d:%-m/%-d}の大商い（{v[d] / base_v[d]:.1f}倍）の安値を割らず{len(after)}日"))

    # 手じまいの気配
    avg20 = v.iloc[-EXIT_DAYS - 20:-EXIT_DAYS].mean()
    low20 = float(p["Low"].iloc[-EXIT_DAYS - 20:-EXIT_DAYS].min())
    for d, r in p.tail(EXIT_DAYS).iterrows():
        body = abs(r["Close"] - r["Open"])
        upper = r["High"] - max(r["Open"], r["Close"])
        if avg20 and r["Volume"] >= avg20 * EXIT_VOL_X and upper >= SHADOW_X * body and upper >= r["Close"] * SHADOW_MIN:
            f.exits.append(f"{d:%-m/%-d} 大商いで長い上ヒゲ")
        if avg20 and r["Volume"] >= avg20 * BREAK_VOL_X and r["Close"] < low20:
            f.exits.append(f"{d:%-m/%-d} 出来高を伴う安値割れ")
    return f
