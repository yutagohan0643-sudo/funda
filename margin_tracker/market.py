"""地合い（マクロ）のまとめ。

ルールの「見るもの」= VIX・米国・日経（週足まで）・グロース・金利・原油 に、ドル円と米国のセクター別の騰落を加える。
データは Yahoo Finance の日足（遅延あり）。米国の値は日本時間の朝に確定した前日（米国時間）の終値。

判定の基準値はこのファイルの上の方にまとめてある。数字を変えると「追い風／向かい風」の出方が変わる。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd
import requests

# ---- 判定の基準値 ----
EQUITY_MOVE = 0.005  # 株価指数: 前日比 ±0.5% 以上で追い風／向かい風
VIX_CALM, VIX_FEAR = 20.0, 30.0  # VIX: 20未満は落ち着き、30以上は恐怖
VIX_PEAKOUT = 0.85  # VIX が直近5日の最高値の85%以下まで下がったら「ピークアウト（底の候補）」
RATE_MOVE_BP = 0.05  # 米10年金利: 前日比 ±0.05%（5bp）以上で向かい風／追い風
COMMODITY_MOVE = 0.03  # 原油: 前日比 ±3% 以上で「急変動」
FX_MOVE = 0.01  # ドル円: 前日比 ±1% 以上で「急変動」
CRYPTO_MOVE = 0.05  # ビットコイン: 前日比 ±5% 以上で「急変動」（投機マネーの温度感）
WEEKLY_MA = 13  # 日経平均の週足: 13週移動平均の上か下か

TAIL, NEUTRAL, HEAD, WATCH = "追い風", "中立", "向かい風", "注意"


@dataclass
class Indicator:
    symbol: str
    name: str
    region: str  # "米国" / "日本" / "その他"
    kind: str  # equity / vix / rate / commodity / fx


INDICATORS = [
    Indicator("^GSPC", "S&P500", "米国", "equity"),
    Indicator("^IXIC", "NASDAQ", "米国", "equity"),
    Indicator("^DJI", "ダウ", "米国", "equity"),
    Indicator("^SOX", "半導体（SOX）", "米国", "equity"),
    Indicator("ES=F", "S&P500先物", "米国", "equity"),
    Indicator("IWM", "米小型株（ラッセル2000）", "米国", "equity"),
    Indicator("^VIX", "VIX（恐怖指数）", "米国", "vix"),
    Indicator("^TNX", "米10年金利", "米国", "rate"),
    Indicator("^N225", "日経平均", "日本", "equity"),
    Indicator("NIY=F", "日経先物（CME）", "日本", "equity"),
    Indicator("2516.T", "グロース250（ETF）", "日本", "equity"),
    Indicator("1306.T", "TOPIX（ETF）", "日本", "equity"),
    Indicator("CL=F", "原油（WTI）", "その他", "commodity"),
    Indicator("JPY=X", "ドル円", "その他", "fx"),
    Indicator("BTC-USD", "ビットコイン", "その他", "crypto"),
]

SECTORS = [
    ("XLK", "情報技術"), ("SMH", "半導体"), ("XLC", "通信"), ("XLY", "一般消費財"), ("XLF", "金融"),
    ("XLI", "資本財"), ("XLV", "ヘルスケア"), ("XLP", "生活必需品"), ("XLE", "エネルギー"),
    ("XLB", "素材"), ("XLU", "公益"), ("XLRE", "不動産"),
]

# テーマ・業種別（米国のETFで見る）。(記号, 名前, 分類)
THEMES = [
    ("SOXX", "半導体", "テック"), ("IGV", "ソフトウェア", "テック"), ("SKYY", "クラウド", "テック"),
    ("CIBR", "サイバーセキュリティ", "テック"), ("AIQ", "AI・ビッグデータ", "テック"), ("BOTZ", "ロボット・AI", "テック"),
    ("DTCR", "データセンター", "テック"), ("QTUM", "量子コンピュータ", "テック"), ("FDN", "インターネット", "テック"),
    ("ESPO", "ゲーム・eスポーツ", "テック"), ("XTL", "通信", "テック"), ("MAGS", "マグニフィセント7", "テック"),
    ("ARKK", "イノベーション（ARK）", "テック"),
    ("UFO", "宇宙", "宇宙・防衛"), ("ITA", "防衛・航空宇宙", "宇宙・防衛"),
    ("DRIV", "自動運転・EV", "乗り物"), ("LIT", "電池・リチウム", "乗り物"), ("JETS", "航空", "乗り物"),
    ("IYT", "運輸", "乗り物"),
    ("URA", "原子力・ウラン", "エネルギー"), ("TAN", "太陽光", "エネルギー"), ("ICLN", "クリーンエネルギー", "エネルギー"),
    ("XOP", "石油・ガス開発", "エネルギー"),
    ("GDX", "金鉱", "素材"), ("COPX", "銅", "素材"), ("SLX", "鉄鋼", "素材"), ("XME", "金属・鉱業", "素材"),
    ("XBI", "バイオ", "ヘルスケア"), ("IHI", "医療機器", "ヘルスケア"),
    ("KBE", "銀行", "金融"), ("KRE", "地方銀行", "金融"), ("KIE", "保険", "金融"), ("BITQ", "暗号資産関連", "金融"),
    ("ITB", "住宅建設", "消費・住宅"), ("XRT", "小売", "消費・住宅"),
]
SPARK_URL = "https://query1.finance.yahoo.com/v7/finance/spark"
SPARK_BATCH = 20  # Yahoo の spark は1回20銘柄まで

GROWTH_SYMBOL, TOPIX_SYMBOL = "2516.T", "1306.T"


@dataclass
class Reading:
    ind: Indicator
    close: float
    change: float  # 前日比（金利は差、それ以外は率）
    change5: float  # 5日前比（同上）
    date: pd.Timestamp
    label: str
    note: str = ""


@dataclass
class Market:
    readings: list[Reading] = field(default_factory=list)
    sectors: list[tuple[str, str, float, float, pd.Timestamp]] = field(default_factory=list)  # (記号, 名前, 前日比, 5日比, 日付)
    themes: list[tuple[str, str, float, float, pd.Timestamp, str]] = field(default_factory=list)  # 上に分類を加えたもの
    series: dict[str, pd.DataFrame] = field(default_factory=dict)  # 日足（銘柄の下げと比べる用）
    errors: list[str] = field(default_factory=list)

    def count(self, label: str) -> int:
        return sum(r.label == label for r in self.readings)


def _change(s: pd.Series, n: int, diff: bool) -> float:
    if len(s) <= n:
        return float("nan")
    return float(s.iloc[-1] - s.iloc[-1 - n]) if diff else float(s.iloc[-1] / s.iloc[-1 - n] - 1)


def judge(ind: Indicator, df: pd.DataFrame) -> Reading:
    c = df["Close"].dropna()
    diff = ind.kind == "rate"
    chg, chg5 = _change(c, 1, diff), _change(c, 5, diff)
    label, note = NEUTRAL, ""
    if ind.kind == "equity":
        label = TAIL if chg >= EQUITY_MOVE else HEAD if chg <= -EQUITY_MOVE else NEUTRAL
        if len(c) >= 25:
            note = "25日線の上" if c.iloc[-1] >= c.tail(25).mean() else "25日線の下"
        if ind.symbol == "^N225":
            wk = c.resample("W-FRI").last().dropna()
            if len(wk) >= WEEKLY_MA:
                ma = wk.rolling(WEEKLY_MA).mean()
                up = wk.iloc[-1] >= ma.iloc[-1]
                note += f"／週足: 13週線の{'上' if up else '下'}"
                if not up:
                    note += "（週足で怪しい→利確早め）"
    elif ind.kind == "vix":
        v = float(c.iloc[-1])
        if v >= VIX_FEAR:
            label, note = HEAD, "恐怖"
            if v <= c.tail(5).max() * VIX_PEAKOUT:
                label, note = WATCH, "恐怖からピークアウト＝底の候補"
        elif v >= VIX_CALM:
            label, note = HEAD, "警戒"
        else:
            label, note = TAIL, "落ち着き"
    elif ind.kind == "rate":
        label = HEAD if chg >= RATE_MOVE_BP else TAIL if chg <= -RATE_MOVE_BP else NEUTRAL
        note = "金利上昇はグロースに逆風" if label == HEAD else ""
    elif ind.kind == "commodity":
        label = WATCH if abs(chg) >= COMMODITY_MOVE else NEUTRAL
        note = "急変動" if label == WATCH else ""
    elif ind.kind == "crypto":
        label = WATCH if abs(chg) >= CRYPTO_MOVE else NEUTRAL
        note = "急変動" if label == WATCH else ""
    elif ind.kind == "fx":
        label = WATCH if abs(chg) >= FX_MOVE else NEUTRAL
        note = ("円安" if chg > 0 else "円高") + ("が急進" if label == WATCH else "")
    return Reading(ind, float(c.iloc[-1]), chg, chg5, c.index[-1], label, note)


def spark_closes(symbols: list[str]) -> dict[str, pd.Series]:
    """Yahoo の spark で、終値（直近1か月）を20銘柄ずつまとめて取る。"""
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0"
    out = {}
    for i in range(0, len(symbols), SPARK_BATCH):
        r = s.get(SPARK_URL, params={"symbols": ",".join(symbols[i:i + SPARK_BATCH]), "range": "1mo", "interval": "1d"},
                  timeout=30)
        r.raise_for_status()
        for x in r.json()["spark"]["result"] or []:
            resp = x["response"][0]
            tz = resp["meta"].get("exchangeTimezoneName", "America/New_York")
            idx = pd.to_datetime(resp["timestamp"], unit="s", utc=True).tz_convert(tz).normalize().tz_localize(None)
            out[x["symbol"]] = pd.Series(resp["indicators"]["quote"][0]["close"], index=idx, dtype=float).dropna()
        time.sleep(0.5)
    return out


def fetch(fetcher, start, closes_fetcher=spark_closes) -> Market:
    """fetcher(symbol, start) → 日足 DataFrame（地合いの指標用）。
    closes_fetcher(symbols) → {記号: 終値}（セクター・テーマ用。まとめて取る）。"""
    m = Market()
    for ind in INDICATORS:
        try:
            df = fetcher(ind.symbol, start)
            m.series[ind.symbol] = df
            m.readings.append(judge(ind, df))
        except Exception as e:  # 取れない指標があってもレポートは出す
            m.errors.append(f"{ind.name}: {e}")
        time.sleep(0.3)
    try:
        closes = closes_fetcher([sym for sym, _ in SECTORS] + [sym for sym, _, _ in THEMES])
    except Exception as e:
        closes = {}
        m.errors.append(f"セクター・テーマ: {e}")
    for sym, name in SECTORS:
        c = closes.get(sym)
        if c is None or len(c) < 2:
            m.errors.append(f"{name}: データなし")
            continue
        m.sectors.append((sym, name, _change(c, 1, False), _change(c, 5, False), c.index[-1]))
    for sym, name, group in THEMES:
        c = closes.get(sym)
        if c is None or len(c) < 2:
            m.errors.append(f"{name}: データなし")
            continue
        m.themes.append((sym, name, _change(c, 1, False), _change(c, 5, False), c.index[-1], group))
    m.sectors.sort(key=lambda x: x[2], reverse=True)
    m.themes.sort(key=lambda x: x[2], reverse=True)
    return m


THEME_MOVE = 0.03  # 監視銘柄の関連テーマが前日比 ±3% 以上なら、注目ポイントに出す


def find_theme(m: Market, key: str) -> tuple[str, str, float, float] | None:
    """テーマ・セクターを名前か ETF の記号で探す。→ (名前, 記号, 前日比, 5日比)"""
    k = key.strip().upper()
    for sym, name, c, c5, *_ in list(m.themes) + list(m.sectors):
        if k in (name.upper(), sym.upper()):
            return name, sym, c, c5
    return None


def overall(m: Market) -> str:
    tail, head = m.count(TAIL), m.count(HEAD)
    if head >= tail + 3:
        return "向かい風が強い"
    if tail >= head + 3:
        return "追い風"
    return "まちまち"
