from datetime import date, datetime
import numpy as np

import pandas as pd
import pytest

from margin_tracker import jpx, store
from margin_tracker.report import build, read_watchlist, summarize

# 実際の PDF（2026/11/19 申込み分）から読み取ったテキストの抜粋
SAMPLE = """銘柄別信用取引残高
2026/11/19申込み現在 東京証券取引所株式部 （単位：一株、一円） 2026/11/20
B イオレ 普通株式 グロース 制 23340 JP3131380002株数 Shs. 900 ▲ 20,500 0.0% 12,801,300 1,075,400 29.7% 900 ▲ 20,500 0 0 4,987,800 468,200 7,813,500 607,200
eole Inc. Growth Margin 23340 JP3131380002金額 Val. 504,000 ▲ 10,816,200 - 6,862,195,200 528,726,100 - 504,000 ▲ 10,816,200 0 0 2,586,719,800 232,877,300 4,275,475,400 295,848,800
B ティアフォー 普通株式 グロース 制 593A0 JP3539190003株数 Shs. 0 ▲ 300 0.0% 3,082,800 ▲ 155,300 4.9% 0 ▲ 300 0 0 1,053,900 ▲ 36,400 2,028,900 ▲ 118,900
TIER IV,Inc. Growth Margin 593A0 JP3539190003金額 Val. 0 ▲ 545,400 - 7,294,072,900 ▲ 348,279,100 - 0 ▲ 545,400 0 0 2,422,975,300 ▲ 88,828,300 4,871,097,600 ▲ 259,450,800
A ＮＥＸＴ ＦＵＮＤＳ 崩れた行 投Ｍ信Ｃ等 JP 30受46益51証00券08株数 Shs. 0 0 * 1,482 0 * 0 0 0 0 1,231 0 251 0
"""


def test_numbers_handles_minus_mark_percent_and_blank():
    assert jpx._numbers("900 ▲ 20,500 0.0% 29.7% * -") == [900, -20500, 0.0, pytest.approx(0.297), None, None]


def test_parse_text_reads_stocks_and_skips_broken_rows():
    as_of, df = jpx.parse_text(SAMPLE)
    assert as_of == date(2026, 11, 19)
    assert sorted(df["code"]) == ["2334", "593A"]
    eole = df.set_index("code").loc["2334"]
    assert eole["buy"] == 12_801_300 and eole["buy_chg"] == 1_075_400
    assert eole["sell_chg"] == -20_500
    assert eole["buy_ratio"] == pytest.approx(0.297)
    assert eole["buy_val"] == 6_862_195_200
    assert df.set_index("code").loc["593A", "buy_chg"] == -155_300


def test_list_files_extracts_dates(monkeypatch):
    html = '<a href="/x/tvdiv-att/20261119_mtall.pdf">a</a><a href="/x/tvdiv-att/20261118_mtall.pdf">b</a>'

    class Resp:
        text = html

        def raise_for_status(self):
            pass

    class Sess:
        def get(self, url, timeout):
            return Resp()

    files = jpx.list_files(Sess())
    assert list(files) == [date(2026, 11, 18), date(2026, 11, 19)]
    assert files[date(2026, 11, 19)].startswith("https://www.jpx.co.jp/x/tvdiv-att/")


def _history(tmp_path, days):
    _, df = jpx.parse_text(SAMPLE)
    for i, day in enumerate(days):
        d = df.copy()
        d["buy"] = d["buy"] + i * 100_000
        store.save(day, d, tmp_path)
    return store.history(["2334", "593A"], tmp_path)


def test_store_roundtrip_keeps_watchlist_history(tmp_path):
    days = [date(2026, 11, d) for d in (13, 16, 17)]
    hist = _history(tmp_path, days)
    assert store.saved_days(tmp_path) == days
    eole = hist[hist["code"] == "2334"]
    assert list(eole["buy"]) == [12_801_300, 12_901_300, 13_001_300]


def test_summarize_cost_basis_profit_and_volume_days(tmp_path):
    hist = _history(tmp_path, [date(2026, 11, d) for d in (10, 11, 12, 13, 16, 17)])
    eole = hist[hist["code"] == "2334"].reset_index(drop=True)
    prices = pd.DataFrame({"Close": [500.0] * 20, "Volume": [1_000_000] * 20},
                          index=pd.bdate_range("2026-10-20", periods=20))
    s = summarize(eole, prices)
    assert s["avg_cost"] == pytest.approx(6_862_195_200 / (12_801_300 + 500_000))
    assert s["pl"] == pytest.approx(500 / s["avg_cost"] - 1)
    assert s["buy_5d"] == pytest.approx((12_801_300 + 500_000) / 12_801_300 - 1)
    assert s["days_of_volume"] == pytest.approx((12_801_300 + 500_000) / 1_000_000)


def test_build_report_handles_missing_stock_and_prices(tmp_path):
    hist = _history(tmp_path, [date(2026, 11, 13), date(2026, 11, 16)])
    md = build([("2334", "イオレ"), ("9999", "存在しない")], hist, {"2334": None}, tmp_path / "out",
               now=datetime(2026, 11, 21, 17, 0))
    md = md.replace("\u2060", "")  # 改行防止の記号を除いて比べる
    assert "11/16 申込み分" in md
    assert "### イオレ（2334）" in md and "| 分析 |" in md
    assert "データなし" in md and "存在しない（9999）" in md
    assert "## 今日のまとめ" not in md and "| 買残 | 万株 | 直近 | 万株 |" in md and "| ファンダ |" in md
    assert (tmp_path / "out" / "charts" / "2334.png").exists()


def test_signals_flags_surge_deep_loss_and_heavy_volume():
    from margin_tracker.report import signals

    s = {"buy": 1_100_000, "buy_chg": 100_000, "pl": -0.3, "avg_cost": 700.0, "days_of_volume": 4.0, "dist": None}
    text = " / ".join(signals(s))
    assert "買残が急増 +10.0%" in text and "含み損 -30%" in text and "4.0日分" in text
    quiet = {"buy": 1_000_000, "buy_chg": 10_000, "pl": -0.05, "avg_cost": 500.0, "days_of_volume": 1.0, "dist": None}
    assert signals(quiet) == []


def test_read_watchlist(tmp_path):
    f = tmp_path / "w.txt"
    f.write_text("# コメント\n2334 イオレ\n593a ティアフォー [自動運転・EV, SOXX]  # メモ\n4493 CSC [サイバーセキュリティ、クラウド]\n\n",
                 encoding="utf-8")
    assert read_watchlist(f) == [("2334", "イオレ"), ("593A", "ティアフォー"), ("4493", "CSC")]
    from margin_tracker.report import read_themes

    assert read_themes(f) == {"593A": ["自動運転・EV", "SOXX"], "4493": ["サイバーセキュリティ", "クラウド"]}


def test_theme_text_finds_by_name_or_symbol_and_flags_big_moves():
    from margin_tracker.report import _theme_text

    m = mk.Market()
    m.themes = [("CIBR", "サイバーセキュリティ", -0.051, -0.038, pd.Timestamp("2026-11-20"), "テック")]
    m.sectors = [("XLK", "情報技術", 0.01, 0.02, pd.Timestamp("2026-11-20"))]
    text, sig = _theme_text(m, ["サイバーセキュリティ", "xlk", "存在しない"])
    assert "サイバーセキュリティ **-5.1%**（5日 -3.8%）" in text and "情報技術 **+1.0%**" in text
    assert "存在しない（テーマ名が見つからない）" in text
    assert sig == ["前夜の米国で関連テーマが大きく下落：サイバーセキュリティ -5.1%"]
    assert _theme_text(m, []) == ("", [])


# ---- 価格帯別の保有者分布 ----
from margin_tracker.cost_dist import estimate as estimate_cost, estimate_listed_shares  # noqa: E402


def _bars(highs, lows, closes, volumes):
    idx = pd.bdate_range("2026-01-05", periods=len(closes))
    return pd.DataFrame({"High": highs, "Low": lows, "Close": closes, "Volume": volumes}, index=idx)


def test_listed_shares_from_rounded_ratios():
    # 実際の上場株式数 1,000万株。上場比は 0.1% 単位に丸められている
    hist = pd.DataFrame({"buy": [523_000, 547_000, 581_000], "buy_ratio": [0.052, 0.055, 0.058],
                         "sell": [0, 0, 0], "sell_ratio": [0.0, 0.0, 0.0]})
    assert estimate_listed_shares(hist) == pytest.approx(10_000_000, rel=0.01)
    assert estimate_listed_shares(hist.assign(buy_ratio=0.0)) is None


def test_full_turnover_moves_all_holders_to_latest_prices():
    # 最初は 1000円付近、その後に上場株数と同じ出来高で 500円付近に移る → 全員 500円付近
    px = _bars([1010, 510], [990, 490], [1000, 500], [0, 1_000_000])
    d = estimate_cost(px, listed_shares=1_000_000)
    assert d.avg_cost == pytest.approx(500, abs=10)
    assert d.profit_ratio == pytest.approx(0.5, abs=0.1)


def test_partial_turnover_mixes_old_and_new_holders():
    px = _bars([1010, 510], [990, 490], [1000, 500], [0, 250_000])
    d = estimate_cost(px, listed_shares=1_000_000)
    # 25% が 500円付近、75% が 1000円付近
    assert d.avg_cost == pytest.approx(0.75 * 1000 + 0.25 * 500, abs=15)
    assert d.weights[d.prices > 900].sum() == pytest.approx(0.75, abs=0.01)
    assert d.profit_ratio == pytest.approx(0.125, abs=0.05)
    bands = sorted(d.bands)
    assert len(bands) == 2 and bands[0][1] < 600 and bands[1][0] > 900
    assert abs(sum(w for _, _, w in d.bands) - 1) < 0.5


def test_calibrate_recovers_the_decay_that_matches_margin_cost():
    from margin_tracker.cost_dist import calibrate

    rng = np.random.default_rng(3)
    close = 1000 * np.exp(rng.normal(0, 0.03, 120).cumsum())
    px = _bars(close * 1.02, close * 0.98, close, rng.integers(50_000, 200_000, 120).astype(float))
    listed = 2_000_000
    days = px.index[-3:]
    # 係数 0.3 のモデルの平均コストを「実際の平均建値」とする
    costs = [estimate_cost(px[px.index <= d], listed, decay=0.3).avg_cost for d in days]
    hist = pd.DataFrame({"date": days, "buy": [100_000] * 3, "buy_val": [c * 100_000 for c in costs]})
    cal = calibrate(px, listed, hist)
    assert cal.decay == 0.3 and abs(cal.error) < 1e-9
    assert set(cal.errors) >= {0.1, 1.0}


def test_calibrate_prefers_default_when_difference_is_small():
    from margin_tracker.cost_dist import calibrate

    px = _bars([1010.0] * 30, [990.0] * 30, [1000.0] * 30, [100_000.0] * 30)  # 値動きなし → どの係数も同じ
    hist = pd.DataFrame({"date": px.index[-2:], "buy": [1000, 1000], "buy_val": [1_000_000, 1_000_000]})
    assert calibrate(px, 1_000_000, hist).decay == 1.0


# ---- 抵抗になりやすい位置 ----
from margin_tracker.levels import resistances  # noqa: E402


def _ohlc(rows):
    idx = pd.bdate_range("2026-08-03", periods=len(rows))
    df = pd.DataFrame(rows, columns=["High", "Low", "Close"], index=idx)
    df["Open"] = df["Close"]
    df["Volume"] = 1000.0
    return df


def test_resistance_finds_swing_high_gap_and_margin_cost_nearest_first():
    rows = [(100, 98, 99)] * 10 + [(130, 120, 125)] + [(110, 100, 105)] * 10 + [(98, 95, 96)] + [(97, 94, 95)] * 8
    # 10日目に高値130の山、21日目に「前日安値100 > 当日高値98」の下落の窓（上端100）
    px = _ohlc(rows)
    levels = resistances(px, close=95.0, dist=None, margin_cost=140.0, top=5)
    prices = [lv.low for lv in levels]
    assert prices == sorted(prices)
    assert any(lv.low <= 100 <= lv.high and any("下落の窓" in r for r in lv.reasons) for lv in levels)
    assert any(lv.low == pytest.approx(130) and any("跳ね返された高値" in r for r in lv.reasons) for lv in levels)
    assert all(lv.low > 95 for lv in levels)


def test_resistance_merges_nearby_reasons_and_counts_kinds_once():
    # 安値は前後の日と重なるようにして、窓ができないようにする
    rows = [(100, 98, 99)] * 8 + [(120, 99, 115)] + [(100, 98, 99)] * 6 + [(121, 99, 115)] + [(100, 98, 99)] * 6
    px = _ohlc(rows)
    levels = resistances(px, close=99.0, dist=None, margin_cost=121.5)
    lv = next(lv for lv in levels if lv.low >= 119)
    assert lv.low == pytest.approx(120) and lv.high == pytest.approx(121.5)
    assert lv.score == 3  # 2回止められた高値（根拠2つ分）＋信用建値
    assert lv.stars == "★★★"
    assert lv.reasons[0].startswith("2回止められた高値（")


def test_no_resistance_when_price_at_top():
    px = _ohlc([(100 + i, 99 + i, 100 + i) for i in range(30)])
    assert resistances(px, close=200.0, dist=None, margin_cost=float("nan")) == []


def test_supports_find_rebound_low_and_gap_up_nearest_first():
    from margin_tracker.levels import supports

    # 上昇の窓（前日高値 80 < 当日安値 85）→ 110 まで上げて 92 まで押し、105 まで反発（+14%）→ 現在 100
    rows = ([(70, 60, 65)] * 30 + [(80, 75, 78)] + [(95, 85, 90)]
            + [(112, 100, 110)] * 6 + [(100, 92, 95)] + [(105, 96, 100)] * 7 + [(104, 99, 100)])
    px = _ohlc(rows)
    levels = supports(px, close=100.0, dist=None, margin_cost=float("nan"), top=6)
    nears = [lv.near for lv in levels]
    assert nears == sorted(nears, reverse=True)  # 株価に近い順（高い順）
    assert all(n < 100 for n in nears)
    rebound = next(lv for lv in levels if any("反発した安値" in r for r in lv.reasons))
    assert rebound.low == pytest.approx(92)
    assert any("+14%" in r for r in rebound.reasons)
    reasons = " / ".join(r for lv in levels for r in lv.reasons)
    assert "上昇の窓" in reasons
    assert all(lv.side == "down" and lv.distance(100) < 0 for lv in levels)


def test_supports_ignore_small_rebounds_and_lows_older_than_half_year():
    from margin_tracker.levels import supports

    # 7か月前の安値 50（半年より前）と、反発が +2% しかない安値 95 は数えない
    rows = ([(60, 50, 55)] + [(70, 65, 68)] * 20 + [(110, 105, 108)] * 130
            + [(100, 95, 97)] + [(97, 96, 97)] * 8 + [(98, 96, 97)] * 5)
    levels = supports(_ohlc(rows), close=97.0, dist=None, margin_cost=float("nan"))
    reasons = " / ".join(r for lv in levels for r in lv.reasons)
    assert "反発した安値" not in reasons


def test_supports_count_two_rebounds_as_stronger():
    from margin_tracker.levels import supports

    dip = [(100, 90, 95)] + [(110, 100, 105)] * 10
    rows = [(110, 100, 105)] * 10 + dip + [(110, 100, 105)] * 5 + [(101, 91, 96)] + [(110, 100, 105)] * 10
    levels = supports(_ohlc(rows), close=105.0, dist=None, margin_cost=float("nan"))
    lv = next(lv for lv in levels if any("反発した安値" in r for r in lv.reasons))
    assert lv.low == pytest.approx(90) and 91 <= lv.high <= 90 * 1.02
    assert any(r.startswith("2回反発した安値") for r in lv.reasons)
    assert lv.score >= 2


# ---- 地合い ----
from margin_tracker import energy as en, market as mk  # noqa: E402


def _series(closes, start="2026-01-05"):
    idx = pd.bdate_range(start, periods=len(closes))
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c, "Volume": 1000.0}, index=idx)


def test_market_labels_equity_vix_and_rate():
    ind = {i.symbol: i for i in mk.INDICATORS}
    up = mk.judge(ind["^GSPC"], _series([100] * 30 + [101]))
    down = mk.judge(ind["^GSPC"], _series([100] * 30 + [98]))
    assert up.label == mk.TAIL and down.label == mk.HEAD and down.note == "25日線の下"
    assert mk.judge(ind["^VIX"], _series([15] * 10)).label == mk.TAIL
    assert mk.judge(ind["^VIX"], _series([20] * 9 + [35])).note == "恐怖"
    peak = mk.judge(ind["^VIX"], _series([20] * 5 + [40, 38, 36, 34, 33]))
    assert peak.label == mk.WATCH and "ピークアウト" in peak.note
    rate = mk.judge(ind["^TNX"], _series([4.0] * 10 + [4.1]))
    assert rate.label == mk.HEAD and rate.change == pytest.approx(0.1)


def test_market_fetch_sorts_sectors_and_survives_failures(monkeypatch):
    monkeypatch.setattr(mk.time, "sleep", lambda s: None)

    def fetcher(sym, start):
        if sym == "CL=F":
            raise ValueError("取得失敗")
        return _series([100] * 10 + [100 + hash(sym) % 7 - 3])

    def closes(symbols):
        return {s: pd.Series([100.0] * 6 + [100 + hash(s) % 7 - 3], index=pd.bdate_range("2026-01-05", periods=7))
                for s in symbols if s != "QTUM"}

    m = mk.fetch(fetcher, "2026-01-01", closes)
    assert any("量子" in e for e in m.errors) and len(m.themes) == len(mk.THEMES) - 1
    assert [c for *_, c, _, _, _ in m.themes] == sorted((c for *_, c, _, _, _ in m.themes), reverse=True)
    assert any("原油" in e for e in m.errors)
    days = [c for _, _, c, _, _ in m.sectors]
    assert days == sorted(days, reverse=True)
    assert mk.overall(m) in ("向かい風が強い", "追い風", "まちまち")


# ---- 下げの状態 ----
def _decline_prices():
    """上昇→高値→出来高を伴う投げ→細って下ヒゲ、の形"""
    closes = list(np.linspace(80, 100, 60)) + [96, 92, 85, 80, 81, 82]
    df = _series(closes, start="2026-06-01")
    df.loc[df.index[-4], "Volume"] = 6000.0  # 投げの大商い
    df.loc[df.index[-3:], "Volume"] = 1200.0
    last = df.index[-1]
    df.loc[last, ["Open", "Close", "Low", "High"]] = [82, 82.5, 76, 83]  # 長い下ヒゲ
    return df














# ---- ファンダ ----
from margin_tracker import fundamentals as fd  # noqa: E402


def _quarters(rev, op, net, shares, equity=None, cash=None, debt=None):
    idx = pd.to_datetime(["2025-06-30", "2025-09-30", "2025-12-31", "2026-03-31", "2026-06-30"])
    data = {"revenue": rev, "op": op, "net": net, "shares": shares}
    for k, v in (("equity", equity), ("cash", cash), ("debt", debt)):
        if v is not None:
            data[k] = v
    return pd.DataFrame(data, index=idx, dtype=float)


TODAY = pd.Timestamp("2026-08-20")


def test_fundamentals_growth_and_valuation():
    q = _quarters([100, 105, 110, 115, 130], [10, 11, 12, 13, 16], [7, 8, 8, 9, 11], [1e6] * 5, equity=[500] * 5)
    q[["revenue", "op", "net", "equity"]] *= 1e6
    f = fd.evaluate(q, close=1000.0, today=TODAY)
    assert f.verdict == "増収増益"
    assert f.op_yoy == pytest.approx(0.6) and f.rev_yoy == pytest.approx(0.3)
    assert f.market_cap == pytest.approx(1e9)
    assert f.per == pytest.approx(1e9 / 36e6)  # 直近4四半期の純利益 8+8+9+11
    assert f.pbr == pytest.approx(2.0)
    assert f.period == "2026年4-6月期"
    assert "増収増益" in f.positives and f.concerns == []


def test_fundamentals_flags_concerns():
    q = _quarters([100, 100, 100, 100, 120], [10, 5, 3, 2, 5], [5, -1, -2, -3, -1], [1.0e6, 1.0e6, 1.0e6, 1.0e6, 1.1e6],
                  equity=[100, 90, 80, 70, 60], debt=[50, 60, 70, 80, 90])
    f = fd.evaluate(q, close=100.0, today=TODAY)
    text = " / ".join(f.concerns)
    assert f.verdict == "増収減益"
    assert "営業利益 -50%（前年比）" in text
    assert "増収減益" in text and "利益率" in text
    assert "株数 +10%/年" in text
    assert "負債が自己資本の1.5倍" in text
    assert "純損失" in text


def test_fundamentals_loss_runway_and_stale():
    q = _quarters([50] * 5, [-10, -10, -12, -14, -16], [-10] * 5, [1e6] * 5, cash=[100, 90, 80, 70, 60])
    f = fd.evaluate(q, close=100.0, today=pd.Timestamp("2027-01-31"))
    text = " / ".join(f.concerns)
    assert f.verdict == "営業赤字が続く（拡大）"
    assert "営業赤字" in text and "現金 約5四半期分" in text
    assert "未反映" in text


def test_fundamentals_turnaround():
    q = _quarters([50] * 5, [-5, -2, 0, 3, 4], [1] * 5, [1e6] * 5)
    assert fd.evaluate(q, close=100.0, today=TODAY).verdict == "営業黒字に転換"


def test_build_report_shows_fundamentals_and_listed_ratio(tmp_path):
    hist = _history(tmp_path, [date(2026, 11, 13), date(2026, 11, 16)])
    q = _quarters([100, 105, 110, 115, 120], [10, 9, 8, 7, 6], [5, 5, 4, 4, 3], [1e7, 1e7, 1e7, 1e7, 1.2e7],
                  equity=[2000] * 5)
    q[["revenue", "op", "net", "equity"]] *= 1e6
    md = build([("2334", "イオレ")], hist, {"2334": _series([500.0] * 40)}, tmp_path / "out",
               now=datetime(2026, 11, 21, 17, 0), funds={"2334": q})
    md = md.replace("\u2060", "")  # 改行防止の記号を除いて比べる
    assert "| 上場株数比 | **29.7%** |" in md
    assert "| PER |" in md and "26/4-6月 **増収減益**" in md
    assert "懸念" in md and "株数 +20%/年" in md
    assert "| 期 | 営業利益 |" in md and "| 25/4-6 |" in md and "<br>" not in md.split("## 銘柄")[1].split("各項目の見方")[0]
    assert "信用残の推移**（直近3日" not in md and "| 項目 | 値 |" not in md  # 詳細の表は出さない


# ---- 上値を押さえる要因 ----
def test_overhang_lockup_dates_and_signals():
    from margin_tracker import overhang as oh

    px = _series([1000.0] * 90, start="2026-07-22")
    px.attrs["first_trade"] = pd.Timestamp("2026-07-22")
    o = oh.check(px, pd.Timestamp("2026-10-15"), "株数が1年で +10%（希薄化）", 0.12)
    text = " / ".join(o.items)
    assert "上場日 2026/7/22" in text and "90日後 2026/10/20（あと5日）" in text and "180日後 2027/1/18（あと95日）" in text
    assert "初値 1,000円" in text
    assert "株数が1年で +10%" in text and "上場株数の12.0%" in text
    assert o.signals == ["ロックアップ解除の目安（上場90日後 10/20）が近い：VC・大株主の売りに注意"]


def test_overhang_ignores_old_listings():
    from margin_tracker import overhang as oh

    px = _series([1000.0] * 30)
    px.attrs["first_trade"] = pd.Timestamp("2005-03-01")
    o = oh.check(px, pd.Timestamp("2026-11-21"), None, 0.02)
    assert o.items == [] and o.signals == []


def test_margin_points_recent_days_and_weeks_back():
    from margin_tracker.report import _history_table, _trend_line, margin_points

    days = pd.bdate_range("2026-10-26", "2026-11-19")
    h = pd.DataFrame({"date": days, "buy": [1_000_000 + i * 10_000 for i in range(len(days))], "sell": 5000})
    pts = margin_points(h)
    labels = [label for label, _ in pts]
    assert labels == ["3週前", "2週前", "1週前", "11/17", "11/18", "11/19"]
    assert pts[2][1]["date"] == pd.Timestamp("2026-11-12")  # 11/19 の7日前
    assert pts[0][1]["date"] == pd.Timestamp("2026-10-29")  # 21日前
    line = _trend_line(h)
    assert line.startswith("3週前 ") and "**11/19 " in line and " → " in line
    table = _history_table(h)
    assert "| 1週前 | 11/12 |" in table and "（最新）" in table


def test_margin_points_without_old_data():
    from margin_tracker.report import _history_table, _trend_line, margin_points

    h = pd.DataFrame({"date": pd.bdate_range("2026-11-13", periods=5), "buy": [100_000] * 5, "sell": [0] * 5})
    pts = margin_points(h)
    assert [r is None for _, r in pts[:3]] == [True, True, True]
    assert "3週前 ―・2週前 ―・1週前 ―" in _trend_line(h)
    assert "データなし" in _history_table(h)


def test_fundamentals_cache_limits_fetches_and_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(fd.time, "sleep", lambda s: None)
    q = _quarters([1] * 5, [1] * 5, [1] * 5, [1] * 5)
    calls = []

    def ok(code):
        calls.append(code)
        return q

    now = pd.Timestamp("2026-11-21 17:00")
    codes = ["A", "B", "C", "D", "E"]
    first = fd.get_all(codes, ok, tmp_path, now)
    assert len(calls) == fd.MAX_FETCH_PER_RUN  # 1回に3銘柄まで
    assert sum(v is not None for v in first.values()) == 3

    calls.clear()
    second = fd.get_all(codes, ok, tmp_path, now)
    assert sorted(calls) == ["D", "E"]  # まだ取れていない銘柄から取る（保存済みは問い合わせない）
    assert all(v is not None for v in second.values())

    def ng(code):
        raise RuntimeError("429")

    later = now + pd.Timedelta(days=fd.REFRESH_DAYS + 1)
    msgs = []
    third = fd.get_all(codes, ng, tmp_path, later, log=msgs.append)
    assert all(v is not None for v in third.values())  # 断られても保存済みのデータを使う
    assert len(msgs) == fd.MAX_FETCH_PER_RUN and "保存済み" in msgs[0]










# ---- 引力・圧力・妥当株価・反発 ----
def test_gravity_direction():
    down = {"2516.T": _series([100.0] * 30 + [90.0] * 5), "^N225": _series([100.0] * 30 + [92.0] * 5),
            "^IXIC": _series([100.0] * 30 + [95.0]), "^VIX": _series([20.0] * 10 + [35.0])}
    g = en.gravity(down)
    assert g.direction == en.DOWN and any("VIX 35" in x for x in g.reasons)
    up = {"2516.T": _series([90.0] * 30 + [100.0] * 5), "^N225": _series([90.0] * 30 + [100.0] * 5),
          "^VIX": _series([15.0] * 10)}
    assert en.gravity(up).direction == en.UP
    peak = {"^VIX": _series([20.0] * 5 + [40, 38, 36, 34, 33])}
    assert "ピークアウト" in en.gravity(peak).turning


def test_pressure_balance():
    base = [100.0] * 30
    falling = _series(base + [99, 97, 95, 93, 92])
    falling.loc[falling.index[-5:], "Volume"] = 1500.0
    up_margin = pd.DataFrame({"date": pd.bdate_range("2026-11-13", periods=5), "buy": [100, 104, 108, 112, 117]})
    p = en.pressure(falling, up_margin)
    assert p.level == "助長（強い）" and p.margin == pytest.approx(0.17) and "買残+17%" in p.breakdown
    down_margin = up_margin.assign(buy=[117, 112, 108, 104, 100])
    assert en.pressure(falling, down_margin).level == "解放中"
    quiet = _series(base + [99, 97, 95, 93, 92])
    quiet.loc[quiet.index[-5:], "Volume"] = 500.0
    flat_margin = up_margin.assign(buy=[100] * 5)
    assert en.pressure(quiet, flat_margin).level == "くすぶる"
    rising = _series(base + [101, 103, 105, 106, 108])
    assert en.pressure(rising, down_margin).level == "軽い"


def test_valuation_from_per_and_pbr():
    assert en.valuation(8.0, 0.8) == "割安（PER 8倍）"
    assert en.valuation(15.0, 2.0).startswith("普通")
    assert en.valuation(30.0, 5.0).startswith("やや割高")
    assert en.valuation(275.0, 11.6).startswith("割高")
    assert en.valuation(float("nan"), 10.7) == "割高（赤字のため PBR 10.7倍）"
    assert en.valuation(None, None) == ""


def test_overhangs_pick_dilution_materials(tmp_path):
    (tmp_path / "20261120.csv").write_text(
        "time,code,name,kind,score,facts,title,pdf\n"
        "17:00,4882,P,MSワラント・行使価額修正条項付,-67,調達 4.7億円 / 希薄化 約25%,t,u\n"
        "15:30,4882,P,業績予想の上方修正,10,,t,u\n", encoding="utf-8")
    heavy = en.overhangs("4882", pd.Timestamp("2026-11-21"), tmp_path)
    assert heavy == [(pd.Timestamp("2026-11-20"), "MSワラント・行使価額修正条項付", -67.0, "調達 4.7億円 / 希薄化 約25%")]
    others = en.materials("4882", pd.Timestamp("2026-11-21"), tmp_path, exclude=en.OVERHANG_KINDS)
    assert [k for _, k, _, _ in others] == ["業績予想の上方修正"]



def test_materials_from_ir_data(tmp_path):
    (tmp_path / "20261120.csv").write_text(
        "time,code,name,kind,score,facts,title,pdf\n"
        "15:30,4493,CSC,業績予想の上方修正,22.0,通期営業利益 +28%,t,u\n"
        "15:30,9999,X,増配,10,,t,u\n", encoding="utf-8")
    (tmp_path / "20260901.csv").write_text("time,code,name,kind,score,facts,title,pdf\n15:30,4493,CSC,増配,10,,t,u\n",
                                           encoding="utf-8")
    m = en.materials("4493", pd.Timestamp("2026-11-21"), tmp_path)
    assert m == [(pd.Timestamp("2026-11-20"), "業績予想の上方修正", 22.0, "通期営業利益 +28%")]


def test_rebound_marks_and_against_market():
    prev = pd.Series({"Open": 100, "High": 102, "Low": 98, "Close": 100})
    day = lambda o, h, l, c: pd.Series({"Open": o, "High": h, "Low": l, "Close": c})  # noqa: E731
    assert en.rebound_mark(prev, day(100, 104, 99, 103)) == "◎"
    assert en.rebound_mark(prev, day(100, 101.5, 99, 101)) == "○"
    assert en.rebound_mark(prev, day(99, 100, 95, 99)) == "△"
    assert en.rebound_mark(prev, day(99, 99.5, 95, 95.5)) == "×"
    px = _series([100.0] * 10)
    px.loc[px.index[-1], ["Open", "Close"]] = [99.0, 100.5]
    growth = _series([100.0] * 9 + [97.0])
    rb = en.rebound(px, growth)
    assert rb.days_text.endswith("逆") and rb.latest.startswith("逆行陽線")


# ---- 温度感 ----
from margin_tracker import temperature as tp  # noqa: E402


def test_margin_pl_and_coverage():
    day = pd.DataFrame({"code": ["A", "B", "C"], "buy": [100, 200, 50], "buy_val": [100_000, 100_000, 50_000]})
    pl, cov = tp.margin_pl(day, {"A": 800.0, "B": 450.0})  # A: 8万/10万, B: 9万/10万
    assert pl == pytest.approx(170_000 / 200_000 - 1)
    assert cov == pytest.approx(200_000 / 250_000)
    assert tp.margin_pl(day, {}) is None
    assert tp.pl_label(-0.22).startswith("底") and tp.pl_label(-0.02).startswith("天井") and tp.pl_label(-0.10) == "普通"


def test_update_history_records_missing_days(tmp_path):
    path = tmp_path / "mm.csv"
    day = pd.Timestamp("2026-11-19")
    df = pd.DataFrame({"code": ["1111", "2222"], "buy": [100, 100], "buy_val": [100_000, 50_000]})

    def closes(symbols):
        idx = pd.to_datetime(["2026-11-18", "2026-11-19"])
        return {"1111.T": pd.Series([900.0, 900.0], index=idx), "2222.T": pd.Series([500.0, 450.0], index=idx)}

    hist = tp.update_history({day: df}, closes, path)
    assert len(hist) == 1 and hist["pl"].iloc[0] == pytest.approx(135_000 / 150_000 - 1)
    calls = []
    tp.update_history({day: df}, lambda s: calls.append(s) or {}, path)
    assert calls == []  # 記録済みの日は計算しない


def test_temperature_gauges():
    falling = _series(list(np.linspace(100, 120, 260)) + list(np.linspace(120, 100, 15)))
    gauges = {g.name: g for g in tp.nikkei_gauges(falling)}
    assert gauges["25日乖離"].heat == -1 and gauges["RSI"].heat == -1
    hist = pd.DataFrame({"date": pd.to_datetime(["2026-11-13", "2026-11-19"]), "pl": [-0.10, -0.18], "coverage": [0.8, 0.8]})
    t = tp.build(hist, falling)
    assert t.summary == "冷えている（悲観）"
    assert t.gauges[0].value == "-18.0%" and t.gauges[1].value == "-10.0%" and t.gauges[1].note == "11/13申込"



def test_band_reason_shows_range_and_share():
    from margin_tracker.levels import Level, Point, UP

    lv = Level(1701.0, 1701.0, [Point(1701.0, "band", "", band=(1629.0, 1774.0, 0.484), weight=2)], UP)
    assert lv.short_reasons == "厚い帯 1,629〜1,774（48%）"


# ---- 本尊の気配 ----
def test_footprints_detect_accumulation_signs(tmp_path):
    n = 60
    closes = [100.0] * 40 + [100, 102, 101, 103, 102, 104, 103, 105, 104, 106, 105, 107, 106, 108, 107, 109, 108, 110, 109, 111]
    px = _series(closes[:n])
    vol = px["Volume"].copy()
    for i in range(41, n):  # 上げた日は商い3倍
        vol.iloc[i] = 3000.0 if closes[i] > closes[i - 1] else 1000.0
    px["Volume"] = vol
    growth = _series([100.0] * 50 + list(np.linspace(100, 90, 10)))  # 地合いは下げ
    margin = pd.DataFrame({"date": pd.bdate_range("2026-11-13", periods=6), "buy": [100_000] * 6,
                           "sell": [10_000, 11_000, 12_000, 13_000, 14_000, 15_000]})
    (tmp_path / "20261120.csv").write_text("time,code,name,kind,score,facts,title,pdf\n15:30,1111,A,大株主の買い集め,10,,t,u\n",
                                           encoding="utf-8")
    f = en.footprints(px, margin, growth, "1111", pd.Timestamp("2026-11-21"), tmp_path)
    names = [n for n, _ in f.hits]
    assert "出来高" in names and "逆行" in names and "売残" in names and "買い集め" in names
    assert f.level.startswith("あり")


def test_footprints_exit_and_small_sell():
    px = _series([100.0] * 59 + [101.0])
    last = px.index[-1]
    px.loc[last, ["Open", "High", "Low", "Close", "Volume"]] = [100, 110, 99, 101, 5000]  # 大商いで長い上ヒゲ
    margin = pd.DataFrame({"date": pd.bdate_range("2026-11-13", periods=2), "buy": [100_000, 100_000], "sell": [100, 300]})
    f = en.footprints(px, margin, None)
    assert any("長い上ヒゲ" in e for e in f.exits)
    assert all(n != "売残" for n, _ in f.hits)  # 売残が買残の5%未満なら数えない
