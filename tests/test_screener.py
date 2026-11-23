import numpy as np
import pandas as pd

from screener import core
from screener.report import build


def _bars(n=60, crash=True, vol=1_000_000):
    idx = pd.bdate_range("2026-08-01", periods=n)
    rng = np.random.default_rng(0)
    close = 1000 + rng.normal(0, 5, n).cumsum()
    if crash:
        close[-1] = close[-2] * 0.80  # 最終日に急落
    df = pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close,
                       "Volume": float(vol)}, index=idx)
    df.attrs["name"] = "テスト"
    return df


def _quarters(**over):
    idx = pd.to_datetime(["2025-09-30", "2025-12-31", "2026-03-31", "2026-06-30", "2026-09-30"])
    q = pd.DataFrame({"revenue": [100, 100, 100, 100, 110], "op": [10, 10, 10, 10, 12], "net": [7, 7, 7, 7, 8],
                      "equity": [500] * 5, "shares": [1e6] * 5, "cash": [200] * 5, "debt": [50] * 5}, index=idx, dtype=float)
    for k, v in over.items():
        q[k] = v
    return q


def test_touch_detected_after_crash():
    t = core.find_touch("1234", _bars())
    assert t is not None and t.low_sigma <= -3 and t.date == _bars().index[-1]


def test_no_touch_without_crash_and_illiquid_skipped():
    assert core.find_touch("1234", _bars(crash=False)) is None
    assert core.find_touch("1234", _bars(vol=10)) is None  # 売買代金が少ない
    fund = _bars()
    fund.attrs["name"] = "Canadian Solar Infrastructure Fund, Inc."
    assert core.find_touch("9284", fund) is None


def test_pre_screen_uses_close_only():
    assert core.pre_screen({"A": _bars()["Close"], "B": _bars(crash=False)["Close"]}) == ["A"]


def test_judge_passes_healthy_company():
    f, fatal, _ = core.judge(_quarters(), 900, pd.Timestamp("2026-11-20"))
    assert f is not None and fatal == [] and f.verdict == "増収増益"


def test_judge_flags_fatal():
    now = pd.Timestamp("2026-11-20")
    assert "債務超過" in core.judge(_quarters(equity=[-1.0] * 5), 900, now)[1]
    _, fatal, _ = core.judge(_quarters(op=[-10, -10, -10, -10, -20], cash=[30.0] * 5), 900, now)
    assert any(x.startswith("資金繰り") for x in fatal) and "営業赤字が1年続き、拡大中" in fatal
    assert any(x.startswith("大幅な希薄化") for x in core.judge(_quarters(shares=[1e6, 1e6, 1e6, 1e6, 1.2e6]), 900, now)[1])
    assert any(x.startswith("売上の急減") for x in core.judge(_quarters(revenue=[100, 100, 100, 100, 60.0]), 900, now)[1])
    assert core.judge(None, 900, now)[0] is None


def test_report_sections():
    t = core.find_touch("1234", _bars())
    good = core.Result(t, *core.judge(_quarters(), t.close, pd.Timestamp("2026-11-20")))
    bad = core.Result(t, *core.judge(_quarters(equity=[-1.0] * 5), t.close, pd.Timestamp("2026-11-20")))
    unknown = core.Result(t, None)
    uni = pd.DataFrame({"buy_ratio": [0.12]}, index=["1234"])
    md = build([good, bad, unknown], uni, pd.Timestamp("2026-11-20 17:00"), 3.0, 1, 3700, 1e7)
    assert "致命的なファンダの問題がない **1 銘柄**" in md
    assert "債務超過" in md and "ファンダ未確認" in md and "12.0% ⚠" in md


def test_sub_report_links_and_title():
    t = core.find_touch("1234", _bars(), sigma=2.0)
    r = core.Result(t, *core.judge(_quarters(), t.close, pd.Timestamp("2026-11-20")))
    uni = pd.DataFrame({"buy_ratio": [0.01]}, index=["1234"])
    sub_md = build([r], uni, pd.Timestamp("2026-11-20 17:00"), 2.0, 1, 3700, 1e7, upper=3.0)
    assert "-2σ にタッチし、-3σ には届かなかった" in sub_md and "[README.md](README.md)" in sub_md
    main_md = build([], uni, pd.Timestamp("2026-11-20 17:00"), 3.0, 1, 3700, 1e7, sub=(2.0, 1))
    assert "[2sigma.md](2sigma.md)" in main_md


def test_names_cache_only_fetches_missing(tmp_path, monkeypatch):
    from screener import names

    path = tmp_path / "names.csv"
    names.save({"3551": "ダイニック"}, path)
    asked = []
    monkeypatch.setattr(names, "fetch_one", lambda code, s: asked.append(code) or "帝国通信工業")
    monkeypatch.setattr(names, "INTERVAL", 0)
    assert names.get(["3551", "6763"], path) == {"3551": "ダイニック", "6763": "帝国通信工業"}
    assert asked == ["6763"] and names.load(path)["6763"] == "帝国通信工業"


def _stock_and_index(event=False, market_fall=False, n=200):
    idx = pd.bdate_range("2026-02-02", periods=n)
    rng = np.random.default_rng(1)
    m = 0.01 * rng.normal(size=n)
    s = 1.0 * m + 0.003 * rng.normal(size=n)
    if market_fall:
        m[-5:] = -0.03
        s[-5:] = -0.03
    else:
        s[-5:] = -0.03
    mc, sc = 1000 * np.cumprod(1 + m), 1000 * np.cumprod(1 + s)
    vol = np.full(n, 1e6)
    op = sc.copy()
    if event:
        op[-1] = sc[-2] * 0.90
        sc[-1] = sc[-2] * 0.88
        vol[-1] = 5e6
    stock = pd.DataFrame({"Open": op, "High": np.maximum(op, sc) * 1.01, "Low": np.minimum(op, sc) * 0.99,
                          "Close": sc, "Volume": vol}, index=idx)
    index = pd.DataFrame({"Close": mc}, index=idx)
    return stock, {"TOPIX": index}


def _result(stock):
    from screener import core as c
    t = c.find_touch("1234", stock, sigma=2.0) or c.find_touch("1234", _bars(), sigma=2.0)
    return c.Result(t, *c.judge(_quarters(), t.close, pd.Timestamp("2026-11-20")))


def test_judge_drop_bad_news_is_justified():
    from screener import rank, tdnet
    stock, idx = _stock_and_index(market_fall=True)
    verdict, reasons, *_ = rank.judge_drop(_result(stock), tdnet.News(bad=["11/14 下方修正"]), stock, idx, stock.index[-1])
    assert verdict == rank.JUSTIFIED and reasons == ["11/14 下方修正"]


def test_judge_drop_market_is_unjustified_and_event_is_unclear():
    from screener import rank, tdnet
    stock, idx = _stock_and_index(market_fall=True)
    verdict, reasons, r5, share = rank.judge_drop(_result(stock), tdnet.News(), stock, idx, stock.index[-1])
    assert verdict == rank.UNJUSTIFIED and share >= 0.5 and reasons[0].startswith("地合い")
    stock, idx = _stock_and_index(event=True)
    verdict, *_ = rank.judge_drop(_result(stock), tdnet.News(), stock, idx, stock.index[-1])
    assert verdict == rank.UNCLEAR


def test_tdnet_parse_and_classify():
    from screener import tdnet
    page = ('<td class="oddnew-L kjTime" noWrap>15:00</td>\n<td class="oddnew-M kjCode" noWrap>35510</td>\n'
            '<td class="oddnew-M kjName" noWrap>ダイニック </td>\n<td class="oddnew-M kjTitle" align="left">'
            '<a href="x.pdf">通期業績予想の下方修正に関するお知らせ</a></td>\n'
            '<td class="evennew-L kjTime" noWrap>15:00</td>\n<td class="evennew-M kjCode" noWrap>35510</td>\n'
            '<td class="evennew-M kjName" noWrap>ダイニック </td>\n<td class="evennew-M kjTitle" align="left">'
            '<a href="y.pdf">自己株式の取得に係る事項の決定に関するお知らせ</a></td>')
    df = tdnet.parse(page)
    assert list(df["code"]) == ["3551", "3551"]
    df.insert(0, "date", pd.Timestamp("2026-11-14"))
    n = tdnet.classify(df, "3551")
    assert n.bad == ["11/14 下方修正"] and n.good == ["11/14 自社株買い"]


def test_adjust_splits_multiplies_shares_before_split():
    q = _quarters()
    adj = core.adjust_splits(q, [(pd.Timestamp("2026-11-01"), 2.0)])
    assert (adj["shares"] == 2e6).all() and (q["shares"] == 1e6).all()
    assert core.adjust_splits(q, []) is q


def test_ranking_splits_by_average_volume():
    from screener import rank
    from screener.report import build_ranking

    def ranked(code, vol):
        t = core.find_touch(code, _bars(vol=vol))
        r = core.Result(t, *core.judge(_quarters(), t.close, pd.Timestamp("2026-11-20")))
        return rank.Ranked(r, 70, 30, 20, 20, rank.UNJUSTIFIED, ["地合い"])

    md = build_ranking([ranked("1111", 2_000_000), ranked("2222", 50_000)], pd.Timestamp("2026-11-20 17:00"), 3.0, 1, 30)
    big, small = md.split("## 平均出来高 50万株未満")[:2]
    assert "1111" in big.split("## 平均出来高 50万株以上")[1] and "2222" not in big
    assert "2222" in small and "1111" not in small


def _prow(**kw):
    base = {"code": "1111", "name": "テスト", "close": 1000.0, "low_sigma": -3.0, "avg_volume": 100000, "total": 70,
            "fund": 30, "hope": 20, "rebound": 20, "verdict": "不当な下げ", "fund_verdict": "増収増益", "per": 9.0,
            "pbr": 0.8, "market_cap": 5e10, "buy_ratio": 0.03, "concerns": "", "reasons": "地合い", "good_news": ""}
    base.update(kw)
    return base


def test_promote_filters_and_sorts():
    from screener import promote

    rows = [
        _prow(code="A", total=70),
        _prow(code="B", total=80),
        _prow(code="C", verdict="然るべき下げ"),
        _prow(code="D", total=60),
        _prow(code="E", fund_verdict="増収減益"),
        _prow(code="F", per=20.0),
        _prow(code="G", per=None),
        _prow(code="H", buy_ratio=0.15),
        _prow(code="I", buy_ratio=None, fund_verdict="営業黒字に転換"),
    ]
    assert [r["code"] for r in promote.pick(rows)] == ["B", "A", "I"]


def test_promote_report():
    import pandas as pd
    from screener import promote

    md = promote.build([_prow(code="8508", name="Ｊトラスト", total=78)], pd.Timestamp("2026-11-28 07:00"), 826)
    assert "826 銘柄から" in md and "**1 銘柄**" in md and "| 8508 Ｊトラスト | 1,000 |" in md
    assert "60点以上なら監視を続け" in md
    assert "基準を満たす銘柄なし" in promote.build([], pd.Timestamp("2026-11-28"), 10)
