"""使い方:
  python -m screener              # 最新の日に -3σ にタッチした銘柄を探して screener/README.md を作る
  python -m screener --days 5     # 直近5営業日のうちにタッチした銘柄
  python -m screener --sigma 2.5  # -2.5σ で探す（候補を広げたいとき）
  python -m screener --sub-sigma 0  # -2σ の予備軍（screener/2sigma.md）を出さない（速い）
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

from . import fundamentals, jpx, store

from . import core, names, promote, rank, tdnet
from .report import build, build_ranking

SPARK_URL = "https://query1.finance.yahoo.com/v7/finance/spark"
SPARK_BATCH = 20  # Yahoo の spark は1回20銘柄まで
MAX_FUND_FETCH = 1500  # 1回の実行で決算データを問い合わせるのはこの銘柄数まで（残りは保存済みか「未確認」）
MAX_FUND_FAIL = 5  # 続けてこの回数失敗したら、Yahoo に断られているとみて問い合わせをやめる
FUND_INTERVAL = 3  # 決算データの問い合わせの間隔（秒）


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


def refresh_margin(root: Path) -> None:
    """JPX の最新の信用残データ（銘柄の一覧に使う）がまだ保存されていなければ取って保存する。"""
    try:
        files = jpx.list_files()
    except Exception as e:  # 取れなければ保存済みの一覧を使う
        log(f"JPX の一覧を取れませんでした（保存済みを使います）: {e}")
        return
    if not files:
        return
    day = max(files)
    if day in store.saved_days(root):
        return
    try:
        as_of, df = jpx.parse_text(jpx.pdf_to_text(jpx.download(files[day])))
    except Exception as e:
        log(f"JPX の信用残データを読めませんでした（保存済みを使います）: {e}")
        return
    if not df.empty:
        store.save(as_of or day, df, root)
        log(f"JPX の信用残データ {as_of or day} を保存")


def universe(root: Path) -> tuple[pd.Timestamp, pd.DataFrame]:
    """JPX の信用残データの最新日に載っている銘柄（上場比が空欄の ETF・REIT などは除く）。"""
    refresh_margin(root)
    days = store.saved_days(root)
    if not days:
        raise SystemExit(f"{root} に信用残データがなく、JPX からも取れませんでした")
    df = pd.read_csv(store.path_for(days[-1], root), dtype={"code": str})
    return pd.Timestamp(days[-1]), df[df["buy_ratio"].notna()].set_index("code")


def spark_closes(codes: list[str], rng: str = "3mo") -> dict[str, pd.Series]:
    """全銘柄の終値を20銘柄ずつまとめて取る（一次ふるい用）。"""
    from backtest.data import to_symbol

    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0"
    sym2code = {to_symbol(c): c for c in codes}
    syms = list(sym2code)
    out = {}
    for i in range(0, len(syms), SPARK_BATCH):
        batch = syms[i:i + SPARK_BATCH]
        for wait in (20, 60, None):  # 断られたら間をおいて取り直す
            try:
                r = s.get(SPARK_URL, params={"symbols": ",".join(batch), "range": rng, "interval": "1d"}, timeout=30)
                if r.status_code == 429 and wait:
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                break
            except requests.RequestException as e:
                if wait is None:
                    log(f"終値の取得に失敗: {batch[0]}〜: {e}")
                    r = None
                    break
                time.sleep(wait)
        if r is not None and r.ok:
            for x in r.json()["spark"]["result"] or []:
                resp = (x.get("response") or [{}])[0]
                if not resp.get("timestamp"):
                    continue
                idx = pd.to_datetime(resp["timestamp"], unit="s", utc=True).tz_convert("Asia/Tokyo").normalize().tz_localize(None)
                c = pd.Series(resp["indicators"]["quote"][0]["close"], index=idx, dtype=float).dropna()
                out[sym2code.get(x["symbol"], x["symbol"])] = c[~c.index.duplicated(keep="last")]
        if i // SPARK_BATCH % 20 == 19:
            log(f"  終値: {i + len(batch):,} / {len(syms):,} 銘柄")
        time.sleep(0.3)
    return out


def daily_bars(codes: list[str]) -> dict[str, pd.DataFrame]:
    from backtest.data import fetch_yahoo

    start = (pd.Timestamp.now() - pd.Timedelta(days=400)).date()  # 52週高値まで見る
    out = {}
    for code in codes:
        try:
            out[code] = fetch_yahoo(code, start)
        except Exception as e:
            log(f"日足の取得に失敗: {code}: {e}")
        time.sleep(0.4)
    return out


def get_funds(codes: list[str], now: pd.Timestamp) -> dict[str, pd.DataFrame | None]:
    """保存済み（7日以内）の決算データを使い、なければ問い合わせて data/fundamentals に保存する。"""
    out, tried, fails = {}, 0, 0
    for code in codes:
        path = fundamentals.cache_path(code)
        cached = pd.read_csv(path, index_col=0, parse_dates=True) if path.exists() else None
        when = fundamentals.fetched_at(code)
        fresh = cached is not None and when is not None and (now - when).days < fundamentals.REFRESH_DAYS
        if not fresh and tried < MAX_FUND_FETCH and fails < MAX_FUND_FAIL:
            if tried:
                time.sleep(FUND_INTERVAL)
            tried += 1
            try:
                cached = fundamentals.fetch(code)
                fundamentals.save_cache(code, cached)
                fails = 0
            except Exception as e:
                fails += 1
                log(f"決算データの取得に失敗: {code}: {e}" + ("（続けて失敗したので、残りは問い合わせない）" if fails >= MAX_FUND_FAIL else ""))
            if tried % 100 == 0:
                log(f"  決算データ: {tried} 銘柄を問い合わせた")
        out[code] = cached
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m screener", description="ボリンジャーバンド -3σ タッチ × 致命的なファンダなし の銘柄を探します。")
    ap.add_argument("--sigma", type=float, default=core.BB_SIGMA, help=f"タッチを見る σ（初期値: {core.BB_SIGMA:g}）")
    ap.add_argument("--sub-sigma", type=float, default=core.SUB_SIGMA,
                    help=f"予備軍として別のファイルに出す σ（初期値: {core.SUB_SIGMA:g}、0 で出さない）")
    ap.add_argument("--days", type=int, default=core.LOOKBACK, help=f"直近何営業日のタッチを拾うか（初期値: {core.LOOKBACK}）")
    ap.add_argument("--min-turnover", type=float, default=core.MIN_TURNOVER / 1e4,
                    help=f"平均売買代金の下限・万円（初期値: {core.MIN_TURNOVER / 1e4:,.0f}）")
    ap.add_argument("--data", default=str(store.DAILY_DIR), help="信用残データの場所（銘柄の一覧に使う）")
    ap.add_argument("--out", default="screener", help="レポートの出力先フォルダ")
    args = ap.parse_args(argv)

    now = pd.Timestamp.now(tz="Asia/Tokyo").tz_localize(None)
    as_of, uni = universe(Path(args.data))
    codes = list(uni.index)
    log(f"対象: {len(codes):,} 銘柄（JPX 信用残 {as_of:%Y-%m-%d} の一覧）")

    closes = spark_closes(codes)
    log(f"終値を取れた銘柄: {len(closes):,}")
    data_day = max(c.index[-1] for c in closes.values() if len(c))  # 株価の最新の営業日（保存するファイルの名前に使う）
    sub_sigma = args.sub_sigma if 0 < args.sub_sigma < args.sigma else None
    scan = sub_sigma or args.sigma  # 浅い方の σ でタッチを探し、深さで振り分ける
    pre = core.pre_screen(closes, sigma=scan - core.PRE_MARGIN, lookback=args.days)
    log(f"一次ふるい（終値 -{scan - core.PRE_MARGIN:g}σ 以下）: {len(pre)} 銘柄")

    bars = daily_bars(pre)
    touches = [t for code, df in bars.items()
               if (t := core.find_touch(code, df, sigma=scan, lookback=args.days, min_turnover=args.min_turnover * 1e4))]
    touches.sort(key=lambda t: t.low_sigma)
    deep = [t for t in touches if t.low_sigma <= -args.sigma]
    shallow = [t for t in touches if t.low_sigma > -args.sigma]
    log(f"-{args.sigma:g}σ タッチ: {len(deep)} 銘柄" + (f"、-{sub_sigma:g}σ タッチ（-{args.sigma:g}σ 未満）: {len(shallow)} 銘柄" if sub_sigma else ""))

    funds = get_funds([t.code for t in deep + shallow], now)  # 深い方から先に問い合わせる
    jp = names.get([t.code for t in deep + shallow], log=log)  # 日本語の銘柄名（取れなければ英語名のまま）
    for t in deep + shallow:
        t.name = jp.get(t.code, t.name)

    def judge_all(ts):
        return [core.Result(t, *core.judge(core.adjust_splits(funds.get(t.code), bars[t.code].attrs.get("splits", [])),
                                           t.close, now)) for t in ts]

    results, sub_results = judge_all(deep), judge_all(shallow)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    common = dict(now=now, days=args.days, scanned=len(closes), min_turnover=args.min_turnover * 1e4)
    md = build(results, uni, sigma=args.sigma, sub=(sub_sigma, len(sub_results)) if sub_sigma else None, **common)
    (out_dir / "README.md").write_text(md, encoding="utf-8")
    if sub_sigma:
        (out_dir / "2sigma.md").write_text(build(sub_results, uni, sigma=sub_sigma, upper=args.sigma, **common), encoding="utf-8")
    hist_dir = Path("data/screener")
    hist_dir.mkdir(parents=True, exist_ok=True)

    # ランキング（ファンダ・思惑・下げの性質）
    from backtest.data import fetch_yahoo

    start = (now - pd.Timedelta(days=400)).date()
    indices = {}
    for sym, label in (("1306.T", "TOPIX"), ("2516.T", "グロース250")):
        try:
            indices[label] = fetch_yahoo(sym, start)
        except Exception as e:
            log(f"指数の取得に失敗: {sym}: {e}")
    disclosures = tdnet.load(now, log=log)
    log(f"TDnet: 直近{tdnet.DAYS}日の開示 {len(disclosures):,} 件")
    news = {t.code: tdnet.classify(disclosures, t.code) for t in deep + shallow}
    ranked = rank.rank(results + sub_results, bars, news, indices, uni, now)
    (out_dir / "ranking.md").write_text(build_ranking(ranked, now=now, sigma=args.sigma, days=args.days,
                                                      tdnet_days=tdnet.DAYS), encoding="utf-8")
    rows = [promote.row(x, uni) for x in ranked]
    pd.DataFrame([{"rank": i + 1, **r} for i, r in enumerate(rows)]).to_csv(
        hist_dir / f"ranking_{data_day:%Y%m%d}.csv", index=False)
    (out_dir / "昇格候補.md").write_text(promote.build(promote.pick(rows), now, len(rows)), encoding="utf-8")
    log(f"ランキング: {out_dir / 'ranking.md'}（{len(ranked)} 銘柄）・昇格候補: {out_dir / '昇格候補.md'}")
    rows = [{
        "level": f"-{args.sigma:g}σ" if r.touch.low_sigma <= -args.sigma else f"-{sub_sigma:g}σ",
        "code": r.touch.code, "name": r.touch.name, "touch_date": r.touch.date.date(), "close": r.touch.close,
        "low_sigma": round(r.touch.low_sigma, 2), "close_sigma": round(r.touch.close_sigma, 2),
        "result": "通過" if r.passed else ("除外" if r.fatal else "未確認"),
        "fatal": " / ".join(r.fatal), "concerns": " / ".join(r.concerns),
    } for r in results + sub_results]
    pd.DataFrame(rows, columns=["level", "code", "name", "touch_date", "close", "low_sigma", "close_sigma", "result", "fatal", "concerns"]
                 ).to_csv(hist_dir / f"{data_day:%Y%m%d}.csv", index=False)
    log(f"レポート: {out_dir / 'README.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
