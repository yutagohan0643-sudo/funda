"""使い方:
  python -m margin_tracker            # JPX から新しい日の信用残を取り込み、レポートを作り直す
  python -m margin_tracker --no-fetch # 取り込みはせず、レポートだけ作り直す
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import jpx, market, store, temperature
from .report import build, read_themes, read_watchlist


def fetch_new(root: Path) -> list:
    """公表ページにあって、まだ保存していない日の PDF を取り込む。"""
    session = jpx._session()
    have = set(store.saved_days(root))
    added = []
    for day, url in jpx.list_files(session).items():
        if day in have:
            continue
        print(f"取得中: {day} {url}", file=sys.stderr)
        as_of, df = jpx.parse_text(jpx.pdf_to_text(jpx.download(url, session)))
        if as_of and as_of != day:
            print(f"  注意: ファイル名の日付 {day} と中身の日付 {as_of} が違います。中身の日付を使います", file=sys.stderr)
            day = as_of
        if len(df) < 1000:
            raise RuntimeError(f"{day}: 読み取れた銘柄が {len(df)} 件しかありません。PDF の形式が変わった可能性があります")
        store.save(day, df, root)
        print(f"  保存: {len(df):,} 銘柄", file=sys.stderr)
        added.append(day)
        time.sleep(2)  # JPX のサーバーに負担をかけないよう間を空ける
    return added


def fetch_prices(codes: list[str]) -> dict:
    from .prices import fetch_yahoo
    import pandas as pd

    # 保有者分布の推定に2年分の日足を使う
    start = (pd.Timestamp.now() - pd.Timedelta(days=730)).date()
    out = {}
    for code in codes:
        try:
            out[code] = fetch_yahoo(code, start)
        except Exception as e:  # 株価が取れなくても信用残のレポートは出す
            print(f"株価の取得に失敗: {code}: {e}", file=sys.stderr)
            out[code] = None
        time.sleep(0.5)
    return out


def fetch_funds(codes: list[str]) -> dict:
    from . import fundamentals

    return fundamentals.get_all(codes, log=lambda msg: print(msg, file=sys.stderr))


def fetch_temperature(root: Path, mkt: market.Market) -> temperature.Temperature:
    """市場全体の信用評価損益率（記録していない日だけ計算して足す）と、日経平均のテクニカル。"""
    import pandas as pd

    have = set()
    if temperature.HISTORY.exists():
        have = set(pd.read_csv(temperature.HISTORY, parse_dates=["date"])["date"].dt.normalize())
    days = {}
    for day in store.saved_days(root)[-20:]:  # 株価は直近1か月分しか取らないので、直近の日だけ
        if pd.Timestamp(day) not in have:
            days[pd.Timestamp(day)] = pd.read_csv(store.path_for(day, root), dtype={"code": str})
    hist = temperature.update_history(days, market.spark_closes, log=lambda m: print(m, file=sys.stderr))
    return temperature.build(hist, mkt.series.get("^N225"))


def fetch_market() -> market.Market:
    from .prices import fetch_yahoo
    import pandas as pd

    start = (pd.Timestamp.now() - pd.Timedelta(days=400)).date()  # 週足13週線と25日線に足りる長さ
    m = market.fetch(fetch_yahoo, start)
    for e in m.errors:
        print(f"地合いの取得に失敗: {e}", file=sys.stderr)
    return m


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m margin_tracker", description="監視銘柄の信用残推移を更新します。")
    ap.add_argument("--watchlist", default="watchlist.txt", help="監視銘柄のファイル（初期値: watchlist.txt）")
    ap.add_argument("--data", default=str(store.DAILY_DIR), help="日次データの保存先")
    ap.add_argument("--out", default="margin", help="レポートの出力先フォルダ（初期値: margin）")
    ap.add_argument("--no-fetch", action="store_true", help="JPX から取り込まない")
    args = ap.parse_args(argv)

    root = Path(args.data)
    if not args.no_fetch:
        added = fetch_new(root)
        print(f"新しく取り込んだ日: {', '.join(map(str, added)) or 'なし'}", file=sys.stderr)

    watchlist = read_watchlist(args.watchlist)
    codes = [c for c, _ in watchlist]
    hist = store.history(codes, root)
    out_dir = Path(args.out)
    mkt = fetch_market()
    temp = fetch_temperature(root, mkt)
    md = build(watchlist, hist, fetch_prices(codes), out_dir, market=mkt, temp=temp,
               themes=read_themes(args.watchlist), funds=fetch_funds(codes))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README.md").write_text(md, encoding="utf-8")
    print(f"レポート: {out_dir / 'README.md'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
