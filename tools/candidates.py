"""株価目線の候補探し。TDnet の全銘柄の開示 × 株価の反応から、3つのタイプを拾う。

  A. 企業再評価      中計・資本コスト・還元方針・事業の組み替えなどの開示のあと、株価が市場より強い
  B. 好決算なのに売られた  決算短信で営業増益（＋予想も増益 or 上方修正）なのに、発表後に市場より弱い
  C. 大きな思惑      AI・提携・受注・TOB などの開示で、出来高を伴って大きく上がった（PER は問わない）

使い方（リポジトリの一番上で）:
  python tools/candidates.py            # 直近 21 日の開示で候補を出す → candidates/YYYY-MM-DD.md
  python tools/candidates.py --days 31  # TDnet に残っているのは約1か月分

データ: TDnet 適時開示一覧（www.release.tdnet.info）、決算短信 XBRL サマリー、JPX 上場銘柄一覧、Yahoo Finance（株価・時価総額）
キャッシュ: cache/（git に入れない）
"""
from __future__ import annotations

import argparse
import io
import json
import re
import time
import unicodedata
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / 'cache'
TDNET = 'https://www.release.tdnet.info/inbs/'
JPX_LIST = 'https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xlsx'
UA = {'User-Agent': 'Mozilla/5.0 (funda candidates)'}
CLOSE_HHMM = '15:30'  # 東証の大引け（2024/11〜）。これ以降の開示は翌営業日から反応

# ---- 開示の分類（表題の正規表現） ----
RERATE = re.compile(
    r'中期経営計画|中期経営方針|長期ビジョン|資本コストや株価を意識|資本収益性|株主還元方針|配当方針|累進配当|DOE|'
    r'配当性向|自己株式の消却|自己株式(?:の)?取得(?!状況|結果)|株式分割|市場区分の変更|プライム市場への|'
    r'事業ポートフォリオ|政策保有株式|持株会社体制|商号の変更|商号変更|事業の譲受|事業譲渡|完全子会社化|'
    r'株主優待制度の(?:新設|導入)|親会社等の異動|主要株主.*異動|資本業務提携')
THEME = re.compile(
    r'AI|ＡＩ|人工知能|生成|データセンター|GPU|ＧＰＵ|半導体|量子|宇宙|衛星|防衛|核融合|原子力|'
    r'ステーブルコイン|暗号資産|ビットコイン|ブロックチェーン|Web3|ロボット|ドローン|全固体|ペロブスカイト|'
    r'業務提携|資本提携|共同開発|受注|採択|特許|治験|承認|ライセンス|独占|販売契約|基本合意|'
    r'公開買付け|ＴＯＢ|TOB|買集め|大量保有|株式取得|子会社化|新規事業|参入', re.I)
# TOB される側（株価が TOB 価格に張り付くので上値がない）
TOB_TARGET = re.compile(r'当社株式.*公開買付け|公開買付け.*賛同|証券コード.*公開買付け|普通株式に対する公開買付け|'
                        r'株券等に対する公開買付け|MBO|ＭＢＯ|上場廃止|株式併合|売渡請求')
# 再評価のなかでも「本気度」が高い開示（自社株買い・分割・優待より重く見る）
RERATE_STRONG = re.compile(r'中期経営計画|中期経営方針|長期ビジョン|資本コストや株価を意識|資本収益性|株主還元方針|配当方針|'
                           r'累進配当|DOE|事業ポートフォリオ|政策保有株式|市場区分の変更|プライム市場への|持株会社体制|'
                           r'商号|資本業務提携|親会社等の異動|主要株主.*異動')
BUYBACK = re.compile(r'自己株式')
# 定型の開示（材料になりにくい）
ROUTINE = re.compile(r'決算短信|説明資料|説明会|招集|株主総会|ガバナンス|自己株式|役員|人事|定款|異動|月次|'
                     r'有価証券報告書|半期報告書|四半期報告書|内部統制|独立役員|ストック・?オプション|新株予約権の(?:発行|行使)|'
                     r'譲渡制限付株式|行使状況|配当|IR|ＩＲ')
UPREV = re.compile(r'上方修正|業績予想.*修正|増配|過去最高')
TANSHIN = re.compile(r'決算短信')
SKIP_TITLE = re.compile(r'訂正|英文|Notice|Summary|説明会資料|説明資料')


def _get(url: str, retries: int = 3) -> bytes:
    for i in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            time.sleep(2 * (i + 1))
        except Exception:
            time.sleep(2 * (i + 1))
    raise RuntimeError(url)


def _clean(s: str) -> str:
    return unicodedata.normalize('NFKC', re.sub(r'<[^>]+>', '', s)).strip()


# ---------------- TDnet ----------------
ROW = re.compile(r'kjTime"[^>]*>(.*?)</td>.*?kjCode"[^>]*>(.*?)</td>.*?kjName"[^>]*>(.*?)</td>.*?'
                 r'kjTitle"[^>]*>(.*?)</td>.*?kjXbrl"[^>]*>(.*?)</td>', re.S)


def tdnet_day(d: date) -> list[dict]:
    p = CACHE / 'tdnet' / f'{d:%Y%m%d}.json'
    if p.exists() and d < date.today():
        return json.loads(p.read_text())
    rows, page = [], 1
    while True:
        try:
            h = _get(f'{TDNET}I_list_{page:03d}_{d:%Y%m%d}.html').decode('utf-8', 'ignore')
        except urllib.error.HTTPError:
            break
        found = ROW.findall(h)
        for t, code, name, title, xbrl in found:
            pdf = re.search(r'href="([^"]+\.pdf)"', title)
            zp = re.search(r'href="([^"]+\.zip)"', xbrl)
            # 一覧には過去の開示が再掲されることがある。PDF 番号の日付が離れていたら捨てる
            if pdf and re.match(r'\d{4}(\d{8})', pdf.group(1)):
                pd_ = datetime.strptime(re.match(r'\d{4}(\d{8})', pdf.group(1)).group(1), '%Y%m%d').date()
                if abs((d - pd_).days) > 3:
                    continue
            rows.append({'date': f'{d:%Y-%m-%d}', 'time': _clean(t), 'code': _clean(code)[:4], 'name': _clean(name),
                         'title': _clean(title), 'pdf': TDNET + pdf.group(1) if pdf else '',
                         'xbrl': TDNET + zp.group(1) if zp else ''})
        if not found or f'I_list_{page + 1:03d}_' not in h:
            break
        page += 1
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rows, ensure_ascii=False))
    return rows


def tdnet(days: int) -> pd.DataFrame:
    ds = [date.today() - timedelta(days=i) for i in range(days)]
    with ThreadPoolExecutor(4) as ex:
        out = [r for rows in ex.map(tdnet_day, ds) for r in rows]
    return pd.DataFrame(out)


# ---------------- 決算短信 XBRL サマリー ----------------
def _num(a: str, body: str) -> float | None:
    v = _clean(body).replace(',', '')
    if not v or v in '-－':
        return None
    try:
        x = float(v)
    except ValueError:
        return None
    sc = re.search(r'scale="(-?\d+)"', a)
    x *= 10 ** int(sc.group(1)) if sc else 1
    return -x if 'sign="-"' in a else x


def xbrl_summary(url: str) -> dict:
    """営業利益（累計）・前年比・通期予想の前年比を取り出す。連結を優先、なければ単体"""
    key = re.sub(r'\W', '', url.rsplit('/', 1)[-1])
    p = CACHE / 'xbrl' / f'{key}.json'
    if p.exists():
        return json.loads(p.read_text())
    out: dict = {}
    try:
        z = zipfile.ZipFile(io.BytesIO(_get(url)))
        name = next(n for n in z.namelist() if '/Summary/' in n and n.endswith('-ixbrl.htm'))
        h = z.read(name).decode('utf-8', 'ignore')
        vals = {}
        for m in re.finditer(r'<ix:nonFraction([^>]*)>(.*?)</ix:nonFraction>', h, re.S):
            a = m.group(1)
            n = re.search(r'name="[^:"]+:([^"]+)"', a)
            c = re.search(r'contextRef="([^"]+)"', a)
            if n and c:
                vals[(n.group(1), c.group(1))] = _num(a, m.group(2))
        for scope in ('ConsolidatedMember', 'NonConsolidatedMember'):
            cur = [c for (n, c) in vals if n == 'OperatingIncome' and c.startswith('Current') and
                   scope in c and 'ResultMember' in c and c.split('_')[1] == scope]
            if not cur:
                continue
            ctx = cur[0]
            prior = ctx.replace('Current', 'Prior', 1)
            fc = f'CurrentYearDuration_{scope}_ForecastMember' if 'Year' not in ctx.split('_')[0] else f'NextYearDuration_{scope}_ForecastMember'
            g = lambda n, c: vals.get((n, c))
            out = {'period': ctx.split('_')[0].replace('CurrentAccumulated', '').replace('Duration', ''),
                   'sales': g('NetSales', ctx), 'op': g('OperatingIncome', ctx), 'op_prior': g('OperatingIncome', prior),
                   'op_yoy': g('ChangeInOperatingIncome', ctx), 'sales_yoy': g('ChangeInNetSales', ctx),
                   'op_fc': g('OperatingIncome', fc), 'op_fc_yoy': g('ChangeInOperatingIncome', fc)}
            if out['op_yoy'] is None and out['op'] and out['op_prior'] and out['op_prior'] > 0:
                out['op_yoy'] = out['op'] / out['op_prior'] - 1
            break
    except Exception as e:  # noqa: BLE001
        out = {'error': str(e)[:80]}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out))
    return out


# ---------------- 銘柄・株価 ----------------
def jpx_list() -> pd.DataFrame:
    p = CACHE / 'data_j.xlsx'
    if not p.exists() or time.time() - p.stat().st_mtime > 7 * 86400:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(_get(JPX_LIST))
    df = pd.read_excel(p, dtype={'コード': str})
    df = df[df['市場・商品区分'].str.contains('内国株式', na=False)]
    return pd.DataFrame({'code': df['コード'].str.strip(), '社名': df['銘柄名'],
                         '市場': df['市場・商品区分'].str.replace('（内国株式）', '', regex=False),
                         '業種': df['33業種区分']})


def prices(codes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    syms = [f'{c}.T' for c in codes] + ['1306.T']
    closes, vols = [], []
    for i in range(0, len(syms), 200):
        d = yf.download(syms[i:i + 200], period='1y', interval='1d', auto_adjust=True,
                        group_by='column', progress=False, threads=True)
        closes.append(d['Close'])
        vols.append(d['Volume'])
    c = pd.concat(closes, axis=1)
    v = pd.concat(vols, axis=1)
    c.columns = [x.replace('.T', '') for x in c.columns]
    v.columns = [x.replace('.T', '') for x in v.columns]
    return c.loc[:, ~c.columns.duplicated()], v.loc[:, ~v.columns.duplicated()]


def mcaps() -> pd.Series:
    from yfinance import EquityQuery as Q
    q = Q('eq', ['region', 'jp'])
    rows, off = [], 0
    while True:
        r = yf.screen(q, size=250, offset=off, sortField='intradaymarketcap', sortAsc=False)
        qs = r.get('quotes', [])
        rows += [{'code': x['symbol'][:-2], 'mcap': x.get('marketCap'), 'per': x.get('trailingPE'),
                  'pbr': x.get('priceToBook')} for x in qs if x['symbol'].endswith('.T')]
        off += len(qs)
        if not qs or off >= r.get('total', 0):
            break
    return pd.DataFrame(rows).drop_duplicates('code').set_index('code')


def reaction(ev: pd.DataFrame, close: pd.DataFrame, vol: pd.DataFrame) -> pd.DataFrame:
    """開示の直前の終値を基準に、今日までの騰落・市場（TOPIX連動 ETF）との差・最大上昇・出来高の倍率"""
    idx = close.index.tz_localize(None) if close.index.tz is not None else close.index
    close.index, vol.index = idx, idx
    topix = close['1306']
    out = []
    for r in ev.itertuples():
        if r.code not in close.columns:
            out.append({})
            continue
        d = pd.Timestamp(r.date)
        after_close = r.time >= CLOSE_HHMM
        base_i = idx.searchsorted(d, side='right') - (0 if after_close else 1) - 1
        if base_i < 60 or base_i >= len(idx) - 1:
            out.append({})
            continue
        s, ts, vs = close[r.code], topix, vol[r.code]
        b, tb = s.iloc[base_i], ts.iloc[base_i]
        post = s.iloc[base_i + 1:]
        if pd.isna(b) or post.dropna().empty:
            out.append({})
            continue
        last = post.dropna().iloc[-1]
        v_pre = vs.iloc[base_i - 60:base_i].mean()
        out.append({'基準日': idx[base_i].date(), '株価': last, '騰落': last / b - 1,
                    '対TOPIX': (last / b - 1) - (ts.dropna().iloc[-1] / tb - 1),
                    '最大上昇': post.max() / b - 1, '初日': post.iloc[0] / b - 1,
                    '出来高倍率': (vs.iloc[base_i + 1:base_i + 6].mean() / v_pre) if v_pre else None,
                    '200日線比': last / s.iloc[max(0, base_i - 200):base_i].mean() - 1})
    return pd.concat([ev.reset_index(drop=True), pd.DataFrame(out)], axis=1)


# ---------------- 3タイプの判定 ----------------
def pick(ev: pd.DataFrame, min_mcap: float, max_mcap: float) -> dict[str, pd.DataFrame]:
    ev = ev[ev['mcap'].fillna(0).between(min_mcap, max_mcap)].dropna(subset=['騰落'])
    tob = set(ev.loc[ev['title'].str.contains(TOB_TARGET), 'code'])  # 期間内に TOB 等の対象になった会社は全部外す
    ev = ev[~ev['code'].isin(tob)]
    t = ev['title']
    res = {}

    # B. 好決算なのに売られた
    b = ev[t.str.contains(TANSHIN) & ~t.str.contains(SKIP_TITLE) & (ev['xbrl'] != '')].copy()
    up = set(ev.loc[t.str.contains(UPREV), 'code'])
    b['上方修正等'] = b['code'].isin(up)
    good = (b['op'] > 0) & (b['op_yoy'] >= 0.20) & ((b['op_fc_yoy'].fillna(0) >= 0) | b['上方修正等'])
    b = b[good & (b['対TOPIX'] <= -0.05)].copy()
    b['点'] = b['op_yoy'].clip(upper=2.0).rank(pct=True) + (-b['対TOPIX']).rank(pct=True) + b['上方修正等'] * 0.3
    res['B'] = b.sort_values('点', ascending=False).drop_duplicates('code')

    # A. 企業再評価
    a = ev[t.str.contains(RERATE) & ~t.str.contains(TOB_TARGET) & ~t.str.contains(SKIP_TITLE)].copy()
    a = a[(a['対TOPIX'] >= 0.05) & (a['200日線比'] > 0)]
    a['本気度'] = a['title'].str.contains(RERATE_STRONG)
    a['点'] = a['対TOPIX'].clip(upper=1.0).rank(pct=True) + a['出来高倍率'].clip(upper=10).rank(pct=True) * 0.5 \
        + a['本気度'] * 0.6
    res['A'] = a.sort_values('点', ascending=False).drop_duplicates('code')

    # C. 大きな思惑（PER は問わない）
    base = ev[~t.str.contains(TOB_TARGET) & ~t.str.contains(SKIP_TITLE) & ~t.str.contains(TANSHIN)
              & ~t.str.contains(BUYBACK)]
    theme = base[base['title'].str.contains(THEME)]
    theme = theme[((theme['最大上昇'] >= 0.20) | (theme['初日'] >= 0.10)) & (theme['出来高倍率'] >= 3)]
    # 表題にキーワードは無いが、出来高を伴って急騰した（材料は開示を読んで確かめる）
    other = base[~base['title'].str.contains(THEME) & ~base['title'].str.contains(ROUTINE)]
    other = other[(other['最大上昇'] >= 0.30) & (other['出来高倍率'] >= 5)]
    c = pd.concat([theme.assign(材料='テーマ'), other.assign(材料='急騰')])
    c['点'] = c['最大上昇'].clip(upper=3).rank(pct=True) + c['出来高倍率'].clip(upper=30).rank(pct=True) \
        + c['対TOPIX'].rank(pct=True) * 0.5
    res['C'] = c.sort_values('点', ascending=False).drop_duplicates('code')
    return res


# ---------------- 出力 ----------------
def _p(x, d=1):
    return '-' if pd.isna(x) else f'{x * 100:+.{d}f}%'


def _f(x, fmt='{:.1f}'):
    return '-' if x is None or pd.isna(x) else fmt.format(x)


def _row(r, extra: str) -> str:
    title = r['title'] if len(r['title']) <= 48 else r['title'][:47] + '…'
    return (f"| {r['code']} | {r['社名']} | {r['市場']} | {_f(r['mcap'] / 1e8 if r['mcap'] else None, '{:,.0f}')} | "
            f"{_f(r['per'])} | {r['date'][5:]} | [{title}]({r['pdf']}) | {extra} | {_p(r['騰落'])} | {_p(r['対TOPIX'])} |")


def write(res: dict, out: Path, days: int, top: int, cap: tuple) -> None:
    L = [f'# 株価目線の候補（{date.today():%Y/%m/%d}、直近{days}日の開示）', '',
         f'対象: 内国株、時価総額 {cap[0]:,.0f}億〜{cap[1]:,.0f}億円。', '',
         'TDnet の全銘柄の開示と株価の反応から、機械的に拾った**入口のリスト**。ここから気になる銘柄を選んで `コード_基礎.md` を作る。',
         '騰落・対TOPIX は「開示の直前の終値」から最新の終値まで。対TOPIX は TOPIX 連動 ETF（1306）との差。', '']
    head = '| コード | 社名 | 市場 | 時価総額(億) | PER | 開示日 | 開示 | {} | 騰落 | 対TOPIX |\n|---|---|---|--:|--:|---|---|---|--:|--:|'
    sec = [
        ('A', 'A. 企業再評価の路線', '中計・資本コスト・還元方針・事業の組み替えなどの開示のあと、株価が TOPIX より5%以上強く、200日線の上にある。◎ は中計・資本コスト・配当方針など本気度の高い開示（自社株買い・分割・優待より上位に置く）。',
         '出来高倍率', lambda r: _f(r['出来高倍率'], '{:.1f}倍') + (' ◎' if r['本気度'] else '')),
        ('B', 'B. 好決算なのに地合いで売られた', '決算短信で営業利益（累計）が前年より20%以上増え、通期予想も増益（または上方修正・増配）なのに、発表後に TOPIX より5%以上弱い。',
         '営業益 前年比 / 通期予想', lambda r: f"{_p(r['op_yoy'], 0)} / {_p(r['op_fc_yoy'], 0)}{' ★修正' if r['上方修正等'] else ''}"),
        ('C', 'C. 大きな思惑（PER は問わない）', 'AI・半導体・宇宙・防衛・提携・受注・他社の買収などの開示のあと（TOB される側・自社株買いは除く）、最大20%以上（または初日10%以上）上がり、出来高が普段の3倍以上。? は表題にキーワードは無いが、出来高5倍以上で30%以上急騰したもの（材料は開示を読んで確かめる）。',
         '最大上昇 / 出来高', lambda r: f"{_p(r['最大上昇'], 0)} / {_f(r['出来高倍率'], '{:.0f}倍')}" + (' ?' if r['材料'] == '急騰' else '')),
    ]
    for k, h, note, col, fn in sec:
        df = res[k]
        L += [f'## {h}（{len(df)}銘柄）', '', note, '', head.format(col)]
        L += [_row(r, fn(r)) for _, r in df.head(top).iterrows()]
        L.append('')
    L += ['> 決算の数字は決算短信の XBRL サマリー（累計・前年同期比）。株価・時価総額・PER は Yahoo Finance。',
          '> 機械的な抽出なので、開示の中身と株価が動いた本当の理由は必ず読んで確かめる。投資判断は自分で。', '']
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(L), encoding='utf-8')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--days', type=int, default=21)
    ap.add_argument('--min-mcap', type=float, default=20, help='時価総額の下限（億円）')
    ap.add_argument('--max-mcap', type=float, default=3000, help='時価総額の上限（億円）。大型株も見るなら大きく')
    ap.add_argument('--top', type=int, default=25)
    a = ap.parse_args()

    print('1/5 TDnet の開示一覧')
    ev = tdnet(a.days)
    lst = jpx_list()
    ev = ev.merge(lst, on='code', how='inner')
    print(f'    {len(ev)} 件（内国株）')

    print('2/5 決算短信の XBRL')
    tan = ev[ev['title'].str.contains(TANSHIN) & ~ev['title'].str.contains(SKIP_TITLE) & (ev['xbrl'] != '')]
    with ThreadPoolExecutor(6) as ex:
        sm = dict(zip(tan.index, ex.map(xbrl_summary, tan['xbrl'])))
    ev = ev.join(pd.DataFrame.from_dict(sm, orient='index'))
    for c in ('op', 'op_yoy', 'op_fc_yoy', 'sales_yoy'):
        if c not in ev:
            ev[c] = None
    # XBRL の前年比は % 表記（scale=-2 で小数になっている）
    print(f'    {len(tan)} 件')

    print('3/5 株価')
    close, vol = prices(sorted(ev['code'].unique()))

    print('4/5 時価総額・PER')
    ev = ev.join(mcaps(), on='code')

    print('5/5 判定')
    ev = reaction(ev, close, vol)
    res = pick(ev, a.min_mcap * 1e8, a.max_mcap * 1e8)
    out = ROOT / 'candidates' / f'{date.today():%Y-%m-%d}.md'
    write(res, out, a.days, a.top, (a.min_mcap, a.max_mcap))
    print(f"    A {len(res['A'])} / B {len(res['B'])} / C {len(res['C'])} → {out.relative_to(ROOT)}")


if __name__ == '__main__':
    main()
