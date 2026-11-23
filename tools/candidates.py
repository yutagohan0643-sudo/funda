"""材料の中身から候補を探す。TDnet の全銘柄の開示を読み、材料そのものの大きさで3つのタイプを拾う（株価の動きは見ない）。
A（次の決算に向けた候補・勝率ランキング）と C（割安で放置）は、この一覧と決算短信をもとに _次の決算.md で人が作る。

  D. 企業再評価      中計・資本コスト・還元方針・事業の組み替え・自社株買い。自社株買いは取得株数の割合で大きさを見る
  E. 大きな材料      受注・契約・提携・買収・承認など。本文の金額 ÷ 年間売上で大きさを見る
  B. 決算・業績が非常に良い  業績予想の上方修正（XBRL の前回予想→今回予想）と、決算短信の増益率・通期予想への進捗

使い方（リポジトリの一番上で）:
  python tools/candidates.py            # 直近 21 日の開示 → candidates/YYYY-MM-DD.md（機械的な一覧）
  python tools/candidates.py --days 31  # TDnet に残っているのは約1か月分
  一覧の上位は本文を読み、candidates/YYYY-MM-DD_次の決算.md（A・B・C）と _材料.md（直近・D・E）に人の頭で評価を書く

データ: TDnet 適時開示一覧と PDF・XBRL（www.release.tdnet.info）、JPX 上場銘柄一覧、Yahoo Finance（時価総額だけ）
キャッシュ: cache/（git に入れない）
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
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
                           r'商号|資本業務提携')
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


# ---------------- XBRL ----------------
FACT = re.compile(r'<ix:nonFraction([^>]*?)(?:/>|>(.*?)</ix:nonFraction>)', re.S)


def xbrl_facts(url: str) -> dict:
    """XBRL サマリーの数値を {名前|コンテキスト: 値} で返す（空欄の値は飛ばす）"""
    key = re.sub(r'\W', '', url.rsplit('/', 1)[-1])
    p = CACHE / 'xbrl2' / f'{key}.json'
    if p.exists():
        return json.loads(p.read_text())
    out = {}
    try:
        z = zipfile.ZipFile(io.BytesIO(_get(url)))
        names = [n for n in z.namelist() if n.endswith('-ixbrl.htm') and ('/Summary/' in n or 'rvfc' in n or 'rvdf' in n)]
        for n in names:
            h = z.read(n).decode('utf-8', 'ignore')
            for m in FACT.finditer(h):
                a, body = m.group(1), m.group(2)
                if body is None or 'xsi:nil="true"' in a:
                    continue
                nm = re.search(r'name="[^:"]+:([^"]+)"', a)
                cx = re.search(r'contextRef="([^"]+)"', a)
                v = _num(a, body)
                if nm and cx and v is not None:
                    out[f'{nm.group(1)}|{cx.group(1)}'] = v
    except Exception as e:  # noqa: BLE001
        out = {'_error': str(e)[:80]}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out))
    return out


def _num(a: str, body: str) -> float | None:
    v = _clean(body).replace(',', '')
    try:
        x = float(v)
    except ValueError:
        return None
    sc = re.search(r'scale="(-?\d+)"', a)
    x *= 10 ** int(sc.group(1)) if sc else 1
    return -x if 'sign="-"' in a else x


def _scope(f: dict) -> str:
    return 'ConsolidatedMember' if any('_ConsolidatedMember' in k for k in f) else 'NonConsolidatedMember'


def tanshin_metrics(f: dict) -> dict:
    """決算短信: 累計の売上・営業利益と前年比、通期予想と前年比、通期予想への進捗"""
    sc = _scope(f)
    cur = sorted({k.split('|')[1] for k in f if k.startswith('OperatingIncome|Current') and f'_{sc}_' in k + '_'
                  and 'ResultMember' in k})
    if not cur:
        return {}
    ctx = cur[0]
    period = ctx.split('_')[0]
    annual = 'Year' in period and 'Accumulated' not in period
    fc = f'{"NextYearDuration" if annual else "CurrentYearDuration"}_{sc}_ForecastMember'
    g = lambda n, c: f.get(f'{n}|{c}')
    op, op_p = g('OperatingIncome', ctx), g('OperatingIncome', ctx.replace('Current', 'Prior', 1))
    yoy = g('ChangeInOperatingIncome', ctx)
    if yoy is None and op is not None and op_p and op_p > 0:
        yoy = op / op_p - 1
    fop = g('OperatingIncome', fc)
    q = re.search(r'Q(\d)', period)
    return {'期': '通期' if annual else (f'{q.group(1)}Q累計' if q else period),
            '売上': g('NetSales', ctx), '営業益': op, '営業益_前年': op_p, '営業益_前年比': yoy,
            '売上_前年比': g('ChangeInNetSales', ctx), '予想売上': g('NetSales', fc), '予想営業益': fop,
            '予想営業益_前年比': g('ChangeInOperatingIncome', fc),
            '進捗': (op / fop) if (op and fop and fop > 0 and not annual) else None,
            '標準進捗': (int(q.group(1)) / 4) if (q and not annual) else None}


def revision_metrics(f: dict) -> dict:
    """業績予想の修正: 前回予想 → 今回予想（レンジ予想は中央値）"""
    sc = _scope(f)
    def val(name, which):
        base = f'CurrentYearDuration_{sc}_{which}Member'
        v = f.get(f'{name}|{base}_ForecastMember')
        if v is None:
            lo, hi = f.get(f'{name}|{base}_LowerMember'), f.get(f'{name}|{base}_UpperMember')
            v = (lo + hi) / 2 if lo is not None and hi is not None else None
        return v
    out = {}
    for name, jp in (('NetSales', '売上'), ('OperatingIncome', '営業益'), ('OrdinaryIncome', '経常益'),
                     ('ProfitAttributableToOwnersOfParent', '純利益'), ('ProfitAttributableToOwners', '純利益')):
        a, b = val(name, 'Previous'), val(name, 'Current')
        if b is None or jp + '_今回' in out:
            continue
        out[f'{jp}_前回'], out[f'{jp}_今回'] = a, b
        out[f'{jp}_修正率'] = (b / a - 1) if (a and a > 0) else None
        prior = f.get(f'{name}|PriorYearDuration_{sc}_CurrentMember_ResultMember') or \
            f.get(f'{name}|PriorYearDuration_{sc}_ResultMember')
        out[f'{jp}_前期'] = prior
    return out


def qual_text(url: str, n: int = 15000) -> str:
    """決算短信 XBRL の定性的情報（経営成績の説明など）を文字にして返す"""
    key = re.sub(r'\W', '', url.rsplit('/', 1)[-1])
    p = CACHE / 'qual' / f'{key}.txt'
    if p.exists():
        return p.read_text()
    txt = ''
    try:
        z = zipfile.ZipFile(io.BytesIO(_get(url)))
        name = next((x for x in z.namelist() if x.endswith('qualitative.htm')), None)
        if name:
            h = z.read(name).decode('utf-8', 'ignore')
            h = re.sub(r'<(script|style)[^>]*>.*?</\1>', ' ', h, flags=re.S)
            txt = re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', re.sub(r'<[^>]+>', ' ', h)))[:n]
    except Exception:  # noqa: BLE001
        txt = ''
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(txt)
    return txt


def forecast_note(q: str) -> str:
    """定性的情報のうち「業績予想などの将来予測情報に関する説明」の本文（目次は飛ばす）"""
    for m in re.finditer('将来予測情報', q):
        if not re.match(r'[^。]{0,40}?(?:\.{4,}|…{2,})', q[m.end():m.end() + 60]):
            return q[m.start():m.start() + 600]
    return ''


# ---------------- PDF 本文 ----------------
def pdf_text(url: str, pages: int = 3) -> str:
    key = re.sub(r'\W', '', url.rsplit('/', 1)[-1])
    p = CACHE / 'text' / f'{key}.txt'
    if p.exists():
        return p.read_text()
    try:
        raw = _get(url)
        with tempfile.NamedTemporaryFile(suffix='.pdf') as tf:
            tf.write(raw)
            tf.flush()
            txt = subprocess.run(['pdftotext', '-l', str(pages), '-layout', tf.name, '-'],
                                 capture_output=True, text=True, timeout=60).stdout
    except Exception:  # noqa: BLE001
        txt = ''
    txt = unicodedata.normalize('NFKC', txt)
    txt = re.sub(r'[ \t\u3000]+', ' ', txt)
    txt = re.sub(r'\n\s*\n+', '\n', txt)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(txt)
    return txt


AMOUNT = re.compile(r'(?:約|総額|合計)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*(兆|億|百万|千)\s*円')
UNIT = {'兆': 1e12, '億': 1e8, '百万': 1e6, '千': 1e3}
KEY_AMOUNT = re.compile(r'受注(?:金額|額|高)|契約(?:金額|総額)|取得価額|取得価格|取得金額|譲受価額|譲渡価額|買収金額|'
                        r'取得総額|取得価額の総額|調達|払込金額の総額|出資(?:金額|額)|投資(?:金額|額)|事業規模|総額')
SHARE_RATIO = re.compile(r'(?:発行済株式総数|自己株式を除く)[^\n%]{0,40}?(?:割合|比率)?[^\n%]{0,10}?([0-9]+(?:\.[0-9]+)?)\s*%')


def amounts(txt: str) -> tuple[float | None, str]:
    """本文から材料の金額を取る。受注金額・取得価額などの行を優先。概要表（資本金・総資産など）は除く"""
    best, line_best = None, ''
    for line in txt.splitlines():
        if re.search(r'資本金|総資産|純資産|売上高|営業利益|経常利益|当期純利益|大株主|資本準備金', line):
            continue
        if not KEY_AMOUNT.search(line):
            continue
        for m in AMOUNT.finditer(line):
            v = float(m.group(1).replace(',', '')) * UNIT[m.group(2)]
            if best is None or v > best:
                best, line_best = v, line.strip()[:80]
    return best, line_best


def excerpt(txt: str, n: int = 220) -> str:
    body = txt.split('記', 1)[1] if '\n記\n' in txt else txt
    body = re.sub(r'\s+', ' ', body)
    return body[:n]


REVISION = re.compile(r'業績予想.*修正|予想の修正|上方修正|業績予想と実績値')
DIVIDEND = re.compile(r'配当予想の修正|増配|配当方針|記念配当|特別配当')
DOWN = re.compile(r'下方修正|減配|無配|損失|赤字|減損')
# お金の手当て（借入・増資）は材料ではない。ただし「提携」を含むものは残す
FINANCE = re.compile(r'借入|ローン|コミットメント|当座貸越|社債|優先株式|新株予約権|第三者割当|売出し|上場承認|'
                     r'担保|保証|リース契約|資金の借|金銭消費貸借')
MIN_OP = 2e8  # 営業利益がこれ未満の会社は、増益率が大きくても材料として小さい
MATERIAL = re.compile(r'受注|契約|提携|共同開発|ライセンス|承認|採択|特許|子会社化|株式(?:の)?取得|事業の譲受|買収|'
                      r'合弁|出資|新規事業|参入|量産|販売開始|上市|AI|ＡＩ|データセンター|半導体|宇宙|防衛|量子|ロボット|暗号資産', re.I)


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



# ---------------- 材料の評価 ----------------
def evaluate(ev: pd.DataFrame, workers: int = 6) -> dict[str, pd.DataFrame]:
    t = ev['title']
    common = ~t.str.contains(SKIP_TITLE)
    # 年間売上（材料の大きさの物差し）: 期間内の決算短信の通期予想売上 → 業績修正の今回予想売上
    sales = {}

    # --- B. 決算・業績 ---
    tan = ev[t.str.contains(TANSHIN) & common & (ev['xbrl'] != '')]
    rev = ev[t.str.contains(REVISION) & common & (ev['xbrl'] != '') & ~t.str.contains(TANSHIN)]
    with ThreadPoolExecutor(workers) as ex:
        tf = dict(zip(tan.index, ex.map(xbrl_facts, tan['xbrl'])))
        rf = dict(zip(rev.index, ex.map(xbrl_facts, rev['xbrl'])))
    T = pd.DataFrame({i: tanshin_metrics(f) for i, f in tf.items()}).T
    R = pd.DataFrame({i: revision_metrics(f) for i, f in rf.items()}).T
    for i, r in T.iterrows():
        if pd.notna(r.get('予想売上')):
            sales.setdefault(ev.at[i, 'code'], r['予想売上'])
    for i, r in R.iterrows():
        if pd.notna(r.get('売上_今回')):
            sales.setdefault(ev.at[i, 'code'], r['売上_今回'])

    b_rev = ev.loc[R.index].join(R) if len(R) else ev.iloc[0:0]
    if len(b_rev):
        b_rev = b_rev[(b_rev['営業益_修正率'] >= 0.10) & (b_rev['営業益_今回'] >= MIN_OP)].copy()
        b_rev['種類'] = '上方修正'
        b_rev['大きさ'] = b_rev['営業益_修正率'].clip(upper=2) * 1.2
    b_tan = ev.loc[T.index].join(T) if len(T) else ev.iloc[0:0]
    if len(b_tan):
        lead = (b_tan['進捗'] - b_tan['標準進捗']).astype(float)
        b_tan['進捗の上乗せ'] = lead
        # 会社予想が前年より大きく減益のまま、進捗だけ高い = 予想が古い・保守的（上方修正の余地）
        b_tan['予想が保守的'] = (lead >= 0.20) & (b_tan['予想営業益_前年比'].fillna(0) < 0.10)
        # 別の開示を出さず、決算短信の中で上方修正する会社もある（例: サンコール 2026/11/13）
        with ThreadPoolExecutor(workers) as ex:
            qs = list(ex.map(qual_text, b_tan['xbrl']))
        b_tan['短信で上方修正'] = [bool(re.search(r'上方修正', forecast_note(q))) for q in qs]
        grow = (b_tan['営業益_前年比'] >= 0.30) & (b_tan['営業益_前年'] >= 0.5e8) & (b_tan['予想営業益_前年比'].fillna(0) >= 0.10)
        revised = b_tan['短信で上方修正'] & (b_tan['予想営業益_前年比'].fillna(0) >= 0.20)
        ok = (b_tan['営業益'] >= MIN_OP) & (grow | (lead >= 0.20) | revised)
        b_tan = b_tan[ok].copy()
        b_tan['種類'] = '決算'
        yoy = b_tan['営業益_前年比'].where(b_tan['営業益_前年'] >= 0.5e8).clip(lower=0, upper=1.5).fillna(0)
        b_tan['大きさ'] = yoy + b_tan['進捗の上乗せ'].fillna(0).clip(lower=0, upper=0.6) * 2 \
            + b_tan['予想営業益_前年比'].fillna(0).clip(lower=0, upper=1.0) * 0.5 + b_tan['短信で上方修正'] * 0.3
    B = pd.concat([b_rev, b_tan]).sort_values('大きさ', ascending=False)

    # --- A・C は本文を読む ---
    # 自社株買いは「新しく決めた」ものだけ（取得の結果・状況の報告、消却だけの開示、売出しは除く）
    a = ev[t.str.contains(RERATE) & common & ~t.str.contains(TOB_TARGET)
           & ~t.str.contains(r'取得結果|取得状況|進捗状況|取得終了|売出し')
           & ~(t.str.contains('消却') & ~t.str.contains('取得'))].copy()
    c = ev[t.str.contains(MATERIAL) & common & ~t.str.contains(TOB_TARGET) & ~t.str.contains(TANSHIN)
           & ~t.str.contains(REVISION) & ~t.str.contains(BUYBACK) & ~t.str.contains(DOWN)
           & (~t.str.contains(FINANCE) | t.str.contains('提携'))
           & ~t.str.contains(ROUTINE.pattern.replace('|異動', '').replace('配当|', ''))].copy()
    need = pd.concat([a, c])['pdf'].drop_duplicates()
    with ThreadPoolExecutor(workers) as ex:
        texts = dict(zip(need, ex.map(pdf_text, need)))

    a['本文'] = a['pdf'].map(texts).fillna('')
    a['本気度'] = a['title'].str.contains(RERATE_STRONG)
    a['取得割合'] = a['本文'].map(lambda x: max([float(v) for v in SHARE_RATIO.findall(x) if float(v) < 50] or [None])
                                 if '自己株式' in x else None)
    a['金額'] = a['本文'].map(lambda x: amounts(x)[0])
    a['対時価'] = a['金額'] / a['mcap']
    a['大きさ'] = a['本気度'] * 1.0 + a['取得割合'].fillna(0).clip(upper=15) / 5
    # 自社株買いだけで小さいもの（2%未満）は落とす
    a = a[a['本気度'] | (a['取得割合'].fillna(0) >= 2)].sort_values('大きさ', ascending=False)

    c['本文'] = c['pdf'].map(texts).fillna('')
    am = c['本文'].map(amounts)
    c['金額'] = am.map(lambda x: x[0])
    c['金額の行'] = am.map(lambda x: x[1])
    c['年間売上'] = c['code'].map(sales)
    c['対売上'] = c['金額'] / c['年間売上']
    c['対時価'] = c['金額'] / c['mcap']
    c['大きさ'] = c['対売上'].clip(upper=3).fillna(c['対時価'].clip(upper=3) * 0.5).fillna(0)
    C = c.sort_values('大きさ', ascending=False)
    return {'A': a, 'B': B, 'C': C}


# ---------------- 出力 ----------------
def _p(x, d=0):
    return '-' if x is None or pd.isna(x) else f'{x * 100:+.{d}f}%'


def _oku(x):
    return '-' if x is None or pd.isna(x) else f'{x / 1e8:,.1f}億'


def _ratio(x):
    return '-' if x is None or pd.isna(x) else f'{x * 100:.0f}%'


RECENT: set = set()  # 直近の営業日（write で決める）


def _link(r):
    title = r['title'] if len(r['title']) <= 44 else r['title'][:43] + '…'
    new = '【新】' if r['date'] in RECENT else ''
    return f"{new}[{title}]({r['pdf']})"


def write(res: dict, out: Path, days: int, top: int, cap: tuple, recent_days: int = 3) -> None:
    L = [f'# 材料から見た候補（{date.today():%Y/%m/%d}、直近{days}日の開示）', '',
         f'TDnet の全銘柄の開示を読み、**材料そのものの大きさ**で並べた機械的な一覧（株価の動きは見ていない）。'
         f'対象は内国株・時価総額 {cap[0]:,.0f}億〜{cap[1]:,.0f}億円。TOB される側は除く。',
         '上位の本文を読んだ評価は同じ日付の `_次の決算.md` と `_材料.md`。', '']
    # 直近3営業日の開示を先頭にまとめる（鮮度の高いものから見る）
    global RECENT
    days = sorted({d for k in ('A', 'B', 'C') for d in res[k]['date']}, reverse=True)[:recent_days]
    RECENT = set(days)
    if days:
        L += [f'## 直近{len(days)}営業日（{days[-1][5:].replace("-", "/")}〜{days[0][5:].replace("-", "/")}）の開示', '',
              '下の B・D・E から、直近の開示だけを抜き出した（各表では【新】）。大引け 15:30 以降の開示は、翌営業日から株価が反応する。', '',
              '| 日付 | 時刻 | タイプ | コード | 社名 | 時価総額 | 開示 |', '|---|---|---|---|---|--:|---|']
        for k, name in (('B', 'B 決算・業績'), ('A', 'D 企業再評価'), ('C', 'E 大きな材料')):
            df = res[k][res[k]['date'].isin(RECENT)].drop_duplicates('code')
            for _, r in df.sort_values(['date', '大きさ'], ascending=[False, False]).iterrows():
                L.append(f"| {r['date'][5:].replace('-', '/')} | {r['time']} | {name} | {r['code']} | {r['社名']} | {_oku(r['mcap'])} | {_link(r)} |")
        L.append('')

    B = res['B'].drop_duplicates('code').head(top)
    L += [f'## B. 決算・業績が非常に良い（{len(res["B"].drop_duplicates("code"))}社）', '',
          '業績予想の上方修正（営業利益 +10% 以上。決算短信の中での上方修正で通期予想 +20% 以上も含む）と、決算短信で営業利益 +30% 以上かつ通期予想も +10% 以上、または通期予想への進捗が標準より20ポイント以上先行（会社予想が減益のままなら「予想が保守的」＝上方修正の余地）。営業利益2億円未満は除く。', '',
          '| コード | 社名 | 時価総額 | 開示日 | 開示 | 中身 |', '|---|---|--:|---|---|---|']
    for _, r in B.iterrows():
        if r['種類'] == '上方修正':
            what = (f"営業益 {_oku(r['営業益_前回'])}→{_oku(r['営業益_今回'])}（{_p(r['営業益_修正率'])}）"
                    f"、前期 {_oku(r.get('営業益_前期'))}")
        else:
            what = (f"{r['期']} 営業益 {_oku(r['営業益'])}（前年比 {_p(r['営業益_前年比'])}）"
                    f"、通期予想 {_oku(r['予想営業益'])}（{_p(r['予想営業益_前年比'])}）"
                    + (f"、進捗 {r['進捗'] * 100:.0f}%（標準 {r['標準進捗'] * 100:.0f}%）" if pd.notna(r.get('進捗')) else '')
                    + ('、**予想が保守的**' if r.get('予想が保守的') is True else '')
                    + ('、**短信で上方修正**' if r.get('短信で上方修正') is True else ''))
        L.append(f"| {r['code']} | {r['社名']} | {_oku(r['mcap'])} | {r['date'][5:]} | {_link(r)} | {what} |")

    A = res['A'].drop_duplicates('code').head(top)
    L += ['', f'## D. 企業再評価（{len(res["A"].drop_duplicates("code"))}社）', '',
          '中計・資本コスト・配当方針・事業の組み替えなど（◎）と、発行済株式の2%以上の自社株買い。', '',
          '| コード | 社名 | 時価総額 | 開示日 | 開示 | 自社株買い（株数の割合 / 金額） |', '|---|---|--:|---|---|---|']
    for _, r in A.iterrows():
        bb = f"{r['取得割合']:.1f}% / {_oku(r['金額'])}" if pd.notna(r['取得割合']) else '-'
        L.append(f"| {r['code']} | {r['社名']} | {_oku(r['mcap'])} | {r['date'][5:]} | {'◎ ' if r['本気度'] else ''}{_link(r)} | {bb} |")

    C = res['C'].drop_duplicates('code').head(top)
    L += ['', f'## E. 大きな材料（{len(res["C"].drop_duplicates("code"))}社）', '',
          '受注・契約・提携・買収・承認など。本文の金額（受注金額・取得価額など）÷ 年間売上（期間内の決算・修正の通期予想）で大きさを見る。金額の無い材料は下に来るので、表題で拾い読みする。', '',
          '| コード | 社名 | 時価総額 | 開示日 | 開示 | 金額 | 対売上 | 対時価 |', '|---|---|--:|---|---|--:|--:|--:|']
    for _, r in C.iterrows():
        L.append(f"| {r['code']} | {r['社名']} | {_oku(r['mcap'])} | {r['date'][5:]} | {_link(r)} | {_oku(r['金額'])} | "
                 f"{_ratio(r['対売上'])} | {_ratio(r['対時価'])} |")
    L += ['', '> 数字は TDnet の XBRL と開示 PDF から機械的に取ったもの。金額の取り違えがあり得るので、必ず本文で確かめる。', '']
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(L), encoding='utf-8')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--days', type=int, default=21)
    ap.add_argument('--min-mcap', type=float, default=20, help='時価総額の下限（億円）')
    ap.add_argument('--max-mcap', type=float, default=3000, help='時価総額の上限（億円）')
    ap.add_argument('--top', type=int, default=30)
    ap.add_argument('--recent', type=int, default=3, help='先頭にまとめる直近の営業日数')
    a = ap.parse_args()

    print('1/3 TDnet の開示一覧')
    ev = tdnet(a.days).merge(jpx_list(), on='code', how='inner')
    ev = ev.join(mcaps(), on='code')
    tob = set(ev.loc[ev['title'].str.contains(TOB_TARGET), 'code'])
    ev = ev[ev['mcap'].fillna(0).between(a.min_mcap * 1e8, a.max_mcap * 1e8) & ~ev['code'].isin(tob)]
    print(f'    {len(ev)} 件（内国株・時価総額で絞り、TOB される側を除く）')

    print('2/3 XBRL と本文を読む')
    res = evaluate(ev)

    print('3/3 書き出し')
    out = ROOT / 'candidates' / f'{date.today():%Y-%m-%d}.md'
    write(res, out, a.days, a.top, (a.min_mcap, a.max_mcap), a.recent)
    print(f"    B（決算） {res['B']['code'].nunique()} / D（再評価） {res['A']['code'].nunique()} / E（材料） {res['C']['code'].nunique()} → {out.relative_to(ROOT)}")


if __name__ == '__main__':
    main()
