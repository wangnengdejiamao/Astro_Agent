"""
WISE / NEOWISE 红外测光与光变曲线
===================================
AllWISE: W1 (3.4um), W2 (4.6um), W3 (12um), W4 (22um)
NEOWISE: W1/W2 多历元光变曲线

用法:
    from astro_toolbox.wise import get_photometry, query_lightcurve
    phot = get_photometry(190.305, 2.596)
    lc = query_lightcurve(190.305, 2.596)
"""
import numpy as np
import pandas as pd
from . import config, utils


def get_photometry(ra, dec, radius_arcsec=config.SEARCH_RADIUS_ARCSEC):
    """
    查询 AllWISE 测光 (W1-W4)。

    Returns:
        dict: {band_name: (mag, mag_err, wave_A)}
    """
    tbl = utils.query_vizier('II/328/allwise', ra, dec, radius_arcsec,
                             columns=['W1mag', 'e_W1mag', 'W2mag', 'e_W2mag',
                                      'W3mag', 'e_W3mag', 'W4mag', 'e_W4mag'])
    if tbl is None:
        return {}

    row = tbl[0]
    bands = {
        'WISE_W1': ('W1mag', 'e_W1mag'),
        'WISE_W2': ('W2mag', 'e_W2mag'),
        'WISE_W3': ('W3mag', 'e_W3mag'),
        'WISE_W4': ('W4mag', 'e_W4mag'),
    }
    phot = {}
    for band_name, (mag_col, err_col) in bands.items():
        try:
            mag = float(row[mag_col])
        except (ValueError, KeyError, np.ma.MaskError):
            continue
        if not (0 < mag < 30):
            continue
        # 误差列常为 masked/null (W3/W4 尤其如此), 用默认值 0.1 mag 代替
        try:
            err = float(row[err_col])
            if not (err > 0):
                err = 0.1
        except (ValueError, KeyError, np.ma.MaskError):
            err = 0.1
        wave = config.BAND_INFO[band_name]['wave_A']
        phot[band_name] = (mag, err, wave)
    return phot


def query_lightcurve(ra, dec, radius_arcsec=config.SEARCH_RADIUS_ARCSEC):
    """
    查询 NEOWISE 多历元光变曲线 (W1/W2)。
    通过 IRSA TAP 查询 neowiser_p1bs_psd 表 (比 Irsa.query_region 快很多)。

    Returns:
        dict: {'W1': DataFrame(mjd, mag, magerr), 'W2': ...}
    """
    import requests as req

    # 使用 IRSA TAP 服务，比 astroquery 的 Irsa.query_region 快 10-50 倍
    tap_url = "https://irsa.ipac.caltech.edu/TAP/sync"
    adql = (
        f"SELECT mjd, w1mpro, w1sigmpro, w2mpro, w2sigmpro "
        f"FROM neowiser_p1bs_psd "
        f"WHERE CONTAINS(POINT('ICRS', ra, dec), "
        f"CIRCLE('ICRS', {ra}, {dec}, {radius_arcsec/3600.0})) = 1 "
        f"AND qual_frame > 0 "
        f"ORDER BY mjd"
    )
    params = {
        'REQUEST': 'doQuery',
        'LANG': 'ADQL',
        'FORMAT': 'csv',
        'QUERY': adql,
    }

    try:
        session = utils.get_session(tap_url)
        resp = session.post(tap_url, data=params, timeout=utils.get_timeout())
        resp.raise_for_status()
        text = resp.text.strip()
        if not text or 'mjd' not in text.lower():
            return None

        import io
        df = pd.read_csv(io.StringIO(text))
        if len(df) == 0:
            return None
    except Exception as e:
        print(f"NEOWISE TAP 查询失败: {e}")
        # 回退到 astroquery
        return _query_lightcurve_astroquery(ra, dec, radius_arcsec)

    result = {'ra': ra, 'dec': dec, 'survey': 'NEOWISE'}

    for band, mag_col, err_col in [('W1', 'w1mpro', 'w1sigmpro'),
                                    ('W2', 'w2mpro', 'w2sigmpro')]:
        try:
            mjd = np.array(df['mjd'], dtype=float)
            mag = np.array(df[mag_col], dtype=float)
            err = np.array(df[err_col], dtype=float)
            mask = np.isfinite(mag) & np.isfinite(err) & (err > 0)
            if np.sum(mask) > 0:
                band_df = pd.DataFrame({'mjd': mjd[mask], 'mag': mag[mask],
                                        'magerr': err[mask]})
                band_df = band_df.sort_values('mjd').reset_index(drop=True)
                result[band] = band_df
        except (KeyError, ValueError):
            continue

    # 观测时间范围摘要
    all_mjds = []
    for band in ('W1', 'W2'):
        if band in result:
            all_mjds.extend(result[band]['mjd'].tolist())
    if all_mjds:
        result['obs_mjd_min'] = float(min(all_mjds))
        result['obs_mjd_max'] = float(max(all_mjds))
        result['n_epochs'] = len(all_mjds)

    return result if ('W1' in result or 'W2' in result) else None


def _query_lightcurve_astroquery(ra, dec, radius_arcsec=config.SEARCH_RADIUS_ARCSEC):
    """回退: 通过 astroquery 查询 NEOWISE (较慢)"""
    from astroquery.ipac.irsa import Irsa
    import astropy.units as u

    c = utils.coord(ra, dec)
    try:
        tbl = Irsa.query_region(c, catalog='neowiser_p1bs_psd',
                                radius=radius_arcsec * u.arcsec)
    except Exception as e:
        print(f"NEOWISE 查询失败: {e}")
        return None

    if tbl is None or len(tbl) == 0:
        return None

    result = {'ra': ra, 'dec': dec, 'survey': 'NEOWISE'}

    for band, mag_col, err_col in [('W1', 'w1mpro', 'w1sigmpro'),
                                    ('W2', 'w2mpro', 'w2sigmpro')]:
        try:
            mjd = np.array(tbl['mjd'], dtype=float)
            mag = np.array(tbl[mag_col], dtype=float)
            err = np.array(tbl[err_col], dtype=float)
            mask = np.isfinite(mag) & np.isfinite(err) & (err > 0)
            if np.sum(mask) > 0:
                df = pd.DataFrame({'mjd': mjd[mask], 'mag': mag[mask],
                                   'magerr': err[mask]})
                df = df.sort_values('mjd').reset_index(drop=True)
                result[band] = df
        except (KeyError, ValueError):
            continue

    return result if ('W1' in result or 'W2' in result) else None


def plot_lightcurve(result, save_path=None):
    """绘制 NEOWISE W1/W2 光变曲线"""
    if result is None:
        return None
    fig, ax = utils.setup_lightcurve_plot()
    colors = {'W1': 'blue', 'W2': 'red'}
    for band in ('W1', 'W2'):
        if band not in result:
            continue
        df = result[band]
        ax.errorbar(df['mjd'], df['mag'], yerr=df['magerr'],
                    fmt='.', color=colors[band], markersize=2, elinewidth=0.5,
                    alpha=0.6, label=f'WISE {band} ({len(df)} pts)')
    ax.set_title(f"NEOWISE Light Curve  RA={result['ra']:.4f} DEC={result['dec']:.4f}")
    ax.legend()
    fig.tight_layout()
    utils.save_and_close(fig, save_path)
    return fig


# ================================================================
#  AllWISE 质量诊断 (重点判别 W3/W4 真伪)
# ================================================================
#
# AllWISE qph (ph_qual) 4 字符, 每个波段一位:
#   A: SNR >= 10        B: 3 <= SNR < 10
#   C: 2 <= SNR < 3     U: SNR < 2 (仅 95% 上限, 非探测)
#   X / Z: 测量异常
#
# AllWISE ccf (contamination/confusion) 4 字符:
#   '0': 干净
#   小写 d/p/h/o : 衍射尖 / 持续效应 / 散射晕 / 光鬼 (轻度污染)
#   大写 D/P/H/O : 同类型严重污染 — 大概率虚假
#
# nb: PSF profile fit 中的源数 (>1 表示与邻源混叠)
# ex: 形态扩展度 (0=点源, >0=扩展, W3/W4 可能是 cirrus/星系)

_CCF_DESC = {
    'D': 'diffraction spike (severe)', 'd': 'diffraction spike',
    'P': 'persistence (severe)',       'p': 'persistence',
    'H': 'halo (severe)',              'h': 'halo',
    'O': 'optical ghost (severe)',     'o': 'optical ghost',
    '0': 'clean',
}


def _safe_str(val, n=4, fill='U'):
    """把 masked / NaN 安全转成 4 字符质量码"""
    try:
        s = str(val).strip()
    except Exception:
        s = ''
    if not s or s.lower() in ('nan', 'none', 'masked', '--'):
        s = fill * n
    return (s + fill * n)[:n]


def _safe_int(val, default=0):
    try:
        v = int(float(val))
        return v
    except (ValueError, TypeError):
        return default


def _safe_float(val):
    try:
        f = float(val)
        return f if np.isfinite(f) else None
    except (ValueError, TypeError):
        return None


def check_reliability(ra, dec, radius_arcsec=config.SEARCH_RADIUS_ARCSEC):
    """
    检查 AllWISE W1-W4 测光的可靠性 (重点针对 W3/W4 真伪)。

    判定规则 (按优先级):
        upper_limit: ph_qual='U' 或 SNR<2  -> 仅 95% 上限, 不是真实探测
        spurious   : ccf 大写 (严重 artefact) 或 nb>1 (混叠)
        marginal   : ph_qual='C', SNR 2-5, ccf 小写, 或 W3/W4 形态扩展
        good       : ph_qual A/B, ccf=0, nb=1, SNR>=5

    Returns:
        dict: {'W1': {...}, 'W2': {...}, 'W3': {...}, 'W4': {...}}
            每个值含: mag, err, snr, ph_qual, ccf, nb, na, ex, var,
                     level, reliable, reasons
            没有 AllWISE 匹配时返回 {}.
    """
    cols = ['W1mag', 'e_W1mag', 'W2mag', 'e_W2mag',
            'W3mag', 'e_W3mag', 'W4mag', 'e_W4mag',
            'qph', 'ccf', 'ex', 'var', 'nb', 'na']
    tbl = utils.query_vizier('II/328/allwise', ra, dec, radius_arcsec,
                             columns=cols)
    if tbl is None or len(tbl) == 0:
        return {}

    row = tbl[0]
    qph = _safe_str(row['qph'] if 'qph' in tbl.colnames else '', fill='U')
    ccf = _safe_str(row['ccf'] if 'ccf' in tbl.colnames else '', fill='0')
    var = _safe_str(row['var'] if 'var' in tbl.colnames else '', fill='0')

    nb = _safe_int(row['nb'] if 'nb' in tbl.colnames else 1, default=1)
    na = _safe_int(row['na'] if 'na' in tbl.colnames else 0, default=0)
    ex = _safe_int(row['ex'] if 'ex' in tbl.colnames else 0, default=0)

    out = {}
    for i, band in enumerate(('W1', 'W2', 'W3', 'W4')):
        mag = _safe_float(row[f'{band}mag'])
        err = _safe_float(row[f'e_{band}mag'])
        # SNR 由星等误差反推: sigma_mag ≈ 1.0857 / SNR
        snr = (1.0857 / err) if (err is not None and err > 0) else None
        q = qph[i]
        c = ccf[i]
        v = var[i]

        reasons = []
        if q == 'U' or (snr is not None and snr < 2):
            level = 'upper_limit'
            reasons.append(f'ph_qual={q} (SNR<2, 95% upper limit only)')
        elif c.isupper() and c != '0':
            level = 'spurious'
            reasons.append(f'ccf={c} ({_CCF_DESC.get(c, "severe artefact")})')
        elif nb > 1:
            level = 'spurious'
            reasons.append(f'nb={nb} (blended with {nb - 1} neighbour(s))')
        elif q in ('X', 'Z'):
            level = 'spurious'
            reasons.append(f'ph_qual={q} (measurement error)')
        elif q == 'C' or (snr is not None and snr < 5):
            level = 'marginal'
            note = f'ph_qual={q}'
            if snr is not None:
                note += f', SNR={snr:.1f}'
            reasons.append(note)
        elif c.islower() and c != '0':
            level = 'marginal'
            reasons.append(f'ccf={c} ({_CCF_DESC.get(c, "minor artefact")})')
        elif ex > 0 and band in ('W3', 'W4'):
            level = 'marginal'
            reasons.append(f'ex={ex} (extended — possible cirrus/galaxy)')
        else:
            level = 'good'

        out[band] = {
            'mag': mag, 'err': err, 'snr': snr,
            'ph_qual': q, 'ccf': c, 'nb': nb, 'na': na,
            'ex': ex, 'var': v,
            'level': level,
            'reliable': level == 'good',
            'reasons': reasons,
        }
    return out


def print_reliability_report(quality_info, source_name=None):
    """打印 W1-W4 质量诊断报告 (人读)"""
    if not quality_info:
        print("[WISE] 没有 AllWISE 匹配源")
        return

    tag = {'good': '[OK]  ', 'marginal': '[MARG]',
           'upper_limit': '[UL]  ', 'spurious': '[SPUR]'}

    header = f"AllWISE quality report"
    if source_name:
        header += f" — {source_name}"
    print(header)
    print(f"{'Band':<5} {'mag':>7} {'SNR':>6} {'qph':>4} {'ccf':>4} "
          f"{'nb':>3} {'ex':>3}  status")
    print('-' * 78)
    for band in ('W1', 'W2', 'W3', 'W4'):
        if band not in quality_info:
            continue
        q = quality_info[band]
        mag = f"{q['mag']:.2f}" if q['mag'] is not None else '  N/A'
        snr = f"{q['snr']:.1f}" if q['snr'] is not None else '  N/A'
        status = tag.get(q['level'], '[?]   ') + ' ' + q['level']
        if q['reasons']:
            status += ' — ' + '; '.join(q['reasons'])
        print(f"{band:<5} {mag:>7} {snr:>6} {q['ph_qual']:>4} {q['ccf']:>4} "
              f"{q['nb']:>3} {q['ex']:>3}  {status}")


def save_photometry_csv(result, output_dir):
    """保存 AllWISE 测光数据为 CSV"""
    df = utils.photometry_to_dataframe(result)
    return utils.write_csv(df, output_dir, 'wise_photometry.csv')


def save_lightcurve_csv(result, output_dir):
    """保存 NEOWISE 光变曲线为 CSV"""
    df = utils.lightcurve_to_dataframe(result, ['W1', 'W2'])
    return utils.write_csv(df, output_dir, 'wise_lightcurve.csv')
