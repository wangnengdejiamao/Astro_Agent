"""公共工具: 下载、坐标、绘图、查询封装、代理管理"""
import os
import time
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
import requests
from urllib.parse import urlparse
from . import config

warnings.filterwarnings('ignore')


# ================================================================
#  代理管理: 根据域名决定是否使用代理
# ================================================================

_proxy_alive = None  # 缓存代理可用性, 整个进程生命周期只检测一次


def _check_proxy_alive():
    """快速 TCP 检测代理端口是否在监听, 结果缓存整个进程周期"""
    global _proxy_alive
    if _proxy_alive is not None:
        return _proxy_alive
    import socket
    proxy_url = config.PROXY_URL
    if not proxy_url:
        _proxy_alive = False
        return False
    try:
        parsed = urlparse(proxy_url)
        host = parsed.hostname or '127.0.0.1'
        port = parsed.port or 7890
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        s.connect((host, port))
        s.close()
        _proxy_alive = True
    except Exception:
        _proxy_alive = False
        print(f"  [proxy] {proxy_url} 不可用, 本次全部直连")
    return _proxy_alive


def _should_use_proxy(url):
    """判断该 URL 是否需要走代理"""
    if not config.PROXY_URL:
        return False
    if not _check_proxy_alive():
        return False
    hostname = urlparse(url).hostname or ''
    # 明确不走代理的域名
    for domain in config.NO_PROXY_DOMAINS:
        if domain in hostname:
            return False
    # 需要走代理的域名
    for domain in config.PROXY_DOMAINS:
        if domain in hostname:
            return True
    # 默认: 非中国域名走代理
    return True


def get_session(url=None):
    """
    获取 requests.Session, 根据 URL 自动配置代理。

    用法:
        session = get_session('https://data.desi.lbl.gov/...')
        resp = session.get(url, timeout=config.TIMEOUT)
    """
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=config.MAX_WORKERS,
        pool_maxsize=config.MAX_WORKERS + 2,
        max_retries=requests.adapters.Retry(
            total=2, backoff_factor=1,
            status_forcelist=[502, 503, 504],
        ),
    )
    session.mount('https://', adapter)
    session.mount('http://', adapter)

    if url and _should_use_proxy(url):
        proxy = config.PROXY_URL
        session.proxies = {
            'http': proxy,
            'https': proxy,
        }
        # 代理可能需要跳过 SSL 验证
        session.verify = True

    return session


def get_session_no_proxy():
    """获取不使用任何代理的 Session (用于代理回退)"""
    session = requests.Session()
    session.trust_env = False
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=config.MAX_WORKERS,
        pool_maxsize=config.MAX_WORKERS + 2,
        max_retries=requests.adapters.Retry(
            total=2, backoff_factor=1,
            status_forcelist=[502, 503, 504],
        ),
    )
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    return session


def get_timeout():
    """返回 (connect_timeout, read_timeout) 元组"""
    return (config.CONNECT_TIMEOUT, config.TIMEOUT)


def ensure_dir(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def coord(ra, dec):
    """统一创建 SkyCoord 对象"""
    return SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')


def download_file(url, path, timeout=None, retries=config.MAX_RETRIES,
                  chunk_size=config.CHUNK_SIZE):
    """带重试、代理和原子写入的文件下载"""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    ensure_dir(os.path.dirname(path))
    partial = path + '.partial'
    timeout = timeout or get_timeout()
    session = get_session(url)
    for attempt in range(retries):
        try:
            resp = session.get(url, stream=True, timeout=timeout)
            resp.raise_for_status()
            with open(partial, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
            os.rename(partial, path)
            return path
        except Exception as e:
            if os.path.exists(partial):
                os.remove(partial)
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"Download failed after {retries} retries: {e}")


def query_vizier(catalog_id, ra, dec, radius_arcsec=config.SEARCH_RADIUS_ARCSEC,
                 columns=None, retries=3):
    """Vizier 查询封装, 返回 astropy Table 或 None, 带重试"""
    from astroquery.vizier import Vizier
    v = Vizier(columns=columns or ['**'], row_limit=50,
               timeout=config.TIMEOUT)
    c = coord(ra, dec)
    for attempt in range(retries):
        try:
            result = v.query_region(c, radius=radius_arcsec * u.arcsec,
                                    catalog=catalog_id)
            if result and len(result) > 0 and len(result[0]) > 0:
                return result[0]
            return None
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
    return None


def query_simbad(ra, dec, radius_arcsec=config.SEARCH_RADIUS_ARCSEC):
    """SIMBAD 查询, 返回对象类型和名称"""
    from astroquery.simbad import Simbad
    c = coord(ra, dec)
    result = Simbad.query_region(c, radius=radius_arcsec * u.arcsec)
    if result and len(result) > 0:
        return result
    return None


def query_simbad_references(ra, dec, radius_arcsec=config.SEARCH_RADIUS_ARCSEC,
                            max_refs=20):
    """
    查询 SIMBAD 中天体的文献引用, 并尝试通过 ADS 获取摘要。

    Args:
        ra, dec: 坐标 (度)
        radius_arcsec: 搜索半径
        max_refs: 最多返回的文献数

    Returns:
        dict: {
            'main_id': SIMBAD 主标识符,
            'otype': 天体类型,
            'n_refs': 文献总数,
            'references': [
                {
                    'bibcode': bibcode,
                    'title': 标题,
                    'authors': 作者列表字符串,
                    'journal': 期刊,
                    'year': 年份,
                    'abstract': 摘要 (如果获取到),
                    'url': ADS 链接,
                },
                ...
            ],
        }
        或 None (未找到天体)
    """
    from astroquery.simbad import Simbad

    # 第一步: 用 query_region 获取主标识符和天体类型
    c = coord(ra, dec)
    s = Simbad()
    s.add_votable_fields('otype')
    result = s.query_region(c, radius=radius_arcsec * u.arcsec)
    if result is None or len(result) == 0:
        return None

    # 列名兼容: 新版 astroquery 返回全大写
    main_id_col = 'MAIN_ID' if 'MAIN_ID' in result.colnames else 'main_id'
    main_id = str(result[main_id_col][0]).strip()
    otype = ''
    for col in ('OTYPE', 'otype'):
        if col in result.colnames:
            otype = str(result[col][0]).strip()
            break

    # 第二步: 用 CONTAINS/CIRCLE 空间查询直接获取 bibcodes
    # (通过 main_id JOIN 的方式在新版 SIMBAD TAP 中返回空表)
    radius_deg = radius_arcsec / 3600.0
    bibcodes = []
    try:
        bib_result = Simbad.query_tap(
            f"""SELECT ref.bibcode
                FROM ref
                JOIN has_ref ON ref.oidbib = has_ref.oidbibref
                JOIN basic ON has_ref.oidref = basic.oid
                WHERE CONTAINS(
                    POINT('ICRS', basic.ra, basic.dec),
                    CIRCLE('ICRS', {ra}, {dec}, {radius_deg})
                ) = 1"""
        )
        if bib_result is not None and len(bib_result) > 0:
            bib_col = 'bibcode' if 'bibcode' in bib_result.colnames else 'BIBCODE'
            bibcodes = [str(row[bib_col]).strip()
                        for row in bib_result if str(row[bib_col]).strip()]
    except Exception:
        pass

    # 回退: 旧式 query_bibobj
    if not bibcodes:
        try:
            bib_result = Simbad.query_bibobj(main_id)
            if bib_result and len(bib_result) > 0:
                bib_col = 'bibcode' if 'bibcode' in bib_result.colnames else 'BIBCODE'
                bibcodes = [str(row[bib_col]).strip() for row in bib_result]
        except Exception:
            pass

    if not bibcodes:
        return {
            'main_id': main_id,
            'otype': otype,
            'n_refs': 0,
            'references': [],
        }

    # 提取 bibcode 列表
    n_total = len(bibcodes)

    # 按年份倒序 (bibcode 前4位是年份)
    bibcodes.sort(key=lambda b: b[:4], reverse=True)
    bibcodes = bibcodes[:max_refs]

    # 第三步: 通过 SIMBAD TAP 或 ADS 获取文献详情
    references = []
    # 尝试用 astroquery 的 ADS 获取详情和摘要
    try:
        _refs = _fetch_bibdetails_ads(bibcodes)
        if _refs:
            references = _refs
    except Exception:
        pass

    # 如果 ADS 失败, 用 SIMBAD query_bibcode 获取基本信息 (无摘要)
    if not references:
        try:
            _refs = _fetch_bibdetails_simbad(bibcodes)
            if _refs:
                references = _refs
        except Exception:
            pass

    # 最后兜底: 仅返回 bibcode 和 ADS 链接
    if not references:
        for bib in bibcodes:
            references.append({
                'bibcode': bib,
                'title': '',
                'authors': '',
                'journal': '',
                'year': bib[:4],
                'abstract': '',
                'url': f'https://ui.adsabs.harvard.edu/abs/{bib}',
            })

    return {
        'main_id': main_id,
        'otype': otype,
        'n_refs': n_total,
        'references': references,
    }


def _fetch_bibdetails_ads(bibcodes):
    """通过 NASA ADS API 获取文献详情 (含摘要)"""
    ads_token = config.ADS_TOKEN

    if not ads_token:
        # 尝试从 ~/.ads/dev_key 读取
        token_path = os.path.expanduser('~/.ads/dev_key')
        if os.path.exists(token_path):
            with open(token_path) as f:
                ads_token = f.read().strip()

    if not ads_token:
        return None

    # 使用 ADS API 批量查询
    session = get_session('https://api.adsabs.harvard.edu')
    headers = {
        'Authorization': f'Bearer {ads_token}',
        'Content-Type': 'application/json',
    }

    references = []
    # ADS API 支持批量查询 bibcode
    bib_query = ' OR '.join(f'bibcode:"{b}"' for b in bibcodes)
    url = 'https://api.adsabs.harvard.edu/v1/search/query'
    params = {
        'q': bib_query,
        'fl': 'bibcode,title,author,pub,year,abstract',
        'rows': len(bibcodes),
        'sort': 'date desc',
    }

    try:
        resp = session.get(url, headers=headers, params=params,
                           timeout=get_timeout())
        resp.raise_for_status()
        data = resp.json()
        docs = data.get('response', {}).get('docs', [])

        for doc in docs:
            bib = doc.get('bibcode', '')
            authors = doc.get('author', [])
            author_str = '; '.join(authors[:5])
            if len(authors) > 5:
                author_str += f' et al. (+{len(authors)-5})'

            references.append({
                'bibcode': bib,
                'title': doc.get('title', [''])[0] if doc.get('title') else '',
                'authors': author_str,
                'journal': doc.get('pub', ''),
                'year': doc.get('year', bib[:4]),
                'abstract': doc.get('abstract', ''),
                'url': f'https://ui.adsabs.harvard.edu/abs/{bib}',
            })
    except Exception:
        return None

    return references if references else None


def _fetch_bibdetails_simbad(bibcodes):
    """通过 SIMBAD query_bibcode 获取基本文献信息 (不含摘要)"""
    from astroquery.simbad import Simbad

    references = []
    # 批量查询
    try:
        bib_info = Simbad.query_bibcode_list(bibcodes)
    except AttributeError:
        # 旧版 astroquery 没有 query_bibcode_list, 逐个查询
        bib_info = None
        for bib in bibcodes:
            try:
                r = Simbad.query_bibcode(bib, wildcard=False)
                if r and len(r) > 0:
                    row = r[0]
                    references.append({
                        'bibcode': bib,
                        'title': str(_row_get(row, 'title', '')).strip() if 'title' in r.colnames else '',
                        'authors': '',
                        'journal': str(_row_get(row, 'journal', '')).strip() if 'journal' in r.colnames else '',
                        'year': bib[:4],
                        'abstract': '',
                        'url': f'https://ui.adsabs.harvard.edu/abs/{bib}',
                    })
            except Exception:
                references.append({
                    'bibcode': bib,
                    'title': '',
                    'authors': '',
                    'journal': '',
                    'year': bib[:4],
                    'abstract': '',
                    'url': f'https://ui.adsabs.harvard.edu/abs/{bib}',
                })
        return references if references else None

    if bib_info and len(bib_info) > 0:
        for row in bib_info:
            bib = str(row['bibcode']).strip() if 'bibcode' in bib_info.colnames else ''
            references.append({
                'bibcode': bib,
                'title': str(_row_get(row, 'title', '')).strip() if 'title' in bib_info.colnames else '',
                'authors': '',
                'journal': str(_row_get(row, 'journal', '')).strip() if 'journal' in bib_info.colnames else '',
                'year': bib[:4] if bib else '',
                'abstract': '',
                'url': f'https://ui.adsabs.harvard.edu/abs/{bib}',
            })

    return references if references else None


def mag_to_flux_jy(mag, mag_err=None, zero_jy=3631.0):
    """AB 星等转 Jy 流量"""
    flux = zero_jy * 10 ** (-0.4 * mag)
    if mag_err is not None:
        flux_err = flux * 0.4 * np.log(10) * mag_err
        return flux, flux_err
    return flux


def mag_to_flux_cgs(mag, wave_A, mag_err=None, zero_jy=3631.0):
    """AB 星等转 erg/s/cm^2/Hz, 再转 erg/s/cm^2/A"""
    flux_jy = zero_jy * 10 ** (-0.4 * mag)
    # Jy → erg/s/cm^2/Hz: 1 Jy = 1e-23 erg/s/cm^2/Hz
    flux_hz = flux_jy * 1e-23
    # f_nu → f_lambda: f_lambda = f_nu * c / lambda^2
    c_A = 2.99792458e18  # c in Angstrom/s
    flux_lambda = flux_hz * c_A / wave_A ** 2
    if mag_err is not None:
        flux_lambda_err = flux_lambda * 0.4 * np.log(10) * mag_err
        return flux_lambda, flux_lambda_err
    return flux_lambda


def setup_spectrum_plot():
    """返回标准光谱图的 fig, ax"""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlabel('Wavelength (A)')
    ax.set_ylabel('Flux')
    ax.grid(True, alpha=0.3)
    return fig, ax


def set_spectrum_axes(ax, wave, flux, model=None, margin_x=0.02, margin_y=0.10):
    """
    根据光谱数据设置 tight 轴范围。

    Args:
        ax: matplotlib Axes
        wave: 波长数组 (Angstrom)
        flux: 流量数组
        model: 可选的模型流量数组 (拟合曲线), 纳入 y 轴范围
        margin_x: x 轴两侧留白比例 (default 2%)
        margin_y: y 轴两侧留白比例 (default 10%)
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)

    valid = np.isfinite(wave) & np.isfinite(flux) & (wave > 0)
    if valid.sum() < 2:
        return

    w, f = wave[valid], flux[valid]

    # X 轴: 紧凑围绕波长范围
    wmin, wmax = w.min(), w.max()
    dw = max((wmax - wmin) * margin_x, 10)  # 至少留 10 A
    ax.set_xlim(wmin - dw, wmax + dw)

    # Y 轴: 用 1-99 百分位排除极端异常值
    all_f = [f]
    if model is not None:
        m = np.asarray(model, dtype=float)
        m_valid = m[np.isfinite(m)]
        if len(m_valid) > 0:
            all_f.append(m_valid)
    combined = np.concatenate(all_f)
    combined = combined[np.isfinite(combined)]
    if len(combined) < 2:
        return

    flo, fhi = np.percentile(combined, [1, 99])
    df = max((fhi - flo) * margin_y, abs(fhi) * 0.01)
    ax.set_ylim(flo - df, fhi + df)


def setup_lightcurve_plot():
    """返回标准光变曲线图的 fig, ax"""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlabel('Time (MJD)')
    ax.set_ylabel('Magnitude')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    return fig, ax


def save_and_close(fig, path):
    """保存图片并关闭"""
    if path:
        ensure_dir(os.path.dirname(path))
        fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ================================================================
#  MHAOV 周期分析 (Multi-Harmonic Analysis of Variance)
# ================================================================

def mhaov(time, mag, magerr=None, freq_min=None, freq_max=None,
          n_freq=100000, n_harmonics=1):
    """
    Multi-Harmonic Analysis of Variance (MHAOV) 周期搜索算法。

    基于 Schwarzenberg-Czerny (1996):
    对每个试探频率，用截断 Fourier 级数 (n_harmonics 阶) 加权最小二乘拟合，
    计算 F 统计量。峰值对应最佳周期。

    Args:
        time:     观测时间数组 (MJD)
        mag:      星等数组
        magerr:   星等误差 (None 则等权)
        freq_min: 最小频率 (1/day), 默认 2/(time span)
        freq_max: 最大频率 (1/day), 默认 Nyquist 估计
        n_freq:   频率网格数 (越大越精细, 默认 100000)
        n_harmonics: Fourier 谐波阶数 (默认3)

    Returns:
        dict: {
            'best_period': 最佳周期 (day),
            'best_freq':   最佳频率 (1/day),
            'periods':     周期数组,
            'freqs':       频率数组,
            'power':       MHAOV 统计量 (theta),
            'fap':         近似虚警概率,
        }
    """
    time = np.asarray(time, dtype=np.float64)
    mag = np.asarray(mag, dtype=np.float64)

    mask = np.isfinite(time) & np.isfinite(mag)
    if magerr is not None:
        magerr = np.asarray(magerr, dtype=np.float64)
        mask &= np.isfinite(magerr) & (magerr > 0)
        magerr = magerr[mask]
        w = 1.0 / magerr ** 2
    else:
        w = None
    time = time[mask]
    mag = mag[mask]
    N = len(time)
    p = 2 * n_harmonics  # 模型参数数

    if N < p + 2:
        return None

    # 对大数据集自适应降低频率网格, 保持计算时间可控。
    # 这里是 Python 循环 + QR 分解，过密网格会很慢；精细搜索由 LS 辅助。
    max_ops = 5e7
    max_n_freq = int(max_ops / (N * n_harmonics))
    if n_freq > max_n_freq:
        n_freq = max(max_n_freq, 3000)  # 至少 3000 个频率

    # 加权平均 & 中心化
    if w is not None:
        w_sum = np.sum(w)
        mean_mag = np.sum(w * mag) / w_sum
        y = mag - mean_mag
        sw = np.sqrt(w)       # sqrt weights for factoring into design matrix
    else:
        mean_mag = np.mean(mag)
        y = mag - mean_mag
        sw = np.ones(N)
        w = np.ones(N)
        w_sum = float(N)

    # 加权总方差 SS_total = sum(w * y^2)
    SS_total = np.sum(w * y ** 2)

    # 频率范围
    T = time.max() - time.min()
    if freq_min is None:
        freq_min = 2.0 / T
    if freq_max is None:
        # 对地面巡天的不均匀采样，用第 5 百分位数的 dt 来估计 Nyquist
        dt_sorted = np.diff(np.sort(time))
        dt_sorted = dt_sorted[dt_sorted > 0]
        dt_ref = np.percentile(dt_sorted, 5)  # 取短间隔
        freq_max = 0.5 / dt_ref
        # 限制最大频率不超过 720/day (周期 > 2 分钟, 覆盖所有已知 DWD)
        freq_max = min(freq_max, 720.0)

    freqs = np.linspace(freq_min, freq_max, n_freq)

    # 加权 y 向量
    wy = sw * y  # shape (N,)
    SS_total = np.sum(wy ** 2)

    # MHAOV 统计量 — 向量化
    theta = np.zeros(n_freq)
    t0 = time - time[0]

    # 预计算 2*pi*t0 用于频率缩放
    twopi_t0 = 2.0 * np.pi * t0  # shape (N,)

    for i, f in enumerate(freqs):
        phase = f * twopi_t0  # shape (N,)

        # 构建加权设计矩阵 (已乘 sqrt(w))
        X = np.empty((N, p))
        for h in range(1, n_harmonics + 1):
            hp = h * phase
            X[:, 2*(h-1)]   = sw * np.cos(hp)
            X[:, 2*(h-1)+1] = sw * np.sin(hp)

        # QR 分解: SS_model = |Q^T wy|^2
        Q, _ = np.linalg.qr(X, mode='reduced')
        proj = Q.T @ wy
        SS_model = np.dot(proj, proj)
        SS_res = SS_total - SS_model

        if SS_res > 0 and N > p + 1:
            theta[i] = (SS_model / p) / (SS_res / (N - p - 1))

    # 最佳频率
    best_idx = np.argmax(theta)
    best_freq = freqs[best_idx]

    # 子谐波消歧: 食双星常被检测为 P, 真实轨道周期其实是 2P
    # 同时也检查 P/2, P/3 处是否有更高的峰
    # 保存原始最佳频率，避免迭代中被篡改
    orig_best_freq = best_freq
    orig_best_idx = best_idx
    orig_best_theta = theta[best_idx]

    candidates = [(best_freq, best_idx, theta[best_idx])]

    for div in (2, 3):
        # 检查频率的整数倍 (对应周期的 1/div): P_test = P/div
        test_freq = orig_best_freq * div
        if test_freq <= freqs[-1]:
            idx = np.argmin(np.abs(freqs - test_freq))
            candidates.append((freqs[idx], idx, theta[idx]))

        # 检查频率的整数分之一 (对应周期的 div 倍): P_test = P*div
        test_freq2 = orig_best_freq / div
        if test_freq2 >= freqs[0]:
            idx2 = np.argmin(np.abs(freqs - test_freq2))
            candidates.append((freqs[idx2], idx2, theta[idx2]))

    # 选择 theta 最高的候选
    best_freq, best_idx, _ = max(candidates, key=lambda x: x[2])

    best_period = 1.0 / best_freq
    periods = 1.0 / freqs

    # FAP (F 分布)
    from scipy.stats import f as f_dist
    peak_theta = theta[best_idx]
    N_eff = n_freq  # 保守估计
    single_p = f_dist.sf(peak_theta, p, N - p - 1)
    fap = 1.0 - (1.0 - single_p) ** N_eff
    fap = max(fap, 0.0)

    return {
        'best_period': best_period,
        'best_freq': best_freq,
        'periods': periods,
        'freqs': freqs,
        'power': theta,
        'fap': fap,
        'n_points': N,
        'n_harmonics': n_harmonics,
        'method': 'MHAOV',
    }


def lomb_scargle(time, mag, magerr=None, freq_min=None, freq_max=None,
                 n_freq=100000):
    """
    Lomb-Scargle 周期搜索。与 mhaov() 返回同构字段，便于统一比较。

    LS 对近似正弦信号很稳，但对食双星/尖窄脉冲常容易选到 P/2 或别名；
    因此周期管线会把它和 MHAOV 一起使用，而不是单独采用 LS 峰值。
    """
    time = np.asarray(time, dtype=np.float64)
    mag = np.asarray(mag, dtype=np.float64)
    mask = np.isfinite(time) & np.isfinite(mag)
    if magerr is not None:
        magerr = np.asarray(magerr, dtype=np.float64)
        mask &= np.isfinite(magerr) & (magerr > 0)
        magerr = magerr[mask]
    time = time[mask]
    mag = mag[mask]
    N = len(time)
    if N < 5:
        return None

    T = time.max() - time.min()
    if not np.isfinite(T) or T <= 0:
        return None
    if freq_min is None:
        freq_min = 2.0 / T
    if freq_max is None:
        dt_sorted = np.diff(np.sort(time))
        dt_sorted = dt_sorted[dt_sorted > 0]
        if len(dt_sorted) == 0:
            return None
        dt_ref = np.percentile(dt_sorted, 5)
        freq_max = min(0.5 / dt_ref, 720.0)
    if not np.isfinite(freq_min + freq_max) or freq_max <= freq_min:
        return None

    max_ops = 4e8
    max_n_freq = int(max_ops / max(N, 1))
    if n_freq > max_n_freq:
        n_freq = max(max_n_freq, 10000)
    freqs = np.linspace(freq_min, freq_max, int(n_freq))

    try:
        from astropy.timeseries import LombScargle
        ls = LombScargle(time - time[0], mag, dy=magerr)
        power = ls.power(freqs, method='fast', assume_regular_frequency=True)
        best_idx = int(np.nanargmax(power))
        best_freq = float(freqs[best_idx])
        try:
            fap = float(ls.false_alarm_probability(power[best_idx]))
        except Exception:
            fap = np.nan
    except Exception:
        return None

    periods = 1.0 / freqs
    return {
        'best_period': 1.0 / best_freq,
        'best_freq': best_freq,
        'periods': periods,
        'freqs': freqs,
        'power': power,
        'fap': fap,
        'n_points': N,
        'n_harmonics': 1,
        'method': 'LS',
    }


def plot_period_analysis(period_result, time, mag, magerr=None, save_path=None,
                         title=''):
    """
    绘制周期分析结果: 周期图 + 折叠光变曲线

    Args:
        period_result: mhaov() 返回的 dict
        time, mag, magerr: 原始数据
        save_path: 保存路径
        title: 图标题前缀
    """
    if period_result is None:
        return None

    P = period_result['best_period']
    fap = period_result['fap']
    P_h = P * 24.0
    P_min = P_h * 60.0

    fig, axes = plt.subplots(2, 1, figsize=(12, 8),
                              gridspec_kw={'height_ratios': [1, 1]})

    # 上图: 周期图
    ax1 = axes[0]
    p_label = f'Best P = {P_h:.4f} h ({P:.6f} d; {P_min:.2f} min)'
    method = period_result.get('method', 'MHAOV')
    periods = period_result.get('periods')
    power = period_result.get('power')
    if periods is not None and power is not None:
        periods = np.asarray(periods, dtype=float)
        periods_h = periods * 24.0
        power = np.asarray(power, dtype=float)
        # 按周期排序绘图
        sort_idx = np.argsort(periods_h)
        ax1.plot(periods_h[sort_idx], power[sort_idx], 'k-', lw=0.6)
        ax1.axvline(P_h, color='red', ls='--', lw=1.5, label=p_label)
        # 标注谐波
        for h in (2, 3):
            ax1.axvline(P_h * h, color='orange', ls=':', lw=0.8, alpha=0.6,
                         label=f'{h}P = {P_h*h:.4f} h' if h == 2 else None)
            ax1.axvline(P_h / h, color='blue', ls=':', lw=0.8, alpha=0.6,
                         label=f'P/{h} = {P_h/h:.4f} h' if h == 2 else None)
        ax1.set_xlabel('Period (hour)')
        ax1.set_ylabel(f'{method} Power')
        ax1.set_xscale('log')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
    else:
        ax1.axis('off')
        note = period_result.get(
            'fap_note',
            'Native periodogram arrays are not stored for this method.')
        two_p = period_result.get('two_period_day')
        two_p_note = ''
        if two_p is not None:
            two_p_note = f'\n2P candidate = {two_p * 24.0:.4f} h ({two_p:.6f} d)'
        ax1.text(
            0.02, 0.72,
            f'{method} period result\n{p_label}{two_p_note}\n{note}',
            transform=ax1.transAxes, ha='left', va='top', fontsize=11)
    ax1.set_title(f'{title}  {method} Periodogram  (N={period_result.get("n_points", len(time))}, '
                  f'FAP={fap:.2e})')

    # 下图: 折叠光变曲线
    ax2 = axes[1]
    time = np.asarray(time, dtype=float)
    mag = np.asarray(mag, dtype=float)
    phase = ((time - time.min()) / P) % 1.0

    if magerr is not None:
        magerr = np.asarray(magerr, dtype=float)
        ax2.errorbar(phase, mag, yerr=magerr, fmt='.', color='black',
                     markersize=5, elinewidth=0.45, alpha=0.58)
        ax2.errorbar(phase + 1, mag, yerr=magerr, fmt='.', color='gray',
                     markersize=5, elinewidth=0.45, alpha=0.32)
    else:
        ax2.scatter(phase, mag, s=16, c='black', alpha=0.58)
        ax2.scatter(phase + 1, mag, s=16, c='gray', alpha=0.32)

    ax2.set_xlabel('Phase')
    ax2.set_ylabel('Magnitude')
    ax2.invert_yaxis()
    ax2.set_xlim(0, 2)
    ax2.set_title(f'Phase-folded Light Curve  P = {P_h:.4f} h ({P:.6f} d)')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    save_and_close(fig, save_path)
    return fig


# ================================================================
#  CSV 保存工具
# ================================================================

def write_csv(df, output_dir, filename):
    """将 DataFrame 写入 CSV 文件。"""
    import pandas as pd
    if df is None or (hasattr(df, 'empty') and df.empty) or len(df) == 0:
        return None
    ensure_dir(output_dir)
    path = os.path.join(output_dir, filename)
    df.to_csv(path, index=False)
    return path


def photometry_to_dataframe(phot_dict):
    """测光 dict {band: (mag, err, wave)} → DataFrame。"""
    import pandas as pd
    if not phot_dict:
        return pd.DataFrame()
    rows = []
    for band_name, (mag, err, wave) in phot_dict.items():
        rows.append({'band': band_name, 'mag': mag, 'mag_err': err, 'wave_A': wave})
    return pd.DataFrame(rows)


def spectrum_to_dataframe(result, wave_key='wavelength', flux_key='flux',
                          error_key='error', extra_keys=None):
    """光谱 dict → DataFrame。"""
    import pandas as pd
    if result is None or wave_key not in result:
        return pd.DataFrame()
    data = {'wavelength_A': result[wave_key], 'flux': result[flux_key]}
    if error_key and error_key in result and result[error_key] is not None:
        data['error'] = result[error_key]
    if extra_keys:
        for key in extra_keys:
            if key in result and result[key] is not None:
                data[key] = result[key]
    return pd.DataFrame(data)


def lightcurve_to_dataframe(result, band_keys, band_col_name='band'):
    """多波段光变曲线 dict → 合并 DataFrame (增加 band 列)。"""
    import pandas as pd
    if result is None:
        return pd.DataFrame()
    all_dfs = []
    for band in band_keys:
        if band in result and hasattr(result[band], 'columns'):
            df = result[band].copy()
            df[band_col_name] = band
            all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


def keyvalue_to_dataframe(result, keys=None):
    """单行键值 dict → 单行 DataFrame。"""
    import pandas as pd
    if result is None:
        return pd.DataFrame()
    row = {}
    for k, v in result.items():
        if keys and k not in keys:
            continue
        if isinstance(v, (int, float, str, bool, np.integer, np.floating)):
            row[k] = v
    return pd.DataFrame([row]) if row else pd.DataFrame()


# ================================================================
#  数据来源 / Provenance (HST / JWST 等空间任务)
# ================================================================
#
# Provenance 信息的三个来源 (按优先级):
#   1. MAST 观测表 (Observations.query_criteria 返回的行):
#        proposal_id, proposal_pi, proposal_type, obs_title,
#        target_name, t_min, t_exptime, instrument_name, dataURL
#   2. FITS Primary Header (下载文件后 hdul[0].header):
#        HST: PROPOSID, PR_INV_L/F/M, OBSERVER, LINENUM, ASN_ID,
#             DATE-OBS, TIME-OBS, EXPTIME, CAL_VER
#        JWST: PROGRAM, TITLE, PI_NAME, CATEGORY, OBSERVTN, VISIT,
#              DATE-OBS, EXPSTART, EFFEXPTM, CAL_VER
#   3. 派生信息: MJD → UTC ISO 字符串, MAST DOI 模板, 致谢文本

_HST_ACK = (
    "Based on observations made with the NASA/ESA Hubble Space Telescope, "
    "obtained from the Mikulski Archive for Space Telescopes (MAST) "
    "at the Space Telescope Science Institute (STScI). "
    "STScI is operated by AURA under NASA contract NAS 5-26555."
)
_JWST_ACK = (
    "Based on observations made with the NASA/ESA/CSA James Webb Space Telescope, "
    "obtained from the Mikulski Archive for Space Telescopes (MAST) "
    "at the Space Telescope Science Institute (STScI)."
)
_SDSS_ACK = (
    "Funding for the Sloan Digital Sky Survey has been provided by the Alfred P. "
    "Sloan Foundation, the U.S. Department of Energy Office of Science, and the "
    "Participating Institutions. SDSS acknowledges support and resources from the "
    "Center for High Performance Computing at the University of Utah. "
    "The SDSS website is www.sdss.org."
)
_KOA_ACK = (
    "Some of the data presented herein were obtained at Keck Observatory, which "
    "is a private 501(c)3 non-profit organization operated as a scientific "
    "partnership among the California Institute of Technology, the University of "
    "California, and the National Aeronautics and Space Administration. "
    "Data were retrieved from the Keck Observatory Archive (KOA), operated by "
    "W. M. Keck Observatory and the NASA Exoplanet Science Institute (NExScI)."
)


def _normalize_proposal_id(pid_raw):
    """整数化 proposal_id; 失败保留字符串; 都失败返回 0。"""
    if pid_raw is None:
        return 0
    s = str(pid_raw).strip()
    if not s or s.lower() in ('nan', 'none', '--', 'masked'):
        return 0
    try:
        return int(s)
    except ValueError:
        return s  # KOA PROGID 这种 'U092LR' 字符串


def mjd_to_isot(mjd):
    """MJD → UTC ISO 字符串 'YYYY-MM-DDTHH:MM:SS.sss'。失败返回 ''。"""
    try:
        from astropy.time import Time
        if mjd is None or not np.isfinite(float(mjd)) or float(mjd) <= 0:
            return ''
        return Time(float(mjd), format='mjd', scale='utc').isot
    except Exception:
        return ''


def _hkey(header, *keys, default=None):
    """从 FITS header 中取第一个非空键值。"""
    if header is None:
        return default
    for k in keys:
        if k in header:
            v = header[k]
            if v is None:
                continue
            s = str(v).strip()
            if s and s.lower() not in ('none', 'nan', 'unknown', 'n/a'):
                return v
    return default


def _row_get(obs_row, key, default=None):
    """从 MAST 观测行 (astropy Row 或 dict) 中安全取值。"""
    if obs_row is None:
        return default
    try:
        v = obs_row[key]
    except (KeyError, IndexError, ValueError, TypeError):
        return default
    # 处理 masked
    try:
        import numpy.ma as ma
        if ma.is_masked(v):
            return default
    except Exception:
        pass
    if v is None:
        return default
    s = str(v).strip()
    if s.lower() in ('', 'none', 'nan', '--', 'masked'):
        return default
    return v


def _resolve_pi_name(obs_row, header):
    """优先级: MAST proposal_pi > header PR_INV_L (+F) > OBSERVER > PI_NAME。"""
    pi = _row_get(obs_row, 'proposal_pi')
    if pi:
        return str(pi).strip()
    last = _hkey(header, 'PR_INV_L')
    first = _hkey(header, 'PR_INV_F')
    if last:
        return f"{last}, {first}".strip(', ') if first else str(last).strip()
    name = _hkey(header, 'PI_NAME', 'OBSERVER')
    return str(name).strip() if name else ''


def build_provenance(survey, obs_row=None, header=None,
                     ra=None, dec=None, override=None):
    """
    构造统一的来源 (provenance) 信息 dict。

    Args:
        survey: 'HST' / 'JWST' / 其它任务名
        obs_row: MAST Observations 表的行 (可选)
        header: FITS Primary Header (可选)
        ra, dec: 查询坐标
        override: 调用方已知的字段 (优先于 obs_row/header)

    Returns:
        dict: 包含 mission, instrument, detector, grating, filter,
              proposal_id, proposal_pi, proposal_type, title,
              obs_id, target_name, exptime_s,
              obs_mjd, obs_date_utc, ra, dec,
              archive, archive_url, citation
    """
    survey = (survey or '').upper()
    o = override or {}

    # --- 仪器 / 探测器 / 光栅 / 滤光片
    instrument = (o.get('instrument')
                  or _row_get(obs_row, 'instrument_name')
                  or _hkey(header, 'INSTRUME')
                  or '')
    detector = (o.get('detector')
                or _hkey(header, 'DETECTOR')
                or '')
    grating = (o.get('grating')
               or _hkey(header, 'GRATING', 'OPT_ELEM')
               or '')
    filt = (o.get('filter')
            or _hkey(header, 'FILTER', 'FILTER1')
            or _row_get(obs_row, 'filters')
            or '')

    # --- proposal 信息
    pid_raw = (o.get('proposal_id')
               or _row_get(obs_row, 'proposal_id')
               or _hkey(header, 'PROPOSID', 'PROGRAM', 'PROGID'))
    proposal_id = _normalize_proposal_id(pid_raw)

    proposal_pi = o.get('proposal_pi') or _resolve_pi_name(obs_row, header)
    proposal_type = (o.get('proposal_type')
                     or _row_get(obs_row, 'proposal_type')
                     or '')

    title = (o.get('title')
             or _row_get(obs_row, 'obs_title')
             or _hkey(header, 'TITLE')
             or '')

    # --- 观测标识
    obs_id = (o.get('obs_id')
              or _row_get(obs_row, 'obs_id')
              or _hkey(header, 'ROOTNAME', 'FILENAME', 'OBSERVTN')
              or '')

    target_name = (o.get('target_name')
                   or _row_get(obs_row, 'target_name')
                   or _hkey(header, 'TARGNAME', 'TARGET')
                   or '')

    # --- 时间 (MJD + UTC)
    obs_mjd = o.get('obs_mjd')
    if obs_mjd is None:
        for src in (header, obs_row):
            if src is None:
                continue
            for key in ('EXPSTART', 'MJD-BEG', 'TEXPSTRT'):
                v = _hkey(src, key) if src is header else _row_get(src, key)
                if v is not None:
                    try:
                        obs_mjd = float(v)
                        break
                    except (ValueError, TypeError):
                        pass
            if obs_mjd is not None:
                break
        if obs_mjd is None:
            v = _row_get(obs_row, 't_min')
            if v is not None:
                try:
                    obs_mjd = float(v)
                except (ValueError, TypeError):
                    obs_mjd = None
    obs_date_utc = mjd_to_isot(obs_mjd) if obs_mjd is not None else ''
    if not obs_date_utc:
        obs_date_utc = str(_hkey(header, 'DATE-OBS') or '')

    # --- 曝光时间
    exptime = (o.get('exptime_s')
               or _hkey(header, 'EXPTIME', 'EFFEXPTM')
               or _row_get(obs_row, 't_exptime'))
    try:
        exptime_s = float(exptime) if exptime is not None else None
    except (ValueError, TypeError):
        exptime_s = None

    # --- 致谢/引用文本
    if survey == 'HST':
        ack, archive, archive_url = _HST_ACK, 'MAST/STScI', 'https://mast.stsci.edu/'
    elif survey == 'JWST':
        ack, archive, archive_url = _JWST_ACK, 'MAST/STScI', 'https://mast.stsci.edu/'
    elif survey.startswith('SDSS'):
        ack, archive, archive_url = _SDSS_ACK, 'SDSS', 'https://www.sdss.org/'
    elif survey.startswith('KOA') or survey.startswith('KECK'):
        ack, archive, archive_url = _KOA_ACK, 'KOA/NExScI', 'https://koa.ipac.caltech.edu/'
    else:
        ack, archive, archive_url = '', '', ''

    # 拼一句简短的引用 (用于绘图标题/论文脚注)
    cite_parts = []
    if survey:
        cite_parts.append(survey)
    if instrument:
        cite_parts.append(str(instrument))
    if proposal_id:
        cite_parts.append(f"PID {proposal_id}")
    if proposal_pi:
        cite_parts.append(f"PI {str(proposal_pi).split(',')[0].strip()}")
    citation_short = ' / '.join(cite_parts)

    prov = {
        'mission': survey,
        'instrument': str(instrument) if instrument else '',
        'detector': str(detector) if detector else '',
        'grating': str(grating) if grating else '',
        'filter': str(filt) if filt else '',
        'proposal_id': proposal_id,
        'proposal_pi': str(proposal_pi) if proposal_pi else '',
        'proposal_type': str(proposal_type) if proposal_type else '',
        'title': str(title) if title else '',
        'obs_id': str(obs_id) if obs_id else '',
        'target_name': str(target_name) if target_name else '',
        'exptime_s': exptime_s,
        'obs_mjd': float(obs_mjd) if obs_mjd is not None else None,
        'obs_date_utc': obs_date_utc,
        'ra': float(ra) if ra is not None else None,
        'dec': float(dec) if dec is not None else None,
        'archive': archive,
        'archive_url': archive_url,
        'citation_short': citation_short,
        'acknowledgement': ack,
    }

    # 把 override 中的额外字段 (例如 SDSS plate/fiberid, KOA koaid/semester)
    # 直接合并到 provenance 顶层, 不会覆盖标准字段
    for k, v in o.items():
        if k not in prov and v is not None:
            prov[k] = v
    return prov


def format_provenance_text(prov, multi_line=True):
    """把 provenance dict 渲染成人读字符串 (用于绘图标题 / 控制台)。"""
    if not prov:
        return ''
    if multi_line:
        lines = []
        head = f"{prov.get('mission', '')} {prov.get('instrument', '')}"
        if prov.get('grating'):
            head += f" {prov['grating']}"
        if prov.get('filter') and prov.get('filter') != prov.get('grating'):
            head += f" [{prov['filter']}]"
        lines.append(head.strip())
        if prov.get('proposal_id'):
            tag = f"PID {prov['proposal_id']}"
            if prov.get('proposal_pi'):
                tag += f"  PI: {prov['proposal_pi']}"
            if prov.get('proposal_type'):
                tag += f"  ({prov['proposal_type']})"
            lines.append(tag)
        if prov.get('title'):
            t = prov['title']
            lines.append(t if len(t) < 80 else t[:77] + '...')
        when = []
        if prov.get('obs_date_utc'):
            when.append(prov['obs_date_utc'][:19])
        if prov.get('obs_mjd'):
            when.append(f"MJD {prov['obs_mjd']:.3f}")
        if prov.get('exptime_s'):
            when.append(f"t_exp={prov['exptime_s']:.0f}s")
        if when:
            lines.append('  '.join(when))
        if prov.get('obs_id'):
            lines.append(f"obs_id: {prov['obs_id']}")
        return '\n'.join(lines)
    else:
        return prov.get('citation_short', '')


def add_provenance_columns(df, prov, columns=None):
    """把 provenance 的若干字段作为列追加到 DataFrame 每一行。

    默认列: mission, instrument, proposal_id, proposal_pi, obs_id, obs_mjd
    """
    if df is None or len(df) == 0 or not prov:
        return df
    if columns is None:
        columns = ['mission', 'instrument', 'proposal_id', 'proposal_pi',
                   'obs_id', 'obs_mjd']
    df = df.copy()
    for col in columns:
        v = prov.get(col)
        df[col] = v if v is not None else ''
    return df


def write_provenance_json(prov, output_dir, filename):
    """把 provenance dict 写成 JSON 边车文件 (sidecar)。"""
    import json
    if not prov:
        return None
    ensure_dir(output_dir)
    if not filename.endswith('.json'):
        filename = filename + '.json'
    path = os.path.join(output_dir, filename)

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(x) for x in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o) if np.isfinite(o) else None
        return o

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(_clean(prov), f, indent=2, ensure_ascii=False)
    return path
