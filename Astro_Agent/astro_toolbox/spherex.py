"""
SPHEREx 近红外低分辨率光谱
============================
NASA SPHEREx 于 2025 年 3 月发射。
波长覆盖: 0.75 - 5.0 um (6 个探测器, R~40-130)
数据通过 IRSA (irsa.ipac.caltech.edu) 发布。

数据获取方式:
1. IRSA SIA v2 搜索 SPHEREx spectral image 切片
2. IRSA IBE 接口下载目标坐标处的 cutout
3. 从各通道 cutout 中提取中心像素流量, 组装低分辨率光谱

SPHEREx Level 2 数据: 每个 FITS 文件是一个波长通道的 sky image
  - IMAGE: 流量图 (MJy/sr)
  - VARIANCE: 方差图
  - WCS-WAVE: 波长信息 (um)

用法:
    from astro_toolbox.spherex import query_spectrum, get_photometry
    spec = query_spectrum(232.3955, 29.4672)
"""
import numpy as np
import io
from . import config, utils

# IRSA SIA v2 端点
SIA_V2_URL = "https://irsa.ipac.caltech.edu/SIA/v2"

# SPHEREx SIA v2 collection 名称 (按优先级: 最新的在前)
SPHEREX_COLLECTIONS = ['spherex_qr2', 'spherex_qr2_deep', 'SPHEREx']

# 物理常数
_MJY_SR_TO_CGS_PER_A = None  # 延迟计算


def _mjy_sr_to_cgs(flux_mjy_sr, wave_um, pixel_sr):
    """
    MJy/sr → erg/s/cm^2/A (f_lambda)

    1 MJy/sr = 1e-17 erg/s/cm^2/Hz/sr
    f_nu (per pixel) = flux_mjy_sr * pixel_sr * 1e-17  [erg/s/cm^2/Hz]
    f_lambda = f_nu * c / lambda^2
    """
    wave_cm = wave_um * 1e-4
    wave_A = wave_um * 1e4
    c_cgs = 2.99792458e10  # cm/s
    # f_nu per pixel
    f_nu = flux_mjy_sr * pixel_sr * 1e-17  # erg/s/cm^2/Hz
    # f_lambda = f_nu * c / lambda^2  (with lambda in cm, result in erg/s/cm^2/cm)
    f_lambda_per_cm = f_nu * c_cgs / wave_cm ** 2
    # 转为 per Angstrom
    f_lambda = f_lambda_per_cm * 1e-8
    return f_lambda


def query_spectrum(ra, dec, radius_arcsec=None,
                   collection=None, cutout_size=5):
    """
    查询 SPHEREx 低分辨率光谱。

    通过 IRSA SIA v2 搜索覆盖目标的所有波长通道,
    下载每个通道的小 cutout, 提取中心像素流量组装光谱。

    Args:
        ra, dec: 目标坐标 (度)
        collection: SIA v2 collection 名称 (默认自动尝试)
        cutout_size: cutout 边长 (像素, 默认 5)

    Returns:
        dict: {
            'wavelength': array (Angstrom),
            'flux': array (erg/s/cm^2/A),
            'error': array,
            'flux_mjy_sr': array (原始 MJy/sr),
            'survey': 'SPHEREx',
            'n_channels': int,
        }
        或 None
    """
    # 1. SIA v2 搜索: 找到覆盖目标的所有 spectral image 切片
    image_list = _query_sia_v2(ra, dec, collection)
    if image_list is None or len(image_list) == 0:
        print("  SPHEREx: 该坐标处无数据覆盖")
        return None

    print(f"  SPHEREx: 找到 {len(image_list)} 个通道, 正在提取光谱...")

    # 2. 按波长去重: 同一波段多次观测取最近的
    unique_images = _deduplicate_by_wavelength(image_list)

    # 3. 下载 cutout 并提取中心像素流量
    wavelengths = []
    fluxes_mjy = []
    errors_mjy = []

    session = utils.get_session(SIA_V2_URL)

    for img_info in unique_images:
        result = _extract_pixel_from_cutout(
            img_info, ra, dec, cutout_size, session)
        if result is not None:
            wavelengths.append(result['wave_um'])
            fluxes_mjy.append(result['flux_mjy_sr'])
            errors_mjy.append(result['error_mjy_sr'])

    if len(wavelengths) == 0:
        print("  SPHEREx: 无法从 cutout 中提取有效流量")
        return None

    # 按波长排序
    wave_um = np.array(wavelengths)
    flux_mjy = np.array(fluxes_mjy)
    err_mjy = np.array(errors_mjy)

    sort_idx = np.argsort(wave_um)
    wave_um = wave_um[sort_idx]
    flux_mjy = flux_mjy[sort_idx]
    err_mjy = err_mjy[sort_idx]

    # 转换为 CGS (erg/s/cm^2/A)
    # SPHEREx 像素为 6.2 arcsec, pixel solid angle
    pixel_arcsec = 6.2
    pixel_sr = (pixel_arcsec / 206265.0) ** 2

    wave_A = wave_um * 1e4
    flux_cgs = _mjy_sr_to_cgs(flux_mjy, wave_um, pixel_sr)
    err_cgs = _mjy_sr_to_cgs(np.abs(err_mjy), wave_um, pixel_sr)

    # 背景扣除后流量可能为负/零 (源太暗), 只保留正值通道
    positive = flux_cgs > 0
    n_positive = np.sum(positive)
    n_total = len(wave_A)
    if n_positive == 0:
        print(f"  SPHEREx: 背景扣除后无正流量通道 ({n_total} channels), 源可能太暗")
        return None

    if n_positive < n_total:
        print(f"  SPHEREx: 背景扣除后 {n_total - n_positive}/{n_total} 通道流量<=0, 已排除")
        wave_A = wave_A[positive]
        flux_cgs = flux_cgs[positive]
        err_cgs = err_cgs[positive]
        flux_mjy = flux_mjy[positive]

    print(f"  SPHEREx: 提取到 {len(wave_A)} 个有效通道 (已扣除背景)")

    return {
        'survey': 'SPHEREx',
        'ra': ra, 'dec': dec,
        'wavelength': wave_A,
        'flux': flux_cgs,
        'error': err_cgs,
        'flux_mjy_sr': flux_mjy,
        'n_channels': len(wave_A),
    }


def _query_sia_v2(ra, dec, collection=None):
    """
    通过 IRSA SIA v2 查询覆盖目标坐标的 SPHEREx spectral image 列表。

    Returns:
        list of dict, 每个包含 access_url, em_min, em_max, band_name
    """
    collections = [collection] if collection else SPHEREX_COLLECTIONS

    for coll in collections:
        try:
            params = {
                'COLLECTION': coll,
                'POS': f'CIRCLE {ra} {dec} 0.01',
                'RESPONSEFORMAT': 'votable',
            }
            session = utils.get_session(SIA_V2_URL)
            resp = session.get(SIA_V2_URL, params=params,
                               timeout=utils.get_timeout())
            if resp.status_code != 200:
                continue

            from astropy.io.votable import parse
            vot = parse(io.BytesIO(resp.content))

            images = []
            for resource in vot.resources:
                if resource.type != 'results':
                    continue
                for table in resource.tables:
                    t = table.to_table()
                    if len(t) == 0:
                        continue

                    # 获取列名映射 (VOTable field name → column index)
                    field_names = [f.name for f in table.fields]
                    col_names = t.colnames

                    # 关键列索引
                    idx_map = {}
                    for fname, ucd_target in [
                        ('access_url', 'meta.ref.url'),
                        ('em_min', 'em.wl;stat.min'),
                        ('em_max', 'em.wl;stat.max'),
                    ]:
                        for j, f in enumerate(table.fields):
                            if f.name == fname or (f.ucd and ucd_target in f.ucd):
                                idx_map[fname] = col_names[j]
                                break

                    if 'access_url' not in idx_map:
                        continue

                    url_col = idx_map['access_url']
                    em_min_col = idx_map.get('em_min')
                    em_max_col = idx_map.get('em_max')
                    # band name column
                    band_col = None
                    for j, f in enumerate(table.fields):
                        if f.name == 'energy_bandpassname':
                            band_col = col_names[j]
                            break

                    for row in t:
                        url = str(row[url_col])
                        if not url or url == '--':
                            continue
                        info = {'access_url': url}
                        if em_min_col:
                            info['em_min'] = float(row[em_min_col])
                            info['em_max'] = float(row[em_max_col])
                            # 中心波长 (um)
                            info['wave_um'] = (info['em_min'] +
                                               info['em_max']) / 2.0 * 1e6
                        if band_col:
                            info['band'] = str(row[band_col])
                        images.append(info)

            if images:
                return images

        except Exception as e:
            print(f"  SPHEREx SIA v2 查询失败 ({coll}): {e}")
            continue

    return None


def _deduplicate_by_wavelength(image_list, tol_um=0.01):
    """
    按中心波长去重: 同一波长多次观测只保留一个。
    """
    if not image_list:
        return []

    # 按波长排序
    sorted_imgs = sorted(image_list, key=lambda x: x.get('wave_um', 0))

    unique = [sorted_imgs[0]]
    for img in sorted_imgs[1:]:
        w = img.get('wave_um', 0)
        w_last = unique[-1].get('wave_um', 0)
        if abs(w - w_last) > tol_um:
            unique.append(img)
    return unique


def _extract_pixel_from_cutout(img_info, ra, dec, cutout_size, session):
    """
    下载 SPHEREx FITS cutout, 提取中心像素的流量和波长。
    **带背景扣除**: 用 cutout 外环像素中位数估计天空背景, 减去后得到纯源流量。

    SPHEREx 像素为 6.2", 对于暗源 (>18 mag) 中心像素被背景主导,
    不做背景扣除会导致流量严重偏高。

    SPHEREx Level 2 FITS 结构:
      [1] IMAGE: 流量 (MJy/sr)
      [3] VARIANCE: 方差
      [6] WCS-WAVE: 波长信息, VALUES shape (ny, nx, 2) = [wave_um, bandwidth_um]

    Returns:
        dict: {wave_um, bandwidth_um, flux_mjy_sr, error_mjy_sr} 或 None
    """
    url = img_info['access_url']
    cutout_url = f"{url}?center={ra},{dec}&size={cutout_size}pix"

    try:
        resp = session.get(cutout_url, timeout=(30, 120))
        if resp.status_code != 200:
            return None

        from astropy.io import fits as pyfits
        hdul = pyfits.open(io.BytesIO(resp.content))

        # IMAGE HDU (index 1)
        if len(hdul) < 2 or hdul[1].data is None:
            hdul.close()
            return None

        img = hdul[1].data
        ny, nx = img.shape
        cy, cx = ny // 2, nx // 2

        # 中心像素原始流量 (MJy/sr)
        raw_flux = float(img[cy, cx])
        if not np.isfinite(raw_flux):
            hdul.close()
            return None

        # 背景估计: 用外环像素 (排除中心 1 像素) 的中位数
        # 构建掩膜: 中心像素=False, 其余=True
        mask = np.ones((ny, nx), dtype=bool)
        mask[cy, cx] = False
        # 也排除无效值
        valid_bg = mask & np.isfinite(img) & (img != 0)
        if np.sum(valid_bg) >= 4:
            bg = np.median(img[valid_bg])
        else:
            bg = 0.0

        flux = raw_flux - bg

        # 方差 (VARIANCE HDU, index 3)
        error = 0.0
        if len(hdul) > 3 and hdul[3].data is not None:
            var_center = float(hdul[3].data[cy, cx])
            if np.isfinite(var_center) and var_center > 0:
                # 背景方差: 外环像素方差的中位数 / N_bg_pixels (背景估计的误差)
                var_img = hdul[3].data
                valid_bg_var = mask & np.isfinite(var_img) & (var_img > 0)
                if np.sum(valid_bg_var) >= 4:
                    bg_var = np.median(var_img[valid_bg_var]) / np.sum(valid_bg_var)
                else:
                    bg_var = 0.0
                error = np.sqrt(var_center + bg_var)

        # 波长 (WCS-WAVE HDU, index 6)
        # VALUES shape: (ny_grid, nx_grid, 2) → [wavelength_um, bandwidth_um]
        wave_um = img_info.get('wave_um', 0)
        bandwidth_um = 0.03  # default
        if len(hdul) > 6 and hdul[6].data is not None:
            try:
                wcs_data = hdul[6].data[0]
                values = wcs_data['VALUES']
                # shape is (ny, nx, 2): last dim is [wave, bandwidth]
                if values.ndim == 3 and values.shape[-1] == 2:
                    # 取中心位置的波长
                    wy = values.shape[0] // 2
                    wx = values.shape[1] // 2
                    wave_um = float(values[wy, wx, 0])
                    bandwidth_um = float(values[wy, wx, 1])
            except Exception:
                pass

        hdul.close()

        if wave_um <= 0.1:
            return None

        return {
            'wave_um': wave_um,
            'bandwidth_um': bandwidth_um,
            'flux_mjy_sr': flux,
            'error_mjy_sr': error,
        }

    except Exception:
        return None


def get_photometry(ra, dec, radius_arcsec=config.SEARCH_RADIUS_ARCSEC):
    """
    获取 SPHEREx 宽带测光 (从光谱合成)。

    从 SPHEREx 低分辨率光谱中合成几个关键波段的测光,
    用于 SED 拟合。

    Returns:
        dict: {band_name: (mag, mag_err, wave_A)}
    """
    spec = query_spectrum(ra, dec)
    if spec is None:
        return {}

    wave = spec['wavelength']  # Angstrom
    flux = spec['flux']        # erg/s/cm^2/A
    error = spec.get('error', np.zeros_like(flux))

    # 合成宽带: 在各波长范围内取加权平均
    synth_bands = {
        'SPHEREx_1.0': (7500, 12000),     # D1: ~0.75-1.2 um
        'SPHEREx_1.5': (12000, 17000),     # D2: ~1.1-1.7 um
        'SPHEREx_2.0': (17000, 25000),     # D3: ~1.6-2.5 um
        'SPHEREx_3.0': (25000, 40000),     # D4: ~2.4-3.8 um + D5: ~3.8-4.4 um
        'SPHEREx_4.5': (40000, 51000),     # D6: ~4.4-5.0 um
    }

    phot = {}
    for band_name, (w_lo, w_hi) in synth_bands.items():
        mask = (wave >= w_lo) & (wave <= w_hi) & np.isfinite(flux) & (flux > 0)
        if np.sum(mask) >= 1:
            mean_flux = np.mean(flux[mask])
            mean_err = np.sqrt(np.mean(error[mask] ** 2)) if np.any(error[mask] > 0) else mean_flux * 0.1
            center_wave = (w_lo + w_hi) / 2.0
            # flux_lambda → f_nu (Jy): f_nu = f_lambda * lambda^2 / c * 1e23
            c_A = 2.99792458e18  # Angstrom/s
            f_nu_jy = mean_flux * center_wave ** 2 / c_A * 1e23
            if f_nu_jy > 0:
                mag = -2.5 * np.log10(f_nu_jy / 3631.0)
                # 误差传播
                f_nu_err = mean_err * center_wave ** 2 / c_A * 1e23
                mag_err = 2.5 / np.log(10) * f_nu_err / f_nu_jy if f_nu_jy > 0 else 0.1
                if 0 < mag < 30:
                    phot[band_name] = (mag, mag_err, center_wave)

    return phot


def plot_spectrum(spec, save_path=None):
    """绘制 SPHEREx 低分辨率光谱"""
    if spec is None:
        return None

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 5))

    wave = spec['wavelength'] / 10000.0  # Angstrom → um
    flux = spec.get('flux_mjy_sr', spec['flux'])
    error = spec.get('error')

    # 使用 MJy/sr 绘图 (如果有)
    if 'flux_mjy_sr' in spec:
        ylabel = 'Flux (MJy/sr)'
        flux = spec['flux_mjy_sr']
    else:
        ylabel = r'$f_\lambda$ (erg s$^{-1}$ cm$^{-2}$ A$^{-1}$)'

    valid = np.isfinite(flux) & (flux != 0)
    ax.plot(wave[valid], flux[valid], 'b.-', lw=1.0, ms=4,
            alpha=0.8, label='SPHEREx spectrum')
    if error is not None and np.any(error > 0):
        err_valid = error[valid] if len(error) == len(flux) else None
        if err_valid is not None:
            ax.fill_between(wave[valid], flux[valid] - err_valid,
                            flux[valid] + err_valid,
                            color='blue', alpha=0.15)

    ax.set_xlabel('Wavelength (um)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"SPHEREx Spectrum  RA={spec['ra']:.4f} DEC={spec['dec']:.4f}  "
                 f"({spec.get('n_channels', 0)} channels)", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 轴范围紧凑到光谱数据 (单位是 um, 直接设置)
    w_valid = wave[valid]
    f_valid = flux[valid]
    if len(w_valid) > 1:
        wmin, wmax = w_valid.min(), w_valid.max()
        dw = max((wmax - wmin) * 0.02, 0.01)
        ax.set_xlim(wmin - dw, wmax + dw)
        flo, fhi = np.percentile(f_valid[np.isfinite(f_valid)], [1, 99])
        df = max((fhi - flo) * 0.1, abs(fhi) * 0.01)
        ax.set_ylim(flo - df, fhi + df)

    fig.tight_layout()
    utils.save_and_close(fig, save_path)
    return fig


def save_spectrum_csv(result, output_dir):
    """保存 SPHEREx 光谱为 CSV"""
    import pandas as pd
    if result is None or 'wavelength' not in result:
        return None
    data = {'wavelength_A': result['wavelength'], 'flux': result['flux']}
    if 'error' in result and result['error'] is not None:
        data['error'] = result['error']
    if 'flux_mjy_sr' in result:
        data['flux_mjy_sr'] = result['flux_mjy_sr']
    df = pd.DataFrame(data)
    return utils.write_csv(df, output_dir, 'spherex_spectrum.csv')


def save_photometry_csv(result, output_dir):
    """保存 SPHEREx 合成测光为 CSV"""
    df = utils.photometry_to_dataframe(result)
    return utils.write_csv(df, output_dir, 'spherex_photometry.csv')
