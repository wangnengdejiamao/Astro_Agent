"""
SDSS DR18 光谱查询与下载
========================
波长覆盖: 3800-9200 A (光学)

用法:
    from astro_toolbox.sdss import query_spectrum, plot_spectrum
    result = query_spectrum(190.305, 2.596)
    plot_spectrum(result, save_path='sdss_spec.png')
"""
import numpy as np
from astropy.io import fits
from . import config, utils


def query_spectrum(ra, dec, radius_arcsec=config.SEARCH_RADIUS_ARCSEC):
    """
    按坐标查询 SDSS 光谱 (含 provenance: plate/mjd/fiberid/specobjid/run2d/programname)。

    Returns:
        dict: 标准光谱 dict; 'provenance' 字段包含 SDSS 特有的标识
        或 None
    """
    from astroquery.sdss import SDSS
    c = utils.coord(ra, dec)
    import astropy.units as u

    # 查询光谱观测
    result = SDSS.query_region(c, radius=radius_arcsec * u.arcsec,
                               spectro=True)
    if result is None or len(result) == 0:
        return None

    row = result[0]
    plate = int(row['plate'])
    mjd = int(row['mjd'])
    fiberid = int(row['fiberID'])

    # 从 SDSS 元数据表中提取尽可能多的来源信息
    def _row_str(key):
        try:
            v = row[key]
        except Exception:
            return ''
        s = str(v).strip()
        return '' if s.lower() in ('nan', '--', 'none', '', 'masked') else s

    specobjid = _row_str('specobjid') or _row_str('specobjID')
    run2d = _row_str('run2d')
    instrument_meta = _row_str('instrument')
    programname = _row_str('programname') or _row_str('programName')
    survey_name = _row_str('survey')
    z = float(row.get('z', 0)) if 'z' in row.colnames else 0.0
    cls = _row_str('class')
    subcls = _row_str('subClass') or _row_str('subclass')

    # 下载光谱
    sp = SDSS.get_spectra(matches=result[:1])
    if sp is None or len(sp) == 0:
        return None

    hdul = sp[0]
    primary_header = hdul[0].header
    # SDSS 光谱: HDU1 = coadd spectrum
    data = hdul[1].data
    # 波长: 10^loglam
    wavelength = 10 ** data['loglam']
    flux = data['flux']
    ivar = data['ivar']
    error = np.where(ivar > 0, 1.0 / np.sqrt(ivar), 0.0)
    model = data['model'] if 'model' in data.names else None

    # 主头里有更可靠的来源信息
    h = primary_header
    if not run2d:
        run2d = str(h.get('RUN2D', '')).strip()
    if not instrument_meta:
        instrument_meta = str(h.get('INSTRUME', 'SDSS')).strip()
    if not programname:
        programname = str(h.get('PROGRAM', '')).strip()
    if not survey_name:
        survey_name = str(h.get('SURVEY', '')).strip()
    plate_quality = str(h.get('PLATEQUA', '')).strip()
    exptime_s = None
    for k in ('EXPTIME', 'XCSAO_TS'):
        v = h.get(k)
        if v is not None:
            try:
                exptime_s = float(v)
                break
            except (ValueError, TypeError):
                pass
    # 数据释放版本: SDSS DR18 是 astroquery 当前默认
    data_release = str(h.get('DATAVERS', 'DR18')).strip() or 'DR18'

    # 把 SDSS 特定字段塞进 provenance.override (会保留到顶层)
    survey_tag = f"SDSS-{instrument_meta or 'SDSS'}".upper()
    prov = utils.build_provenance(
        survey_tag, header=primary_header, ra=ra, dec=dec,
        override={
            'instrument': instrument_meta or 'SDSS',
            'proposal_id': plate,            # SDSS 用 plate 充当 proposal 概念
            'proposal_type': programname,    # eBOSS / BOSS / SEGUE 等
            'title': f"SDSS spectrum (plate={plate}, mjd={mjd}, "
                     f"fiber={fiberid})",
            'obs_id': specobjid or f"{plate}-{mjd}-{fiberid:04d}",
            'obs_mjd': float(mjd),
            'exptime_s': exptime_s,
            # extras (会保留为 provenance 顶层键, 写入 JSON 与 CSV 列):
            'plate': plate,
            'mjd': mjd,
            'fiberid': fiberid,
            'specobjid': specobjid,
            'run2d': run2d,
            'programname': programname,
            'sdss_survey': survey_name,
            'plate_quality': plate_quality,
            'data_release': data_release,
            'redshift': z,
            'class': cls,
            'subclass': subcls,
        })

    result = {
        'survey': 'SDSS',
        'ra': ra, 'dec': dec,
        'plate': plate, 'mjd': mjd, 'fiberid': fiberid,
        'specobjid': specobjid, 'run2d': run2d,
        'programname': programname, 'data_release': data_release,
        'obs_mjd': mjd,
        'exptime_s': exptime_s,
        'z': z,
        'class': cls,
        'subclass': subcls,
        'wavelength': wavelength,
        'flux': flux,
        'error': error,
        'model': model,
        'provenance': prov,
    }
    try:
        from .diagnostics import analyze_spectrum
        result['diagnostics'] = analyze_spectrum(
            wavelength, flux, error, survey='SDSS', metadata=result)
    except Exception:
        pass
    return result


def get_photometry(ra, dec, radius_arcsec=config.SEARCH_RADIUS_ARCSEC):
    """
    获取 SDSS ugriz 测光 (SED 用)。

    Returns:
        dict: {band_name: (mag, mag_err, wave_A)} 或空 dict
    """
    from astroquery.sdss import SDSS
    import astropy.units as u
    c = utils.coord(ra, dec)
    result = SDSS.query_region(c, radius=radius_arcsec * u.arcsec,
                               photoobj_fields=['ra', 'dec',
                                                'psfMag_u', 'psfMagErr_u',
                                                'psfMag_g', 'psfMagErr_g',
                                                'psfMag_r', 'psfMagErr_r',
                                                'psfMag_i', 'psfMagErr_i',
                                                'psfMag_z', 'psfMagErr_z'])
    if result is None or len(result) == 0:
        return {}

    row = result[0]
    bands = {
        'SDSS_u': ('psfMag_u', 'psfMagErr_u'),
        'SDSS_g': ('psfMag_g', 'psfMagErr_g'),
        'SDSS_r': ('psfMag_r', 'psfMagErr_r'),
        'SDSS_i': ('psfMag_i', 'psfMagErr_i'),
        'SDSS_z': ('psfMag_z', 'psfMagErr_z'),
    }
    phot = {}
    for band_name, (mag_col, err_col) in bands.items():
        mag = float(row[mag_col])
        err = float(row[err_col])
        if 0 < mag < 30 and err > 0:
            wave = config.BAND_INFO[band_name]['wave_A']
            phot[band_name] = (mag, err, wave)
    return phot


def plot_spectrum(result, save_path=None):
    """绘制 SDSS 光谱"""
    if result is None:
        return None
    fig, ax = utils.setup_spectrum_plot()
    ax.plot(result['wavelength'], result['flux'], 'k-', lw=0.6, label='Flux')
    if result.get('model') is not None:
        ax.plot(result['wavelength'], result['model'], 'r-', lw=0.5,
                alpha=0.7, label='Best-fit model')
    ax.fill_between(result['wavelength'],
                    result['flux'] - result['error'],
                    result['flux'] + result['error'],
                    color='gray', alpha=0.2)
    prov = result.get('provenance', {})
    title_lines = [
        f"SDSS {prov.get('instrument', 'SDSS')} Spectrum  "
        f"plate={result['plate']} mjd={result['mjd']} fiber={result['fiberid']}",
    ]
    extras = []
    if prov.get('programname'):
        extras.append(f"prog={prov['programname']}")
    if prov.get('run2d'):
        extras.append(f"run2d={prov['run2d']}")
    if prov.get('data_release'):
        extras.append(prov['data_release'])
    if extras:
        title_lines.append('  '.join(extras))
    cls_line = f"class={result['class']}"
    if result.get('subclass'):
        cls_line += f"/{result['subclass']}"
    if result.get('z'):
        cls_line += f"  z={result['z']:.4f}"
    title_lines.append(cls_line)
    ax.set_title('\n'.join(title_lines), fontsize=10)
    diag = result.get('diagnostics')
    if diag and diag.get('flags'):
        ax.text(0.01, 0.96, ', '.join(diag['flags'][:4]),
                transform=ax.transAxes, ha='left', va='top', fontsize=8,
                color='crimson',
                bbox=dict(facecolor='white', edgecolor='crimson',
                          alpha=0.8, boxstyle='round,pad=0.25'))
    ax.set_ylabel('Flux (1e-17 erg/s/cm2/A)')
    ax.legend()

    # 轴范围紧凑到光谱数据
    utils.set_spectrum_axes(ax, result['wavelength'], result['flux'],
                            model=result.get('model'))

    fig.tight_layout()
    utils.save_and_close(fig, save_path)
    return fig


def save_spectrum_csv(result, output_dir):
    """保存 SDSS 光谱为 CSV (附带 provenance 列 + 边车 JSON)。"""
    df = utils.spectrum_to_dataframe(result, extra_keys=['model'])
    prov = (result or {}).get('provenance')
    df = utils.add_provenance_columns(
        df, prov,
        columns=['mission', 'instrument', 'plate', 'mjd', 'fiberid',
                 'specobjid', 'run2d', 'programname', 'data_release'])
    csv_path = utils.write_csv(df, output_dir, 'sdss_spectrum.csv')
    if prov:
        utils.write_provenance_json(prov, output_dir,
                                    'sdss_spectrum_provenance.json')
    return csv_path


def save_photometry_csv(result, output_dir):
    """保存 SDSS 测光为 CSV"""
    df = utils.photometry_to_dataframe(result)
    return utils.write_csv(df, output_dir, 'sdss_photometry.csv')
