"""
KOA / Keck local spectrum extraction
====================================

This module turns KOA-downloaded, already extracted 1D spectra into the same
standard dict shape used by the rest of astro_toolbox:

    {'wavelength': array, 'flux': array, 'error': array, ...}

The main supported local format is the LRIS/PypeIt-like 1D FITS table with
columns such as wave, flux, mask.  It also keeps enough metadata from KOA
readme.txt files to recover MJD, exposure time, arm, grating and coordinates.
"""
import glob
import os
import re
import shutil
import subprocess
import tempfile

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits

from . import config, utils


BALMER_LINES = {
    'Halpha': 6562.8,
    'Hbeta': 4861.3,
    'Hgamma': 4340.5,
    'Hdelta': 4101.7,
    'HeII 4686': 4685.7,
    'HeI 5876': 5875.6,
    'HeI 6678': 6678.2,
}
TELLURIC_LINES = {
    'telluric B': 6867.0,
    'telluric A': 7594.0,
}

KOA_INSTRUMENTS = ('lris', 'hires', 'deimos', 'esi', 'kcwi', 'mosfire',
                   'nirspec', 'nirc2', 'nires', 'osiris')
LRIS_LAMP_KEYS = (
    'MERCURY', 'NEON', 'ARGON', 'CADMIUM', 'ZINC', 'KRYPTON', 'XENON',
    'FEARGON', 'DEUTERI', 'FLAMP1', 'FLAMP2', 'HALOGEN'
)

_KOA_TABLE_FORMATS = {
    'ipac': 'ascii.ipac',
    'tbl': 'ascii.ipac',
    'csv': 'ascii.csv',
    'tsv': 'ascii.tab',
    'votable': 'votable',
    'xml': 'votable',
    'fits': 'fits',
}


def _clean_key(value):
    return str(value).strip().lower()


def _as_float(value, default=np.nan):
    try:
        value = float(str(value).strip())
    except Exception:
        return default
    return value if np.isfinite(value) else default


def _as_str(value):
    if value is None:
        return ''
    try:
        if hasattr(value, 'item'):
            value = value.item()
    except Exception:
        pass
    return str(value).strip()


def parse_target_coordinates(target):
    """
    Parse compact IAU-style target names such as ZTFJ035352.96+431525.16.

    Returns
    -------
    (ra_deg, dec_deg) or None
    """
    if not target:
        return None
    text = re.sub(r'\s+', '', str(target).strip())
    match = re.search(
        r'J(\d{2})(\d{2})(\d{2}(?:\.\d*)?)([+-])'
        r'(\d{2})(\d{2})(\d{2}(?:\.\d*)?)',
        text,
        flags=re.IGNORECASE)
    if not match:
        return None
    hh, mm, ss, sign, dd, dm, ds = match.groups()
    ra = 15.0 * (float(hh) + float(mm) / 60.0 + float(ss) / 3600.0)
    dec_abs = float(dd) + float(dm) / 60.0 + float(ds) / 3600.0
    dec = dec_abs if sign == '+' else -dec_abs
    return float(ra), float(dec)


def resolve_target_coordinates(target=None, ra=None, dec=None,
                               allow_name_resolve=False):
    """Resolve explicit coordinates or parse a J-name into decimal degrees."""
    if ra is not None and dec is not None:
        return float(ra), float(dec)

    parsed = parse_target_coordinates(target)
    if parsed is not None:
        return parsed

    if allow_name_resolve and target:
        try:
            from astropy.coordinates import SkyCoord
            coord = SkyCoord.from_name(str(target))
            return float(coord.ra.deg), float(coord.dec.deg)
        except Exception as exc:
            raise ValueError(f'Could not resolve target name: {target}') from exc

    raise ValueError('Provide ra/dec or a parseable target name like '
                     'ZTFJ035352.96+431525.16')


def _natural_key(path):
    name = os.path.basename(path)
    return [int(x) if x.isdigit() else x.lower()
            for x in re.split(r'(\d+)', name)]


def _file_index(path):
    match = re.match(r'^(\d+)_', os.path.basename(path))
    return int(match.group(1)) if match else None


def _read_readme(readme_path):
    """Read KOA-style pipe-separated readme.txt metadata."""
    if not readme_path or not os.path.exists(readme_path):
        return None
    import pandas as pd
    try:
        df = pd.read_csv(readme_path, sep='|', engine='python')
    except Exception:
        return None
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(lambda x: str(x).strip())
    return df


def _meta_from_series(row):
    if row is None:
        return {}
    meta = {}
    for key, value in row.items():
        clean = str(key).strip()
        if clean:
            meta[clean] = value
    for key in ('ra', 'dec', 'mjd', 'airmass', 'exptime', 'frameno',
                'calib', 'comb_id', 'bkg_id', 'cenwave', 'dispangle'):
        if key in meta:
            meta[key] = _as_float(meta[key])
    return meta


def _readme_meta_for_file(path):
    """Attach one readme row to an extracted 1D file when possible."""
    directory = os.path.dirname(path)
    readme = os.path.join(directory, 'readme.txt')
    df = _read_readme(readme)
    if df is None or df.empty:
        return {}

    science = df[df['frametype'].astype(str).str.contains(
        'science', case=False, na=False)] if 'frametype' in df.columns else df
    science = science.reset_index(drop=True)

    idx = _file_index(path)
    if idx is not None and 1 <= idx <= len(science):
        return _meta_from_series(science.iloc[idx - 1])

    # Unnumbered files are usually per-arm coadds; use representative metadata.
    meta = {}
    if len(science) > 0:
        first = _meta_from_series(science.iloc[0])
        meta.update(first)
        for key in ('mjd', 'airmass', 'exptime'):
            if key in science.columns:
                vals = np.array([_as_float(v) for v in science[key]],
                                dtype=float)
                vals = vals[np.isfinite(vals)]
                if len(vals):
                    meta[key] = float(np.nanmean(vals))
        meta['n_input_exposures'] = int(len(science))
        if 'mjd' in science.columns:
            vals = np.array([_as_float(v) for v in science['mjd']],
                            dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals):
                meta['mjd_min'] = float(vals.min())
                meta['mjd_max'] = float(vals.max())
    return meta


def _record_for_file(path, root):
    parent = os.path.basename(os.path.dirname(path)).lower()
    obs_id = os.path.basename(os.path.dirname(os.path.dirname(path)))
    idx = _file_index(path)
    meta = _readme_meta_for_file(path)
    record = {
        'path': path,
        'filename': os.path.basename(path),
        'obs_id': obs_id,
        'arm': parent if parent in ('blue', 'red') else '',
        'exposure_index': idx,
        'is_coadd': idx is None,
        'local_root': root,
    }
    record.update(meta)
    if not meta:
        try:
            with fits.open(path, memmap=False) as hdul:
                header_meta = _extract_keck_header_meta(hdul)
            if np.isfinite(_as_float(header_meta.get('targ_ra'))):
                record['ra'] = _as_float(header_meta.get('targ_ra'))
            if np.isfinite(_as_float(header_meta.get('targ_dec'))):
                record['dec'] = _as_float(header_meta.get('targ_dec'))
            if np.isfinite(_as_float(header_meta.get('mjd_obs'))):
                record['mjd'] = _as_float(header_meta.get('mjd_obs'))
            if np.isfinite(_as_float(header_meta.get('exptime'))):
                record['exptime'] = _as_float(header_meta.get('exptime'))
            if header_meta.get('date_obs'):
                record['dateobs'] = header_meta.get('date_obs')
            filename = os.path.basename(path)
            if not record.get('arm'):
                if '_LRISb_' in filename or filename.startswith('spec1d_LB.'):
                    record['arm'] = 'blue'
                elif '_LRISr_' in filename or filename.startswith('spec1d_LR.'):
                    record['arm'] = 'red'
        except Exception:
            pass
    record['instrument'] = 'LRIS'
    record['survey'] = 'KOA_LRIS'
    return record


def discover_local_spectra(root=None, obs_id=None, arm=None,
                           include_coadds=True, include_individual=True):
    """
    Discover extracted KOA 1D FITS under a local directory.

    Parameters
    ----------
    root : str
        Directory such as /Users/ljm/Desktop/DWD/speutrem.
    obs_id : str or None
        Optional subdirectory/KOA set id, e.g. 97587.
    arm : {'blue', 'red'} or None
        Optional LRIS arm filter.
    """
    root = os.path.abspath(root or config.KOA_LOCAL_ROOT)
    paths = sorted(glob.glob(os.path.join(root, '**', '*.fits'),
                             recursive=True), key=_natural_key)
    records = []
    for path in paths:
        base = os.path.basename(path)
        if base.startswith('.'):
            continue
        rec = _record_for_file(path, root)
        if obs_id is not None and str(rec.get('obs_id')) != str(obs_id):
            continue
        if arm is not None and rec.get('arm') != str(arm).lower():
            continue
        if rec['is_coadd'] and not include_coadds:
            continue
        if (not rec['is_coadd']) and not include_individual:
            continue
        records.append(rec)
    return records


def _target_match(record, target):
    if not target:
        return True
    needle = re.sub(r'[^a-z0-9]+', '', str(target).lower())
    hay = ' '.join(_as_str(record.get(k)) for k in (
        'target', 'filename', 'obs_id')).lower()
    hay = re.sub(r'[^a-z0-9]+', '', hay)
    if needle in hay or hay in needle:
        return True
    # ZTFJ035352.96+431525.16 and ZTFJ0353 4315 should still match loosely.
    if needle.startswith('ztfj') and 'ztfj' in hay:
        return needle[4:8] in hay and needle[-6:-2] in hay
    return False


def match_local_spectra(ra=None, dec=None, target=None, root=None,
                        radius_arcsec=None, include_coadds=True,
                        include_individual=True, match_all=False):
    """Find local extracted spectra matching a sky position or target name."""
    radius_arcsec = (config.KOA_SEARCH_RADIUS_ARCSEC if radius_arcsec is None
                     else radius_arcsec)
    records = discover_local_spectra(
        root=root, include_coadds=include_coadds,
        include_individual=include_individual)
    if match_all:
        return records

    matched = []
    for rec in records:
        ok = False
        if ra is not None and dec is not None:
            rra = _as_float(rec.get('ra'))
            rdec = _as_float(rec.get('dec'))
            if np.isfinite(rra) and np.isfinite(rdec):
                sep = utils.coord(float(ra), float(dec)).separation(
                    utils.coord(rra, rdec)).arcsec
                rec['separation_arcsec'] = float(sep)
                ok = sep <= radius_arcsec
        if not ok and target:
            ok = _target_match(rec, target)
        if ok:
            matched.append(rec)
    return sorted(matched, key=lambda r: (
        r.get('obs_id', ''), r.get('arm', ''), not r.get('is_coadd', False),
        r.get('exposure_index') or 0))


def _find_column(data, candidates):
    if not hasattr(data, 'names') or not data.names:
        return None
    names = {_clean_key(name): name for name in data.names}
    for cand in candidates:
        key = _clean_key(cand)
        if key in names:
            return names[key]
    return None


def _find_table_column(table, candidates):
    names = getattr(table, 'colnames', None)
    if not names:
        return None
    lookup = {_clean_key(name): name for name in names}
    for cand in candidates:
        key = _clean_key(cand)
        if key in lookup:
            return lookup[key]
    return None


_KECK_HEADER_KEYS = {
    # 程序/proposal 信息 (Keck 用 PROGID 而非 HST 的 PROPOSID)
    'progid': ('PROGID', 'PROPID'),
    'progpi': ('PROGPI', 'PR_INV_L'),
    'progtitl': ('PROGTITL', 'TITLE'),
    'proginst': ('PROGINST',),
    'semester': ('SEMESTER',),
    'koaid': ('KOAID',),
    # 仪器信息
    'instrume': ('INSTRUME',),
    'detector': ('DETECTOR',),
    'graname': ('GRANAME', 'GRATNAME', 'GRATING', 'XDISPERS'),
    'grangle': ('GRANGLE', 'GRATING_ANGLE'),
    'slitname': ('SLITNAME', 'DECKNAME', 'SLMSKNAM'),
    'dichname': ('DICHNAME',),
    'filter': ('FILTER', 'FILTER1'),
    # 时间 / 曝光
    'date_obs': ('DATE-OBS', 'DATE_OBS', 'DATE'),
    'ut': ('UT', 'UTC'),
    'mjd_obs': ('MJD-OBS', 'MJD_OBS'),
    'exptime': ('EXPTIME', 'ELAPTIME', 'TELAPSE'),
    'airmass': ('AIRMASS',),
    # 目标
    'targname': ('TARGNAME', 'OBJECT', 'OBJNAME'),
    'targ_ra': ('TARGRA', 'RA'),
    'targ_dec': ('TARGDEC', 'DEC'),
    # 观测者
    'observer': ('OBSERVER',),
}


def _extract_keck_header_meta(hdul):
    """从 KOA/Keck FITS 文件 (任意 HDU) 中收集 program/instrument/time 元数据。"""
    meta = {}
    for hdu in hdul:
        h = hdu.header
        if h is None:
            continue
        for out_key, candidates in _KECK_HEADER_KEYS.items():
            if out_key in meta:
                continue
            for k in candidates:
                if k in h:
                    v = h[k]
                    if v is None:
                        continue
                    s = str(v).strip()
                    if s and s.lower() not in ('none', 'nan', 'unknown', ''):
                        meta[out_key] = s
                        break
    # 数值字段做软转换
    for key in ('exptime', 'mjd_obs', 'airmass'):
        if key in meta:
            try:
                meta[key] = float(meta[key])
            except (ValueError, TypeError):
                pass
    return meta


def read_spectrum(path):
    """
    Read a 1D spectrum from a KOA/LRIS extracted FITS file.

    Supports binary tables with wave/flux/mask or image spectra with a simple
    FITS wavelength WCS. The primary FITS header is also scanned for Keck
    program metadata (PROGID/PROGPI/PROGTITL/KOAID/...) and stored under
    'header_meta'.
    """
    with fits.open(path, memmap=False) as hdul:
        # 先采集来源元数据 (即使 spectrum 部分提取失败也能保留 header)
        header_meta = _extract_keck_header_meta(hdul)
        for hdu in hdul:
            data = hdu.data
            if data is None:
                continue

            if hasattr(data, 'names') and data.names:
                wcol = _find_column(data, ['wave', 'wavelength', 'lambda',
                                           'lam', 'OPT_WAVE', 'BOX_WAVE',
                                           'wave_grid_mid', 'wave_grid'])
                fcol = _find_column(data, ['flux', 'flam',
                                           'OPT_FLAM', 'BOX_FLAM',
                                           'counts',
                                           'OPT_COUNTS', 'BOX_COUNTS'])
                if wcol is None or fcol is None:
                    continue
                wave = np.asarray(data[wcol], dtype=float)
                flux = np.asarray(data[fcol], dtype=float)
                ecol = _find_column(
                    data,
                    ['error', 'err', 'sigma', 'flux_err', 'flam_sig',
                     'opt_flam_sig', 'box_flam_sig', 'opt_counts_sig',
                     'box_counts_sig', 'ivar', 'inverse_var',
                     'opt_flam_ivar', 'box_flam_ivar', 'opt_counts_ivar',
                     'box_counts_ivar'])
                error = np.zeros_like(flux, dtype=float)
                if ecol is not None:
                    vals = np.asarray(data[ecol], dtype=float)
                    if 'ivar' in _clean_key(ecol) or _clean_key(ecol) in (
                            'ivar', 'inverse_var'):
                        error = np.where(vals > 0, 1.0 / np.sqrt(vals), 0.0)
                    else:
                        error = vals
                mcol = _find_column(
                    data,
                    ['mask', 'goodpix', 'flag', 'OPT_MASK', 'BOX_MASK',
                     'OPT_FLAM_MASK', 'BOX_FLAM_MASK', 'OPT_COUNTS_MASK',
                     'BOX_COUNTS_MASK'])
                mask = np.ones_like(flux, dtype=bool)
                if mcol is not None:
                    mask_vals = np.asarray(data[mcol])
                    mask = mask_vals.astype(float) > 0
                break

            arr = np.asarray(data, dtype=float).squeeze()
            if arr.ndim != 1:
                continue
            header = hdu.header
            crval = _as_float(header.get('CRVAL1'))
            cdelt = _as_float(header.get('CDELT1', header.get('CD1_1')))
            crpix = _as_float(header.get('CRPIX1', 1.0), default=1.0)
            if not np.isfinite(crval) or not np.isfinite(cdelt):
                wave = np.arange(len(arr), dtype=float)
            else:
                pix = np.arange(len(arr), dtype=float) + 1.0
                wave = crval + (pix - crpix) * cdelt
            flux = arr
            error = np.zeros_like(flux, dtype=float)
            mask = np.ones_like(flux, dtype=bool)
            fcol = 'image'
            break
        else:
            raise ValueError(f'No readable 1D spectrum found in {path}')

    valid = np.isfinite(wave) & np.isfinite(flux) & (wave > 0) & mask
    order = np.argsort(wave[valid])
    wave = wave[valid][order]
    flux = flux[valid][order]
    error = error[valid][order] if error is not None else np.zeros_like(flux)
    flux_col = str(fcol)
    clean_flux_col = _clean_key(flux_col)
    flux_unit = ('erg s^-1 cm^-2 A^-1'
                 if ('flam' in clean_flux_col or clean_flux_col == 'flux')
                 else 'counts')
    return {
        'path': path,
        'filename': os.path.basename(path),
        'wavelength': wave,
        'flux': flux,
        'error': error,
        'flux_column': flux_col,
        'flux_unit': flux_unit,
        'n_points': int(len(wave)),
        'wave_min': float(np.nanmin(wave)) if len(wave) else np.nan,
        'wave_max': float(np.nanmax(wave)) if len(wave) else np.nan,
        'flux_median': float(np.nanmedian(flux)) if len(flux) else np.nan,
        'header_meta': header_meta,
    }


def _interp_median_ratio(ref_wave, ref_flux, wave, flux):
    lo = max(np.nanmin(ref_wave), np.nanmin(wave))
    hi = min(np.nanmax(ref_wave), np.nanmax(wave))
    if not np.isfinite(lo + hi) or hi <= lo:
        return np.nan
    mask = (ref_wave >= lo) & (ref_wave <= hi) & np.isfinite(ref_flux)
    if mask.sum() < 20:
        return np.nan
    other = np.interp(ref_wave[mask], wave, flux, left=np.nan, right=np.nan)
    good = np.isfinite(other) & (other != 0) & np.isfinite(ref_flux[mask])
    if good.sum() < 20:
        return np.nan
    ratio = np.nanmedian(ref_flux[mask][good] / other[good])
    return ratio if np.isfinite(ratio) and ratio > 0 else np.nan


def combine_spectra(spectra, resample_step=None, scale_to_overlap=True):
    """Scale overlapping spectra and stack them onto a common wavelength grid."""
    spectra = [s for s in spectra if s and len(s.get('wavelength', [])) > 0]
    if not spectra:
        return None
    resample_step = config.KOA_RESAMPLE_STEP_A if resample_step is None else resample_step

    ref_wave = np.asarray(spectra[0]['wavelength'], dtype=float)
    ref_flux = np.asarray(spectra[0]['flux'], dtype=float)

    all_wave, all_flux, all_err = [], [], []
    scales = []
    for spec in spectra:
        wave = np.asarray(spec['wavelength'], dtype=float)
        flux = np.asarray(spec['flux'], dtype=float)
        err = np.asarray(spec.get('error', np.zeros_like(flux)), dtype=float)
        scale = 1.0
        if scale_to_overlap and spec is not spectra[0]:
            ratio = _interp_median_ratio(ref_wave, ref_flux, wave, flux)
            if np.isfinite(ratio):
                scale = float(ratio)
        spec['scale_applied'] = scale
        scales.append(scale)
        all_wave.append(wave)
        all_flux.append(flux * scale)
        all_err.append(err * abs(scale))

    wave = np.concatenate(all_wave)
    flux = np.concatenate(all_flux)
    err = np.concatenate(all_err)
    good = np.isfinite(wave) & np.isfinite(flux) & (wave > 0)
    if good.sum() < 2:
        return None
    wave, flux, err = wave[good], flux[good], err[good]

    if resample_step is None or resample_step <= 0:
        idx = np.argsort(wave)
        return wave[idx], flux[idx], err[idx], np.ones(idx.sum(), dtype=int), scales

    wmin = np.floor(np.nanmin(wave) / resample_step) * resample_step
    wmax = np.ceil(np.nanmax(wave) / resample_step) * resample_step
    edges = np.arange(wmin, wmax + resample_step, resample_step)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bins = np.digitize(wave, edges) - 1
    out_f = np.full_like(centers, np.nan, dtype=float)
    out_e = np.full_like(centers, np.nan, dtype=float)
    n = np.zeros_like(centers, dtype=int)

    for i in range(len(centers)):
        m = bins == i
        if not np.any(m):
            continue
        vals = flux[m]
        out_f[i] = np.nanmedian(vals)
        n[i] = int(np.isfinite(vals).sum())
        e = err[m]
        e = e[np.isfinite(e) & (e > 0)]
        if len(e):
            out_e[i] = float(np.sqrt(np.nansum(e ** 2)) / len(e))
        elif n[i] > 1:
            mad = np.nanmedian(np.abs(vals - out_f[i]))
            out_e[i] = float(1.4826 * mad / np.sqrt(n[i]))
        else:
            out_e[i] = 0.0

    keep = np.isfinite(out_f) & (n > 0)
    return centers[keep], out_f[keep], out_e[keep], n[keep], scales


def _select_records(records, prefer_coadds=True):
    if prefer_coadds:
        coadds = [r for r in records if r.get('is_coadd')]
        if coadds:
            return coadds
    return records


def query_spectrum(ra=None, dec=None, target=None, local_root=None,
                   radius_arcsec=None, prefer_coadds=True,
                   resample_step=None, scale_to_overlap=True,
                   match_all=False):
    """
    Query local KOA extracted spectra and return a standard spectrum dict.

    This intentionally does not reduce raw KOA frames.  It consumes 1D products
    already extracted from KOA/Keck data, which is the format present in
    /Users/ljm/Desktop/DWD/speutrem.
    """
    records = match_local_spectra(
        ra=ra, dec=dec, target=target, root=local_root,
        radius_arcsec=radius_arcsec, include_coadds=True,
        include_individual=True, match_all=match_all)
    if not records:
        return None

    selected = _select_records(records, prefer_coadds=prefer_coadds)
    spectra = []
    for rec in selected:
        try:
            spec = read_spectrum(rec['path'])
        except Exception:
            continue
        spec.update(rec)
        spectra.append(spec)
    if not spectra:
        return None

    combined = combine_spectra(
        spectra, resample_step=resample_step,
        scale_to_overlap=scale_to_overlap)
    if combined is None:
        return None
    wave, flux, error, n_contrib, scales = combined

    mjds = np.array([_as_float(s.get('mjd')) for s in spectra], dtype=float)
    mjds = mjds[np.isfinite(mjds)]
    obs_ids = sorted(set(_as_str(s.get('obs_id')) for s in spectra
                         if _as_str(s.get('obs_id'))))
    arms = sorted(set(_as_str(s.get('arm')) for s in spectra
                      if _as_str(s.get('arm'))))

    prov = _aggregate_keck_provenance(
        spectra, ra=ra, dec=dec, target=target,
        obs_mjd=(float(np.nanmin(mjds)) if len(mjds) else None))

    result = {
        'survey': 'KOA_LRIS',
        'instrument': prov.get('instrument') or 'LRIS',
        'source': 'local_extracted_1d',
        'local_root': os.path.abspath(local_root or config.KOA_LOCAL_ROOT),
        'target': target or _as_str(spectra[0].get('target')),
        'ra': float(ra) if ra is not None else _as_float(spectra[0].get('ra')),
        'dec': float(dec) if dec is not None else _as_float(spectra[0].get('dec')),
        'wavelength': wave,
        'flux': flux,
        'error': error,
        'n_contributors': n_contrib,
        'n_files': len(spectra),
        'n_matched_files': len(records),
        'obs_ids': ','.join(obs_ids),
        'arms': ','.join(arms),
        'obs_mjd': float(np.nanmedian(mjds)) if len(mjds) else np.nan,
        'obs_mjd_min': float(np.nanmin(mjds)) if len(mjds) else np.nan,
        'obs_mjd_max': float(np.nanmax(mjds)) if len(mjds) else np.nan,
        'spectra': spectra,
        'matched_records': records,
        'scale_to_overlap': bool(scale_to_overlap),
        'provenance': prov,
        'flux_unit': spectra[0].get('flux_unit', 'counts') if spectra else 'counts',
        'flux_columns': ','.join(sorted(set(
            _as_str(s.get('flux_column')) for s in spectra
            if _as_str(s.get('flux_column'))))),
    }
    return result


def _aggregate_keck_provenance(spectra, ra=None, dec=None, target=None,
                               obs_mjd=None):
    """从多个 1D 文件的 header_meta 汇总 KOA/Keck program 元数据。"""
    from collections import Counter

    def _pick(key):
        vals = [_as_str(s.get('header_meta', {}).get(key)) for s in spectra]
        vals = [v for v in vals if v]
        if not vals:
            return ''
        # 取出现次数最多的; 并列则取第一个
        return Counter(vals).most_common(1)[0][0]

    progid = _pick('progid')
    progpi = _pick('progpi')
    progtitl = _pick('progtitl')
    proginst = _pick('proginst')
    semester = _pick('semester')
    instrume = _pick('instrume') or 'LRIS'
    detector = _pick('detector')
    graname = _pick('graname')
    slitname = _pick('slitname')
    dichname = _pick('dichname')
    targname = _pick('targname') or _as_str(target)
    observer = _pick('observer')

    # 收集所有不同的 KOAID (per exposure) 与 PROGID
    koaids = sorted({_as_str(s.get('header_meta', {}).get('koaid'))
                     for s in spectra
                     if _as_str(s.get('header_meta', {}).get('koaid'))})
    progids = sorted({_as_str(s.get('header_meta', {}).get('progid'))
                      for s in spectra
                      if _as_str(s.get('header_meta', {}).get('progid'))})

    # 累计曝光
    total_exptime = 0.0
    n_with_exp = 0
    for s in spectra:
        m = s.get('header_meta', {})
        if isinstance(m.get('exptime'), (int, float)) and np.isfinite(m['exptime']):
            total_exptime += float(m['exptime'])
            n_with_exp += 1
        elif isinstance(s.get('exptime'), (int, float)) and np.isfinite(s['exptime']):
            total_exptime += float(s['exptime'])
            n_with_exp += 1
    exptime_total = total_exptime if n_with_exp else None

    survey_tag = 'KOA_' + (instrume or 'LRIS').upper().split('/')[0]
    prov = utils.build_provenance(
        survey_tag, ra=ra, dec=dec,
        override={
            'instrument': instrume,
            'detector': detector,
            'grating': graname,
            'filter': dichname,
            'proposal_id': progid,            # Keck 用字符串 PROGID
            'proposal_pi': progpi,
            'proposal_type': proginst,
            'title': progtitl,
            'obs_id': koaids[0] if koaids else '',
            'target_name': targname,
            'obs_mjd': obs_mjd,
            'exptime_s': exptime_total,
            # extras
            'semester': semester,
            'slit': slitname,
            'observer': observer,
            'koa_ids': koaids,
            'proposal_ids': progids,
            'n_exposures': len(spectra),
        })
    return prov


def _require_pykoa():
    try:
        from pykoa.koa import Koa
    except Exception as exc:
        raise RuntimeError(
            'pykoa is required for online KOA query/download. Install it with '
            '`pip install pykoa`, then retry. Local extracted spectra can still '
            'be read without pykoa.') from exc
    return Koa


def _koa_astropy_format(fmt=None, path=None):
    fmt = _clean_key(fmt or '')
    if not fmt and path:
        suffix = os.path.splitext(str(path))[1].lower().lstrip('.')
        fmt = suffix or 'ipac'
    return _KOA_TABLE_FORMATS.get(fmt, fmt or 'ascii.ipac')


def read_koa_table(path, fmt=None):
    """Read a KOA metadata table produced by PyKOA."""
    from astropy.table import Table

    candidates = []
    if fmt:
        candidates.append(_koa_astropy_format(fmt))
    candidates.extend([
        _koa_astropy_format(path=path),
        'ascii.ipac', 'ascii.csv', 'ascii.tab', 'votable', 'fits'
    ])
    tried = []
    for table_format in candidates:
        if table_format in tried:
            continue
        tried.append(table_format)
        try:
            return Table.read(path, format=table_format)
        except Exception:
            continue

    try:
        import pandas as pd
        return Table.from_pandas(pd.read_csv(path))
    except Exception as exc:
        raise ValueError(f'Could not read KOA table: {path}') from exc


def write_koa_table(table, path, fmt='ipac'):
    """Write a KOA metadata table in a format accepted by PyKOA.download."""
    table.write(path, format=_koa_astropy_format(fmt), overwrite=True)
    return path


def _row_value(row, column, default=''):
    if not column:
        return default
    try:
        return row[column]
    except Exception:
        return default


def _row_text(row, column):
    return _as_str(_row_value(row, column, ''))


def _science_row_mask(table):
    type_col = _find_table_column(
        table, ['koaimtyp', 'imagetyp', 'imgtyp', 'frametype', 'obstype',
                'data_type'])
    if type_col is None:
        return np.ones(len(table), dtype=bool)

    good = []
    bad_tokens = ('bias', 'flat', 'arc', 'dark', 'lamp', 'focus', 'test')
    for row in table:
        text = _row_text(row, type_col).lower()
        if not text:
            good.append(True)
            continue
        is_science = ('science' in text or 'object' in text or text == 'sci')
        is_bad = any(tok in text for tok in bad_tokens)
        good.append(is_science and not is_bad)
    return np.array(good, dtype=bool)


def filter_koa_table(table, science_only=True, target=None, row_limit=None):
    """
    Keep the rows most useful for downloading a science spectrum.

    This filter is intentionally conservative: if KOA does not expose an image
    type column for an instrument, rows are kept instead of silently discarded.
    """
    keep = np.ones(len(table), dtype=bool)
    if science_only:
        keep &= _science_row_mask(table)

    if target:
        name_col = _find_table_column(
            table, ['targname', 'target', 'object', 'ofname', 'koaid'])
        if name_col is not None:
            needle = re.sub(r'[^a-z0-9]+', '', str(target).lower())
            matched = []
            for row in table:
                hay = re.sub(r'[^a-z0-9]+', '',
                             _row_text(row, name_col).lower())
                matched.append((not needle) or needle in hay or hay in needle)
            matched = np.array(matched, dtype=bool)
            if np.any(matched):
                keep &= matched

    out = table[keep]
    if row_limit is not None and int(row_limit) > 0:
        out = out[:int(row_limit)]
    return out


def query_koa_metadata(ra, dec, instrument='lris', radius_arcsec=8.0,
                       outpath=None, fmt='ipac', maxrec=2000,
                       cookiepath=None, overwrite=True):
    """
    Query official KOA metadata through PyKOA.

    Public metadata can be queried without a cookie. Proprietary downloads
    require KOA credentials/cookies handled by PyKOA.
    """
    Koa = _require_pykoa()
    instrument = str(instrument).lower()
    if instrument not in KOA_INSTRUMENTS:
        raise ValueError(f'Unsupported/unknown KOA instrument: {instrument}')

    suffix = 'tbl' if _clean_key(fmt) in ('ipac', 'tbl') else _clean_key(fmt)
    outpath = outpath or os.path.join(tempfile.gettempdir(),
                                      f'koa_{instrument}_query.{suffix}')
    if overwrite and os.path.exists(outpath):
        os.remove(outpath)

    radius_deg = float(radius_arcsec) / 3600.0
    pos = f'circle {float(ra)} {float(dec)} {radius_deg}'
    kwargs = {'format': fmt, 'maxrec': maxrec}
    if cookiepath:
        kwargs['cookiepath'] = cookiepath
    try:
        Koa.query_position(instrument, pos, outpath, **kwargs)
    except TypeError:
        kwargs.pop('cookiepath', None)
        Koa.query_position(instrument, pos, outpath, **kwargs)
    if not os.path.exists(outpath):
        raise RuntimeError(
            f'KOA query did not create an output table: {outpath}. '
            'Check network access, KOA availability, and credentials/cookies '
            'for proprietary data.')
    return outpath


def download_koa_files(table_path, output_dir, fmt=None, lev0file=True,
                       calibfile=False, lev1file=False, cookiepath=None):
    """
    Download KOA files listed in a PyKOA metadata table.

    The table must contain the official PyKOA columns required by KOA download
    (normally instrume, koaid and filehand).
    """
    Koa = _require_pykoa()
    fmt = fmt or ('csv' if str(table_path).lower().endswith('.csv') else 'ipac')
    utils.ensure_dir(output_dir)

    before = set(glob.glob(os.path.join(output_dir, '**', '*'),
                           recursive=True))
    kwargs = {
        'lev0file': int(bool(lev0file)),
        'calibfile': int(bool(calibfile)),
        'lev1file': int(bool(lev1file)),
    }
    if cookiepath:
        kwargs['cookiepath'] = cookiepath
    try:
        Koa.download(table_path, fmt, output_dir, **kwargs)
    except TypeError:
        kwargs.pop('cookiepath', None)
        Koa.download(table_path, fmt, output_dir, **kwargs)

    after = set(glob.glob(os.path.join(output_dir, '**', '*'),
                          recursive=True))
    files = sorted(p for p in (after - before) if os.path.isfile(p))
    if not files:
        files = sorted(p for p in after if os.path.isfile(p))
    return files


def _default_koa_work_dir(target=None, ra=None, dec=None):
    root = os.path.join(config.CACHE_DIR, 'koa')
    if target:
        safe = re.sub(r'[^A-Za-z0-9_.+-]+', '_', str(target)).strip('_')
    else:
        safe = f'ra{float(ra):.6f}_dec{float(dec):.6f}'
    return os.path.join(root, safe or 'target')


def prepare_koa_download(ra=None, dec=None, target=None,
                         instruments=('lris',), radius_arcsec=None,
                         work_dir=None, fmt='ipac', maxrec=2000,
                         science_only=True, row_limit=None, download=True,
                         lev0file=True, calibfile=True, lev1file=False,
                         cookiepath=None, allow_name_resolve=False):
    """
    Query official KOA metadata and optionally download the selected files.

    This function handles the archive side only. Turning raw LRIS frames into a
    science-grade 1D spectrum still requires a reducer such as PypeIt; extracted
    products are standardized by query_spectrum/download_and_extract_spectrum.
    """
    ra, dec = resolve_target_coordinates(
        target=target, ra=ra, dec=dec,
        allow_name_resolve=allow_name_resolve)
    radius_arcsec = (config.KOA_SEARCH_RADIUS_ARCSEC if radius_arcsec is None
                     else radius_arcsec)
    if isinstance(instruments, str):
        instruments = [instruments]
    work_dir = os.path.abspath(work_dir or _default_koa_work_dir(target, ra, dec))
    meta_dir = os.path.join(work_dir, 'metadata')
    raw_dir = os.path.join(work_dir, 'download')
    utils.ensure_dir(meta_dir)
    utils.ensure_dir(raw_dir)

    products = []
    for instrument in instruments:
        inst = str(instrument).lower()
        suffix = 'tbl' if _clean_key(fmt) in ('ipac', 'tbl') else _clean_key(fmt)
        meta_path = os.path.join(meta_dir, f'{inst}_query.{suffix}')
        selected_path = os.path.join(meta_dir, f'{inst}_selected.{suffix}')
        query_koa_metadata(
            ra, dec, instrument=inst, radius_arcsec=radius_arcsec,
            outpath=meta_path, fmt=fmt, maxrec=maxrec,
            cookiepath=cookiepath, overwrite=True)
        table = read_koa_table(meta_path, fmt=fmt)
        selected = filter_koa_table(
            table, science_only=science_only, target=target,
            row_limit=row_limit)
        write_koa_table(selected, selected_path, fmt=fmt)

        downloaded = []
        if download and len(selected) > 0:
            inst_download_dir = os.path.join(raw_dir, inst)
            downloaded = download_koa_files(
                selected_path, inst_download_dir, fmt=fmt,
                lev0file=lev0file, calibfile=calibfile, lev1file=lev1file,
                cookiepath=cookiepath)

        products.append({
            'instrument': inst,
            'metadata_path': meta_path,
            'selected_path': selected_path,
            'n_metadata_rows': int(len(table)),
            'n_selected_rows': int(len(selected)),
            'download_dir': os.path.join(raw_dir, inst),
            'downloaded_files': downloaded,
        })

    return {
        'target': target,
        'ra': float(ra),
        'dec': float(dec),
        'radius_arcsec': float(radius_arcsec),
        'work_dir': work_dir,
        'products': products,
    }


def run_pypeit_reduction(pypeit_file, redux_path=None, extra_args=None,
                         command='run_pypeit'):
    """
    Run an existing PypeIt reduction file and return the subprocess result.

    The all-in-one KOA workflow can download official files automatically, but
    a science-grade LRIS extraction requires a checked PypeIt setup. This helper
    executes that setup when PypeIt is installed and a .pypeit file is supplied.
    """
    exe = shutil.which(command)
    if exe is None:
        raise RuntimeError(
            f'{command} was not found. Install PypeIt and create/check a '
            '.pypeit setup before reducing raw KOA frames.')
    cmd = [exe, pypeit_file]
    if redux_path:
        cmd.extend(['-r', redux_path])
    if extra_args:
        cmd.extend([str(a) for a in extra_args])
    env = os.environ.copy()
    env.setdefault('MPLCONFIGDIR', '/tmp')
    cwd = os.path.dirname(os.path.abspath(pypeit_file)) or None
    return subprocess.run(
        cmd, check=True, capture_output=True, text=True, env=env, cwd=cwd)


def _fits_date(path):
    try:
        with fits.open(path, memmap=False) as hdul:
            value = hdul[0].header.get('DATE-OBS', '')
    except Exception:
        return ''
    return str(value)


def _lris_spectrograph_for_arm(arm, raw_files):
    """Infer the correct PypeIt LRIS spectrograph identifier."""
    dates = [_fits_date(p) for p in raw_files]
    date = sorted(d for d in dates if d)[:1]
    date = date[0] if date else ''
    compact = date.replace('-', '')

    arm = str(arm).lower()
    if compact and compact < '20090501':
        return 'keck_lris_blue_orig' if arm == 'blue' else 'keck_lris_red_orig'
    if arm == 'red' and compact and compact >= '20220501':
        return 'keck_lris_red_mark4'
    return 'keck_lris_blue' if arm == 'blue' else 'keck_lris_red'


def _find_lris_raw_files(raw_root):
    paths = sorted(glob.glob(os.path.join(raw_root, '**', '*.fits'),
                             recursive=True), key=_natural_key)
    paths += sorted(glob.glob(os.path.join(raw_root, '**', '*.fits.gz'),
                              recursive=True), key=_natural_key)
    blue, red = [], []
    for path in paths:
        base = os.path.basename(path)
        if base.startswith('LB.'):
            blue.append(path)
        elif base.startswith('LR.'):
            red.append(path)
    return {'blue': blue, 'red': red}


def _fits_data_readable(path):
    try:
        with fits.open(path, memmap=False, ignore_missing_end=True) as hdul:
            for hdu in hdul:
                if hdu.data is not None:
                    # Accessing the shape catches truncated KOA downloads before
                    # PypeIt spends time setting up an unusable file.
                    _ = hdu.data.shape
        return True, ''
    except Exception as exc:
        return False, f'{type(exc).__name__}: {exc}'


def _normalize_lris_header_for_pypeit(header):
    """Normalize KOA/LRIS header cards that commonly confuse PypeIt typing."""
    koaimtyp = _as_str(header.get('KOAIMTYP', header.get('OBSTYPE', ''))).lower()
    if 'object' in koaimtyp or 'science' in koaimtyp:
        header['OBSTYPE'] = ('object', 'Normalized for PypeIt')
        header['TRAPDOOR'] = ('open', 'Normalized for PypeIt science')
        for key in LRIS_LAMP_KEYS:
            if key in header:
                header[key] = ('off', 'Normalized for PypeIt science')
    elif 'flat' in koaimtyp:
        header['OBSTYPE'] = ('flat', 'Normalized for PypeIt')
        header['TRAPDOOR'] = ('closed', 'Normalized for PypeIt flat')
        for key in LRIS_LAMP_KEYS:
            if key in header:
                header[key] = ('on' if key == 'HALOGEN' else 'off',
                               'Normalized for PypeIt flat')
    elif 'arc' in koaimtyp or 'lamp' in koaimtyp:
        header['OBSTYPE'] = ('arc', 'Normalized for PypeIt')
        header['TRAPDOOR'] = ('closed', 'Normalized for PypeIt arc')
        for key in LRIS_LAMP_KEYS:
            if key in header:
                header[key] = ('on' if key in ('MERCURY', 'NEON', 'ARGON',
                                                'CADMIUM', 'ZINC')
                               else 'off', 'Normalized for PypeIt arc')
    elif 'bias' in koaimtyp or 'dark' in koaimtyp:
        header['OBSTYPE'] = ('bias', 'Normalized for PypeIt bias')
        header['TRAPDOOR'] = ('closed', 'Normalized for PypeIt bias')
        for key in LRIS_LAMP_KEYS:
            if key in header:
                header[key] = ('off', 'Normalized for PypeIt bias')

    if 'EXPTIME' not in header and header.get('ELAPTIME') is not None:
        header['EXPTIME'] = (header.get('ELAPTIME'),
                             'Copied from ELAPTIME for PypeIt')


def normalize_lris_headers_for_pypeit(raw_root, output_root=None,
                                      in_place=False, overwrite=True,
                                      skip_bad=True):
    """
    Prepare KOA/LRIS raw files for PypeIt frame typing.

    Some KOA LRIS raw files have KOAIMTYP=object but OBSTYPE/lamp cards that
    make PypeIt treat science frames as unusable calibration frames.  This
    helper creates a normalized staging tree by default, or edits headers
    in-place when in_place=True.  Pixel data are not changed.
    """
    raw_root = os.path.abspath(raw_root)
    output_root = raw_root if in_place else os.path.abspath(
        output_root or os.path.join(os.path.dirname(raw_root),
                                    'pypeit_raw_normalized'))
    if not in_place:
        utils.ensure_dir(output_root)

    rows = []
    for path in _find_lris_raw_files(raw_root)['blue'] + _find_lris_raw_files(raw_root)['red']:
        rel = os.path.relpath(path, raw_root)
        outpath = path if in_place else os.path.join(output_root, rel)
        readable, read_error = _fits_data_readable(path)
        if skip_bad and not readable:
            rows.append({
                'input_path': path,
                'output_path': '',
                'status': 'skipped_bad_fits',
                'error': read_error,
            })
            continue
        if not in_place:
            utils.ensure_dir(os.path.dirname(outpath))
            if overwrite or not os.path.exists(outpath):
                shutil.copy2(path, outpath)

        try:
            with fits.open(outpath, mode='update', memmap=False,
                           checksum=False, do_not_scale_image_data=True,
                           ignore_missing_end=True) as hdul:
                for hdu in hdul:
                    for key in ('CHECKSUM', 'DATASUM'):
                        if key in hdu.header:
                            del hdu.header[key]
                _normalize_lris_header_for_pypeit(hdul[0].header)
                hdul.flush(output_verify='silentfix')
            rows.append({
                'input_path': path,
                'output_path': outpath,
                'status': 'normalized',
                'error': '',
            })
        except Exception as exc:
            rows.append({
                'input_path': path,
                'output_path': outpath,
                'status': 'error',
                'error': f'{type(exc).__name__}: {exc}',
            })
    return {
        'raw_root': output_root,
        'in_place': bool(in_place),
        'rows': rows,
        'n_normalized': sum(r['status'] == 'normalized' for r in rows),
        'n_skipped_bad': sum(r['status'] == 'skipped_bad_fits' for r in rows),
        'n_errors': sum(r['status'] == 'error' for r in rows),
    }


def repair_lris_pypeit_frame_types(pypeit_file, min_science_exptime=30.0):
    """Mark KOA object rows left as None by pypeit_setup as science frames."""
    if not pypeit_file or not os.path.exists(pypeit_file):
        return {'pypeit_file': pypeit_file, 'n_repaired': 0}

    with open(pypeit_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # First pass: find a default calibration group for each data block.
    block_defaults = {}
    found_types = set()
    in_data = False
    block_index = -1
    header = None
    calib_idx = None
    for line in lines:
        stripped = line.strip()
        if stripped == 'data read':
            in_data = True
            block_index += 1
            header = None
            calib_idx = None
            block_defaults[block_index] = None
            continue
        if stripped == 'data end':
            in_data = False
            header = None
            calib_idx = None
            continue
        if not in_data:
            continue
        if 'filename' in line and 'frametype' in line:
            header = [part.strip() for part in line.split('|')]
            try:
                calib_idx = header.index('calib')
            except ValueError:
                calib_idx = None
            continue
        if (header is None or calib_idx is None or not stripped
                or line.lstrip().startswith('#') or '|' not in line):
            continue
        parts = line.rstrip('\n').split('|')
        if len(parts) != len(header):
            continue
        found_types.update(
            token.strip() for token in parts[header.index('frametype')].split(',')
            if token.strip())
        calib_text = parts[calib_idx].strip()
        if calib_text and calib_text.lower() != 'none':
            block_defaults[block_index] = calib_text

    # Second pass: repair frame types and assign the default calib group.
    in_data = False
    data_path = None
    header = None
    frame_idx = exp_idx = file_idx = calib_idx = None
    block_index = -1
    repaired = 0
    out = []
    for line in lines:
        stripped = line.strip()
        if stripped == 'data read':
            in_data = True
            block_index += 1
            data_path = None
            header = None
            frame_idx = exp_idx = file_idx = calib_idx = None
        elif stripped == 'data end':
            in_data = False
            data_path = None
            header = None
        elif in_data and stripped.startswith('path '):
            data_path = stripped.split(None, 1)[1]
        elif in_data and 'filename' in line and 'frametype' in line:
            header = [part.strip() for part in line.split('|')]
            try:
                file_idx = header.index('filename')
                frame_idx = header.index('frametype')
                exp_idx = header.index('exptime')
                calib_idx = header.index('calib')
            except ValueError:
                file_idx = frame_idx = exp_idx = calib_idx = None
        elif (in_data and header and frame_idx is not None
              and line.strip() and not line.lstrip().startswith('#')
              and '|' in line):
            parts = line.rstrip('\n').split('|')
            if len(parts) == len(header):
                frame_type = parts[frame_idx].strip()
                filename = parts[file_idx].strip()
                try:
                    exptime = float(parts[exp_idx].strip())
                except Exception:
                    exptime = np.nan
                is_koa_object = False
                if data_path:
                    raw_path = os.path.join(data_path, filename)
                    try:
                        hdr = fits.getheader(raw_path, 0)
                        is_koa_object = 'object' in _as_str(
                            hdr.get('KOAIMTYP', hdr.get('OBSTYPE', ''))).lower()
                    except Exception:
                        is_koa_object = False
                if (frame_type in ('None', '', 'none')
                        and is_koa_object
                        and np.isfinite(exptime)
                        and exptime >= float(min_science_exptime)):
                    parts[frame_idx] = 'science'.center(len(parts[frame_idx]))
                    default_calib = block_defaults.get(block_index)
                    if calib_idx is not None and default_calib is not None:
                        parts[calib_idx] = default_calib.center(len(parts[calib_idx]))
                    line = '|'.join(parts) + '\n'
                    repaired += 1
        out.append(line)

    if repaired:
        with open(pypeit_file, 'w', encoding='utf-8') as f:
            f.writelines(out)
        lines = out

    injected = []
    if 'bias' not in found_types and '[baseprocess]' not in ''.join(lines):
        inject_at = None
        for idx, line in enumerate(lines):
            if line.startswith('# Setup'):
                inject_at = idx
                break
        if inject_at is not None:
            injected.extend(['use_biasimage = False'])
            lines = (lines[:inject_at]
                     + ['[baseprocess]\n', '    use_biasimage = False\n', '\n']
                     + lines[inject_at:])
    if injected:
        with open(pypeit_file, 'w', encoding='utf-8') as f:
            f.writelines(lines)

    return {
        'pypeit_file': pypeit_file,
        'n_repaired': repaired,
        'injected_cfg': ','.join(injected),
    }


def _pypeit_search_roots_for_arm(files, arm):
    """Build the `pypeit_setup -r` roots for one LRIS arm."""
    if not files:
        return []
    prefix = 'LB' if str(arm).lower() == 'blue' else 'LR'
    roots = []
    for path in sorted(set(files), key=_natural_key):
        root = os.path.join(os.path.dirname(path), prefix)
        if root not in roots:
            roots.append(root)
    return roots


def setup_pypeit_lris(raw_root, setup_dir=None, cfg_split='all',
                      overwrite=True, run=False, extra_setup_args=None,
                      extra_run_args=None, normalize_headers=False,
                      normalize_in_place=False, normalized_raw_root=None,
                      min_science_exptime=30.0):
    """
    Generate LRIS PypeIt reduction files from KOA-downloaded raw data.

    This follows the PypeIt LRIS convention of treating blue and red arms as
    separate spectrographs. It returns generated .pypeit files and, when run is
    True, executes run_pypeit on each file.
    """
    setup_exe = shutil.which('pypeit_setup')
    if setup_exe is None:
        raise RuntimeError(
            'pypeit_setup was not found. Install PypeIt before reducing raw '
            'KOA LRIS frames. The KOA download and local 1D standardization '
            'steps do not require PypeIt.')
    setup_dir = os.path.abspath(setup_dir or os.path.join(raw_root, 'pypeit_setup'))
    utils.ensure_dir(setup_dir)

    normalization = None
    setup_raw_root = raw_root
    if normalize_headers:
        normalization = normalize_lris_headers_for_pypeit(
            raw_root,
            output_root=normalized_raw_root or os.path.join(
                setup_dir, 'pypeit_raw'),
            in_place=normalize_in_place)
        setup_raw_root = normalization['raw_root']

    raw_by_arm = _find_lris_raw_files(setup_raw_root)
    results = []
    for arm, files in raw_by_arm.items():
        if not files:
            continue
        spectrograph = _lris_spectrograph_for_arm(arm, files)
        roots = _pypeit_search_roots_for_arm(files, arm)
        if not roots:
            continue
        arm_setup_dir = os.path.join(setup_dir, arm)
        utils.ensure_dir(arm_setup_dir)
        cmd = [
            setup_exe, '-s', spectrograph, '-r', *roots,
            '-d', arm_setup_dir, '-c', str(cfg_split)
        ]
        if overwrite:
            cmd.append('-o')
        if extra_setup_args:
            cmd.extend([str(a) for a in extra_setup_args])
        env = os.environ.copy()
        env.setdefault('MPLCONFIGDIR', '/tmp')
        try:
            setup_proc = subprocess.run(
                cmd, check=True, capture_output=True, text=True, env=env)
        except subprocess.CalledProcessError as exc:
            detail = '\n'.join(x for x in (
                exc.stdout[-4000:] if exc.stdout else '',
                exc.stderr[-4000:] if exc.stderr else '') if x)
            raise RuntimeError(
                f'pypeit_setup failed for {arm} arm with command: '
                f'{" ".join(cmd)}\n{detail}') from exc

        pypeit_files = sorted(glob.glob(
            os.path.join(arm_setup_dir, '**', '*.pypeit'), recursive=True))
        repairs = [
            repair_lris_pypeit_frame_types(
                pypeit_file, min_science_exptime=min_science_exptime)
            for pypeit_file in pypeit_files
        ]
        run_results = []
        if run:
            for pypeit_file in pypeit_files:
                proc = run_pypeit_reduction(
                    pypeit_file, extra_args=extra_run_args)
                run_results.append({
                    'pypeit_file': pypeit_file,
                    'stdout': proc.stdout,
                    'stderr': proc.stderr,
                })
        results.append({
            'arm': arm,
            'spectrograph': spectrograph,
            'raw_root': ';'.join(roots),
            'n_raw_files': len(files),
            'setup_dir': arm_setup_dir,
            'pypeit_files': pypeit_files,
            'setup_stdout': setup_proc.stdout,
            'setup_stderr': setup_proc.stderr,
            'frame_type_repairs': repairs,
            'normalization': normalization,
            'run_results': run_results,
        })
    if not results:
        raise RuntimeError(f'No LRIS raw files found under {raw_root}')
    return results


def download_and_extract_spectrum(ra=None, dec=None, target=None,
                                  instruments=('lris',), radius_arcsec=None,
                                  work_dir=None, extracted_root=None,
                                  download=True, calibfile=True,
                                  lev0file=True, lev1file=False,
                                  row_limit=None, pypeit_file=None,
                                  auto_pypeit=False, pypeit_setup_only=False,
                                  output_dir=None,
                                  prefer_coadds=True, resample_step=None,
                                  scale_to_overlap=True, cookiepath=None,
                                  allow_name_resolve=False):
    """
    One-stop KOA workflow: official query/download plus 1D standardization.

    If extracted_root or the download work_dir already contains 1D FITS products
    (PypeIt/spec1d/coadds or simple wave/flux tables), they are combined into the
    standard astro_toolbox spectrum dict. Raw LRIS frames are not silently
    reduced; provide pypeit_file to run a checked PypeIt reduction first.
    """
    ra, dec = resolve_target_coordinates(
        target=target, ra=ra, dec=dec,
        allow_name_resolve=allow_name_resolve)
    work_dir = os.path.abspath(work_dir or _default_koa_work_dir(target, ra, dec))

    download_info = None
    if download:
        download_info = prepare_koa_download(
            ra=ra, dec=dec, target=target, instruments=instruments,
            radius_arcsec=radius_arcsec, work_dir=work_dir,
            row_limit=row_limit, download=True, lev0file=lev0file,
            calibfile=calibfile, lev1file=lev1file, cookiepath=cookiepath,
            allow_name_resolve=False)

    pypeit_result = None
    if pypeit_file:
        pypeit_result = run_pypeit_reduction(pypeit_file)
    pypeit_setup = None
    pypeit_error = None
    if auto_pypeit:
        raw_root = os.path.join(work_dir, 'download', 'lris')
        try:
            pypeit_setup = setup_pypeit_lris(
                raw_root,
                setup_dir=os.path.join(work_dir, 'pypeit_setup'),
                run=not pypeit_setup_only)
        except Exception as exc:
            pypeit_error = str(exc)

    roots = []
    if extracted_root:
        roots.append(extracted_root)
    roots.append(work_dir)
    roots.extend(os.path.join(work_dir, name) for name in ('Science', 'redux',
                                                           'download'))

    spectrum = None
    for root in roots:
        if not root or not os.path.exists(root):
            continue
        spectrum = query_spectrum(
            ra=ra, dec=dec, target=target, local_root=root,
            radius_arcsec=radius_arcsec, prefer_coadds=prefer_coadds,
            resample_step=resample_step,
            scale_to_overlap=scale_to_overlap)
        if spectrum:
            break

    if spectrum:
        spectrum['source'] = 'koa_download_or_extracted_1d'
        spectrum['koa_download'] = download_info
        spectrum['pypeit_setup'] = pypeit_setup
        spectrum['pypeit_error'] = pypeit_error
        if pypeit_result is not None:
            spectrum['pypeit_stdout'] = pypeit_result.stdout
            spectrum['pypeit_stderr'] = pypeit_result.stderr
        if output_dir:
            utils.ensure_dir(output_dir)
            spectrum['saved_csv'] = save_csv(spectrum, output_dir)
            spectrum['saved_exposures'] = save_exposure_table(spectrum, output_dir)
            spectrum['saved_report'] = save_report(spectrum, output_dir)
            spectrum['saved_plot'] = os.path.join(output_dir, 'koa_spectrum.png')
            plot_spectrum(spectrum, save_path=spectrum['saved_plot'])
        return spectrum

    return {
        'survey': 'KOA',
        'source': 'koa_download_no_extracted_1d',
        'target': target,
        'ra': float(ra),
        'dec': float(dec),
        'work_dir': work_dir,
        'koa_download': download_info,
        'pypeit_setup': pypeit_setup,
        'pypeit_error': pypeit_error,
        'message': ('KOA metadata/download completed, but no readable 1D '
                    'spectrum was found. For LRIS, KOA does not provide level-1 '
                    'spectra; reduce the raw frames with PypeIt or put extracted '
                    '1D FITS products under extracted_root/work_dir.')
    }


def plot_spectrum(result, save_path=None, show_components=True):
    if not result:
        return None
    wave = np.asarray(result['wavelength'], dtype=float)
    flux = np.asarray(result['flux'], dtype=float)
    fig, ax = plt.subplots(figsize=(14, 5.5))

    if show_components:
        for spec in result.get('spectra', []):
            w = np.asarray(spec['wavelength'], dtype=float)
            f = np.asarray(spec['flux'], dtype=float) * spec.get('scale_applied', 1.0)
            color = '#4C78A8' if spec.get('arm') == 'blue' else '#E45756'
            ax.plot(w, f, color=color, lw=0.45, alpha=0.35)

    ax.plot(wave, flux, color='black', lw=0.9, label='KOA/LRIS combined')
    valid = np.isfinite(wave) & np.isfinite(flux)
    if valid.sum() > 10:
        flo, fhi = np.nanpercentile(flux[valid], [1, 99])
        pad = max((fhi - flo) * 0.15, abs(fhi) * 0.02)
        ax.set_ylim(flo - pad, fhi + pad)
        ax.set_xlim(np.nanmin(wave[valid]), np.nanmax(wave[valid]))

    ymin, ymax = ax.get_ylim()
    for name, line_wave in BALMER_LINES.items():
        if np.nanmin(wave) <= line_wave <= np.nanmax(wave):
            ax.axvline(line_wave, color='tab:red', ls=':', lw=0.8, alpha=0.65)
            ax.text(line_wave, ymax - 0.05 * (ymax - ymin), name,
                    rotation=90, va='top', ha='right', fontsize=8,
                    color='tab:red')
    for name, line_wave in TELLURIC_LINES.items():
        if np.nanmin(wave) <= line_wave <= np.nanmax(wave):
            ax.axvline(line_wave, color='0.35', ls='--', lw=0.8, alpha=0.55)
            ax.text(line_wave, ymin + 0.05 * (ymax - ymin), name,
                    rotation=90, va='bottom', ha='left', fontsize=8,
                    color='0.35')

    prov = result.get('provenance', {}) or {}
    inst = prov.get('instrument') or result.get('instrument', 'LRIS')
    grat = prov.get('grating', '')
    head = f"KOA/{inst}"
    if grat:
        head += f"  {grat}"
    if result.get('target'):
        head += f"  {result['target']}"
    title_lines = [head]
    pid_line = ''
    if prov.get('proposal_id'):
        pid_line = f"PROGID {prov['proposal_id']}"
        if prov.get('proposal_pi'):
            pid_line += f"  PI: {prov['proposal_pi']}"
        if prov.get('semester'):
            pid_line += f"  ({prov['semester']})"
    if pid_line:
        title_lines.append(pid_line)
    if prov.get('title'):
        t = prov['title']
        title_lines.append(t if len(t) < 80 else t[:77] + '...')
    if result.get('obs_ids'):
        title_lines.append(f"obs_set={result['obs_ids']}  arms={result.get('arms','')}")
    ax.set_title('\n'.join(title_lines), fontsize=10)
    ax.set_xlabel('Wavelength (A)')
    flux_unit = result.get('flux_unit') or 'counts'
    if str(flux_unit).lower() == 'counts':
        ax.set_ylabel('Flux (PypeIt counts, not flux calibrated)')
    else:
        ax.set_ylabel(r'$f_\lambda$ (erg s$^{-1}$ cm$^{-2}$ A$^{-1}$)')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best', fontsize=9)
    fig.tight_layout()
    if save_path:
        utils.save_and_close(fig, save_path)
    return fig


def save_csv(result, output_dir, filename='koa_spectrum.csv'):
    if not result:
        return None
    import pandas as pd
    df = pd.DataFrame({
        'wavelength_A': result['wavelength'],
        'flux': result['flux'],
        'error': result.get('error', np.zeros_like(result['wavelength'])),
        'n_contributors': result.get(
            'n_contributors', np.ones_like(result['wavelength'], dtype=int)),
    })
    df['flux_unit'] = result.get('flux_unit', '')
    df['flux_columns'] = result.get('flux_columns', '')
    prov = result.get('provenance')
    df = utils.add_provenance_columns(
        df, prov,
        columns=['mission', 'instrument', 'grating', 'proposal_id',
                 'proposal_pi', 'obs_id', 'obs_mjd', 'semester'])
    csv_path = utils.write_csv(df, output_dir, filename)
    if prov:
        prov_name = filename.replace('.csv', '_provenance.json')
        if not prov_name.endswith('.json'):
            prov_name = filename + '_provenance.json'
        utils.write_provenance_json(prov, output_dir, prov_name)
    return csv_path


def save_exposure_table(result, output_dir, filename='koa_exposures.csv'):
    if not result:
        return None
    import pandas as pd
    rows = []
    for spec in result.get('spectra', []):
        rows.append({
            'filename': spec.get('filename'),
            'path': spec.get('path'),
            'obs_id': spec.get('obs_id'),
            'arm': spec.get('arm'),
            'is_coadd': spec.get('is_coadd'),
            'target': spec.get('target'),
            'ra': spec.get('ra'),
            'dec': spec.get('dec'),
            'separation_arcsec': spec.get('separation_arcsec'),
            'mjd': spec.get('mjd'),
            'mjd_min': spec.get('mjd_min'),
            'mjd_max': spec.get('mjd_max'),
            'dateobs': spec.get('dateobs'),
            'exptime': spec.get('exptime'),
            'airmass': spec.get('airmass'),
            'dispname': spec.get('dispname'),
            'decker': spec.get('decker'),
            'binning': spec.get('binning'),
            'dichroic': spec.get('dichroic'),
            'scale_applied': spec.get('scale_applied'),
            'wave_min': spec.get('wave_min'),
            'wave_max': spec.get('wave_max'),
            'n_points': spec.get('n_points'),
        })
    if not rows:
        return None
    return utils.write_csv(pd.DataFrame(rows), output_dir, filename)


def save_report(result, output_dir, filename='koa_spectrum_report.txt'):
    if not result:
        return None
    utils.ensure_dir(output_dir)
    path = os.path.join(output_dir, filename)
    lines = []
    lines.append('# KOA / Keck Spectrum Report')
    lines.append(f"survey = {result.get('survey', '')}")
    lines.append(f"instrument = {result.get('instrument', '')}")
    lines.append(f"source = {result.get('source', '')}")
    lines.append(f"local_root = {result.get('local_root', '')}")
    lines.append(f"target = {result.get('target', '')}")
    lines.append(f"obs_ids = {result.get('obs_ids', '')}")
    lines.append(f"arms = {result.get('arms', '')}")
    lines.append(f"n_files_used = {result.get('n_files', 0)}")
    lines.append(f"n_matched_files = {result.get('n_matched_files', 0)}")
    lines.append(f"obs_mjd = {result.get('obs_mjd', np.nan)}")
    lines.append(f"obs_mjd_min = {result.get('obs_mjd_min', np.nan)}")
    lines.append(f"obs_mjd_max = {result.get('obs_mjd_max', np.nan)}")
    lines.append(f"wavelength_min_A = {np.nanmin(result['wavelength']):.3f}")
    lines.append(f"wavelength_max_A = {np.nanmax(result['wavelength']):.3f}")

    prov = result.get('provenance') or {}
    if prov:
        lines.append('')
        lines.append('# Provenance')
        for key in ('mission', 'instrument', 'detector', 'grating', 'filter',
                    'proposal_id', 'proposal_pi', 'proposal_type', 'title',
                    'target_name', 'obs_id', 'semester', 'observer',
                    'exptime_s', 'obs_date_utc', 'obs_mjd', 'archive',
                    'archive_url', 'citation_short'):
            if key in prov and prov[key] not in (None, '', 0):
                lines.append(f"{key} = {prov[key]}")
        if prov.get('koa_ids'):
            lines.append(f"koa_ids = {','.join(map(str, prov['koa_ids']))}")
        if prov.get('proposal_ids'):
            lines.append(f"proposal_ids = {','.join(map(str, prov['proposal_ids']))}")
        if prov.get('acknowledgement'):
            lines.append('')
            lines.append('# Acknowledgement')
            lines.append(prov['acknowledgement'])
    lines.append('')
    lines.append('Files used:')
    for spec in result.get('spectra', []):
        lines.append(
            f"  {spec.get('obs_id','')}/{spec.get('arm','')}/"
            f"{spec.get('filename','')}  MJD={spec.get('mjd', np.nan)}  "
            f"scale={spec.get('scale_applied', 1.0):.4g}")
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    return path


def _main(argv=None):
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description='Query/download KOA data and standardize extracted 1D spectra.')
    parser.add_argument('--target', help='Target name, e.g. ZTFJ035352.96+431525.16')
    parser.add_argument('--ra', type=float, help='Right ascension in degrees')
    parser.add_argument('--dec', type=float, help='Declination in degrees')
    parser.add_argument('--instrument', action='append', default=None,
                        help='KOA instrument to query; may be repeated. Default: lris')
    parser.add_argument('--radius-arcsec', type=float,
                        default=config.KOA_SEARCH_RADIUS_ARCSEC)
    parser.add_argument('--work-dir',
                        help='Directory for KOA metadata/download cache')
    parser.add_argument('--local-root', dest='extracted_root',
                        help='Directory containing already extracted 1D FITS')
    parser.add_argument('--output-dir',
                        help='Directory for final CSV/PNG/report outputs')
    parser.add_argument('--row-limit', type=int,
                        help='Limit selected KOA metadata rows before download')
    parser.add_argument('--no-download', action='store_true',
                        help='Skip online KOA query/download and only standardize local 1D FITS')
    parser.add_argument('--no-lev0', action='store_true',
                        help='Do not download KOA raw level-0 FITS files')
    parser.add_argument('--no-calibfile', action='store_true',
                        help='Do not ask PyKOA to download associated calibration files')
    parser.add_argument('--lev1file', action='store_true',
                        help='Ask PyKOA to download KOA level-1 products when available')
    parser.add_argument('--pypeit-file',
                        help='Run an existing checked .pypeit reduction before standardizing')
    parser.add_argument('--auto-pypeit', action='store_true',
                        help='Generate LRIS PypeIt setup files from downloaded KOA raw frames')
    parser.add_argument('--pypeit-setup-only', action='store_true',
                        help='Create PypeIt setup files but do not run run_pypeit')
    args = parser.parse_args(argv)

    result = download_and_extract_spectrum(
        ra=args.ra, dec=args.dec, target=args.target,
        instruments=args.instrument or ['lris'],
        radius_arcsec=args.radius_arcsec,
        work_dir=args.work_dir,
        extracted_root=args.extracted_root,
        download=not args.no_download,
        lev0file=not args.no_lev0,
        calibfile=not args.no_calibfile,
        lev1file=args.lev1file,
        row_limit=args.row_limit,
        pypeit_file=args.pypeit_file,
        auto_pypeit=args.auto_pypeit,
        pypeit_setup_only=args.pypeit_setup_only,
        output_dir=args.output_dir)

    summary = {
        'survey': result.get('survey'),
        'source': result.get('source'),
        'target': result.get('target'),
        'ra': result.get('ra'),
        'dec': result.get('dec'),
        'n_files': result.get('n_files'),
        'n_points': int(len(result.get('wavelength', [])))
        if 'wavelength' in result else None,
        'obs_ids': result.get('obs_ids'),
        'arms': result.get('arms'),
        'saved_csv': result.get('saved_csv'),
        'saved_plot': result.get('saved_plot'),
        'saved_report': result.get('saved_report'),
        'pypeit_error': result.get('pypeit_error'),
        'message': result.get('message'),
    }
    if result.get('koa_download'):
        summary['koa_download'] = [
            {
                'instrument': p.get('instrument'),
                'n_metadata_rows': p.get('n_metadata_rows'),
                'n_selected_rows': p.get('n_selected_rows'),
                'n_downloaded_files': len(p.get('downloaded_files') or []),
                'download_dir': p.get('download_dir'),
            }
            for p in result['koa_download'].get('products', [])
        ]
    if result.get('pypeit_setup'):
        summary['pypeit_setup'] = [
            {
                'arm': p.get('arm'),
                'spectrograph': p.get('spectrograph'),
                'n_raw_files': p.get('n_raw_files'),
                'pypeit_files': p.get('pypeit_files'),
            }
            for p in result.get('pypeit_setup') or []
        ]
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    _main()
