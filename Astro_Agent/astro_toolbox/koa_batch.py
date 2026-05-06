"""Batch KOA query/download helpers for catalog CSV files."""
import argparse
import glob
import os
import re
import shutil
import time
import traceback

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table

from . import koa, utils


DEFAULT_INSTRUMENTS = ('lris',)
REDUCTION_SUMMARY = 'koa_reduction_summary.csv'


def _safe_name(value, fallback):
    text = str(value or '').strip()
    text = re.sub(r'[^A-Za-z0-9_.+-]+', '_', text).strip('_')
    return text or fallback


def _target_from_row(row, index):
    for col in ('FirstColumn_23chars', 'ren_ZTF_Name', 'target', 'name'):
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            return str(row[col]).strip()
    if 'img_path' in row and pd.notna(row['img_path']):
        match = re.search(r'(ZTFJ\d{6}\.\d+[+-]\d{6}\.\d+)',
                          str(row['img_path']))
        if match:
            return match.group(1)
    return f'row_{index:04d}'


def _coord_from_row(row):
    ra_cols = ('RA_Decimal', 'ra', 'RA', 'raj2000')
    dec_cols = ('Dec_Decimal', 'dec', 'DEC', 'dej2000')
    ra = dec = None
    for col in ra_cols:
        if col in row and pd.notna(row[col]):
            ra = float(row[col])
            break
    for col in dec_cols:
        if col in row and pd.notna(row[col]):
            dec = float(row[col])
            break
    return ra, dec


def _first_image_hdu(hdul):
    for i, hdu in enumerate(hdul):
        data = hdu.data
        if data is None:
            continue
        arr = np.asarray(data)
        if arr.ndim >= 2:
            return i, np.asarray(arr, dtype=float).squeeze()
    return None, None


def make_fits_quicklook(fits_path, png_path, title=None):
    """Save a compact PNG quicklook for a KOA FITS file."""
    utils.ensure_dir(os.path.dirname(png_path))
    try:
        spec = koa.read_spectrum(fits_path)
    except Exception:
        spec = None

    if spec and len(spec.get('wavelength', [])) > 0:
        wave = np.asarray(spec['wavelength'], dtype=float)
        flux = np.asarray(spec['flux'], dtype=float)
        good = np.isfinite(wave) & np.isfinite(flux)
        if good.sum() > 1 and np.nanmax(wave[good]) > np.nanmin(wave[good]):
            fig, ax = plt.subplots(figsize=(9, 3.8))
            ax.plot(wave[good], flux[good], color='black', lw=0.7)
            ax.set_xlabel('Wavelength (A)')
            ax.set_ylabel('Flux')
            ax.set_title(title or os.path.basename(fits_path), fontsize=10)
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            utils.save_and_close(fig, png_path)
            return png_path

    with fits.open(fits_path, memmap=False) as hdul:
        hdu_index, image = _first_image_hdu(hdul)
        if image is None:
            return None
        image = np.asarray(image, dtype=float).squeeze()
        if image.ndim > 2:
            image = image[0]
        finite = np.isfinite(image)
        if not np.any(finite):
            return None
        vmin, vmax = np.nanpercentile(image[finite], [1, 99])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = np.nanmin(image[finite]), np.nanmax(image[finite])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(image, origin='lower', cmap='gray', vmin=vmin, vmax=vmax,
              aspect='auto')
    ax.set_title(title or f'{os.path.basename(fits_path)} [HDU {hdu_index}]',
                 fontsize=10)
    ax.set_xlabel('X pixel')
    ax.set_ylabel('Y pixel')
    fig.tight_layout()
    utils.save_and_close(fig, png_path)
    return png_path


def _relative_png_path(download_dir, fits_path, png_dir):
    rel = os.path.relpath(fits_path, download_dir)
    rel = re.sub(r'\.gz$', '', rel)
    rel = re.sub(r'\.fits?$', '', rel, flags=re.IGNORECASE)
    return os.path.join(png_dir, rel + '.png')


def _write_progress(path, rows):
    if rows:
        pd.DataFrame(rows).sort_values('row_index').to_csv(path, index=False)


def _load_progress(path):
    if os.path.exists(path):
        rows = pd.read_csv(path).to_dict('records')
        return {int(row['row_index']): row for row in rows}
    return {}


def _list_fits(root):
    if not root or not os.path.exists(root):
        return []
    patterns = ('*.fits', '*.fit', '*.fits.gz')
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(os.path.join(root, '**', pattern),
                               recursive=True))
    return sorted({p for p in paths if os.path.isfile(p)})


def _load_reduction_progress(path):
    if os.path.exists(path):
        rows = pd.read_csv(path).to_dict('records')
        out = {}
        for row in rows:
            key = int(row['row_index'])
            target = str(row.get('target', ''))
            out[(key, target)] = row
        return out
    return {}


def _write_reduction_progress(path, rows):
    if rows:
        pd.DataFrame(rows).sort_values(['row_index', 'target']).to_csv(
            path, index=False)


def _target_selected(row, reduce_targets):
    if not reduce_targets:
        return True
    target = str(row.get('target', ''))
    safe = _safe_name(target, '')
    wanted = {str(t).strip() for t in reduce_targets if str(t).strip()}
    return target in wanted or safe in wanted


def _safe_int(value, default=0):
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def _safe_float(value, default=np.nan):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _lris_arm_from_metadata(koaid, instrume=''):
    koaid = str(koaid or '').strip().upper()
    instrume = str(instrume or '').strip().upper()
    if koaid.startswith('LB.') or 'BLUE' in instrume:
        return 'blue'
    if koaid.startswith('LR.') or instrume == 'LRIS':
        return 'red'
    return ''


def _prepare_fast_calib_subset(selected_path, output_dir, prefer_arm='red',
                               max_science_frames=1):
    """
    Build a tiny IPAC selection table for KOA calibration download.

    Fast reduction only needs associated calibrations for the specific science
    frames that will be reduced. Selecting the longest preferred-arm exposure(s)
    keeps KOA downloads much smaller than using the full selected table.
    """
    if not selected_path or not os.path.exists(selected_path):
        return None

    table = Table.read(selected_path, format='ascii.ipac')
    if len(table) == 0:
        return None

    names = {str(name).lower(): str(name) for name in table.colnames}
    koaid_col = names.get('koaid')
    if koaid_col is None:
        return None

    instrume_col = names.get('instrume')
    koaimtyp_col = names.get('koaimtyp')
    exptime_col = names.get('elaptime')
    date_col = names.get('date_obs')
    ut_col = names.get('ut')

    rows = []
    for idx, row in enumerate(table):
        koaid = str(row[koaid_col]).strip()
        if not koaid or koaid.lower() == 'null':
            continue
        koaimtyp = (str(row[koaimtyp_col]).strip().lower()
                    if koaimtyp_col is not None else '')
        if koaimtyp and koaimtyp not in ('object', 'science', 'sci'):
            continue
        instrume = str(row[instrume_col]).strip() if instrume_col else ''
        rows.append({
            'idx': idx,
            'koaid': koaid,
            'arm': _lris_arm_from_metadata(koaid, instrume),
            'exptime': _safe_float(row[exptime_col], default=0.0)
            if exptime_col is not None else 0.0,
            'date_obs': str(row[date_col]).strip() if date_col else '',
            'ut': str(row[ut_col]).strip() if ut_col else '',
        })

    if not rows:
        return None

    prefer_arm = str(prefer_arm or 'red').strip().lower() or 'red'
    candidates = [r for r in rows if r['arm'] == prefer_arm]
    selected_arm = prefer_arm
    if not candidates:
        candidates = [r for r in rows if r['arm']]
        if not candidates:
            candidates = list(rows)
        if candidates:
            selected_arm = candidates[0]['arm'] or 'mixed'

    candidates = sorted(
        candidates,
        key=lambda r: (r['exptime'], r['date_obs'], r['ut'], r['koaid']),
        reverse=True,
    )
    take = max(1, int(max_science_frames or 1))
    chosen = candidates[:take]
    subset = table[[row['idx'] for row in chosen]]

    utils.ensure_dir(output_dir)
    subset_path = os.path.join(
        output_dir, f'lris_selected_fastcal_{selected_arm}.tbl')
    subset.write(subset_path, format='ascii.ipac', overwrite=True)
    return {
        'path': subset_path,
        'arm': selected_arm,
        'n_rows': len(subset),
        'koaid': [row['koaid'] for row in chosen],
    }


def _prepare_fast_science_subset(selected_path, output_dir, prefer_arm='red',
                                 max_science_frames=1):
    """
    Build a tiny IPAC selection table for lev0 science download.

    This is a download-time companion to fast PypeIt extraction: pick the
    longest science exposure(s) from the preferred arm so we can get to a
    usable 1D spectrum without downloading every KOA exposure for the source.
    """
    if not selected_path or not os.path.exists(selected_path):
        return None

    table = Table.read(selected_path, format='ascii.ipac')
    if len(table) == 0:
        return None

    names = {str(name).lower(): str(name) for name in table.colnames}
    koaid_col = names.get('koaid')
    if koaid_col is None:
        return None

    instrume_col = names.get('instrume')
    koaimtyp_col = names.get('koaimtyp')
    exptime_col = names.get('elaptime')
    date_col = names.get('date_obs')
    ut_col = names.get('ut')

    rows = []
    for idx, row in enumerate(table):
        koaid = str(row[koaid_col]).strip()
        if not koaid or koaid.lower() == 'null':
            continue
        koaimtyp = (str(row[koaimtyp_col]).strip().lower()
                    if koaimtyp_col is not None else '')
        if koaimtyp and koaimtyp not in ('object', 'science', 'sci'):
            continue
        instrume = str(row[instrume_col]).strip() if instrume_col else ''
        rows.append({
            'idx': idx,
            'koaid': koaid,
            'arm': _lris_arm_from_metadata(koaid, instrume),
            'exptime': _safe_float(row[exptime_col], default=0.0)
            if exptime_col is not None else 0.0,
            'date_obs': str(row[date_col]).strip() if date_col else '',
            'ut': str(row[ut_col]).strip() if ut_col else '',
        })

    if not rows:
        return None

    prefer_arm = str(prefer_arm or 'red').strip().lower() or 'red'
    candidates = [r for r in rows if r['arm'] == prefer_arm]
    selected_arm = prefer_arm
    if not candidates:
        candidates = [r for r in rows if r['arm']]
        if not candidates:
            candidates = list(rows)
        if candidates:
            selected_arm = candidates[0]['arm'] or 'mixed'

    candidates = sorted(
        candidates,
        key=lambda r: (r['exptime'], r['date_obs'], r['ut'], r['koaid']),
        reverse=True,
    )
    take = max(1, int(max_science_frames or 1))
    chosen = candidates[:take]
    subset = table[[row['idx'] for row in chosen]]

    utils.ensure_dir(output_dir)
    subset_path = os.path.join(
        output_dir, f'lris_selected_fastscience_{selected_arm}.tbl')
    subset.write(subset_path, format='ascii.ipac', overwrite=True)
    return {
        'path': subset_path,
        'arm': selected_arm,
        'n_rows': len(subset),
        'koaid': [row['koaid'] for row in chosen],
    }


def _candidate_1d_roots(target_dir):
    """Roots that may contain PypeIt or user-provided extracted 1D products."""
    names = (
        'spectrum', 'coadd1d', 'spec1d', 'redux', 'Science',
        'pypeit_redux', 'extracted', '1d'
    )
    roots = [os.path.join(target_dir, name) for name in names]
    # PypeIt often writes Science below per-arm setup folders.
    roots.extend(glob.glob(os.path.join(target_dir, 'pypeit_setup', '**',
                                        'Science'), recursive=True))
    roots.extend(glob.glob(os.path.join(target_dir, 'pypeit_setup', '**',
                                        'Coadd'), recursive=True))
    seen = []
    for root in roots:
        root = os.path.abspath(root)
        if os.path.exists(root) and root not in seen:
            seen.append(root)
    return seen


def _spec2d_files(root):
    if not root or not os.path.exists(root):
        return []
    paths = glob.glob(os.path.join(root, '**', 'spec2d_*.fits'),
                      recursive=True)
    return sorted({os.path.abspath(path) for path in paths if os.path.isfile(path)})


def _extract_1d_from_spec2d(spec2d_path, target=None, ra=None, dec=None,
                            aperture_half_width=2):
    spec2d_path = os.path.abspath(spec2d_path)
    components = []
    with fits.open(spec2d_path, memmap=False) as hdul:
        primary = hdul[0].header
        det_roots = sorted({
            hdu.name.rsplit('-', 1)[0]
            for hdu in hdul
            if hdu.name.endswith('-WAVEIMG')
        })
        for det_root in det_roots:
            sci_name = f'{det_root}-SCIIMG'
            sky_name = f'{det_root}-SKYMODEL'
            wave_name = f'{det_root}-WAVEIMG'
            ivar_name = f'{det_root}-IVARRAW'
            if sci_name not in hdul or sky_name not in hdul or wave_name not in hdul:
                continue

            sci = np.asarray(hdul[sci_name].data, dtype=float)
            sky = np.asarray(hdul[sky_name].data, dtype=float)
            wave = np.asarray(hdul[wave_name].data, dtype=float)
            if sci.shape != sky.shape or sci.shape != wave.shape or sci.ndim != 2:
                continue

            resid = sci - sky
            valid = np.isfinite(resid) & np.isfinite(wave) & (wave > 0)
            if int(valid.sum()) < 500:
                continue

            cols = np.where(valid.any(axis=0))[0]
            if len(cols) < max(2 * int(aperture_half_width) + 5, 15):
                continue

            rows = resid.copy()
            rows[~valid] = np.nan
            row_med = np.nanmedian(rows, axis=1)
            rows = rows - row_med[:, None]

            profile = np.nansum(np.clip(rows, 0, None), axis=0)
            smooth = np.convolve(np.nan_to_num(profile), np.ones(7) / 7.0,
                                 mode='same')
            slit_width = int(cols.max() - cols.min() + 1)
            guard = max(5, int(0.15 * slit_width))
            lo = int(cols.min() + guard)
            hi = int(cols.max() - guard)
            if lo >= hi:
                lo = int(cols.min())
                hi = int(cols.max())
            center = int(np.nanargmax(smooth[lo:hi + 1]) + lo)
            ap_lo = max(int(cols.min()), center - int(aperture_half_width))
            ap_hi = min(int(cols.max()), center + int(aperture_half_width))
            ap = np.arange(ap_lo, ap_hi + 1, dtype=int)
            if len(ap) < 2:
                continue

            wave_1d = []
            flux_1d = []
            err_1d = []
            ivar = (np.asarray(hdul[ivar_name].data, dtype=float)
                    if ivar_name in hdul else None)
            for row_idx in range(rows.shape[0]):
                row_mask = valid[row_idx, ap]
                if int(np.count_nonzero(row_mask)) < 2:
                    continue
                wave_row = wave[row_idx, ap][row_mask]
                flux_row = rows[row_idx, ap][row_mask]
                wave_val = np.nanmedian(wave_row)
                flux_val = np.nansum(flux_row)
                if not (np.isfinite(wave_val) and wave_val > 0 and np.isfinite(flux_val)):
                    continue
                if ivar is not None and ivar.shape == rows.shape:
                    ivar_row = ivar[row_idx, ap][row_mask]
                    good_ivar = np.isfinite(ivar_row) & (ivar_row > 0)
                    if np.any(good_ivar):
                        err_val = float(np.sqrt(np.nansum(1.0 / ivar_row[good_ivar])))
                    else:
                        err_val = 0.0
                else:
                    err_val = 0.0
                wave_1d.append(float(wave_val))
                flux_1d.append(float(flux_val))
                err_1d.append(float(err_val))

            if len(wave_1d) < 50:
                continue

            wave_1d = np.asarray(wave_1d, dtype=float)
            flux_1d = np.asarray(flux_1d, dtype=float)
            err_1d = np.asarray(err_1d, dtype=float)
            order = np.argsort(wave_1d)
            wave_1d = wave_1d[order]
            flux_1d = flux_1d[order]
            err_1d = err_1d[order]

            base = os.path.basename(spec2d_path)
            arm = 'blue' if ('LRISb' in base or '/blue/' in spec2d_path
                             or base.startswith('spec2d_LB.')) else 'red'
            component = {
                'filename': base,
                'path': spec2d_path,
                'obs_id': os.path.splitext(base)[0],
                'arm': arm,
                'is_coadd': False,
                'target': target,
                'ra': float(ra) if ra is not None else np.nan,
                'dec': float(dec) if dec is not None else np.nan,
                'mjd': _safe_float(primary.get('MJD')),
                'dateobs': primary.get('DATE-OBS', ''),
                'exptime': _safe_float(primary.get('EXPTIME')),
                'airmass': _safe_float(primary.get('AIRMASS')),
                'dispname': primary.get('DISPNAME', primary.get('GRANAME', '')),
                'decker': primary.get('DECKER', primary.get('SLITNAME', '')),
                'binning': primary.get('BINNING', ''),
                'dichroic': primary.get('DICHNAME', primary.get('DICHROIC', '')),
                'wave_min': float(np.nanmin(wave_1d)),
                'wave_max': float(np.nanmax(wave_1d)),
                'n_points': int(len(wave_1d)),
                'trace_center_col': int(center),
                'wavelength': wave_1d,
                'flux': flux_1d,
                'error': err_1d,
                'scale_applied': 1.0,
            }
            components.append(component)

    if not components:
        return None

    combined = koa.combine_spectra(
        components, resample_step=None, scale_to_overlap=False)
    if combined is None:
        return None
    wave, flux, error, n_contrib, _ = combined
    mjds = np.array([_safe_float(spec.get('mjd')) for spec in components],
                    dtype=float)
    mjds = mjds[np.isfinite(mjds)]
    arms = sorted({str(spec.get('arm', '')).strip()
                   for spec in components if str(spec.get('arm', '')).strip()})
    obs_ids = sorted({str(spec.get('obs_id', '')).strip()
                      for spec in components if str(spec.get('obs_id', '')).strip()})
    return {
        'survey': 'KOA_LRIS',
        'instrument': 'LRIS',
        'source': 'spec2d_fallback_1d',
        'local_root': os.path.dirname(spec2d_path),
        'target': target,
        'ra': float(ra) if ra is not None else np.nan,
        'dec': float(dec) if dec is not None else np.nan,
        'wavelength': wave,
        'flux': flux,
        'error': error,
        'n_contributors': n_contrib,
        'n_files': len(components),
        'n_matched_files': len(components),
        'obs_ids': ','.join(obs_ids),
        'arms': ','.join(arms),
        'obs_mjd': float(np.nanmedian(mjds)) if len(mjds) else np.nan,
        'obs_mjd_min': float(np.nanmin(mjds)) if len(mjds) else np.nan,
        'obs_mjd_max': float(np.nanmax(mjds)) if len(mjds) else np.nan,
        'spectra': components,
        'matched_records': [],
        'scale_to_overlap': False,
        'flux_unit': 'counts',
        'flux_columns': 'spec2d_fallback_sum',
    }


def _standardize_existing_1d(row, target_dir, radius_arcsec=None,
                             prefer_coadds=True, resample_step=None,
                             scale_to_overlap=True):
    """Find extracted 1D products below a target directory and save CSV/PNG."""
    target = row.get('target')
    ra = row.get('ra')
    dec = row.get('dec')
    out_dir = os.path.join(target_dir, 'spectrum')
    roots = _candidate_1d_roots(target_dir)

    for root in roots:
        result = koa.query_spectrum(
            ra=ra, dec=dec, target=target, local_root=root,
            radius_arcsec=radius_arcsec, prefer_coadds=prefer_coadds,
            resample_step=resample_step, scale_to_overlap=scale_to_overlap)
        if result is None:
            # PypeIt products are stored inside this target directory, but their
            # filenames/headers may not carry the catalog target name.
            result = koa.query_spectrum(
                local_root=root, target=target, ra=ra, dec=dec,
                radius_arcsec=radius_arcsec, prefer_coadds=prefer_coadds,
                resample_step=resample_step,
                scale_to_overlap=scale_to_overlap, match_all=True)
        if result:
            result['source'] = 'koa_batch_existing_1d'
            result['target'] = target
            result['ra'] = ra
            result['dec'] = dec
            utils.ensure_dir(out_dir)
            csv_path = koa.save_csv(result, out_dir)
            exp_path = koa.save_exposure_table(result, out_dir)
            report_path = koa.save_report(result, out_dir)
            png_path = os.path.join(out_dir, 'koa_spectrum.png')
            koa.plot_spectrum(result, save_path=png_path)
            return {
                'result': result,
                'root': root,
                'csv_path': csv_path,
                'png_path': png_path,
                'report_path': report_path,
                'exposure_path': exp_path,
            }

    for root in roots:
        for spec2d_path in _spec2d_files(root):
            result = _extract_1d_from_spec2d(
                spec2d_path, target=target, ra=ra, dec=dec)
            if result is None:
                continue
            utils.ensure_dir(out_dir)
            csv_path = koa.save_csv(result, out_dir)
            exp_path = koa.save_exposure_table(result, out_dir)
            report_path = koa.save_report(result, out_dir)
            png_path = os.path.join(out_dir, 'koa_spectrum.png')
            koa.plot_spectrum(result, save_path=png_path)
            return {
                'result': result,
                'root': root,
                'csv_path': csv_path,
                'png_path': png_path,
                'report_path': report_path,
                'exposure_path': exp_path,
            }
    return None


def _parse_pypeit_file(pypeit_file):
    with open(pypeit_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    blocks = []
    current = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped == 'data read':
            current = {
                'paths': [],
                'path_lines': [],
                'header': None,
                'rows': [],
            }
            blocks.append(current)
            continue
        if stripped == 'data end':
            current = None
            continue
        if current is None:
            continue
        if stripped.startswith('path '):
            root = stripped.split(None, 1)[1]
            current['paths'].append(root)
            current['path_lines'].append((idx, root))
            continue
        if 'filename' in line and 'frametype' in line:
            current['header'] = [part.strip() for part in line.split('|')]
            continue
        if (current.get('header') is None or not stripped
                or line.lstrip().startswith('#') or '|' not in line):
            continue
        parts = line.rstrip('\n').split('|')
        header = current['header']
        if len(parts) != len(header):
            continue
        values = {header[i]: parts[i].strip() for i in range(len(header))}
        filename = values.get('filename', '')
        fullpath = ''
        for root in current['paths']:
            candidate = os.path.join(root, filename)
            if os.path.exists(candidate):
                fullpath = candidate
                break
        current['rows'].append({
            'line_idx': idx,
            'filename': filename,
            'frametype': values.get('frametype', ''),
            'exptime': _safe_float(values.get('exptime')),
            'parts': parts,
            'values': values,
            'fullpath': fullpath,
        })

    return lines, blocks


def _pypeit_arm(pypeit_file):
    text = str(pypeit_file).lower()
    if 'lris_red' in text or '/red/' in text:
        return 'red'
    if 'lris_blue' in text or '/blue/' in text:
        return 'blue'
    return ''


def _science_rows_from_blocks(blocks):
    rows = []
    for block in blocks:
        for row in block.get('rows', []):
            frametype = str(row.get('frametype', '')).lower()
            if 'science' in frametype:
                rows.append(row)
    return rows


def _calibration_rows_from_blocks(blocks):
    rows = []
    keywords = ('arc', 'tilt', 'flat', 'trace')
    for block in blocks:
        for row in block.get('rows', []):
            frametype = str(row.get('frametype', '')).lower()
            if any(key in frametype for key in keywords):
                rows.append(row)
    return rows


def _upsert_rdx_detnum(lines, detnum):
    if detnum is None:
        return lines
    start = None
    end = None
    for idx, line in enumerate(lines):
        if line.strip() == '[rdx]':
            start = idx
            continue
        if start is not None and idx > start and re.match(r'^\[[^\[]', line.strip()):
            end = idx
            break
    if start is None:
        return lines
    if end is None:
        end = len(lines)
    for idx in range(start + 1, end):
        if lines[idx].strip().startswith('detnum ='):
            lines[idx] = f'    detnum = {detnum}\n'
            return lines
    lines.insert(end, f'    detnum = {detnum}\n')
    return lines


def _ensure_fast_reduce_block(lines, maxnumber_sci=1, maxnumber_std=1,
                              snr_thresh=20.0):
    if any(line.strip() == '[reduce]' for line in lines):
        return lines
    insert_at = None
    for idx, line in enumerate(lines):
        if line.startswith('# Setup'):
            insert_at = idx
            break
    if insert_at is None:
        insert_at = len(lines)
    block = [
        '[reduce]\n',
        '    [[findobj]]\n',
        f'        maxnumber_sci = {int(maxnumber_sci)}\n',
        f'        maxnumber_std = {int(maxnumber_std)}\n',
        '        skip_second_find = True\n',
        '        skip_final_global = True\n',
        f'        snr_thresh = {float(snr_thresh):.1f}\n',
        '\n',
    ]
    return lines[:insert_at] + block + lines[insert_at:]


def _build_fast_lris_pypeit(pypeit_file, stage_root, max_science_frames=1,
                            snr_thresh=20.0):
    lines, blocks = _parse_pypeit_file(pypeit_file)
    science_rows = _science_rows_from_blocks(blocks)
    if not science_rows:
        return None

    science_rows = sorted(
        science_rows,
        key=lambda row: (-_safe_float(row.get('exptime'), 0.0), row['line_idx']))
    keep_science = {
        row['line_idx'] for row in science_rows[:max(1, int(max_science_frames))]
    }

    path_map = {}
    used_names = {}
    stage_raw_root = os.path.join(stage_root, 'pypeit_raw')
    utils.ensure_dir(stage_raw_root)

    for block in blocks:
        for _, root in block.get('path_lines', []):
            if root in path_map:
                continue
            base = _safe_name(os.path.basename(root), 'root')
            used_names[base] = used_names.get(base, 0) + 1
            suffix = '' if used_names[base] == 1 else f'_{used_names[base]}'
            stage_dir = os.path.join(stage_raw_root, f'{base}{suffix}')
            utils.ensure_dir(stage_dir)
            path_map[root] = stage_dir

    skip_lines = set()
    copied_files = []
    for block in blocks:
        for row in block.get('rows', []):
            frametype = str(row.get('frametype', '')).lower()
            is_science = 'science' in frametype
            is_standard = 'standard' in frametype
            is_calib = any(key in frametype for key in (
                'arc', 'tilt', 'flat', 'trace', 'bias', 'dark'))
            keep_row = (
                (is_science and row['line_idx'] in keep_science)
                or (is_calib and not is_standard)
            )
            if not keep_row:
                skip_lines.add(row['line_idx'])
                continue
            fullpath = row.get('fullpath')
            if not fullpath or not os.path.exists(fullpath):
                raise FileNotFoundError(
                    f'Missing FITS referenced by {pypeit_file}: {row.get("filename")}')
            src_dir = os.path.dirname(fullpath)
            dst_dir = path_map[src_dir]
            dst_path = os.path.join(dst_dir, os.path.basename(fullpath))
            if not os.path.exists(dst_path):
                shutil.copy2(fullpath, dst_path)
            copied_files.append(dst_path)

    path_line_map = {}
    for block in blocks:
        for idx, root in block.get('path_lines', []):
            path_line_map[idx] = root

    out_lines = []
    for idx, line in enumerate(lines):
        if idx in skip_lines:
            continue
        if idx in path_line_map:
            out_lines.append(f' path {path_map[path_line_map[idx]]}\n')
        else:
            out_lines.append(line)

    detnum = 2 if _pypeit_arm(pypeit_file) == 'red' else None
    out_lines = _upsert_rdx_detnum(out_lines, detnum)
    out_lines = _ensure_fast_reduce_block(
        out_lines, maxnumber_sci=1, maxnumber_std=1, snr_thresh=snr_thresh)

    utils.ensure_dir(os.path.join(stage_root, 'setup'))
    fast_name = os.path.splitext(os.path.basename(pypeit_file))[0] + '_fast1.pypeit'
    fast_pypeit = os.path.join(stage_root, 'setup', fast_name)
    with open(fast_pypeit, 'w', encoding='utf-8') as f:
        f.writelines(out_lines)

    return {
        'pypeit_file': fast_pypeit,
        'copied_files': copied_files,
        'selected_science': [
            row.get('filename') for row in science_rows[:max(1, int(max_science_frames))]
        ],
    }


def _science_pypeit_files(pypeit_files, prefer_arm='red'):
    infos = []
    for pypeit_file in sorted(set(pypeit_files)):
        _, blocks = _parse_pypeit_file(pypeit_file)
        science_rows = _science_rows_from_blocks(blocks)
        if not science_rows:
            continue
        calib_rows = _calibration_rows_from_blocks(blocks)
        exptime = max(_safe_float(row.get('exptime'), 0.0) for row in science_rows)
        arm = _pypeit_arm(pypeit_file)
        infos.append({
            'pypeit_file': pypeit_file,
            'arm': arm,
            'n_science': len(science_rows),
            'n_calib': len(calib_rows),
            'max_exptime': exptime,
        })
    preferred = str(prefer_arm or '').strip().lower()
    infos.sort(key=lambda info: (
        0 if info['arm'] == preferred else 1,
        0 if info['n_calib'] > 0 else 1,
        -info['max_exptime'],
        info['pypeit_file']))
    return infos


def _copy_fast_products(redux_root, target_dir):
    science_dir = os.path.join(redux_root, 'Science')
    if not os.path.exists(science_dir):
        return []
    out_dir = os.path.join(target_dir, 'spectrum')
    utils.ensure_dir(out_dir)
    copied = []
    for name in sorted(os.listdir(science_dir)):
        if not (name.startswith('spec1d_') or name.startswith('spec2d_')):
            continue
        src = os.path.join(science_dir, name)
        dst = os.path.join(out_dir, name)
        shutil.copy2(src, dst)
        copied.append(dst)
        txt_src = os.path.splitext(src)[0] + '.txt'
        if os.path.exists(txt_src):
            txt_dst = os.path.join(out_dir, os.path.basename(txt_src))
            shutil.copy2(txt_src, txt_dst)
            copied.append(txt_dst)
    return copied


def _run_fast_lris_1d(target_dir, pypeit_files, target=None, prefer_arm='red',
                      max_science_frames=1, snr_thresh=20.0):
    safe_target = _safe_name(target, 'target')
    runs = []
    for idx, info in enumerate(_science_pypeit_files(
            pypeit_files, prefer_arm=prefer_arm)):
        tag = f'{safe_target}_{info["arm"] or "arm"}_{idx:02d}'
        stage_root = os.path.join('/tmp', f'koa_ascii_{tag}')
        redux_root = os.path.join('/tmp', f'koa_redux_{tag}')
        for path in (stage_root, redux_root):
            if os.path.exists(path):
                shutil.rmtree(path)
        staged = _build_fast_lris_pypeit(
            info['pypeit_file'], stage_root,
            max_science_frames=max_science_frames,
            snr_thresh=snr_thresh)
        if staged is None:
            continue
        run_info = {
            'arm': info['arm'],
            'source_pypeit': info['pypeit_file'],
            'fast_pypeit': staged['pypeit_file'],
            'redux_root': redux_root,
            'selected_science': ';'.join(staged['selected_science']),
            'n_calib': int(info.get('n_calib', 0) or 0),
            'copied_files': [],
            'stdout': '',
            'stderr': '',
            'error': '',
        }
        try:
            proc = koa.run_pypeit_reduction(
                staged['pypeit_file'], redux_path=redux_root, extra_args=['-o'])
            run_info['stdout'] = proc.stdout
            run_info['stderr'] = proc.stderr
            run_info['copied_files'] = _copy_fast_products(redux_root, target_dir)
            run_info['has_spec1d'] = any(
                os.path.basename(path).startswith('spec1d_')
                for path in run_info['copied_files']
            )
        except Exception as exc:
            run_info['error'] = f'{type(exc).__name__}: {exc}'
            run_info['has_spec1d'] = False
        runs.append(run_info)
        if run_info['has_spec1d']:
            break
    return runs


def reduce_existing_metadata(output_root, instruments=DEFAULT_INSTRUMENTS,
                             start=0, limit=None, reduce_targets=None,
                             download_calib=False, pypeit_setup=True,
                             run_pypeit=False, radius_arcsec=8.0,
                             prefer_coadds=True, resample_step=None,
                             scale_to_overlap=True, only_downloaded=True,
                             force=False, normalize_lris_headers=True,
                             patch_raw_headers_in_place=False,
                             science_min_exptime=30.0,
                             fast_pypeit=False, fast_prefer_arm='red',
                             fast_max_science_frames=1,
                             fast_snr_thresh=20.0):
    """
    Reduce/standardize already downloaded KOA target directories.

    The function is intentionally explicit about the expensive part: raw LRIS
    FITS become science-grade 1D spectra only after PypeIt runs with suitable
    calibrations. Without run_pypeit=True it creates/checks PypeIt setup files
    and exports any already-existing 1D products it finds.
    """
    summary_path = os.path.join(output_root, 'koa_batch_summary.csv')
    if not os.path.exists(summary_path):
        raise FileNotFoundError(summary_path)

    reduction_path = os.path.join(output_root, REDUCTION_SUMMARY)
    summary = pd.read_csv(summary_path)
    summary = summary.sort_values('row_index')
    if start is not None:
        summary = summary[summary['row_index'] >= int(start)]
    if limit is not None:
        summary = summary.iloc[:int(limit)]

    progress = _load_reduction_progress(reduction_path)
    for _, row in summary.iterrows():
        row = row.to_dict()
        if str(row.get('status')) != 'ok':
            continue
        if _safe_int(row.get('n_selected_rows', 0)) <= 0:
            continue
        if not _target_selected(row, reduce_targets):
            continue

        row_index = int(row['row_index'])
        target = str(row.get('target', f'row_{row_index:04d}'))
        target_dir = str(row.get('target_dir'))
        key = (row_index, target)
        old = progress.get(key, {})
        if (not force and str(old.get('spectrum_status', '')) in ('found', 'ok')
                and isinstance(old.get('spectrum_csv'), str)
                and os.path.exists(old.get('spectrum_csv'))):
            continue

        status = 'ok'
        error = ''
        message = ''
        setup_status = 'skipped'
        spectrum_status = 'not_found'
        pypeit_files = []
        normalized_roots = []
        n_setup_files = 0
        n_calib_downloaded = 0
        n_header_normalized = 0
        n_bad_fits_skipped = 0
        n_fits_before = 0
        n_fits_after = 0
        spectrum_info = None
        fast_runs = []
        fast_calib_selection = []

        try:
            raw_roots = []
            for instrument in instruments:
                raw_roots.append(os.path.join(
                    target_dir, 'download', str(instrument).lower()))
            n_fits_before = sum(len(_list_fits(root)) for root in raw_roots)
            if only_downloaded and n_fits_before == 0:
                message = 'No downloaded FITS found; skip reduction setup.'
            else:
                if download_calib:
                    for instrument in instruments:
                        inst = str(instrument).lower()
                        selected_path = os.path.join(
                            target_dir, 'metadata', f'{inst}_selected.tbl')
                        download_dir = os.path.join(
                            target_dir, 'download', inst)
                        before = set(_list_fits(download_dir))
                        calib_source = selected_path
                        subset_info = None
                        if fast_pypeit and inst == 'lris':
                            subset_info = _prepare_fast_calib_subset(
                                selected_path,
                                output_dir=os.path.join(target_dir, 'metadata'),
                                prefer_arm=fast_prefer_arm,
                                max_science_frames=fast_max_science_frames,
                            )
                            if subset_info:
                                calib_source = subset_info['path']
                                fast_calib_selection.append(
                                    f'{inst}:{subset_info["arm"]}:'
                                    f'{";".join(subset_info["koaid"])}')
                        if os.path.exists(calib_source):
                            koa.download_koa_files(
                                calib_source, download_dir, fmt='ipac',
                                lev0file=False, calibfile=True,
                                lev1file=False)
                        after = set(_list_fits(download_dir))
                        n_calib_downloaded += len(after - before)

                if pypeit_setup:
                    for instrument, raw_root in zip(instruments, raw_roots):
                        if str(instrument).lower() != 'lris':
                            continue
                        if not _list_fits(raw_root):
                            continue
                        setup = koa.setup_pypeit_lris(
                            raw_root,
                            setup_dir=os.path.join(
                                target_dir, 'pypeit_setup'),
                            run=bool(run_pypeit),
                            normalize_headers=normalize_lris_headers,
                            normalize_in_place=patch_raw_headers_in_place,
                            min_science_exptime=science_min_exptime)
                        norm = setup[0].get('normalization') if setup else {}
                        norm = norm or {}
                        if norm.get('raw_root'):
                            normalized_roots.append(norm.get('raw_root'))
                        n_header_normalized += int(
                            norm.get('n_normalized', 0) or 0)
                        n_bad_fits_skipped += int(
                            norm.get('n_skipped_bad', 0) or 0)
                        for item in setup:
                            pypeit_files.extend(item.get('pypeit_files') or [])
                        setup_status = 'run_ok' if run_pypeit else 'setup_ok'
                    n_setup_files = len(pypeit_files)

                if fast_pypeit and force and pypeit_files:
                    fast_runs = _run_fast_lris_1d(
                        target_dir,
                        pypeit_files,
                        target=target,
                        prefer_arm=fast_prefer_arm,
                        max_science_frames=fast_max_science_frames,
                        snr_thresh=fast_snr_thresh,
                    )
                    if fast_runs:
                        setup_status = 'fast_run_ok'

                spectrum_info = _standardize_existing_1d(
                    row, target_dir, radius_arcsec=radius_arcsec,
                    prefer_coadds=prefer_coadds, resample_step=resample_step,
                    scale_to_overlap=scale_to_overlap)
                if (spectrum_info is None and fast_pypeit and pypeit_files
                        and not fast_runs):
                    fast_runs = _run_fast_lris_1d(
                        target_dir,
                        pypeit_files,
                        target=target,
                        prefer_arm=fast_prefer_arm,
                        max_science_frames=fast_max_science_frames,
                        snr_thresh=fast_snr_thresh,
                    )
                    if fast_runs:
                        setup_status = 'fast_run_ok'
                        spectrum_info = _standardize_existing_1d(
                            row, target_dir, radius_arcsec=radius_arcsec,
                            prefer_coadds=prefer_coadds,
                            resample_step=resample_step,
                            scale_to_overlap=scale_to_overlap)
                if spectrum_info:
                    spectrum_status = 'found'
                    if fast_runs:
                        message = ('Fast LRIS 1D extraction completed from '
                                   f'{len(fast_runs)} PypeIt setup(s).')
                elif not message:
                    message = ('No readable extracted 1D spectrum found yet. '
                               'For LRIS, run PypeIt with calibration frames '
                               'or put spec1d/coadd1d FITS under this target.')
            n_fits_after = sum(len(_list_fits(root)) for root in raw_roots)
        except Exception as exc:
            status = 'error'
            error = f'{type(exc).__name__}: {exc}'
            n_fits_after = n_fits_before
            traceback.print_exc()

        result = spectrum_info.get('result') if spectrum_info else {}
        progress[key] = {
            'row_index': row_index,
            'target': target,
            'ra': row.get('ra'),
            'dec': row.get('dec'),
            'status': status,
            'error': error,
            'setup_status': setup_status,
            'spectrum_status': spectrum_status,
            'message': message,
            'n_selected_rows': _safe_int(row.get('n_selected_rows', 0)),
            'n_downloaded_fits_before': int(n_fits_before),
            'n_downloaded_fits_after': int(n_fits_after),
            'n_calib_downloaded': int(n_calib_downloaded),
            'n_header_normalized': int(n_header_normalized),
            'n_bad_fits_skipped': int(n_bad_fits_skipped),
            'n_pypeit_files': int(n_setup_files),
            'pypeit_files': ';'.join(pypeit_files),
            'normalized_raw_roots': ';'.join(sorted(set(normalized_roots))),
            'spectrum_root': spectrum_info.get('root') if spectrum_info else '',
            'spectrum_csv': spectrum_info.get('csv_path') if spectrum_info else '',
            'spectrum_png': spectrum_info.get('png_path') if spectrum_info else '',
            'spectrum_report': spectrum_info.get('report_path') if spectrum_info else '',
            'spectrum_exposures': spectrum_info.get('exposure_path')
            if spectrum_info else '',
            'n_spectrum_files_used': result.get('n_files', 0)
            if result else 0,
            'n_spectrum_points': int(len(result.get('wavelength', [])))
            if result and 'wavelength' in result else 0,
            'n_fast_runs': len(fast_runs),
            'fast_pypeit_files': ';'.join(
                run.get('fast_pypeit', '') for run in fast_runs),
            'fast_calib_selection': '|'.join(fast_calib_selection),
            'target_dir': target_dir,
        }
        _write_reduction_progress(reduction_path, list(progress.values()))
        print(f'{target}: setup={setup_status}, spectrum={spectrum_status}, '
              f'fits={progress[key]["n_downloaded_fits_after"]}, '
              f'1d_files={progress[key]["n_spectrum_files_used"]}',
              flush=True)

    return reduction_path


def run_batch(csv_path, output_root, instruments=DEFAULT_INSTRUMENTS,
              radius_arcsec=8.0, start=0, limit=None, download=True,
              calibfile=False, lev1file=False, make_png=True,
              row_limit=None, retries=3, retry_sleep=8.0,
              only_errors=False):
    df = pd.read_csv(csv_path)
    utils.ensure_dir(output_root)
    summary_path = os.path.join(output_root, 'koa_batch_summary.csv')
    manifest_path = os.path.join(output_root, 'koa_file_manifest.csv')

    end = len(df) if limit is None else min(len(df), start + int(limit))
    summary_by_index = _load_progress(summary_path)
    manifest_rows = list(_load_progress(manifest_path).values())
    error_indices = {
        idx for idx, row in summary_by_index.items()
        if str(row.get('status')) != 'ok'
    }

    for index in range(start, end):
        if only_errors and index not in error_indices:
            continue
        row = df.iloc[index]
        target = _target_from_row(row, index)
        safe_target = _safe_name(target, f'row_{index:04d}')
        ra, dec = _coord_from_row(row)
        target_dir = os.path.join(output_root, safe_target)
        png_dir = os.path.join(target_dir, 'png')
        status = 'ok'
        error = ''
        result = None

        try:
            if ra is None or dec is None:
                raise ValueError('Missing RA/Dec columns')
            last_exc = None
            for attempt in range(1, int(retries) + 1):
                try:
                    result = koa.prepare_koa_download(
                        ra=ra, dec=dec, target=target,
                        instruments=instruments,
                        radius_arcsec=radius_arcsec,
                        work_dir=target_dir,
                        row_limit=row_limit,
                        download=download,
                        lev0file=True,
                        calibfile=calibfile,
                        lev1file=lev1file,
                    )
                    break
                except Exception as exc:
                    last_exc = exc
                    if attempt >= int(retries):
                        raise
                    print(f'  retry {attempt}/{retries} for {target}: {exc}',
                          flush=True)
                    time.sleep(float(retry_sleep) * attempt)

            if make_png:
                for product in result.get('products', []):
                    download_dir = product.get('download_dir')
                    for path in product.get('downloaded_files', []) or []:
                        if not path.lower().endswith(('.fits', '.fit', '.fits.gz')):
                            continue
                        png_path = _relative_png_path(download_dir, path, png_dir)
                        try:
                            made = make_fits_quicklook(
                                path, png_path,
                                title=f'{target}  {os.path.basename(path)}')
                        except Exception:
                            made = None
                        manifest_rows.append({
                            'row_index': index,
                            'target': target,
                            'ra': ra,
                            'dec': dec,
                            'instrument': product.get('instrument'),
                            'fits_path': path,
                            'png_path': made,
                            'bytes': os.path.getsize(path)
                            if os.path.exists(path) else np.nan,
                        })
        except Exception as exc:
            status = 'error'
            error = f'{type(exc).__name__}: {exc}'
            traceback.print_exc()

        products = result.get('products', []) if result else []
        n_meta = sum(int(p.get('n_metadata_rows', 0)) for p in products)
        n_selected = sum(int(p.get('n_selected_rows', 0)) for p in products)
        n_downloaded = sum(len(p.get('downloaded_files') or [])
                           for p in products)
        summary_by_index[index] = {
            'row_index': index,
            'target': target,
            'ra': ra,
            'dec': dec,
            'status': status,
            'error': error,
            'n_metadata_rows': n_meta,
            'n_selected_rows': n_selected,
            'n_downloaded_files': n_downloaded,
            'target_dir': target_dir,
        }
        _write_progress(summary_path, list(summary_by_index.values()))
        _write_progress(manifest_path, manifest_rows)
        print(f'[{index + 1}/{len(df)}] {target}: {status}, '
              f'selected={n_selected}, downloaded={n_downloaded}',
              flush=True)

    return summary_path, manifest_path


def download_existing_metadata(output_root, instruments=DEFAULT_INSTRUMENTS,
                               calibfile=False, lev1file=False,
                               make_png=True, download_targets=None,
                               fast_download=False, fast_prefer_arm='red',
                               fast_max_science_frames=1):
    """Download FITS from existing per-target selected KOA metadata tables."""
    summary_path = os.path.join(output_root, 'koa_batch_summary.csv')
    manifest_path = os.path.join(output_root, 'koa_file_manifest.csv')
    if not os.path.exists(summary_path):
        raise FileNotFoundError(summary_path)

    summary = pd.read_csv(summary_path)
    manifest_by_path = {}
    if os.path.exists(manifest_path):
        old = pd.read_csv(manifest_path)
        for row in old.to_dict('records'):
            if isinstance(row.get('fits_path'), str):
                manifest_by_path[row['fits_path']] = row

    for idx, row in summary.iterrows():
        if row.get('status') != 'ok' or int(row.get('n_selected_rows', 0)) <= 0:
            continue
        if not _target_selected(row.to_dict(), download_targets):
            continue
        target = row['target']
        target_dir = row['target_dir']
        total_downloaded = 0
        for instrument in instruments:
            inst = str(instrument).lower()
            selected_path = os.path.join(
                target_dir, 'metadata', f'{inst}_selected.tbl')
            download_dir = os.path.join(target_dir, 'download', inst)
            if not os.path.exists(selected_path):
                continue
            download_source = selected_path
            if fast_download and inst == 'lris':
                subset_info = _prepare_fast_science_subset(
                    selected_path,
                    output_dir=os.path.join(target_dir, 'metadata'),
                    prefer_arm=fast_prefer_arm,
                    max_science_frames=fast_max_science_frames,
                )
                if subset_info:
                    download_source = subset_info['path']
            files = koa.download_koa_files(
                download_source, download_dir, fmt='ipac',
                lev0file=True, calibfile=calibfile, lev1file=lev1file)
            fits_files = [
                f for f in files
                if f.lower().endswith(('.fits', '.fit', '.fits.gz'))
            ]
            total_downloaded += len(fits_files)
            if make_png:
                png_dir = os.path.join(target_dir, 'png')
                for path in fits_files:
                    png_path = _relative_png_path(download_dir, path, png_dir)
                    made = None
                    try:
                        made = make_fits_quicklook(
                            path, png_path,
                            title=f'{target}  {os.path.basename(path)}')
                    except Exception:
                        made = None
                    manifest_by_path[path] = {
                        'row_index': int(row['row_index']),
                        'target': target,
                        'ra': row.get('ra'),
                        'dec': row.get('dec'),
                        'instrument': inst,
                        'fits_path': path,
                        'png_path': made,
                        'bytes': os.path.getsize(path)
                        if os.path.exists(path) else np.nan,
                    }
        summary.loc[idx, 'n_downloaded_files'] = total_downloaded
        summary.to_csv(summary_path, index=False)
        pd.DataFrame(manifest_by_path.values()).to_csv(
            manifest_path, index=False)
        print(f'{target}: downloaded/known FITS={total_downloaded}', flush=True)

    return summary_path, manifest_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=('Batch query/download KOA FITS, create PNG quicklooks, '
                     'and prepare/extract 1D spectra.'))
    parser.add_argument('csv_path')
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--instrument', action='append', default=None,
                        help='KOA instrument; repeat for multiple. Default: lris')
    parser.add_argument('--all-instruments', action='store_true')
    parser.add_argument('--radius-arcsec', type=float, default=8.0)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--metadata-only', action='store_true')
    parser.add_argument('--calibfile', action='store_true',
                        help='Also download KOA calibration files.')
    parser.add_argument('--lev1file', action='store_true',
                        help='Also request KOA level-1 files when available.')
    parser.add_argument('--no-png', action='store_true')
    parser.add_argument('--row-limit', type=int,
                        help='Limit KOA rows per target/instrument.')
    parser.add_argument('--retries', type=int, default=3)
    parser.add_argument('--download-existing-metadata', action='store_true',
                        help='Skip querying and download from existing *_selected.tbl files.')
    parser.add_argument('--only-errors', action='store_true',
                        help='Only rerun rows currently marked as errors in the summary.')
    parser.add_argument('--reduce-existing', action='store_true',
                        help='Use existing batch target dirs to create PypeIt setup/run reduction and export 1D spectra.')
    parser.add_argument('--auto-reduce-after-download', action='store_true',
                        help='After querying/downloading, immediately run --reduce-existing on the same output root.')
    parser.add_argument('--reduce-target', action='append', default=None,
                        help='Only reduce this target or safe directory name; repeatable.')
    parser.add_argument('--download-calib-for-reduction', action='store_true',
                        help='Before reduction, ask KOA for associated calibration files from selected metadata.')
    parser.add_argument('--run-pypeit', action='store_true',
                        help='Run PypeIt after setup. Without this, only setup files are generated and existing 1D products are exported.')
    parser.add_argument('--fast-pypeit', action='store_true',
                        help='Run the automated fast LRIS 1D extraction path: ASCII staging, longest science frame, and single-object extraction.')
    parser.add_argument('--fast-download', action='store_true',
                        help='When downloading from existing selected metadata, only fetch the longest science frame(s) from the preferred arm instead of all KOA exposures.')
    parser.add_argument('--fast-prefer-arm', default='red',
                        help='Preferred LRIS arm for --fast-pypeit. Default: red')
    parser.add_argument('--fast-max-science-frames', type=int, default=1,
                        help='How many science frames to keep per PypeIt file in --fast-pypeit mode. Default: 1')
    parser.add_argument('--fast-snr-thresh', type=float, default=20.0,
                        help='Object-finding S/N threshold used in --fast-pypeit mode. Default: 20')
    parser.add_argument('--pypeit-setup-only', action='store_true',
                        help='Alias for the default reduction behavior: create setup files but do not run PypeIt.')
    parser.add_argument('--no-pypeit-setup', action='store_true',
                        help='Only look for already extracted 1D FITS; do not create PypeIt setup files.')
    parser.add_argument('--no-normalize-lris-headers', action='store_true',
                        help='Do not create normalized LRIS FITS headers for PypeIt frame typing.')
    parser.add_argument('--patch-raw-headers-in-place', action='store_true',
                        help='Normalize LRIS headers in downloaded FITS instead of copying to pypeit_setup/pypeit_raw.')
    parser.add_argument('--science-min-exptime', type=float, default=30.0,
                        help='Minimum KOA object-frame exposure time to force as PypeIt science.')
    parser.add_argument('--reduce-all-summary', action='store_true',
                        help='Try reduction setup even when no FITS are currently downloaded.')
    parser.add_argument('--force-reduce', action='store_true',
                        help='Rerun reduction/export even if koa_reduction_summary already has a spectrum.')
    parser.add_argument('--resample-step', type=float,
                        help='Final 1D wavelength grid step in Angstrom; default uses config.KOA_RESAMPLE_STEP_A.')
    parser.add_argument('--no-scale-overlap', action='store_true',
                        help='Do not scale overlapping blue/red spectra before combining.')
    args = parser.parse_args(argv)

    if args.all_instruments:
        instruments = koa.KOA_INSTRUMENTS
    else:
        instruments = tuple(args.instrument or DEFAULT_INSTRUMENTS)

    reduction_path = None

    if args.reduce_existing:
        summary_path = os.path.join(args.output_root, 'koa_batch_summary.csv')
        manifest_path = os.path.join(args.output_root, 'koa_file_manifest.csv')
        reduction_path = reduce_existing_metadata(
            args.output_root,
            instruments=instruments,
            start=args.start,
            limit=args.limit,
            reduce_targets=args.reduce_target,
            download_calib=args.download_calib_for_reduction,
            pypeit_setup=not args.no_pypeit_setup,
            run_pypeit=bool(args.run_pypeit and not args.pypeit_setup_only
                            and not args.fast_pypeit),
            radius_arcsec=args.radius_arcsec,
            resample_step=args.resample_step,
            scale_to_overlap=not args.no_scale_overlap,
            only_downloaded=not args.reduce_all_summary,
            force=args.force_reduce,
            normalize_lris_headers=not args.no_normalize_lris_headers,
            patch_raw_headers_in_place=args.patch_raw_headers_in_place,
            science_min_exptime=args.science_min_exptime,
            fast_pypeit=args.fast_pypeit,
            fast_prefer_arm=args.fast_prefer_arm,
            fast_max_science_frames=args.fast_max_science_frames,
            fast_snr_thresh=args.fast_snr_thresh,
        )
    elif args.download_existing_metadata:
        summary_path, manifest_path = download_existing_metadata(
            args.output_root,
            instruments=instruments,
            calibfile=args.calibfile,
            lev1file=args.lev1file,
            make_png=not args.no_png,
            download_targets=args.reduce_target,
            fast_download=args.fast_download,
            fast_prefer_arm=args.fast_prefer_arm,
            fast_max_science_frames=args.fast_max_science_frames,
        )
    else:
        summary_path, manifest_path = run_batch(
            args.csv_path,
            args.output_root,
            instruments=instruments,
            radius_arcsec=args.radius_arcsec,
            start=args.start,
            limit=args.limit,
            download=not args.metadata_only,
            calibfile=args.calibfile,
            lev1file=args.lev1file,
            make_png=not args.no_png,
            row_limit=args.row_limit,
            retries=args.retries,
            only_errors=args.only_errors,
        )
    if args.auto_reduce_after_download and not args.reduce_existing:
        reduction_path = reduce_existing_metadata(
            args.output_root,
            instruments=instruments,
            start=args.start,
            limit=args.limit,
            reduce_targets=args.reduce_target,
            download_calib=args.download_calib_for_reduction,
            pypeit_setup=not args.no_pypeit_setup,
            run_pypeit=bool(args.run_pypeit and not args.pypeit_setup_only
                            and not args.fast_pypeit),
            radius_arcsec=args.radius_arcsec,
            resample_step=args.resample_step,
            scale_to_overlap=not args.no_scale_overlap,
            only_downloaded=not args.reduce_all_summary,
            force=args.force_reduce,
            normalize_lris_headers=not args.no_normalize_lris_headers,
            patch_raw_headers_in_place=args.patch_raw_headers_in_place,
            science_min_exptime=args.science_min_exptime,
            fast_pypeit=args.fast_pypeit,
            fast_prefer_arm=args.fast_prefer_arm,
            fast_max_science_frames=args.fast_max_science_frames,
            fast_snr_thresh=args.fast_snr_thresh,
        )
    print(f'summary: {summary_path}')
    print(f'manifest: {manifest_path}')
    if reduction_path:
        print(f'reduction: {reduction_path}')


if __name__ == '__main__':
    main()
