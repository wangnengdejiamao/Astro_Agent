#!/usr/bin/env python3
"""Curate sdss_lamost_desi spectra with SIMBAD and rerun local toolbox analyses.

This script is intentionally offline except for the SIMBAD screening step.  It
uses the old ``sdss_lamost_desi/results`` folders only as a source of cached
light curves / SED tables, writes a clean rerun tree, and moves rejected or
duplicate spectra out of ``all_spectra`` into a quarantine directory.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
PARENT_DIR = ROOT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from astro_toolbox import hr_diagram, run_existing_astro_output_analysis, six_dim, wd_fitting  # noqa: E402


DEFAULT_BASE = Path('/Users/ljm/Desktop/csst/desi匹配/未回溯t1234/sdss_lamost_desi')

WD_SIMBAD_TOKENS = (
    'WD', 'WD*', 'WD?', 'WhiteDwarf', 'White Dwarf', 'CV*', 'CataclyV',
    'Cataclysmic', 'AM Her', 'DQ', 'DZ', 'DA', 'DB', 'DO',
)
NORMAL_STAR_TOKENS = (
    'Star', 'PM*', 'V*', 'IR', 'UV', 'HighPM*', 'Candidate_RGB',
)
SPECTRUM_RE = re.compile(
    r'^(?P<survey>[A-Za-z0-9]+)_ra=(?P<ra>[-+]?\d+(?:\.\d+)?)'
    r'_dec=(?P<dec>[-+]?\d+(?:\.\d+)?)(?:_obsid=.*)?\.(?P<ext>[^.]+)$')


def safe_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().lower() in {'true', '1', 'yes', 'y'}


def as_float(value, default=np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def source_dir_name(row: pd.Series) -> str:
    sid = str(int(float(row.get('source_id'))))
    cluster = str(row.get('cluster', '') or '').strip()
    name = f'{cluster}_Gaia_{sid}' if cluster else f'Gaia_{sid}'
    return name.replace(' ', '_').replace('/', '_')


def parse_spectrum_name(path: Path) -> dict | None:
    m = SPECTRUM_RE.match(path.name)
    if not m:
        return None
    d = m.groupdict()
    d['survey'] = d['survey'].lower()
    d['ra'] = float(d['ra'])
    d['dec'] = float(d['dec'])
    d['ext'] = d['ext'].lower()
    return d


def match_catalog(merged: pd.DataFrame, ra: float, dec: float,
                  max_sep_arcsec: float) -> tuple[int | None, float]:
    ras = merged['_ra_float'].to_numpy()
    decs = merged['_dec_float'].to_numpy()
    sep = np.hypot((ras - ra) * np.cos(np.deg2rad(dec)), decs - dec) * 3600.0
    if not np.any(np.isfinite(sep)):
        return None, np.nan
    i = int(np.nanargmin(sep))
    if sep[i] <= max_sep_arcsec:
        return i, float(sep[i])
    return None, float(sep[i])


def read_spectrum_file(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None] | None:
    try:
        if path.suffix.lower() == '.csv':
            df = pd.read_csv(path)
            cols = {c.lower(): c for c in df.columns}
            wcol = cols.get('wavelength_a') or cols.get('wavelength') or cols.get('wave')
            fcol = cols.get('flux') or cols.get('flam')
            ecol = cols.get('error') or cols.get('flux_err') or cols.get('ivar')
            if wcol is None or fcol is None:
                return None
            wave = pd.to_numeric(df[wcol], errors='coerce').to_numpy(float)
            flux = pd.to_numeric(df[fcol], errors='coerce').to_numpy(float)
            err = None
            if ecol is not None:
                vals = pd.to_numeric(df[ecol], errors='coerce').to_numpy(float)
                if ecol.lower() == 'ivar':
                    err = np.where(vals > 0, 1.0 / np.sqrt(vals), np.nan)
                else:
                    err = vals
        else:
            from astropy.io import fits
            with fits.open(path, memmap=False) as hdul:
                data = None
                for hdu in hdul[1:]:
                    if getattr(hdu, 'data', None) is not None:
                        data = hdu.data
                        names = [n.lower() for n in (data.names or [])]
                        if 'flux' in names and ('loglam' in names or 'wavelength' in names):
                            break
                if data is None or data.names is None:
                    return None
                cols = {n.lower(): n for n in data.names}
                if 'loglam' in cols:
                    wave = 10.0 ** np.asarray(data[cols['loglam']], dtype=float)
                elif 'wavelength' in cols:
                    wave = np.asarray(data[cols['wavelength']], dtype=float)
                else:
                    return None
                flux = np.asarray(data[cols['flux']], dtype=float)
                err = None
                if 'ivar' in cols:
                    ivar = np.asarray(data[cols['ivar']], dtype=float)
                    err = np.where(ivar > 0, 1.0 / np.sqrt(ivar), np.nan)
        good = np.isfinite(wave) & np.isfinite(flux)
        if err is not None:
            err = np.asarray(err, dtype=float)
            if len(err) == len(wave):
                good &= np.isfinite(err) | ~np.isfinite(err)
            else:
                err = None
        if np.sum(good) < 50:
            return None
        order = np.argsort(wave[good])
        w = wave[good][order]
        f = flux[good][order]
        e = err[good][order] if err is not None else None
        return w, f, e
    except Exception:
        return None


def spectrum_quality(path: Path) -> dict:
    spec = read_spectrum_file(path)
    if spec is None:
        return {
            'n_pix': 0, 'wave_min': np.nan, 'wave_max': np.nan,
            'median_snr': np.nan, 'quality_score': -np.inf,
        }
    w, f, e = spec
    snr = np.nan
    if e is not None:
        good = np.isfinite(e) & (e > 0) & np.isfinite(f)
        if np.any(good):
            snr = float(np.nanmedian(np.abs(f[good] / e[good])))
    coverage = float(np.nanmax(w) - np.nanmin(w)) if len(w) else 0.0
    score = len(w) + 0.02 * coverage + (snr if np.isfinite(snr) else 0.0)
    return {
        'n_pix': int(len(w)),
        'wave_min': float(np.nanmin(w)),
        'wave_max': float(np.nanmax(w)),
        'median_snr': snr,
        'quality_score': score,
    }


def classify_spectrum_local(path: Path) -> dict:
    spec = read_spectrum_file(path)
    if spec is None:
        return {'local_wd_type': '', 'local_wd_confidence': np.nan}
    w, f, e = spec
    try:
        cl = wd_fitting.classify_wd_type(w, f, e)
        return {
            'local_wd_type': cl.get('spectral_type', ''),
            'local_wd_confidence': cl.get('confidence', np.nan),
            'local_balmer_ew': cl.get('balmer_total_ew', np.nan),
            'local_he_ew': cl.get('he_total_ew', np.nan),
        }
    except Exception:
        return {'local_wd_type': '', 'local_wd_confidence': np.nan}


def query_simbad_one(ra: float, dec: float, radius_arcsec: float) -> dict:
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astroquery.simbad import Simbad

    simbad = Simbad()
    for field in ('otype', 'ids'):
        try:
            simbad.add_votable_fields(field)
        except Exception:
            pass
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    tab = simbad.query_region(coord, radius=radius_arcsec * u.arcsec)
    if tab is None or len(tab) == 0:
        return {
            'simbad_status': 'no_match',
            'simbad_main_id': '',
            'simbad_otype': '',
            'simbad_ids': '',
        }
    row = tab[0]
    names = set(tab.colnames)

    def cell(*cols):
        for col in cols:
            if col in names:
                val = row[col]
                if isinstance(val, bytes):
                    val = val.decode('utf-8', 'ignore')
                return str(val)
        return ''

    return {
        'simbad_status': 'ok',
        'simbad_main_id': cell('MAIN_ID'),
        'simbad_otype': cell('OTYPE', 'OTYPE_S'),
        'simbad_ids': cell('IDS'),
    }


def load_or_query_simbad(unique_sources: pd.DataFrame, cache_path: Path,
                         radius_arcsec: float, sleep_sec: float) -> pd.DataFrame:
    if cache_path.exists():
        cached = pd.read_csv(cache_path, dtype={'source_id': str})
    else:
        cached = pd.DataFrame()
    done = set(cached['source_id'].astype(str)) if len(cached) else set()
    rows = []
    for _, row in unique_sources.iterrows():
        sid = str(row['source_id'])
        if sid in done:
            continue
        try:
            res = query_simbad_one(float(row['ra']), float(row['dec']), radius_arcsec)
        except Exception as exc:
            res = {
                'simbad_status': f'error:{type(exc).__name__}',
                'simbad_main_id': '',
                'simbad_otype': '',
                'simbad_ids': '',
                'simbad_error': str(exc)[:500],
            }
        res.update({
            'source_id': sid,
            'ra': float(row['ra']),
            'dec': float(row['dec']),
        })
        rows.append(res)
        if len(rows) % 20 == 0:
            tmp = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True)
            tmp.to_csv(cache_path, index=False)
            print(f'  SIMBAD cached {len(tmp)}/{len(unique_sources)}', flush=True)
        if sleep_sec > 0:
            time.sleep(sleep_sec)
    if rows:
        cached = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True)
        cached.to_csv(cache_path, index=False)
    return cached


def compute_hr_region(row: pd.Series) -> tuple[str, float]:
    plx = as_float(row.get('parallax'))
    gmag = as_float(row.get('phot_g_mean_mag'))
    bp_rp = as_float(row.get('bp_rp'))
    if not (np.isfinite(plx) and plx > 0 and np.isfinite(gmag) and np.isfinite(bp_rp)):
        return '', np.nan
    m_g = gmag + 5.0 * np.log10(plx / 1000.0) + 5.0
    try:
        analysis = hr_diagram.classify_hr_position(
            bp_rp, m_g, include_wd_model=False)
        region = analysis.get('region', '')
    except Exception:
        region = ''
    return region, float(m_g)


def decide_keep(row: pd.Series) -> tuple[bool, str, str]:
    simbad_text = ' '.join([
        str(row.get('simbad_otype', '')),
        str(row.get('simbad_ids', '')),
        str(row.get('simbad_main_id', '')),
    ])
    simbad_wd = any(tok.lower() in simbad_text.lower() for tok in WD_SIMBAD_TOKENS)
    simbad_normal = any(tok.lower() == str(row.get('simbad_otype', '')).lower()
                        for tok in NORMAL_STAR_TOKENS)
    catalog_text = ' '.join([
        str(row.get('sdss_subclass', '')),
        str(row.get('LRS_subclass', '')),
    ])
    # Do not treat the old fitted spectral_type=DA/DB/DC as a stand-alone
    # keep reason; some normal A/F stars were previously over-fit by WD models.
    catalog_wd = 'wd' in catalog_text.lower()
    hr_region = str(row.get('hr_region', ''))
    hr_keep = hr_region in {'white_dwarf_sequence', 'wd_ms_composite'}
    simple_wd_hr = (
        np.isfinite(as_float(row.get('M_G'))) and as_float(row.get('M_G')) > 8.0
        and np.isfinite(as_float(row.get('bp_rp'))) and as_float(row.get('bp_rp')) < 1.8
    )
    local_type = str(row.get('local_wd_type', '')).upper()
    local_conf = as_float(row.get('local_wd_confidence'))
    local_support = local_type.startswith(('DA', 'DB', 'DC')) and np.isfinite(local_conf) and local_conf >= 0.65

    if simbad_wd:
        return True, 'wd_or_cv', 'SIMBAD WD/CV token'
    if hr_region == 'wd_ms_composite':
        return True, 'wd_ms_candidate', 'Gaia HR WD+MS/composite region'
    if catalog_wd and (hr_keep or simple_wd_hr):
        return True, 'white_dwarf', 'catalog WD plus Gaia HR support'
    if hr_keep or simple_wd_hr:
        return True, 'white_dwarf_candidate', 'Gaia HR WD-region support'
    if catalog_wd and not simbad_normal:
        return True, 'white_dwarf_candidate', 'SDSS/LAMOST subclass WD'
    if local_support and (hr_keep or simple_wd_hr) and not simbad_normal:
        return True, 'white_dwarf_candidate', 'local WD spectral-line plus Gaia HR support'
    return False, 'reject_non_wd', 'no WD/WD+MS support from SIMBAD, Gaia HR, or catalog'


def write_standard_spectrum(src: Path, dst: Path) -> bool:
    spec = read_spectrum_file(src)
    if spec is None:
        return False
    w, f, e = spec
    if e is None or len(e) != len(w):
        med = np.nanmedian(np.abs(f[np.isfinite(f)]))
        e = np.full_like(f, max(0.05 * med, 1.0))
    out = pd.DataFrame({
        'wavelength_A': w,
        'flux': f,
        'error': e,
    })
    out = out[np.isfinite(out['wavelength_A']) & np.isfinite(out['flux'])]
    if len(out) < 50:
        return False
    out.to_csv(dst, index=False)
    return True


def copy_cached_context(old_dir: Path, new_dir: Path) -> None:
    if not old_dir.exists():
        return
    for src in old_dir.glob('*.csv'):
        name = src.name.lower()
        if 'spectrum' in name or name in {'wd_fitting.csv'}:
            continue
        dst = new_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
    for src in old_dir.glob('*.json'):
        dst = new_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
    for src in old_dir.glob('orbit_traceback.txt'):
        dst = new_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)


def build_selected_row(row: pd.Series, final_row: pd.Series | None) -> pd.DataFrame:
    data = row.to_dict()
    if final_row is not None:
        for key, value in final_row.to_dict().items():
            if key not in data or pd.isna(data.get(key)):
                data[key] = value
    if 'cluster_age_gyr' not in data and 'cluster_age_myr' in data:
        data['cluster_age_gyr'] = as_float(data.get('cluster_age_myr')) / 1000.0
    return pd.DataFrame([data])


def make_analysis_args(skip_magnetic: bool, wd_model_grid: str) -> SimpleNamespace:
    return SimpleNamespace(
        skip_combined=False,
        skip_diagnostics=False,
        skip_period=False,
        skip_rv=False,
        skip_rv_correction=False,
        skip_sixdim=False,
        skip_wd=False,
        wd_model_grid=wd_model_grid,
        skip_magnetic=skip_magnetic,
        magnetic_series='Halpha,Hbeta',
        magnetic_b_min_mg=5.0,
        magnetic_b_max_mg=950.0,
        magnetic_n_b_grid=320,
        magnetic_rv_min=-250.0,
        magnetic_rv_max=250.0,
        magnetic_rv_step=25.0,
        magnetic_search_half_width_A=8.0,
        magnetic_min_depth=0.04,
        magnetic_min_snr=3.0,
        magnetic_emission_avoid_A=10.0,
        magnetic_absorption_core_avoid_A=25.0,
        magnetic_baseline_mode='continuum',
        magnetic_wd_model_grid='auto',
        verbose_errors=False,
    )


def choose_best_rv(row: pd.Series, final_row: pd.Series | None) -> tuple[float, float]:
    candidates = []
    for prefix in ('sdss', 'desi', 'lamost'):
        rv = as_float(row.get(f'{prefix}_rv_true_kms'))
        err = as_float(row.get(f'{prefix}_rv_true_err_kms'))
        if np.isfinite(rv):
            candidates.append((err if np.isfinite(err) else 999.0, rv, err))
    if final_row is not None:
        rv = as_float(final_row.get('rv_true'))
        err = as_float(final_row.get('rv_true_err'))
        if np.isfinite(rv):
            candidates.append((err if np.isfinite(err) else 999.0, rv, err))
    if not candidates:
        return np.nan, np.nan
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1], candidates[0][2]


def write_curated_cluster_summary(
    sources: pd.DataFrame,
    summary: pd.DataFrame,
    final_map: dict,
    output_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write final curated source summary and count table from existing analyses."""
    try:
        cluster_cache = six_dim._load_default_cluster_cache()
    except Exception:
        cluster_cache = None
    cluster_age_lookup = {}
    if cluster_cache:
        for cl in cluster_cache:
            try:
                log_age = float(cl.get('logAge50', np.nan))
            except Exception:
                log_age = np.nan
            if np.isfinite(log_age):
                key = str(cl.get('Name', '')).replace('_', ' ').strip().lower()
                cluster_age_lookup[key] = 10.0 ** log_age / 1e9

    if 'keep' in sources.columns:
        keep_mask = sources['keep'].apply(safe_bool)
    else:
        keep_mask = pd.Series(np.ones(len(sources), dtype=bool), index=sources.index)

    final_rows = []
    for _, src in sources[keep_mask].iterrows():
        sid = str(src['source_id'])
        target = source_dir_name(src)
        ana = summary[summary['target'] == target] if 'target' in summary.columns else pd.DataFrame()
        ana_row = ana.iloc[0] if len(ana) else pd.Series(dtype=object)
        final_row = final_map.get(sid)
        rv, rv_err = choose_best_rv(ana_row, final_row)
        dyn = {}
        if np.isfinite(rv):
            try:
                dyn = six_dim.check_6d_match(src, rv, rv_err, cluster_cache=cluster_cache)
            except Exception as exc:
                dyn = {'match_note': f'6d_check_failed:{exc}', 'is_6d_matched': False}
        else:
            dyn = {'match_note': 'no_rv_true', 'is_6d_matched': False}
        tier = str(src.get('tier', ''))
        if final_row is not None:
            old_membership = str(final_row.get('membership', src.get('membership', '')))
        else:
            old_membership = str(src.get('membership', ''))
        old_orbit = (
            safe_bool(src.get('orbit_confirmed'))
            or safe_bool(src.get('orbit_within_tidal'))
            or old_membership in {'backtrack_matched', 'backtrack_only'}
        )
        backtrack_t23 = tier in {'Tier2', 'Tier3'} and old_orbit
        cluster_age = as_float(src.get('cluster_age_gyr'))
        cluster_age_source = 'cluster_age_gyr' if np.isfinite(cluster_age) else ''
        if not np.isfinite(cluster_age):
            cluster_age_myr = as_float(src.get('cluster_age_myr'))
            cluster_age = cluster_age_myr / 1000.0
            cluster_age_source = 'cluster_age_myr' if np.isfinite(cluster_age) else ''
        if not np.isfinite(cluster_age):
            key = str(src.get('cluster', '')).replace('_', ' ').strip().lower()
            cluster_age = cluster_age_lookup.get(key, np.nan)
            cluster_age_source = 'Hunt+2023 logAge50' if np.isfinite(cluster_age) else ''
        cooling_age = as_float(ana_row.get('wd_cooling_age_gyr', src.get('cooling_age_gyr', np.nan)))
        wd_mass = as_float(ana_row.get('wd_mass_msun', src.get('mass', np.nan)))
        progenitor_mass = as_float(
            ana_row.get('wd_progenitor_mass_msun',
                        src.get('m_progenitor',
                                src.get('m_progenitor_msun',
                                        src.get('progenitor_mass_msun', np.nan)))))
        ms_lifetime = as_float(
            ana_row.get('wd_ms_lifetime_gyr',
                        src.get('ms_lifetime_gyr',
                                src.get('single_star_ms_lifetime_gyr', np.nan))))
        total_age = as_float(
            ana_row.get('wd_total_age_gyr',
                        src.get('total_age_with_ms_gyr',
                                src.get('total_age_gyr', np.nan))))
        if (not np.isfinite(ms_lifetime) or not np.isfinite(progenitor_mass)
                or not np.isfinite(total_age)):
            try:
                try:
                    from .wd_age_methods import single_star_ifmr_age
                except Exception:
                    from wd_age_methods import single_star_ifmr_age
                age_info = single_star_ifmr_age(wd_mass, cooling_age)
                if not np.isfinite(progenitor_mass):
                    progenitor_mass = as_float(age_info.get('m_initial_msun'))
                if not np.isfinite(ms_lifetime):
                    ms_lifetime = as_float(age_info.get('ms_lifetime_gyr'))
                if not np.isfinite(total_age):
                    total_age = as_float(age_info.get('total_age_gyr'))
            except Exception:
                pass
        if not np.isfinite(total_age):
            total_age = (
                cooling_age + ms_lifetime
                if np.isfinite(cooling_age) and np.isfinite(ms_lifetime)
                else np.nan
            )
        cooling_age_gt_cluster = bool(
            np.isfinite(cooling_age) and np.isfinite(cluster_age) and cooling_age > cluster_age)
        total_age_gt_cluster = bool(
            np.isfinite(total_age) and np.isfinite(cluster_age) and total_age > cluster_age)
        age_gt_cluster = total_age_gt_cluster or (
            not np.isfinite(total_age) and cooling_age_gt_cluster)
        out = {
            'source_id': sid,
            'target': target,
            'ra': src.get('ra', np.nan),
            'dec': src.get('dec', np.nan),
            'parallax': src.get('parallax', np.nan),
            'pmra': src.get('pmra', np.nan),
            'pmdec': src.get('pmdec', np.nan),
            'phot_g_mean_mag': src.get('phot_g_mean_mag', np.nan),
            'chi2_kin': src.get('chi2_kin', np.nan),
            'chi2_spatial': src.get('chi2_spatial', np.nan),
            'sep_pc': src.get('sep_pc', np.nan),
            'delta_vt_kms': src.get('delta_vt_kms', np.nan),
            'cluster': src.get('cluster', ''),
            'tier': tier,
            'curated_class': src.get('curated_class', ''),
            'curation_reason': src.get('curation_reason', ''),
            'simbad_main_id': src.get('simbad_main_id', ''),
            'simbad_otype': src.get('simbad_otype', ''),
            'hr_region': src.get('hr_region', ''),
            'M_G': src.get('M_G', np.nan),
            'bp_rp': src.get('bp_rp', np.nan),
            'toolbox_status': ana_row.get('status', ''),
            'wd_spectral_type': ana_row.get('wd_spectral_type', src.get('spectral_type', '')),
            'wd_teff': ana_row.get('wd_teff', src.get('teff', np.nan)),
            'wd_logg': ana_row.get('wd_logg', src.get('logg', np.nan)),
            'wd_mass_msun': wd_mass,
            'wd_cooling_age_gyr': cooling_age,
            'wd_progenitor_mass_msun': progenitor_mass,
            'wd_ms_lifetime_gyr': ms_lifetime,
            'wd_total_age_gyr': total_age,
            'cluster_age_gyr': cluster_age,
            'cluster_age_source': cluster_age_source,
            'wd_cooling_age_gt_cluster': cooling_age_gt_cluster,
            'wd_cooling_age_minus_cluster_gyr': (
                cooling_age - cluster_age
                if np.isfinite(cooling_age) and np.isfinite(cluster_age) else np.nan
            ),
            'wd_total_age_gt_cluster': total_age_gt_cluster,
            'age_gt_cluster': age_gt_cluster,
            'rv_true_adopted': rv,
            'rv_true_err_adopted': rv_err,
            'is_6d_matched_new': dyn.get('is_6d_matched', False),
            'is_6d_strict_new': dyn.get('is_6d_strict', False),
            'is_6d_borderline_new': dyn.get('is_6d_borderline', False),
            'rv_match_quality_new': dyn.get('rv_match_quality', ''),
            'match_path_new': dyn.get('match_path', ''),
            'match_note_new': dyn.get('match_note', ''),
            'rv_diff_kms_new': dyn.get('rv_diff_kms', np.nan),
            'rv_sigma_new': dyn.get('rv_sigma', np.nan),
            'cluster_rv_new': dyn.get('cluster_rv', np.nan),
            'cluster_rv_err_new': dyn.get('cluster_rv_err', np.nan),
            'cluster_rv_source_new': dyn.get('cluster_rv_source', ''),
            'cluster_rv_n_new': dyn.get('cluster_rv_n', np.nan),
            'old_membership': old_membership,
            'backtrack_to_cluster_t23': backtrack_t23,
            'reference_period_hour': ana_row.get('reference_period_hour', np.nan),
            'reference_period_source': ana_row.get('reference_period_source', ''),
            'has_sed_photometry': ana_row.get('has_sed_photometry', False),
            'wd_sed_ir_excess_flag': ana_row.get('wd_sed_ir_excess_flag', ''),
            'wd_sed_ir_excess_bands': ana_row.get('wd_sed_ir_excess_bands', ''),
            'spectral_diagnostics_csv': ana_row.get('spectral_diagnostics_csv', ''),
            'period_figures': ana_row.get('period_figures', ''),
            'source_dir': ana_row.get('source_dir', str(output_root / target)),
        }
        final_rows.append(out)

    final_summary = pd.DataFrame(final_rows)
    final_summary.to_csv(output_root / 'curated_wd_cluster_summary.csv', index=False)

    curated_class = final_summary.get('curated_class', '').astype(str)
    is_6d = final_summary.get('is_6d_matched_new', False) == True
    is_6d_strict = final_summary.get('is_6d_strict_new', False) == True
    is_6d_borderline = final_summary.get('is_6d_borderline_new', False) == True
    backtrack_mask = final_summary.get('backtrack_to_cluster_t23', False) == True
    age_gt_cluster_mask = final_summary.get('age_gt_cluster', False) == True
    cooling_age_gt_cluster_mask = final_summary.get('wd_cooling_age_gt_cluster', False) == True
    wd_ms_mask = curated_class.str.contains('wd_ms', na=False)

    final_summary.loc[cooling_age_gt_cluster_mask].to_csv(
        output_root / 'wd_cooling_age_gt_cluster_all_20260430.csv',
        index=False)
    final_summary.loc[cooling_age_gt_cluster_mask & is_6d_strict].to_csv(
        output_root / 'wd_cooling_age_gt_cluster_6d_strict_20260430.csv',
        index=False)
    final_summary.loc[cooling_age_gt_cluster_mask & is_6d_borderline].to_csv(
        output_root / 'wd_cooling_age_gt_cluster_6d_borderline_20260430.csv',
        index=False)
    final_summary.loc[cooling_age_gt_cluster_mask & (is_6d_strict | backtrack_mask)].to_csv(
        output_root / 'wd_cooling_age_gt_cluster_6d_strict_or_t23_20260430.csv',
        index=False)
    final_summary.loc[
        cooling_age_gt_cluster_mask
        & (final_summary.get('tier', '').astype(str) == 'Tier1')
        & ~is_6d
        & ~pd.to_numeric(final_summary.get('rv_true_adopted', np.nan), errors='coerce').notna()
    ].to_csv(
        output_root / 'wd_cooling_age_gt_cluster_tier1_no_rv_20260430.csv',
        index=False)

    counts = []
    for name, mask in {
        'curated_sources': np.ones(len(final_summary), dtype=bool),
        'new_6d_matched': is_6d,
        'new_6d_strict': is_6d_strict,
        'new_6d_borderline_candidate': is_6d_borderline,
        't23_backtrack_to_cluster': backtrack_mask,
        'wd_ms_candidates': wd_ms_mask,
        'age_gt_cluster': age_gt_cluster_mask,
        'wd_cooling_age_gt_cluster': cooling_age_gt_cluster_mask,
        'wd_cooling_age_gt_cluster_and_6d': cooling_age_gt_cluster_mask & is_6d,
        'wd_cooling_age_gt_cluster_and_6d_strict': cooling_age_gt_cluster_mask & is_6d_strict,
        'wd_cooling_age_gt_cluster_and_6d_borderline': cooling_age_gt_cluster_mask & is_6d_borderline,
        't23_backtrack_age_gt_cluster': backtrack_mask & cooling_age_gt_cluster_mask,
        'wd_ms_age_gt_cluster': wd_ms_mask & age_gt_cluster_mask,
        'wd_ms_age_gt_cluster_and_6d_strict': wd_ms_mask & age_gt_cluster_mask & is_6d_strict,
        'wd_ms_age_gt_cluster_and_6d_borderline': wd_ms_mask & age_gt_cluster_mask & is_6d_borderline,
        'sed_ir_excess': final_summary.get('wd_sed_ir_excess_flag', '').astype(str).str.lower().isin({'true', '1'}),
        'has_period': pd.to_numeric(final_summary.get('reference_period_hour', np.nan), errors='coerce').notna(),
    }.items():
        counts.append({'metric': name, 'count': int(np.sum(mask))})
    counts_df = pd.DataFrame(counts)
    counts_df.to_csv(output_root / 'curated_wd_cluster_counts.csv', index=False)
    print('\nFinal counts:')
    print(counts_df.to_string(index=False))
    print(f'\nOutputs: {output_root}')
    return final_summary, counts_df


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, default=DEFAULT_BASE)
    parser.add_argument('--output-root', type=Path, default=None)
    parser.add_argument('--simbad-cache', type=Path, default=None)
    parser.add_argument('--simbad-radius-arcsec', type=float, default=5.0)
    parser.add_argument('--match-radius-arcsec', type=float, default=2.0)
    parser.add_argument('--simbad-sleep-sec', type=float, default=0.05)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--no-quarantine', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--summary-only', action='store_true',
                        help='rebuild final 6D/count CSVs from existing curation and toolbox summary')
    parser.add_argument('--skip-magnetic', action='store_true', default=True)
    parser.add_argument('--wd-model-grid', default='auto')
    args = parser.parse_args(argv)

    base = args.base.expanduser().resolve()
    all_spectra = base / 'all_spectra'
    results_root = base / 'results'
    output_root = args.output_root or (base / 'toolbox_simbad_wd_rerun_20260430')
    output_root.mkdir(parents=True, exist_ok=True)
    quarantine = base / 'all_spectra_quarantine_20260430_simbad_nonwd'
    if not args.no_quarantine:
        quarantine.mkdir(parents=True, exist_ok=True)

    merged = pd.read_csv(base / 'merged_all.csv')
    merged['_ra_float'] = pd.to_numeric(merged['ra'], errors='coerce')
    merged['_dec_float'] = pd.to_numeric(merged['dec'], errors='coerce')
    final_path = base / 'wd_analysis_final.csv'
    final = pd.read_csv(final_path) if final_path.exists() else pd.DataFrame()
    if len(final):
        final['_sid_str'] = final['source_id'].apply(lambda x: str(int(float(x))))
        final_map = {str(s): r for s, r in final.set_index('_sid_str').iterrows()}
    else:
        final_map = {}

    if args.summary_only:
        sources_path = output_root / 'simbad_wd_source_curation.csv'
        summary_path = output_root / 'toolbox_rerun_summary.csv'
        if not sources_path.exists() or not summary_path.exists():
            raise FileNotFoundError(
                f'--summary-only requires {sources_path.name} and {summary_path.name} in {output_root}'
            )
        sources = pd.read_csv(sources_path, dtype={'source_id': str})
        summary = pd.read_csv(summary_path)
        write_curated_cluster_summary(sources, summary, final_map, output_root)
        return 0

    inventory_rows = []
    for path in sorted(all_spectra.iterdir()):
        if not path.is_file():
            continue
        parsed = parse_spectrum_name(path)
        if parsed is None:
            continue
        cat_idx, sep = match_catalog(
            merged, parsed['ra'], parsed['dec'], args.match_radius_arcsec)
        q = spectrum_quality(path)
        row = {
            'path': str(path),
            'filename': path.name,
            **parsed,
            **q,
            'catalog_match_sep_arcsec': sep,
            'source_id': '',
            'cluster': '',
            'tier': '',
        }
        if cat_idx is not None:
            cat = merged.iloc[cat_idx]
            row.update({
                'source_id': str(int(float(cat['source_id']))),
                'cluster': cat.get('cluster', ''),
                'tier': cat.get('tier', ''),
            })
        inventory_rows.append(row)
    inv = pd.DataFrame(inventory_rows)
    inv.to_csv(output_root / 'all_spectra_inventory.csv', index=False)
    matched = inv[inv['source_id'].astype(str) != ''].copy()
    print(f'Inventory: {len(inv)} spectra, matched to catalog: {len(matched)}')

    if matched.empty:
        return 1

    unique_ids = sorted(matched['source_id'].astype(str).unique())
    source_rows = []
    for sid in unique_ids:
        cat = merged[merged['source_id'].apply(lambda x: str(int(float(x)))) == sid].iloc[0]
        region, m_g = compute_hr_region(cat)
        final_row = final_map.get(sid)
        local_best = matched[matched['source_id'].astype(str) == sid].sort_values(
            'quality_score', ascending=False).iloc[0]
        local_cl = classify_spectrum_local(Path(local_best['path']))
        out = cat.to_dict()
        out.update({
            'source_id': sid,
            'hr_region': region,
            'M_G': m_g,
            **local_cl,
        })
        if final_row is not None:
            for col in ('spectral_type', 'teff', 'logg', 'mass', 'radius_rsun',
                        'cooling_age_gyr', 'total_age_gyr', 'ms_lifetime_gyr',
                        'membership', 'is_6d_matched', 'rv_true', 'rv_true_err',
                        'orbit_confirmed', 'orbit_min_sep_pc', 'cluster_age_gyr'):
                if col in final_row:
                    out[col] = final_row[col]
        source_rows.append(out)
    sources = pd.DataFrame(source_rows)
    simbad_cache = args.simbad_cache.expanduser().resolve() if args.simbad_cache else output_root / 'simbad_cache.csv'
    simbad = load_or_query_simbad(
        sources[['source_id', 'ra', 'dec']],
        simbad_cache,
        args.simbad_radius_arcsec,
        args.simbad_sleep_sec)
    sources = sources.merge(
        simbad.drop_duplicates('source_id'), on='source_id', how='left',
        suffixes=('', '_simbad'))

    decisions = []
    for _, row in sources.iterrows():
        keep, cls, reason = decide_keep(row)
        decisions.append({
            'source_id': str(row['source_id']),
            'keep': keep,
            'curated_class': cls,
            'curation_reason': reason,
        })
    decisions = pd.DataFrame(decisions)
    sources = sources.merge(decisions, on='source_id', how='left')
    sources.to_csv(output_root / 'simbad_wd_source_curation.csv', index=False)
    print('Curation counts:')
    print(sources['curated_class'].value_counts(dropna=False).to_string())

    # Pick one best spectrum per source/survey.  Quarantine duplicate CSVs.
    keep_sources = set(sources.loc[sources['keep'] == True, 'source_id'].astype(str))
    selected_paths = set()
    duplicate_csv_paths = set()
    for (sid, survey), grp in matched.groupby([matched['source_id'].astype(str), 'survey']):
        grp = grp.sort_values(['quality_score', 'ext'], ascending=[False, True])
        if sid in keep_sources:
            selected_paths.add(grp.iloc[0]['path'])
        for _, r in grp.iloc[1:].iterrows():
            if str(r['ext']).lower() == 'csv':
                duplicate_csv_paths.add(r['path'])

    rejected_paths = set(matched.loc[~matched['source_id'].astype(str).isin(keep_sources), 'path'])
    quarantine_rows = []
    for p in sorted(rejected_paths | duplicate_csv_paths):
        reason = 'duplicate_csv' if p in duplicate_csv_paths else 'non_wd_rejected'
        quarantine_rows.append({'path': p, 'reason': reason})
    pd.DataFrame(quarantine_rows).to_csv(output_root / 'quarantine_manifest.csv', index=False)

    if not args.no_quarantine:
        for item in quarantine_rows:
            src = Path(item['path'])
            if not src.exists():
                continue
            sub = quarantine / item['reason']
            sub.mkdir(parents=True, exist_ok=True)
            dst = sub / src.name
            if dst.exists():
                dst = sub / f'{src.stem}_{int(time.time())}{src.suffix}'
            shutil.move(str(src), str(dst))
        print(f'Quarantined {len(quarantine_rows)} files -> {quarantine}')

    prepared_dirs = []
    prep_rows = []
    for sid in sorted(keep_sources):
        cat = merged[merged['source_id'].apply(lambda x: str(int(float(x)))) == sid].iloc[0]
        final_row = final_map.get(sid)
        outdir = output_root / source_dir_name(cat)
        outdir.mkdir(parents=True, exist_ok=True)
        olddir = results_root / source_dir_name(cat)
        copy_cached_context(olddir, outdir)
        sel = build_selected_row(cat, final_row)
        sel.to_csv(outdir / 'selected_row.csv', index=False)
        sel.to_csv(outdir / 'sixdim_selected_row.csv', index=False)
        grp = matched[matched['source_id'].astype(str) == sid]
        for survey, sgrp in grp.groupby('survey'):
            sgrp = sgrp.sort_values('quality_score', ascending=False)
            src = Path(sgrp.iloc[0]['path'])
            if survey == 'sdss':
                name = 'sdss_spectrum.csv'
            elif survey == 'desi':
                name = 'desi_spectrum.csv'
            elif survey == 'lamost':
                name = 'lamost_lrs_spectrum.csv'
            else:
                name = f'{survey}_spectrum.csv'
            ok = write_standard_spectrum(src, outdir / name)
            prep_rows.append({
                'source_id': sid,
                'target_dir': str(outdir),
                'survey': survey,
                'source_spectrum': str(src),
                'output_spectrum': str(outdir / name),
                'written': ok,
            })
        prepared_dirs.append(outdir)
    prep = pd.DataFrame(prep_rows)
    prep.to_csv(output_root / 'prepared_spectra_manifest.csv', index=False)
    print(f'Prepared {len(prepared_dirs)} curated source folders -> {output_root}')

    if args.prepare_only:
        return 0

    if args.limit:
        prepared_dirs = prepared_dirs[:args.limit]
    ana_args = make_analysis_args(args.skip_magnetic, args.wd_model_grid)
    rows = []
    for i, d in enumerate(prepared_dirs, 1):
        print(f'[{i}/{len(prepared_dirs)}] toolbox rerun {d.name}', flush=True)
        try:
            row = run_existing_astro_output_analysis.analyze_target(d, ana_args)
        except Exception as exc:
            row = {'target': d.name, 'source_dir': str(d), 'status': 'failed', 'error': str(exc)}
        rows.append(row)
        if i % 10 == 0:
            pd.DataFrame(rows).to_csv(output_root / 'toolbox_rerun_summary_partial.csv', index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(output_root / 'toolbox_rerun_summary.csv', index=False)

    write_curated_cluster_summary(sources, summary, final_map, output_root)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
