#!/usr/bin/env python3
"""Batch magnetic-field screening for astro_output source folders."""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(ROOT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from astro_toolbox import magnetic_field  # noqa: E402


OPTICAL_KEYS = ('sdss', 'desi', 'lamost', 'hst', 'koa')
WAVE_COLS = ('wavelength_A', 'wavelength', 'wave', 'lambda', 'lam')
FLUX_COLS = ('flux', 'flam', 'f_lambda')
ERR_COLS = ('error', 'flux_err', 'ivar_error', 'sigma', 'err')


def _first_existing(columns, candidates):
    for name in candidates:
        if name in columns:
            return name
    return None


def _is_source_dir(path):
    return path.is_dir() and path.name.startswith('ZTFJ')


def _candidate_spectrum_files(source_dir):
    files = []
    for path in sorted(source_dir.iterdir()):
        if not path.is_file():
            continue
        name = path.name.lower()
        if not name.endswith('.csv'):
            continue
        if 'spherex' in name or 'magnetic_field' in name:
            continue
        if 'spectrum' not in name:
            continue
        if not any(key in name for key in OPTICAL_KEYS):
            continue
        files.append(path)
    return files


def _read_spectrum_csv(path):
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return None, f'read_failed:{exc}'
    wave_col = _first_existing(df.columns, WAVE_COLS)
    flux_col = _first_existing(df.columns, FLUX_COLS)
    err_col = _first_existing(df.columns, ERR_COLS)
    if wave_col is None or flux_col is None:
        return None, 'missing_wave_or_flux_column'
    wave = pd.to_numeric(df[wave_col], errors='coerce').to_numpy(float)
    wave = magnetic_field.coerce_wavelength_to_angstrom(wave)
    flux = pd.to_numeric(df[flux_col], errors='coerce').to_numpy(float)
    err = None
    if err_col is not None:
        err = pd.to_numeric(df[err_col], errors='coerce').to_numpy(float)

    good = np.isfinite(wave) & np.isfinite(flux) & (flux != 0)
    if err is not None and err.shape == wave.shape:
        good &= np.isfinite(err) & (err > 0)
    wave = wave[good]
    flux = flux[good]
    err = err[good] if err is not None and err.shape == good.shape else None
    optical = (wave >= 3200.0) & (wave <= 10000.0)
    if np.sum(optical) < 80:
        return None, f'too_few_optical_points:{int(np.sum(optical))}'
    wave = wave[optical]
    flux = flux[optical]
    err = err[optical] if err is not None else None
    if np.nanmax(wave) - np.nanmin(wave) < 250.0:
        return None, 'optical_wavelength_span_too_small'
    return (wave, flux, err), ''


def _spectrum_priority(path):
    name = path.name.lower()
    if 'desi' in name:
        return 0
    if 'sdss' in name:
        return 1
    if 'lamost' in name:
        return 2
    if 'hst' in name:
        return 3
    if 'koa' in name:
        return 4
    return 9


def _load_source_spectra(source_dir):
    usable = []
    skipped = []
    for path in _candidate_spectrum_files(source_dir):
        data, reason = _read_spectrum_csv(path)
        if data is None:
            skipped.append(f'{path.name}:{reason}')
            continue
        wave, flux, err = data
        span = float(np.nanmax(wave) - np.nanmin(wave))
        usable.append({
            'path': path,
            'wave': wave,
            'flux': flux,
            'err': err,
            'n_points': int(len(wave)),
            'span_A': span,
        })
    if not usable:
        return None, None, None, [], skipped

    chosen = min(
        usable,
        key=lambda item: (
            _spectrum_priority(item['path']),
            -item['n_points'],
            -item['span_A'],
            item['path'].name))
    for item in usable:
        if item is chosen:
            continue
        skipped.append(
            f"{item['path'].name}:not_selected_single_spectrum;"
            f"chosen={chosen['path'].name}")

    wave = chosen['wave']
    flux = chosen['flux']
    err = chosen['err']
    good = np.isfinite(wave) & np.isfinite(flux)
    if err is not None:
        good &= np.isfinite(err) & (err > 0)
    order = np.argsort(wave[good])
    return wave[good][order], flux[good][order], (
        err[good][order] if err is not None else None), [chosen['path']], skipped


def _batch_class(result):
    if result is None:
        return 'failed'
    review = str(result.get('magnetic_review_status', ''))
    if review == 'secure_zeeman':
        return 'strong_candidate'
    if review == 'candidate_zeeman':
        return 'candidate'
    if review == 'low_confidence_zeeman':
        return 'low_confidence_candidate'
    quality = str(result.get('quality', ''))
    n_det = int(result.get('n_detected_components', 0) or 0)
    rel_unc = result.get('relative_B_uncertainty', np.nan)
    rv_edge = bool(result.get('rv_at_search_edge', False))
    b_mg = float(result.get('B_MG', np.nan))
    if n_det < 3:
        return 'no_detection'
    if (quality == 'good' and n_det >= 4 and not rv_edge
            and np.isfinite(b_mg) and b_mg >= 50
            and (not np.isfinite(rel_unc) or rel_unc <= 0.45)):
        return 'strong_candidate'
    if quality in ('good', 'candidate') and n_det >= 3:
        return 'candidate'
    return 'no_detection'


RESULT_SUMMARY_COLUMNS = (
    'B_MG',
    'B_err_minus_MG',
    'B_err_plus_MG',
    'B_interval_lower_MG',
    'B_interval_upper_MG',
    'B_interval_contains_best',
    'B_posterior_median_MG',
    'rv_kms',
    'score',
    'quality',
    'field_regime',
    'magnetic_claim',
    'magnetic_review_status',
    'magnetic_review_reasons',
    'magnetic_high_snr_components',
    'magnetic_max_component_snr',
    'magnetic_median_abs_component_offset_A',
    'magnetic_core_side_fraction',
    'magnetic_pit_shape_fraction',
    'magnetic_broad_trough_fraction',
    'n_detected_components',
    'n_usable_components',
    'n_core_side_pits',
    'n_core_side_pit_series',
    'core_side_pit_series',
    'n_core_side_pair_series',
    'core_side_pair_series',
    'n_detected_series',
    'detected_component_labels',
    'series_used',
    'series_detected',
    'rv_at_search_edge',
    'relative_B_uncertainty',
    'baseline_method',
    'wd_spectral_type',
    'wd_template_grid',
    'wd_template_teff',
    'wd_template_logg',
    'wd_template_chi2_red',
    'analysis_regime',
    'field_mode',
    'ambiguous_field_solutions',
    'low_high_boundary_MG',
    'low_field_B_MG',
    'low_field_B_err_minus_MG',
    'low_field_B_err_plus_MG',
    'low_field_rv_kms',
    'low_field_quality',
    'low_field_magnetic_claim',
    'low_field_magnetic_review_status',
    'low_field_magnetic_review_reasons',
    'low_field_n_detected_components',
    'low_field_n_usable_components',
    'low_field_n_core_side_pits',
    'low_field_n_core_side_pit_series',
    'low_field_core_side_pit_series',
    'low_field_n_core_side_pair_series',
    'low_field_core_side_pair_series',
    'low_field_rv_at_search_edge',
    'low_field_score',
    'low_field_ambiguous_field_solutions',
    'high_field_B_MG',
    'high_field_B_err_minus_MG',
    'high_field_B_err_plus_MG',
    'high_field_rv_kms',
    'high_field_quality',
    'high_field_magnetic_claim',
    'high_field_magnetic_review_status',
    'high_field_magnetic_review_reasons',
    'high_field_n_detected_components',
    'high_field_n_usable_components',
    'high_field_n_core_side_pits',
    'high_field_n_core_side_pit_series',
    'high_field_core_side_pit_series',
    'high_field_n_core_side_pair_series',
    'high_field_core_side_pair_series',
    'high_field_rv_at_search_edge',
    'high_field_score',
    'high_field_ambiguous_field_solutions',
)


def _result_summary_values(result):
    return {key: result.get(key) for key in RESULT_SUMMARY_COLUMNS}


def iter_source_dirs(root, include_new_source=True):
    for path in sorted(root.rglob('ZTFJ*')):
        if not _is_source_dir(path):
            continue
        parts = set(path.parts)
        if not include_new_source and 'new_source' in parts:
            continue
        if 'toolbox_run' in parts or 'magnetic_field_batch' in parts:
            continue
        yield path


def run_batch(args):
    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f'Root directory does not exist: {root}')
    source_dirs = list(iter_source_dirs(root, include_new_source=not args.exclude_new_source))
    rows = []
    for idx, source_dir in enumerate(source_dirs, 1):
        t0 = time.time()
        wave, flux, err, used, skipped = _load_source_spectra(source_dir)
        target = source_dir.name
        out_dir = source_dir / args.output_subdir
        row = {
            'target': target,
            'source_dir': str(source_dir),
            'output_dir': str(out_dir),
            'n_input_files_used': len(used),
            'input_files_used': ';'.join(p.name for p in used),
            'skipped_files': ';'.join(skipped),
            'status': 'pending',
        }
        print(f'[{idx}/{len(source_dirs)}] {target}: {len(used)} usable spectra',
              flush=True)
        if wave is None:
            row.update({
                'status': 'skipped',
                'skip_reason': 'no_usable_optical_spectrum',
                'batch_class': 'skipped',
                'elapsed_sec': round(time.time() - t0, 2),
            })
            rows.append(row)
            continue
        if out_dir.exists() and not args.force:
            summary_path = out_dir / 'magnetic_field_summary.csv'
            if summary_path.exists():
                try:
                    prev = pd.read_csv(summary_path).iloc[0].to_dict()
                    row.update(prev)
                    row.update({
                        'status': 'existing',
                        'batch_class': _batch_class(prev),
                        'elapsed_sec': round(time.time() - t0, 2),
                    })
                    rows.append(row)
                    continue
                except Exception:
                    pass
        try:
            rv_grid = np.arange(args.rv_min, args.rv_max + 0.5 * args.rv_step,
                                args.rv_step)
            result = magnetic_field.measure_magnetic_field(
                wave, flux, err=err,
                series=args.series,
                b_min_mg=args.b_min_mg,
                b_max_mg=args.b_max_mg,
                n_b_grid=args.n_b_grid,
                rv_grid_kms=rv_grid,
                search_half_width_A=args.search_half_width_A,
                min_depth=args.min_depth,
                min_snr=args.min_snr,
                min_trough_width_A=args.min_trough_width_A,
                emission_avoid_A=args.emission_avoid_A,
                absorption_core_avoid_A=args.absorption_core_avoid_A,
                baseline_mode=args.baseline_mode,
                wd_model_grid=args.wd_model_grid,
                spectral_type=args.spectral_type,
                field_mode=args.field_mode,
                low_high_boundary_mg=args.low_high_boundary_mg)
            if result is None:
                row.update({
                    'status': 'failed',
                    'skip_reason': 'fit_returned_none',
                    'batch_class': 'failed',
                })
            else:
                files = magnetic_field.save_magnetic_field_outputs(
                    result, str(out_dir))
                row.update(_result_summary_values(result))
                row.update({
                    'status': 'ok',
                    'batch_class': _batch_class(result),
                    'summary_file': files.get('summary'),
                    'plot_file': files.get('plot'),
                    'blue_region_plot': files.get('blue_region_plot_png', ''),
                    'red_region_plot': files.get('red_region_plot_png', ''),
                    'blue_region_nodes': files.get('blue_region_nodes', ''),
                    'red_region_nodes': files.get('red_region_nodes', ''),
                })
        except Exception as exc:
            row.update({
                'status': 'failed',
                'skip_reason': str(exc),
                'batch_class': 'failed',
            })
        row['elapsed_sec'] = round(time.time() - t0, 2)
        rows.append(row)

        if args.limit and len(rows) >= args.limit:
            break

    summary = pd.DataFrame(rows)
    summary_path = root / args.summary_name
    summary.to_csv(summary_path, index=False)
    counts_path = root / args.counts_name
    counts = summary['batch_class'].value_counts(dropna=False).rename_axis(
        'batch_class').reset_index(name='n_targets')
    counts.to_csv(counts_path, index=False)
    print(f'Wrote summary: {summary_path}')
    print(f'Wrote counts: {counts_path}')
    print(counts.to_string(index=False))
    return summary, counts


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Screen astro_output optical spectra for magnetic WD candidates.')
    parser.add_argument('root')
    parser.add_argument('--output-subdir', default='magnetic_field_batch')
    parser.add_argument('--summary-name', default='magnetic_field_batch_summary.csv')
    parser.add_argument('--counts-name', default='magnetic_field_batch_counts.csv')
    parser.add_argument('--exclude-new-source', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--series', default='Halpha,Hbeta',
                        help='Default Halpha,Hbeta; Hgamma is intentionally excluded for batch screening.')
    parser.add_argument('--b-min-mg', type=float, default=5.0)
    parser.add_argument('--b-max-mg', type=float, default=950.0)
    parser.add_argument('--n-b-grid', type=int, default=320)
    parser.add_argument('--rv-min', type=float, default=-250.0)
    parser.add_argument('--rv-max', type=float, default=250.0)
    parser.add_argument('--rv-step', type=float, default=25.0)
    parser.add_argument('--search-half-width-A', type=float, default=8.0)
    parser.add_argument('--min-depth', type=float, default=0.04)
    parser.add_argument('--min-snr', type=float, default=3.0)
    parser.add_argument('--min-trough-width-A', type=float, default=6.0)
    parser.add_argument('--emission-avoid-A', type=float, default=10.0)
    parser.add_argument('--absorption-core-avoid-A', type=float, default=25.0)
    parser.add_argument('--baseline-mode', default='continuum',
                        choices=['template', 'continuum', 'auto'])
    parser.add_argument('--wd-model-grid', default='auto')
    parser.add_argument('--spectral-type', default='auto')
    parser.add_argument('--field-mode', default='auto',
                        choices=['auto', 'split', 'single', 'legacy', 'low', 'high'])
    parser.add_argument('--low-high-boundary-mg', type=float, default=35.0)
    args = parser.parse_args(argv)
    run_batch(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
