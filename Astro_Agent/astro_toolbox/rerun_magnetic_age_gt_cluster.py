#!/usr/bin/env python3
"""Rerun magnetic-field screening for WD-age > cluster-age targets."""

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
from astro_toolbox import run_magnetic_field_batch as batch  # noqa: E402


AGE_COLUMNS_TO_KEEP = (
    'cluster', 'best_cluster', 'cluster_age_gyr', 'cluster_age_myr',
    'cooling_age_gyr', 'wd_cooling_age_gyr', 't_cool_Gyr',
    'total_age_gyr', 'wd_total_age_gyr', 't_total_Gyr',
    'age_gt_cluster', 'wd_cooling_age_gt_cluster',
    'wd_total_age_gt_cluster', 'wd_cooling_age_minus_cluster_gyr',
    'age_excess_gyr', 'has_SDSS', 'has_DESI', 'has_LAMOST',
)


def _truth(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in ('true', '1', 'yes', 'y')


def _as_float(value):
    try:
        out = float(value)
    except Exception:
        return np.nan
    return out if np.isfinite(out) else np.nan


def _metadata_subset(data):
    out = {}
    for key in AGE_COLUMNS_TO_KEEP:
        if key in data:
            out[key] = data.get(key)
    return out


def _strict_selected_rows(root):
    rows = []
    for selected_path in sorted(root.glob('*/selected_row.csv')):
        try:
            selected = pd.read_csv(selected_path)
        except Exception:
            continue
        if selected.empty:
            continue
        data = selected.iloc[0].to_dict()
        flags = [
            _truth(data.get('age_gt_cluster')),
            _truth(data.get('wd_cooling_age_gt_cluster')),
            _truth(data.get('wd_total_age_gt_cluster')),
        ]
        if not any(flags):
            continue
        source_dir = selected_path.parent
        row = {
            'target': data.get('target') or source_dir.name,
            'source_dir': str(source_dir),
            'age_filter_source': 'selected_row.csv',
        }
        row.update(_metadata_subset(data))
        rows.append(row)
    return rows


def _dwd_age_rows(root):
    age_path = root / 'wd_ages_cluster_members.csv'
    membership_path = root / 'cluster_membership_6d.csv'
    if not (age_path.exists() and membership_path.exists()):
        return []
    try:
        ages = pd.read_csv(age_path)
        membership = pd.read_csv(membership_path)
    except Exception:
        return []
    if 'name' not in ages.columns or 'DWD_name' not in membership.columns:
        return []

    keep_cols = [
        c for c in ('DWD_name', 'best_cluster', 'cluster_age_myr',
                    'in_cluster_6D', 'Gaia_source_id', 'min_sep_pc')
        if c in membership.columns
    ]
    merged = ages.merge(
        membership[keep_cols], how='left',
        left_on='name', right_on='DWD_name')

    rows = []
    for _, data in merged.iterrows():
        target = str(data.get('name') or '').strip()
        if not target:
            continue
        cluster_age_gyr = _as_float(data.get('cluster_age_myr')) / 1000.0
        cooling_age_gyr = _as_float(data.get('t_cool_Gyr'))
        total_age_gyr = _as_float(data.get('t_total_Gyr'))
        cooling_gt = (
            np.isfinite(cooling_age_gyr) and np.isfinite(cluster_age_gyr)
            and cooling_age_gyr > cluster_age_gyr)
        total_gt = (
            np.isfinite(total_age_gyr) and np.isfinite(cluster_age_gyr)
            and total_age_gyr > cluster_age_gyr)
        if not (cooling_gt or total_gt):
            continue
        source_dir = root / target
        if not source_dir.is_dir():
            continue
        row = {
            'target': target,
            'source_dir': str(source_dir),
            'age_filter_source': 'wd_ages_cluster_members+cluster_membership_6d',
            'cluster': data.get('best_cluster'),
            'best_cluster': data.get('best_cluster'),
            'cluster_age_gyr': cluster_age_gyr,
            'cluster_age_myr': data.get('cluster_age_myr'),
            'cooling_age_gyr': cooling_age_gyr,
            't_cool_Gyr': cooling_age_gyr,
            'total_age_gyr': total_age_gyr,
            't_total_Gyr': total_age_gyr,
            'age_gt_cluster': bool(cooling_gt or total_gt),
            'wd_cooling_age_gt_cluster': bool(cooling_gt),
            'wd_total_age_gt_cluster': bool(total_gt),
            'age_excess_gyr': (
                max(
                    cooling_age_gyr - cluster_age_gyr
                    if np.isfinite(cooling_age_gyr) else -np.inf,
                    total_age_gyr - cluster_age_gyr
                    if np.isfinite(total_age_gyr) else -np.inf)
                if np.isfinite(cluster_age_gyr) else np.nan),
            'in_cluster_6D': data.get('in_cluster_6D'),
            'Gaia_source_id': data.get('Gaia_source_id'),
            'min_sep_pc': data.get('min_sep_pc'),
        }
        rows.append(row)
    return rows


def age_gt_cluster_rows(root):
    root = Path(root).expanduser().resolve()
    rows = _strict_selected_rows(root)
    if not rows:
        rows = _dwd_age_rows(root)
    seen = set()
    unique = []
    for row in rows:
        source_dir = row.get('source_dir')
        if source_dir in seen:
            continue
        seen.add(source_dir)
        row['root'] = str(root)
        unique.append(row)
    return unique


def _result_row_from_fit(meta, out_dir, used, skipped, result, files, elapsed):
    row = dict(meta)
    row.update({
        'output_dir': str(out_dir),
        'n_input_files_used': len(used),
        'input_files_used': ';'.join(p.name for p in used),
        'skipped_files': ';'.join(skipped),
        'elapsed_sec': round(float(elapsed), 2),
    })
    if result is None:
        row.update({
            'status': 'failed',
            'batch_class': 'failed',
            'skip_reason': 'fit_returned_none',
        })
        return row
    row.update(batch._result_summary_values(result))
    row.update({
        'status': 'ok',
        'batch_class': batch._batch_class(result),
        'summary_file': files.get('summary', ''),
        'components_file': files.get('components', ''),
        'features_file': files.get('full_field_overlay_features', ''),
        'fit_plot': files.get('plot', ''),
        'full_overlay_plot': files.get('full_field_overlay_png', ''),
        'combined_region_plot': files.get('combined_region_plot_png', ''),
        'blue_region_plot': files.get('blue_region_plot_png', ''),
        'red_region_plot': files.get('red_region_plot_png', ''),
    })
    return row


def run_roots(args):
    metas = []
    for root in args.roots:
        metas.extend(age_gt_cluster_rows(root))
    if args.limit:
        metas = metas[:args.limit]
    if not metas:
        raise RuntimeError('No age_gt_cluster targets found.')

    rv_grid = np.arange(args.rv_min, args.rv_max + 0.5 * args.rv_step,
                        args.rv_step)
    rows = []
    for idx, meta in enumerate(metas, 1):
        t0 = time.time()
        source_dir = Path(meta['source_dir'])
        out_dir = source_dir / args.output_subdir
        wave, flux, err, used, skipped = batch._load_source_spectra(source_dir)
        print(f'[{idx}/{len(metas)}] {meta.get("target")}: '
              f'{len(used)} usable spectra', flush=True)
        if wave is None:
            row = dict(meta)
            row.update({
                'status': 'skipped',
                'batch_class': 'skipped',
                'skip_reason': 'no_usable_optical_spectrum',
                'output_dir': str(out_dir),
                'n_input_files_used': 0,
                'input_files_used': '',
                'skipped_files': ';'.join(skipped),
                'elapsed_sec': round(time.time() - t0, 2),
            })
            rows.append(row)
            continue

        if out_dir.exists() and not args.force:
            summary_path = out_dir / 'magnetic_field_summary.csv'
            if summary_path.exists():
                row = dict(meta)
                try:
                    prev = pd.read_csv(summary_path).iloc[0].to_dict()
                    row.update(prev)
                    batch_class = batch._batch_class(prev)
                except Exception:
                    batch_class = 'existing'
                row.update({
                    'status': 'existing',
                    'batch_class': batch_class,
                    'output_dir': str(out_dir),
                    'summary_file': str(summary_path),
                    'n_input_files_used': len(used),
                    'input_files_used': ';'.join(p.name for p in used),
                    'skipped_files': ';'.join(skipped),
                    'elapsed_sec': round(time.time() - t0, 2),
                })
                rows.append(row)
                continue

        try:
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
                emission_avoid_A=args.emission_avoid_A,
                absorption_core_avoid_A=args.absorption_core_avoid_A,
                baseline_mode=args.baseline_mode,
                wd_model_grid=args.wd_model_grid,
                spectral_type=args.spectral_type,
                field_mode=args.field_mode,
                low_high_boundary_mg=args.low_high_boundary_mg)
            files = {}
            if result is not None:
                files = magnetic_field.save_magnetic_field_outputs(
                    result, str(out_dir))
            rows.append(_result_row_from_fit(
                meta, out_dir, used, skipped, result, files,
                time.time() - t0))
        except Exception as exc:
            row = dict(meta)
            row.update({
                'status': 'failed',
                'batch_class': 'failed',
                'skip_reason': str(exc),
                'output_dir': str(out_dir),
                'n_input_files_used': len(used),
                'input_files_used': ';'.join(p.name for p in used),
                'skipped_files': ';'.join(skipped),
                'elapsed_sec': round(time.time() - t0, 2),
            })
            rows.append(row)

    summary = pd.DataFrame(rows)
    for root, part in summary.groupby('root', dropna=False):
        root_path = Path(root)
        summary_path = root_path / args.summary_name
        counts_path = root_path / args.counts_name
        part.to_csv(summary_path, index=False)
        counts = part['batch_class'].value_counts(dropna=False).rename_axis(
            'batch_class').reset_index(name='n_targets')
        counts.to_csv(counts_path, index=False)
        print(f'Wrote summary: {summary_path}', flush=True)
        print(f'Wrote counts: {counts_path}', flush=True)
        print(counts.to_string(index=False), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Rerun magnetic-field fits for WD age > cluster age targets.')
    parser.add_argument('roots', nargs='+')
    parser.add_argument('--output-subdir', default='magnetic_field_age_gt_cluster')
    parser.add_argument('--summary-name',
                        default='magnetic_field_age_gt_cluster_summary.csv')
    parser.add_argument('--counts-name',
                        default='magnetic_field_age_gt_cluster_counts.csv')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--series', default='Halpha,Hbeta')
    parser.add_argument('--b-min-mg', type=float, default=5.0)
    parser.add_argument('--b-max-mg', type=float, default=950.0)
    parser.add_argument('--n-b-grid', type=int, default=320)
    parser.add_argument('--rv-min', type=float, default=-250.0)
    parser.add_argument('--rv-max', type=float, default=250.0)
    parser.add_argument('--rv-step', type=float, default=25.0)
    parser.add_argument('--search-half-width-A', type=float, default=10.0)
    parser.add_argument('--min-depth', type=float, default=0.035)
    parser.add_argument('--min-snr', type=float, default=2.5)
    parser.add_argument('--emission-avoid-A', type=float, default=10.0)
    parser.add_argument('--absorption-core-avoid-A', type=float, default=25.0)
    parser.add_argument('--baseline-mode', default='continuum',
                        choices=['template', 'continuum', 'auto'])
    parser.add_argument('--wd-model-grid', default='auto')
    parser.add_argument('--spectral-type', default='auto')
    parser.add_argument('--field-mode', default='auto',
                        choices=['auto', 'split', 'single', 'legacy', 'low', 'high'])
    parser.add_argument('--low-high-boundary-mg', type=float, default=70.0)
    args = parser.parse_args(argv)
    run_roots(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
