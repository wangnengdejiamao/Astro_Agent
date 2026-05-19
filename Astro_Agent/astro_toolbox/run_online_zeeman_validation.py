#!/usr/bin/env python3
"""Validate the magnetic-field Zeeman fitter on local online SDSS samples."""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(ROOT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from astro_toolbox import magnetic_field  # noqa: E402


DEFAULT_VALIDATION_DIR = (
    '/Users/ljm/Desktop/csst/desi匹配/online_zeeman_validation_20260507')


def _safe_name(name):
    return re.sub(r'[^A-Za-z0-9_.+-]+', '_', str(name)).strip('_')


def _load_reference_fields(validation_dir):
    refs = {}
    amorim_table = validation_dir / 'literature_amorim_2023' / 'table1.dat'
    if amorim_table.exists():
        try:
            with amorim_table.open() as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    name = line[0:19].strip()
                    b_text = line[54:60].strip()
                    try:
                        b = float(b_text)
                    except Exception:
                        continue
                    if name and np.isfinite(b) and b > 0:
                        refs[name] = b
        except Exception:
            pass
    for filename, name_col, b_col in (
            ('online_sources_expected_fields.csv', 'name', 'literature_B_MG'),
            ('online_sdss_zeeman_validation_results.csv', 'source', 'literature_B_MG'),
            ('split_auto_validation_summary.csv', 'source', 'reference_B_MG'),
            ('direct_patch5_validation_summary.csv', 'source', 'reference_B_MG')):
        path = validation_dir / filename
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if name_col not in df.columns or b_col not in df.columns:
            continue
        for _, row in df.iterrows():
            name = str(row.get(name_col, '')).replace('Amorim_', '')
            if not name or name.lower() == 'nan':
                continue
            try:
                b = float(row.get(b_col))
            except Exception:
                continue
            if np.isfinite(b) and b > 0:
                refs[name] = b
    return refs


def _reference_for_source(source, refs):
    if source in refs:
        return refs[source]
    for key, value in refs.items():
        if source in key or key in source:
            return value
    return np.nan


def _score_reference_field(wave, flux, err, reference_b_mg,
                           rv_grid_kms, search_half_width_A,
                           min_depth, min_snr,
                           min_trough_width_A,
                           max_component_offset_A,
                           emission_avoid_A, absorption_core_avoid_A,
                           output_dir=None):
    if not np.isfinite(reference_b_mg):
        return {}
    result = magnetic_field.evaluate_fixed_magnetic_field(
        wave, flux, float(reference_b_mg), err=err,
        series='Halpha,Hbeta', rv_grid_kms=rv_grid_kms,
        search_half_width_A=search_half_width_A,
        min_depth=min_depth, min_snr=min_snr,
        min_trough_width_A=min_trough_width_A,
        max_component_offset_A=max_component_offset_A,
        emission_avoid_A=emission_avoid_A,
        absorption_core_avoid_A=absorption_core_avoid_A,
        baseline_mode='continuum')
    if result is None:
        return {}
    files = {}
    if output_dir is not None:
        files = magnetic_field.save_magnetic_field_outputs(
            result, str(output_dir), prefix='literature_reference',
            blue_reference_b_mg=float(reference_b_mg))
    return {
        'reference_score': result.get('score', np.nan),
        'reference_best_rv_kms': result.get('rv_kms', np.nan),
        'reference_n_detected': result.get('n_detected_components', 0),
        'reference_n_minor_absorption': result.get(
            'n_minor_absorption_components', 0),
        'reference_weighted_detected': result.get(
            'weighted_detected_components', 0.0),
        'reference_n_usable': result.get('n_usable_components', 0),
        'reference_n_core_side_pair_series': result.get(
            'n_core_side_pair_series', 0),
        'reference_quality': result.get('quality', ''),
        'reference_review_status': result.get('magnetic_review_status', ''),
        'reference_review_reasons': result.get('magnetic_review_reasons', ''),
        'reference_components': files.get('components', ''),
        'reference_plot': files.get('combined_region_plot_png', ''),
    }


def run_validation(args):
    validation_dir = Path(args.validation_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    refs = _load_reference_fields(validation_dir)
    spectra = sorted(validation_dir.glob('*_sdss_online_spectrum.csv'))
    if not spectra:
        raise FileNotFoundError(f'No *_sdss_online_spectrum.csv in {validation_dir}')

    rv_grid = np.arange(args.rv_min, args.rv_max + 0.5 * args.rv_step,
                        args.rv_step)
    rows = []
    for path in spectra:
        source = path.name.replace('_sdss_online_spectrum.csv', '')
        reference_b = _reference_for_source(source, refs)
        out = output_dir / _safe_name(source)
        wave, flux, err = magnetic_field.read_spectrum_file(path)
        result = magnetic_field.measure_magnetic_field(
            wave, flux, err=err, series='Halpha,Hbeta',
            b_min_mg=args.b_min_mg, b_max_mg=args.b_max_mg,
            n_b_grid=args.n_b_grid, rv_grid_kms=rv_grid,
            search_half_width_A=args.search_half_width_A,
            min_depth=args.min_depth, min_snr=args.min_snr,
            min_trough_width_A=args.min_trough_width_A,
            emission_avoid_A=args.emission_avoid_A,
            absorption_core_avoid_A=args.absorption_core_avoid_A,
            baseline_mode='continuum',
            field_mode=args.field_mode,
            low_high_boundary_mg=args.low_high_boundary_mg)
        if result is None:
            rows.append({
                'source': source,
                'reference_B_MG': reference_b,
                'measured_B_MG': np.nan,
                'status': 'failed',
                'output_dir': str(out),
            })
            continue
        files = magnetic_field.save_magnetic_field_outputs(result, str(out))
        reference_search_half_width = (
            max(args.search_half_width_A, 35.0)
            if reference_b < args.low_high_boundary_mg
            else max(args.search_half_width_A, 20.0))
        reference_metrics = _score_reference_field(
            wave, flux, err, reference_b, rv_grid,
            search_half_width_A=reference_search_half_width,
            min_depth=args.min_depth, min_snr=args.min_snr,
            min_trough_width_A=max(args.min_trough_width_A, 8.0)
            if reference_b < args.low_high_boundary_mg
            else args.min_trough_width_A,
            max_component_offset_A=args.reference_max_component_offset_A,
            emission_avoid_A=args.emission_avoid_A,
            absorption_core_avoid_A=args.absorption_core_avoid_A,
            output_dir=out)
        usable_broad = (
            str(result.get('magnetic_claim', '')) != 'no_zeeman_detection'
            and int(result.get('n_detected_components', 0) or 0) >= 3
            and str(result.get('magnetic_review_status', '')) in (
                'secure_zeeman', 'candidate_zeeman'))
        adopted_b = result.get('B_MG', np.nan) if usable_broad else np.nan
        delta = adopted_b - reference_b if (
            usable_broad and np.isfinite(reference_b)) else np.nan
        frac_delta = (
            delta / reference_b
            if np.isfinite(delta) and np.isfinite(reference_b) and reference_b
            else np.nan)
        rows.append({
            'source': source,
            'reference_B_MG': reference_b,
            'measured_B_MG': adopted_b,
            'adopted_B_MG': adopted_b,
            'best_trial_B_MG': result.get('B_MG', np.nan),
            'usable_broad_zeeman': bool(usable_broad),
            'delta_B_MG': delta,
            'frac_delta': frac_delta,
            'err_minus_MG': result.get('B_err_minus_MG', np.nan),
            'err_plus_MG': result.get('B_err_plus_MG', np.nan),
            'rv_kms': result.get('rv_kms', np.nan),
            'analysis_regime': result.get('analysis_regime', ''),
            'low_field_B_MG': result.get('low_field_B_MG', np.nan),
            'high_field_B_MG': result.get('high_field_B_MG', np.nan),
            'quality': result.get('quality', ''),
            'claim': result.get('magnetic_claim', ''),
            'review_status': result.get('magnetic_review_status', ''),
            'review_reasons': result.get('magnetic_review_reasons', ''),
            'ambiguous_field_solutions': result.get(
                'ambiguous_field_solutions', False),
            'n_detected': result.get('n_detected_components', 0),
            'n_minor_absorption': result.get(
                'n_minor_absorption_components', 0),
            'weighted_detected': result.get(
                'weighted_detected_components', 0.0),
            'n_core_side_pair_series': result.get(
                'n_core_side_pair_series', 0),
            'core_side_pair_series': result.get('core_side_pair_series', ''),
            'plot': files.get('combined_region_plot_png', ''),
            'field_solutions': files.get('field_solutions', ''),
            'output_dir': str(out),
            **reference_metrics,
        })
    summary = pd.DataFrame(rows)
    summary_path = output_dir / 'validation_summary.csv'
    summary.to_csv(summary_path, index=False)
    return summary_path, summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Run toolbox Zeeman validation on the online sample directory.')
    parser.add_argument('--validation-dir', default=DEFAULT_VALIDATION_DIR)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--field-mode', default='auto',
                        choices=['auto', 'split', 'low', 'high', 'single', 'legacy'])
    parser.add_argument('--b-min-mg', type=float, default=5.0)
    parser.add_argument('--b-max-mg', type=float, default=950.0)
    parser.add_argument('--n-b-grid', type=int, default=420)
    parser.add_argument('--rv-min', type=float, default=-250.0)
    parser.add_argument('--rv-max', type=float, default=250.0)
    parser.add_argument('--rv-step', type=float, default=25.0)
    parser.add_argument('--search-half-width-A', type=float, default=8.0)
    parser.add_argument('--min-depth', type=float, default=0.008)
    parser.add_argument('--min-snr', type=float, default=1.5)
    parser.add_argument('--min-trough-width-A', type=float, default=6.0)
    parser.add_argument('--reference-max-component-offset-A', type=float, default=12.0,
                        help='Fixed literature-B check: reject candidate troughs farther than this from the predicted Zeeman node.')
    parser.add_argument('--emission-avoid-A', type=float, default=10.0)
    parser.add_argument('--absorption-core-avoid-A', type=float, default=25.0)
    parser.add_argument('--low-high-boundary-mg', type=float, default=35.0)
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = str(
            Path(args.validation_dir) / 'toolbox_zeeman_validation_20260512')
    summary_path, summary = run_validation(args)
    cols = [
        'source', 'reference_B_MG', 'measured_B_MG', 'best_trial_B_MG',
        'usable_broad_zeeman', 'delta_B_MG',
        'analysis_regime', 'rv_kms', 'quality', 'review_status',
        'n_detected', 'n_minor_absorption',
        'reference_n_detected', 'reference_n_minor_absorption',
        'ambiguous_field_solutions']
    cols = [c for c in cols if c in summary.columns]
    print(summary[cols].to_string(index=False))
    print(f'saved {summary_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
