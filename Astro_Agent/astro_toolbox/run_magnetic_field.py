#!/usr/bin/env python3
"""Run magnetic WD Zeeman-field measurement on one or more spectrum files."""

import argparse
import os
import sys

import numpy as np


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(ROOT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from astro_toolbox import magnetic_field  # noqa: E402


def _read_one_csv(path, wave_col=None, flux_col=None, err_col=None):
    return magnetic_field.read_spectrum_file(
        path, wave_col=wave_col, flux_col=flux_col, err_col=err_col)


def _load_spectra(paths, wave_col=None, flux_col=None, err_col=None):
    waves = []
    fluxes = []
    errs = []
    has_err = False
    for path in paths:
        w, f, e = _read_one_csv(path, wave_col=wave_col,
                                flux_col=flux_col, err_col=err_col)
        waves.append(w)
        fluxes.append(f)
        if e is not None:
            has_err = True
            errs.append(e)
        else:
            errs.append(np.full_like(f, np.nan, dtype=float))
    wave = np.concatenate(waves)
    flux = np.concatenate(fluxes)
    err = np.concatenate(errs) if has_err else None
    good = np.isfinite(wave) & np.isfinite(flux)
    if err is not None:
        good &= np.isfinite(err) & (err > 0)
    order = np.argsort(wave[good])
    return wave[good][order], flux[good][order], err[good][order] if err is not None else None


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Measure magnetic WD field from Balmer Zeeman split dips.')
    parser.add_argument('--spectrum-csv', '--spectrum-file',
                        dest='spectrum_csv', action='append', required=True,
                        help='Input spectrum file: CSV/TSV or whitespace TXT. Repeat for blue/red arms.')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--template-dir', default=None)
    parser.add_argument('--series', default='Halpha,Hbeta',
                        help='Default Halpha,Hbeta; use auto or include Hgamma only when explicitly needed.')
    parser.add_argument('--b-min-mg', type=float, default=5.0,
                        help='Default 5 MG lets the low-field split search inspect Balmer-core side pits.')
    parser.add_argument('--b-max-mg', type=float, default=950.0)
    parser.add_argument('--n-b-grid', type=int, default=700)
    parser.add_argument('--rv-min', type=float, default=-250.0)
    parser.add_argument('--rv-max', type=float, default=250.0)
    parser.add_argument('--rv-step', type=float, default=25.0)
    parser.add_argument('--search-half-width-A', type=float, default=8.0)
    parser.add_argument('--min-depth', type=float, default=0.008)
    parser.add_argument('--min-snr', type=float, default=1.5)
    parser.add_argument('--min-trough-width-A', type=float, default=6.0,
                        help='Reject line-like features narrower than this; low-field mode uses at least 8 A.')
    parser.add_argument('--emission-avoid-A', type=float, default=10.0)
    parser.add_argument('--absorption-core-avoid-A', type=float, default=25.0,
                        help='Skip normal Balmer/Ca line cores so ordinary DA absorption is not counted as Zeeman splitting.')
    parser.add_argument('--baseline-mode', default='continuum',
                        choices=['template', 'continuum', 'auto'],
                        help='continuum directly searches absorption troughs; template/auto fit WD models only when explicitly requested.')
    parser.add_argument('--field-mode', default='auto',
                        choices=['auto', 'split', 'low', 'high', 'single', 'legacy'],
                        help='auto runs separate low/high magnetic-field searches and selects one.')
    parser.add_argument('--low-high-boundary-mg', type=float, default=35.0,
                        help='Boundary between low-field Balmer-core side-pit search and high-field Zeeman search.')
    parser.add_argument('--wd-model-grid', default='auto',
                        help='WD baseline template grid: auto, koester, nn_da, nn_db.')
    parser.add_argument('--spectral-type', default='auto',
                        help='auto, DA, DB, or DC for the baseline WD fit.')
    parser.add_argument('--blue-reference-b-mg', type=float, default=None,
                        help='Optional B value for the Balmer blue-region reference line; default uses the fitted field.')
    parser.add_argument('--reference-b-mg', type=float, default=None,
                        help='Evaluate a fixed literature/reference B instead of adopting a blind-search B.')
    parser.add_argument('--reference-max-component-offset-A', type=float, default=12.0,
                        help='For --reference-b-mg, reject troughs farther than this from the predicted node.')
    parser.add_argument('--guided-preset', default=None,
                        help='Use a line-ID guided Zeeman fit preset, e.g. cechichang_220, instead of the blind dip-search B.')
    parser.add_argument('--guided-b-min-mg', type=float, default=120.0)
    parser.add_argument('--guided-b-max-mg', type=float, default=300.0)
    parser.add_argument('--guided-n-b-grid', type=int, default=2500)
    parser.add_argument('--wave-col', default=None)
    parser.add_argument('--flux-col', default=None)
    parser.add_argument('--err-col', default=None)
    args = parser.parse_args(argv)

    wave, flux, err = _load_spectra(
        args.spectrum_csv, wave_col=args.wave_col,
        flux_col=args.flux_col, err_col=args.err_col)
    rv_grid = np.arange(args.rv_min, args.rv_max + 0.5 * args.rv_step,
                        args.rv_step)
    if args.reference_b_mg is not None:
        result = magnetic_field.evaluate_fixed_magnetic_field(
            wave, flux, float(args.reference_b_mg), err=err,
            template_dir=args.template_dir, series=args.series,
            rv_grid_kms=rv_grid,
            search_half_width_A=max(float(args.search_half_width_A), 30.0),
            min_depth=args.min_depth, min_snr=args.min_snr,
            min_trough_width_A=max(float(args.min_trough_width_A), 10.0),
            max_component_offset_A=args.reference_max_component_offset_A,
            emission_avoid_A=args.emission_avoid_A,
            absorption_core_avoid_A=args.absorption_core_avoid_A,
            baseline_mode=args.baseline_mode,
            wd_model_grid=args.wd_model_grid,
            spectral_type=args.spectral_type)
        if args.blue_reference_b_mg is None:
            args.blue_reference_b_mg = float(args.reference_b_mg)
    else:
        result = magnetic_field.measure_magnetic_field(
            wave, flux, err=err, template_dir=args.template_dir,
            series=args.series, b_min_mg=args.b_min_mg, b_max_mg=args.b_max_mg,
            n_b_grid=args.n_b_grid, rv_grid_kms=rv_grid,
            search_half_width_A=args.search_half_width_A,
            min_depth=args.min_depth, min_snr=args.min_snr,
            min_trough_width_A=args.min_trough_width_A,
            emission_avoid_A=args.emission_avoid_A,
            absorption_core_avoid_A=args.absorption_core_avoid_A,
            baseline_mode=args.baseline_mode,
            wd_model_grid=args.wd_model_grid,
            spectral_type=args.spectral_type,
            field_mode=args.field_mode,
            low_high_boundary_mg=args.low_high_boundary_mg)
    if result is None:
        print('Magnetic-field fit failed: no usable spectrum/templates')
        return 1
    guided = None
    if args.guided_preset:
        result, guided = magnetic_field.apply_guided_zeeman_fit(
            result, preset=args.guided_preset,
            template_dir=args.template_dir,
            b_min_mg=args.guided_b_min_mg,
            b_max_mg=args.guided_b_max_mg,
            n_b_grid=args.guided_n_b_grid,
            rv_grid_kms=np.array([0.0]))
    files = magnetic_field.save_magnetic_field_outputs(
        result, args.output_dir, blue_reference_b_mg=args.blue_reference_b_mg)
    adopted = magnetic_field.is_adopted_zeeman_result(result)
    print('Magnetic WD field screening OK')
    if adopted:
        print(f"  adopted B = {result['B_MG']:.1f} -{result['B_err_minus_MG']:.1f} +{result['B_err_plus_MG']:.1f} MG")
    else:
        print(f"  no adopted B; best trial = {result['B_MG']:.1f} -{result['B_err_minus_MG']:.1f} +{result['B_err_plus_MG']:.1f} MG")
    print(f"  RV shift = {result['rv_kms']:.1f} km/s")
    print(f"  quality = {result['quality']}")
    print(f"  analysis regime = {result.get('analysis_regime', '')} (mode={result.get('field_mode', '')})")
    if result.get('field_mode') in ('auto', 'split'):
        print(f"  low-field B = {result.get('low_field_B_MG', np.nan):.1f} MG")
        print(f"  high-field B = {result.get('high_field_B_MG', np.nan):.1f} MG")
    if guided is not None:
        print(f"  fit method = guided Zeeman line IDs ({args.guided_preset})")
        print(f"  blind-search B = {result.get('blind_B_MG', np.nan):.1f} MG, quality={result.get('blind_quality', '')}")
        print(f"  guided chi2_red = {guided.get('chi2_red', np.nan):.3f}")
    print(f"  baseline = {result.get('baseline_method')} ({result.get('wd_spectral_type')}, {result.get('wd_template_grid')})")
    print(f"  broad components = {result['n_detected_components']}/{result['n_usable_components']}")
    if result.get('n_minor_absorption_components', 0):
        print(
            "  low-weight minor absorptions = "
            f"{result.get('n_minor_absorption_components', 0)} "
            f"(weighted support={result.get('weighted_detected_components', np.nan):.2f})")
    if result.get('n_core_side_pits', 0):
        print(f"  Balmer core-side pits = {result.get('n_core_side_pits', 0)} ({result.get('core_side_pit_series', '')})")
    print(f"  series detected = {result['series_detected']}")
    print(f"  summary = {files.get('summary')}")
    print(f"  plot = {files.get('plot')}")
    if files.get('blue_region_nodes'):
        print(f"  Balmer blue-region nodes = {files.get('blue_region_nodes')}")
    if files.get('red_region_nodes'):
        print(f"  Balmer red-region nodes = {files.get('red_region_nodes')}")
    if files.get('blue_region_plot_pdf'):
        print(f"  Balmer blue-region PDF = {files.get('blue_region_plot_pdf')}")
    if files.get('blue_region_plot_png'):
        print(f"  Balmer blue-region PNG = {files.get('blue_region_plot_png')}")
    if files.get('red_region_plot_pdf'):
        print(f"  Balmer red-region PDF = {files.get('red_region_plot_pdf')}")
    if files.get('red_region_plot_png'):
        print(f"  Balmer red-region PNG = {files.get('red_region_plot_png')}")
    if files.get('full_field_overlay_png'):
        print(f"  Balmer full-field overlay PNG = {files.get('full_field_overlay_png')}")
    if files.get('full_field_overlay_features'):
        print(f"  Balmer full-field features = {files.get('full_field_overlay_features')}")
    if files.get('combined_region_plot_png'):
        print(f"  Balmer blue+red interpolation PNG = {files.get('combined_region_plot_png')}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
