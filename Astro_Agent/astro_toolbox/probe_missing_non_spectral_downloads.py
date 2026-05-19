#!/usr/bin/env python3
"""Probe/download missing non-spectral products for astro_output folders.

This is deliberately separate from the plotting/analysis rerun.  It skips all
spectral products and only checks local folders for missing light curves and SED
photometry, then optionally saves any newly found data into the same target
folder.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(ROOT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from astro_toolbox import gaia_lc, kepler, sed, tess, wise, ztf  # noqa: E402
from astro_toolbox.run_existing_astro_output_analysis import (  # noqa: E402
    _parse_target_coord,
    iter_target_dirs,
)


DEFAULT_SURVEYS = ('ztf', 'wise', 'tess', 'kepler', 'sed')


def _exists(source_dir, survey):
    patterns = {
        'ztf': ('ztf_lightcurve.csv',),
        'wise': ('wise_lightcurve.csv',),
        'tess': ('tess_lightcurve.csv',),
        'kepler': ('kepler_lightcurve.csv', 'k2_lightcurve.csv'),
        'gaia_lc': ('gaia_lightcurve.csv',),
        'sed': ('sed_photometry.csv',),
    }
    return any((source_dir / name).exists() for name in patterns.get(survey, ()))


def _n_points(result, survey):
    if result is None:
        return 0
    if survey in ('ztf', 'wise', 'gaia_lc'):
        total = 0
        for key, value in result.items():
            if hasattr(value, '__len__') and key not in ('ra', 'dec', 'survey', 'web_url'):
                try:
                    total += len(value)
                except Exception:
                    pass
        return int(total)
    if survey in ('tess', 'kepler'):
        return int(result.get('n_points', 0) or 0)
    if survey == 'sed':
        return int(len(getattr(result, 'flux_data', {}) or {}))
    return 0


def _search_tess(ra, dec):
    import lightkurve as lk
    rows = []
    for author in ('SPOC', 'TESS-SPOC', 'QLP'):
        search = lk.search_lightcurve(f'{ra} {dec}', mission='TESS', author=author)
        n = 0 if search is None else len(search)
        if n > 0:
            rows.append({'author': author, 'n_products': int(n)})
    if not rows:
        return None
    return {'survey': 'TESS', 'ra': ra, 'dec': dec,
            'n_points': int(sum(r['n_products'] for r in rows)),
            'search_products': rows, 'probe_only': True}


def _try_tess_download(ra, dec):
    last_error = ''
    for author in ('SPOC', 'TESS-SPOC', 'QLP'):
        try:
            result = tess.query_lightcurve(ra, dec, author=author)
            if result is not None and result.get('n_points', 0) > 0:
                result['author'] = author
                return result, ''
        except Exception as exc:
            last_error = f'{author}:{exc}'
    return None, last_error


def _search_kepler(ra, dec):
    import lightkurve as lk
    rows = []
    for mission in ('Kepler', 'K2'):
        search = lk.search_lightcurve(f'{ra} {dec}', mission=mission)
        n = 0 if search is None else len(search)
        if n > 0:
            rows.append({'mission': mission, 'n_products': int(n)})
    if not rows:
        return None
    return {'survey': 'Kepler/K2', 'ra': ra, 'dec': dec,
            'n_points': int(sum(r['n_products'] for r in rows)),
            'search_products': rows, 'probe_only': True}


def _try_kepler_download(ra, dec):
    last_error = ''
    for mission in ('Kepler', 'K2'):
        try:
            result = kepler.query_lightcurve(ra, dec, mission=mission)
            if result is not None and result.get('n_points', 0) > 0:
                result['survey'] = mission
                return result, ''
        except Exception as exc:
            last_error = f'{mission}:{exc}'
    return None, last_error


def _query_survey(survey, ra, dec, source_dir, save_found):
    if survey == 'ztf':
        result = ztf.query_lightcurve(ra, dec)
        saved = ztf.save_csv(result, str(source_dir)) if save_found and result else ''
        return result, saved
    if survey == 'wise':
        result = wise.query_lightcurve(ra, dec)
        saved = wise.save_lightcurve_csv(result, str(source_dir)) if save_found and result else ''
        return result, saved
    if survey == 'tess':
        if save_found:
            result, err = _try_tess_download(ra, dec)
            if result is None and err:
                raise RuntimeError(err)
            if result is not None:
                tess.plot_lightcurve(result, str(source_dir / 'tess_lightcurve.png'))
                try:
                    lk_period = tess.analyze_period_lightkurve(result, str(source_dir))
                    if lk_period is not None:
                        result['lightkurve_period_analysis'] = lk_period
                except Exception as exc:
                    print(f"  TESS period analysis failed: {exc}")
        else:
            result = _search_tess(ra, dec)
        saved = tess.save_csv(result, str(source_dir)) if save_found and result else ''
        return result, saved
    if survey == 'kepler':
        if save_found:
            result, err = _try_kepler_download(ra, dec)
            if result is None and err:
                raise RuntimeError(err)
        else:
            result = _search_kepler(ra, dec)
        saved = kepler.save_csv(result, str(source_dir)) if save_found and result else ''
        return result, saved
    if survey == 'gaia_lc':
        result = gaia_lc.query_lightcurve(ra, dec)
        saved = gaia_lc.save_csv(result, str(source_dir)) if save_found and result else ''
        return result, saved
    if survey == 'sed':
        save_path = str(source_dir / 'sed.png') if save_found else None
        fitter = sed.quick_sed(ra, dec, save_path=save_path)
        saved = ''
        if save_found and getattr(fitter, 'flux_data', None):
            saved = fitter.save_csv(str(source_dir)) or ''
            try:
                fitter.save_diagnostics(str(source_dir))
                fitter.save_extinction_report(str(source_dir))
            except Exception:
                pass
        return fitter, saved
    raise ValueError(f'unknown survey: {survey}')


def probe_target(source_dir, surveys, args):
    ra, dec = _parse_target_coord(source_dir.name)
    rows = []
    if not np.isfinite(ra + dec):
        for survey in surveys:
            rows.append({
                'target': source_dir.name,
                'source_dir': str(source_dir),
                'survey': survey,
                'status': 'skipped',
                'reason': 'cannot_parse_coordinates',
            })
        return rows

    for survey in surveys:
        row = {
            'target': source_dir.name,
            'source_dir': str(source_dir),
            'ra': ra,
            'dec': dec,
            'survey': survey,
            'already_exists': _exists(source_dir, survey),
        }
        if row['already_exists'] and not args.force_existing:
            row.update({'status': 'existing', 'n_points': np.nan})
            rows.append(row)
            continue

        t0 = time.time()
        try:
            result, saved = _query_survey(
                survey, ra, dec, source_dir, save_found=args.save_found)
            n = _n_points(result, survey)
            row.update({
                'status': 'available' if n > 0 else 'no_data',
                'n_points': n,
                'saved_file': saved or '',
                'elapsed_sec': round(time.time() - t0, 2),
            })
        except Exception as exc:
            if args.verbose_errors:
                traceback.print_exc()
            row.update({
                'status': 'failed',
                'reason': str(exc),
                'elapsed_sec': round(time.time() - t0, 2),
            })
        rows.append(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Probe missing non-spectral downloads for astro_output target folders.')
    parser.add_argument('root')
    parser.add_argument('--recursive', action='store_true')
    parser.add_argument('--include-archival-reruns', action='store_true')
    parser.add_argument('--surveys', default=','.join(DEFAULT_SURVEYS),
                        help='Comma list from ztf,wise,tess,kepler,gaia_lc,sed.')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--save-found', action='store_true',
                        help='Save successfully found products into each target folder.')
    parser.add_argument('--force-existing', action='store_true',
                        help='Probe even when a local product already exists.')
    parser.add_argument('--summary-name', default='missing_non_spectral_download_probe.csv')
    parser.add_argument('--counts-name', default='missing_non_spectral_download_probe_counts.csv')
    parser.add_argument('--verbose-errors', action='store_true')
    args = parser.parse_args(argv)

    surveys = tuple(s.strip().lower() for s in args.surveys.split(',') if s.strip())
    bad = [s for s in surveys if s not in DEFAULT_SURVEYS + ('gaia_lc',)]
    if bad:
        raise ValueError(f'Unknown surveys: {bad}')

    root = Path(args.root).expanduser().resolve()
    targets = list(iter_target_dirs(
        root,
        recursive=args.recursive,
        include_archival=args.include_archival_reruns))
    if args.limit:
        targets = targets[:args.limit]
    print(f'Probe missing non-spectral data: {len(targets)} targets, surveys={surveys}')

    rows = []
    for idx, source_dir in enumerate(targets, 1):
        print(f'[{idx}/{len(targets)}] {source_dir.name}', flush=True)
        rows.extend(probe_target(source_dir, surveys, args))

    summary = pd.DataFrame(rows)
    summary_name = Path(args.summary_name)
    summary_path = summary_name if summary_name.is_absolute() else root / summary_name
    summary.to_csv(summary_path, index=False)
    counts = (summary.groupby(['survey', 'status'], dropna=False)
              .size().reset_index(name='n_targets'))
    counts_name = Path(args.counts_name)
    counts_path = counts_name if counts_name.is_absolute() else root / counts_name
    counts.to_csv(counts_path, index=False)
    print(f'Wrote summary: {summary_path}')
    print(f'Wrote counts: {counts_path}')
    print(counts.to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
