"""Build a local KOA observation-metadata cache for large catalog matching."""
import argparse
import gzip
import os
import time

import numpy as np
import pandas as pd
from astropy.table import Table

from . import koa, utils


KOA_TABLES = {
    'lris': 'koa_lris',
    'hires': 'koa_hires',
    'deimos': 'koa_deimos',
    'esi': 'koa_esi',
    'kcwi': 'koa_kcwi',
    'mosfire': 'koa_mosfire',
    'nirc2': 'koa_nirc2',
    'nirc': 'koa_nirc',
    'nires': 'koa_nires',
    'nirspec': 'koa_nirspec',
    'osiris': 'koa_osiris',
    'lws': 'koa_lws',
}

PREFERRED_COLUMNS = [
    'koaid', 'filehand', 'filesize_mb', 'propint',
    'instrume', 'instrument', 'date_obs', 'date_beg', 'date_end',
    'ut', 'utc', 'utdatetime',
    'ra', 'dec', 'cra', 'cdec', 'equinox',
    'koaimtyp', 'imtype', 'imagetype', 'datlevel',
    'object', 'targname', 'ofname',
    'progid', 'proginst', 'progpi', 'progtitl', 'semester', 'semid',
    'elaptime', 'exptime', 'itime', 'truitime', 'coadds',
    'frameno', 'framenum', 'obsmode',
    'filter', 'dispers', 'dispname', 'grating', 'graname', 'grisname',
    'slitname', 'deckname', 'decker',
    'waveblue', 'wavecntr', 'wavelen', 'wavered',
    'airmass',
]


def _require_pykoa():
    try:
        from pykoa.koa import Koa
    except Exception as exc:
        raise RuntimeError(
            'pykoa is required. Install with `pip install pykoa` or run with '
            'PYTHONPATH pointing to an installed pykoa directory.') from exc
    return Koa


def _read_ipac(path):
    return Table.read(path, format='ascii.ipac').to_pandas()


def query_table_columns(table_name, cache_dir):
    Koa = _require_pykoa()
    utils.ensure_dir(cache_dir)
    out = os.path.join(cache_dir, f'{table_name}_columns.tbl')
    if not os.path.exists(out):
        query = (
            "select column_name, datatype, description "
            "from TAP_SCHEMA.columns "
            f"where table_name='{table_name}'"
        )
        Koa.query_adql(query, out, format='ipac', maxrec=-1)
    df = _read_ipac(out)
    cols = [str(c).strip().lower() for c in df['column_name'].tolist()]
    return [c for c in cols if c and c != 'null']


def choose_columns(available):
    available = [c.lower() for c in available]
    selected = [c for c in PREFERRED_COLUMNS if c in available]
    for required in ('koaid',):
        if required in available and required not in selected:
            selected.insert(0, required)
    return selected


def _date_filter(column, year):
    start = f'{int(year):04d}-01-01'
    end = f'{int(year) + 1:04d}-01-01'
    return f"{column} >= '{start}' and {column} < '{end}'"


def _adql_select(table_name, columns, year=None, date_column='date_obs'):
    cols = ', '.join(columns)
    query = f'select {cols} from {table_name}'
    if year is not None and date_column in columns:
        query += ' where ' + _date_filter(date_column, year)
    return query


def _normalize_dataframe(df, instrument, table_name):
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    out.insert(0, 'koa_table', table_name)
    out.insert(1, 'koa_instrument_key', instrument)

    if 'ra' not in out.columns:
        out['ra'] = np.nan
    if 'dec' not in out.columns:
        out['dec'] = np.nan
    out['ra_deg'] = pd.to_numeric(out.get('ra'), errors='coerce')
    out['dec_deg'] = pd.to_numeric(out.get('dec'), errors='coerce')

    if 'object' not in out.columns:
        out['object'] = ''
    if 'targname' not in out.columns:
        out['targname'] = ''
    if 'instrume' not in out.columns and 'instrument' in out.columns:
        out['instrume'] = out['instrument']
    if 'instrume' not in out.columns:
        out['instrume'] = instrument.upper()
    return out


def _write_chunk(df, out_base):
    csv_gz = out_base + '.csv.gz'
    df.to_csv(csv_gz, index=False, compression='gzip')
    parquet = None
    try:
        parquet = out_base + '.parquet'
        df.to_parquet(parquet, index=False)
    except Exception:
        parquet = None
    return csv_gz, parquet


def download_instrument_metadata(instrument, output_root, start_year=1993,
                                 end_year=None, overwrite=False,
                                 sleep_sec=1.0):
    Koa = _require_pykoa()
    if instrument not in KOA_TABLES:
        raise ValueError(f'Unknown instrument: {instrument}')
    table_name = KOA_TABLES[instrument]
    end_year = end_year or pd.Timestamp.utcnow().year

    inst_dir = os.path.join(output_root, instrument)
    raw_dir = os.path.join(inst_dir, 'raw_ipac')
    chunk_dir = os.path.join(inst_dir, 'chunks')
    utils.ensure_dir(raw_dir)
    utils.ensure_dir(chunk_dir)

    available = query_table_columns(table_name, inst_dir)
    columns = choose_columns(available)
    if 'date_obs' not in columns:
        raise RuntimeError(f'{table_name} has no date_obs column in selected columns')

    manifest = []
    for year in range(int(start_year), int(end_year) + 1):
        raw_path = os.path.join(raw_dir, f'{table_name}_{year}.tbl')
        base = os.path.join(chunk_dir, f'{table_name}_{year}')
        csv_gz = base + '.csv.gz'
        if os.path.exists(csv_gz) and not overwrite:
            try:
                n_rows = sum(1 for _ in gzip.open(csv_gz, 'rt')) - 1
            except Exception:
                n_rows = np.nan
            manifest.append({
                'instrument': instrument,
                'table_name': table_name,
                'year': year,
                'status': 'exists',
                'n_rows': n_rows,
                'csv_gz': csv_gz,
                'parquet': base + '.parquet'
                if os.path.exists(base + '.parquet') else '',
                'bytes_csv_gz': os.path.getsize(csv_gz),
            })
            continue

        query = _adql_select(table_name, columns, year=year)
        status = 'ok'
        error = ''
        try:
            Koa.query_adql(query, raw_path, format='ipac', maxrec=-1)
            if not os.path.exists(raw_path):
                raise RuntimeError('KOA did not create an output table')
            df = _read_ipac(raw_path)
            df = _normalize_dataframe(df, instrument, table_name)
            csv_gz, parquet = _write_chunk(df, base)
            n_rows = len(df)
            bytes_csv_gz = os.path.getsize(csv_gz)
        except Exception as exc:
            status = 'error'
            error = f'{type(exc).__name__}: {exc}'
            n_rows = 0
            csv_gz = ''
            parquet = ''
            bytes_csv_gz = 0
        manifest.append({
            'instrument': instrument,
            'table_name': table_name,
            'year': year,
            'status': status,
            'error': error,
            'n_rows': n_rows,
            'csv_gz': csv_gz,
            'parquet': parquet or '',
            'bytes_csv_gz': bytes_csv_gz,
        })
        print(f'{instrument} {year}: {status}, rows={n_rows}, '
              f'csv_gz={bytes_csv_gz / 1024 / 1024:.2f} MB',
              flush=True)
        time.sleep(float(sleep_sec))

    manifest_df = pd.DataFrame(manifest)
    manifest_path = os.path.join(inst_dir, f'{table_name}_manifest.csv')
    manifest_df.to_csv(manifest_path, index=False)
    return manifest_path


def summarize_cache(output_root):
    manifests = []
    for path in sorted(os.path.glob(os.path.join(output_root, '*', '*_manifest.csv'))
                       if hasattr(os.path, 'glob') else []):
        manifests.append(pd.read_csv(path))


def build_master_summary(output_root):
    manifest_paths = []
    for inst in KOA_TABLES:
        inst_dir = os.path.join(output_root, inst)
        for name in os.listdir(inst_dir) if os.path.isdir(inst_dir) else []:
            if name.endswith('_manifest.csv'):
                manifest_paths.append(os.path.join(inst_dir, name))
    rows = []
    for path in manifest_paths:
        df = pd.read_csv(path)
        rows.append(df)
    if not rows:
        return None
    all_manifest = pd.concat(rows, ignore_index=True)
    all_path = os.path.join(output_root, 'koa_metadata_manifest.csv')
    all_manifest.to_csv(all_path, index=False)

    summary = (all_manifest.groupby(['instrument', 'table_name'], dropna=False)
               .agg(n_rows=('n_rows', 'sum'),
                    n_chunks=('year', 'count'),
                    n_ok=('status', lambda x: int((x == 'ok').sum()
                                                  + (x == 'exists').sum())),
                    bytes_csv_gz=('bytes_csv_gz', 'sum'))
               .reset_index())
    summary['size_csv_gz_mb'] = summary['bytes_csv_gz'] / 1024 / 1024
    summary_path = os.path.join(output_root, 'koa_metadata_summary.csv')
    summary.to_csv(summary_path, index=False)
    return all_path, summary_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Download compact KOA metadata tables by instrument/year.')
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--instrument', action='append',
                        help='Instrument key, repeatable. Default: lris')
    parser.add_argument('--all-instruments', action='store_true')
    parser.add_argument('--start-year', type=int, default=1993)
    parser.add_argument('--end-year', type=int)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--sleep-sec', type=float, default=1.0)
    args = parser.parse_args(argv)

    instruments = (list(KOA_TABLES.keys()) if args.all_instruments
                   else (args.instrument or ['lris']))
    utils.ensure_dir(args.output_root)

    for inst in instruments:
        download_instrument_metadata(
            inst, args.output_root,
            start_year=args.start_year,
            end_year=args.end_year,
            overwrite=args.overwrite,
            sleep_sec=args.sleep_sec)

    built = build_master_summary(args.output_root)
    if built:
        print('manifest:', built[0])
        print('summary:', built[1])


if __name__ == '__main__':
    main()
