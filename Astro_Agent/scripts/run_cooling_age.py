#!/usr/bin/env python3
"""
对所有逃逸白矮星样本运行冷却年龄分析
====================================
用法:
    python scripts/run_cooling_age.py
"""
import os
import sys
import warnings

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CSV_PATH = os.path.join(PROJECT_ROOT, '逃逸回溯星团的双白矮星', 'spectra_download_urls.csv')
RESULTS_BASE = os.path.join(PROJECT_ROOT, '逃逸回溯星团的双白矮星', 'toolbox_results')

from astro_toolbox.cooling_age import run_cooling_age_analysis, _load_wd_model, _print_summary

import pandas as pd


def main():
    df = pd.read_csv(CSV_PATH)
    df = df.drop_duplicates(subset='source_id', keep='first')

    # 预加载模型
    _load_wd_model()

    # 预加载星团表
    from astro_toolbox.orbit_traceback import load_hunt2023_clusters
    print("预加载星团表...")
    clusters = load_hunt2023_clusters()
    print()

    reports = []
    total = len(df)

    for i, (_, row) in enumerate(df.iterrows()):
        sid = str(int(float(row['source_id'])))
        ra = float(row['ra'])
        dec = float(row['dec'])
        cluster = str(row['cluster']) if pd.notna(row.get('cluster')) else ''

        dir_name = f"{cluster}_Gaia_{sid}" if cluster else f"Gaia_{sid}"
        dir_name = dir_name.replace(' ', '_').replace('/', '_')
        out_dir = os.path.join(RESULTS_BASE, dir_name)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"[{i+1}/{total}] {cluster} Gaia {sid}")
        print(f"  RA={ra:.6f}  DEC={dec:.6f}")
        print(f"{'='*60}")

        # 跳过已完成的
        existing = os.path.join(out_dir, 'cooling_age_analysis.txt')
        if os.path.exists(existing):
            print("  已完成, 跳过")
            # 读取状态
            with open(existing) as f:
                text = f.read()
            status = 'UNKNOWN'
            for line in text.split('\n'):
                if line.startswith('状态:'):
                    status = line.split(':')[1].strip()
                    break
            reports.append({'cluster': cluster, 'status': status,
                           'ra': ra, 'dec': dec, 'source_id': sid})
            continue

        try:
            report = run_cooling_age_analysis(
                ra, dec,
                cluster_name=cluster,
                output_dir=out_dir)
            report['source_id'] = sid
            reports.append(report)
        except Exception as e:
            print(f"  出错: {e}")
            import traceback
            traceback.print_exc()
            reports.append({
                'ra': ra, 'dec': dec, 'cluster': cluster,
                'source_id': sid, 'status': 'ERROR', 'error': str(e)
            })

    _print_summary(reports)

    # 保存汇总 CSV
    summary_path = os.path.join(RESULTS_BASE, 'cooling_age_summary.csv')
    rows = []
    for r in reports:
        row = {
            'source_id': r.get('source_id', ''),
            'cluster': r.get('cluster', ''),
            'ra': r.get('ra', ''),
            'dec': r.get('dec', ''),
            'status': r.get('status', ''),
        }
        if 'wd_params' in r:
            w = r['wd_params']
            row['M_WD'] = f"{w['mass']:.4f}"
            row['Teff'] = f"{w['teff']:.0f}"
            row['logg'] = f"{w['logg']:.4f}"
            row['t_cool_Gyr'] = f"{w['cooling_age_gyr']:.4f}"
        if 'merger' in r:
            m = r['merger']
            row['t_MS_Gyr'] = f"{m['t_ms_gyr']:.4f}"
            row['t_total_Gyr'] = f"{m['t_total_single_gyr']:.4f}"
            row['t_cluster_Gyr'] = f"{m['t_cluster_gyr']:.4f}"
            row['delta_Gyr'] = f"{m['delta_gyr']:.4f}"
            row['merger_flag'] = m['merger_flag']
        rows.append(row)

    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"\n汇总 CSV 已保存: {summary_path}")


if __name__ == '__main__':
    main()
