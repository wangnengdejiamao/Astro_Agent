#!/usr/bin/env python3
"""
逃逸星团白矮星样本 — 全波段工具箱批量查询
==========================================
读取 spectra_download_urls.csv，去重后对每个唯一源跑 astro_toolbox 全套分析。

用法:
    python scripts/run_escaped_wd_batch.py
"""
import os
import sys
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from test_toolbox import AstroQueryAll

# 输入 CSV
CSV_PATH = os.path.join(PROJECT_ROOT, '逃逸回溯星团的双白矮星', 'spectra_download_urls.csv')

# 输出到新文件夹
OUTPUT_BASE = os.path.join(PROJECT_ROOT, '逃逸回溯星团的双白矮星', 'toolbox_results')


def main():
    df = pd.read_csv(CSV_PATH)
    print(f"原始记录: {len(df)} 行")

    # 按 source_id 去重，保留第一条（取唯一坐标即可）
    df_unique = df.drop_duplicates(subset='source_id', keep='first')
    print(f"去重后: {len(df_unique)} 个唯一源\n")

    os.makedirs(OUTPUT_BASE, exist_ok=True)

    total = len(df_unique)
    for i, (_, row) in enumerate(df_unique.iterrows()):
        ra = float(row['ra'])
        dec = float(row['dec'])
        source_id = str(int(float(row['source_id'])))
        cluster = str(row['cluster']) if pd.notna(row.get('cluster')) else ''

        # 目录名: 星团_Gaia_sourceID
        dir_name = f"{cluster}_Gaia_{source_id}" if cluster else f"Gaia_{source_id}"
        dir_name = dir_name.replace(' ', '_').replace('/', '_')
        out_dir = os.path.join(OUTPUT_BASE, dir_name)

        source_label = f"{cluster} Gaia {source_id}" if cluster else f"Gaia {source_id}"

        print(f"\n\n{'#'*70}")
        print(f"  [{i+1}/{total}] {source_label}")
        print(f"  RA={ra:.6f}  DEC={dec:.6f}")
        print(f"  输出: {out_dir}")
        print(f"{'#'*70}")

        querier = AstroQueryAll(ra, dec, output_dir=out_dir)
        querier.source_label = source_label
        querier.query_all()
        querier.save_and_plot_all()

    print(f"\n\n{'='*70}")
    print(f"  批量查询完成! 共 {total} 个源")
    print(f"  输出目录: {OUTPUT_BASE}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
