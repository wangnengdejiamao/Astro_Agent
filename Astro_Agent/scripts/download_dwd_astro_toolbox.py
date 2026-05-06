#!/usr/bin/env python3
"""
用 astro_toolbox 全波段查询所有 DWD 目标。
读取 matched_dwd_with_desi.csv，对每个源运行 AstroQueryAll（SDSS、LAMOST、
DESI、ZTF、WISE、TESS、Kepler、HST、JWST、SED、HR 等全部模块），
输出保存到 匹配上的/DWD/ 下按源名称命名的子目录中。
"""

import os
import sys
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.test_toolbox import AstroQueryAll

INPUT_CSV = os.path.join(PROJECT_ROOT, "匹配上的", "matched_dwd_with_desi.csv")
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "匹配上的", "DWD")


def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"\n{'='*70}")
    print(f"  astro_toolbox 全波段查询: {INPUT_CSV}")
    print(f"  共 {len(df)} 个 DWD 目标")
    print(f"  输出目录: {OUTPUT_BASE}")
    print(f"{'='*70}")

    for i, row in df.iterrows():
        ra = float(row['RA_Decimal'])
        dec = float(row['Dec_Decimal'])

        # 用 img_path 中的 ZTF 名称作为目录名
        src_name = os.path.splitext(str(row['img_path']))[0]
        dir_name = src_name.replace(' ', '_').replace('/', '_')
        out_dir = os.path.join(OUTPUT_BASE, dir_name)

        print(f"\n\n{'#'*70}")
        print(f"  [{i+1}/{len(df)}] {src_name}")
        print(f"  RA={ra:.6f}  DEC={dec:.6f}")
        print(f"  输出: {out_dir}")
        print(f"{'#'*70}")

        querier = AstroQueryAll(ra, dec, output_dir=out_dir)
        querier.source_label = src_name
        querier.query_all()
        querier.save_and_plot_all()

    print(f"\n\n{'='*70}")
    print(f"  全部完成! 共 {len(df)} 个源")
    print(f"  输出目录: {OUTPUT_BASE}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
