#!/usr/bin/env python3
"""
从 matched_dwd_with_desi.csv 下载所有 DWD 的 DESI 光谱。
直接使用 CSV 中的 TARGETID/SURVEY/PROGRAM/HEALPIX，无需重新匹配坐标。
输出保存到 匹配上的/DWD/ 目录。
"""

import os
import sys
import pandas as pd

# 添加项目根目录到 path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from desi_tool.desi_spectrum_tool import DownloadManager, SpectrumExtractor

# ── 路径配置 ──
INPUT_CSV = os.path.join(PROJECT_ROOT, "匹配上的", "matched_dwd_with_desi.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "匹配上的", "DWD")
COADD_CACHE = os.path.join(PROJECT_ROOT, "output", "coadd_cache")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    # 1. 读取 CSV
    df = pd.read_csv(INPUT_CSV)
    print(f"共 {len(df)} 个 DWD 目标\n")

    # 2. 批量下载 coadd 文件（自动去重）
    targets_info = []
    for _, row in df.iterrows():
        targets_info.append({
            'SURVEY': row['DESI_SURVEY'],
            'PROGRAM': row['DESI_PROGRAM'],
            'HEALPIX': int(row['DESI_HEALPIX']),
        })

    downloader = DownloadManager(cache_dir=COADD_CACHE, max_workers=4)
    print("[1/2] 下载 coadd 文件...")
    coadd_map = downloader.download_coadds(targets_info, cli_progress=True)

    # 3. 逐目标提取光谱并保存
    extractor = SpectrumExtractor()
    summary_rows = []

    print(f"\n[2/2] 提取光谱并保存到 {OUTPUT_DIR}")
    for i, row in df.iterrows():
        tid = int(row['DESI_TARGETID'])
        survey = row['DESI_SURVEY']
        program = row['DESI_PROGRAM']
        healpix = int(row['DESI_HEALPIX'])
        src_name = os.path.splitext(row['img_path'])[0] if pd.notna(row.get('img_path')) else f"target_{tid}"

        coadd_key = (survey, program, healpix)
        coadd_path = coadd_map.get(coadd_key)

        status = 'download_failed'
        if coadd_path is not None:
            spectrum = extractor.extract_from_coadd(coadd_path, tid)
            if spectrum is not None:
                info = {'SURVEY': survey, 'PROGRAM': program, 'HEALPIX': healpix}

                fits_path = os.path.join(OUTPUT_DIR, f"spectrum_{tid}.fits")
                extractor.save_spectrum_fits(spectrum, info, fits_path)

                png_path = os.path.join(OUTPUT_DIR, f"spectrum_{tid}.png")
                extractor.plot_spectrum(spectrum, info, save_path=png_path, show=False)

                status = 'ok'
                print(f"  [{i+1:2d}/{len(df)}] {src_name} -> OK")
            else:
                status = 'extract_failed'
                print(f"  [{i+1:2d}/{len(df)}] {src_name} -> 提取失败")
        else:
            print(f"  [{i+1:2d}/{len(df)}] {src_name} -> 下载失败")

        summary_rows.append({
            'source_name': src_name,
            'TARGETID': tid,
            'RA': row['RA_Decimal'],
            'DEC': row['Dec_Decimal'],
            'SURVEY': survey,
            'PROGRAM': program,
            'HEALPIX': healpix,
            'status': status,
        })

    # 4. 保存汇总
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, "summary.csv")
    summary_df.to_csv(summary_path, index=False)

    ok = sum(1 for r in summary_rows if r['status'] == 'ok')
    print(f"\n完成: 成功 {ok}/{len(df)}")
    print(f"汇总表: {summary_path}")


if __name__ == "__main__":
    main()
