#!/usr/bin/env python3
"""
对已完成的逃逸白矮星样本补跑 RV 拟合 + 轨道回溯
================================================
读取 toolbox_results 下每个源的已有光谱数据,
做 CCF RV 测量 (单星+双星) 和轨道回溯到 Hunt+2023 星团。

用法:
    python scripts/run_rv_traceback.py
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

RESULTS_BASE = os.path.join(PROJECT_ROOT, '逃逸回溯星团的双白矮星', 'toolbox_results')
CSV_PATH = os.path.join(PROJECT_ROOT, '逃逸回溯星团的双白矮星', 'spectra_download_urls.csv')


def load_cached_results(out_dir):
    """从已保存的 CSV 文件恢复查询结果"""
    results = {}

    # SDSS
    sdss_csv = os.path.join(out_dir, 'sdss_spectrum.csv')
    if os.path.exists(sdss_csv):
        df = pd.read_csv(sdss_csv)
        if len(df) > 0:
            results['SDSS_spectrum'] = {
                'wavelength': df['wavelength_A'].values,
                'flux': df['flux'].values,
                'error': df['error'].values,
                'z': 0.0,  # 占位, pipeline z 需要从 summary 恢复
            }

    # DESI
    desi_csv = os.path.join(out_dir, 'desi_spectrum.csv')
    if os.path.exists(desi_csv):
        df = pd.read_csv(desi_csv)
        sp = {}
        for band in ('B', 'R', 'Z'):
            bdf = df[df['band'] == band]
            if len(bdf) > 0:
                sp[band] = {
                    'wavelength': bdf['wavelength_A'].values,
                    'flux': bdf['flux'].values,
                    'error': bdf['error'].values,
                }
        if sp:
            results['DESI'] = {'spectrum': sp, 'match': {'z': 0.0}}

    # HST
    hst_csv = os.path.join(out_dir, 'hst_spectrum.csv')
    if os.path.exists(hst_csv):
        df = pd.read_csv(hst_csv)
        if len(df) > 0:
            results['HST_spectrum'] = {
                'wavelength': df['wavelength_A'].values,
                'flux': df['flux'].values,
                'error': df['error'].values,
            }

    return results


def main():
    # 读取源表获取坐标
    df_src = pd.read_csv(CSV_PATH)
    df_src = df_src.drop_duplicates(subset='source_id', keep='first')

    # 源 ID → (ra, dec, cluster) 映射
    src_map = {}
    for _, row in df_src.iterrows():
        sid = str(int(float(row['source_id'])))
        cluster = str(row['cluster']) if pd.notna(row.get('cluster')) else ''
        dir_name = f"{cluster}_Gaia_{sid}" if cluster else f"Gaia_{sid}"
        dir_name = dir_name.replace(' ', '_').replace('/', '_')
        src_map[dir_name] = (float(row['ra']), float(row['dec']), cluster, sid)

    # 扫描已完成的源
    if not os.path.isdir(RESULTS_BASE):
        print(f"未找到结果目录: {RESULTS_BASE}")
        return

    dirs = sorted(os.listdir(RESULTS_BASE))
    total = len(dirs)
    done = 0
    skipped = 0

    # 预加载模版 (只加载一次)
    from astro_toolbox.rv_fitting import _get_best_wd_templates
    print("预加载 WD 模版...")
    templates = _get_best_wd_templates()
    print(f"  {len(templates)} 个模版就绪\n")

    # 预加载星团表
    from astro_toolbox.orbit_traceback import load_hunt2023_clusters
    print("预加载星团表...")
    clusters = load_hunt2023_clusters()
    print()

    for i, dir_name in enumerate(dirs):
        out_dir = os.path.join(RESULTS_BASE, dir_name)
        if not os.path.isdir(out_dir):
            continue

        # 检查是否有 summary (即基础工具箱已完成)
        if not os.path.exists(os.path.join(out_dir, 'summary.txt')):
            print(f"[{i+1}/{total}] {dir_name}: 基础工具箱未完成, 跳过")
            skipped += 1
            continue

        # 检查是否已有 RV
        if os.path.exists(os.path.join(out_dir, 'rv_analysis.txt')):
            print(f"[{i+1}/{total}] {dir_name}: RV 已完成, 跳过")
            done += 1
            continue

        # 获取坐标
        info = src_map.get(dir_name)
        if info is None:
            print(f"[{i+1}/{total}] {dir_name}: 未找到坐标信息, 跳过")
            skipped += 1
            continue

        ra, dec, cluster, sid = info
        print(f"\n{'='*60}")
        print(f"[{i+1}/{total}] {cluster} Gaia {sid}")
        print(f"  RA={ra:.6f}  DEC={dec:.6f}")
        print(f"{'='*60}")

        # 加载已有光谱
        results = load_cached_results(out_dir)
        if not results:
            print("  无光谱数据, 跳过 RV")
            skipped += 1
            continue

        spec_names = list(results.keys())
        print(f"  可用光谱: {spec_names}")

        # RV 分析
        from astro_toolbox.rv_fitting import run_rv_analysis
        try:
            rv_report = run_rv_analysis(results, output_dir=out_dir,
                                         ra=ra, dec=dec)
        except Exception as e:
            print(f"  RV 分析出错: {e}")
            rv_report = None

        # 轨道回溯
        if rv_report and rv_report.get('best_rv') is not None:
            from astro_toolbox.orbit_traceback import run_traceback_analysis
            try:
                tb = run_traceback_analysis(results, rv_report,
                                             output_dir=out_dir,
                                             ra=ra, dec=dec)
            except Exception as e:
                print(f"  轨道回溯出错: {e}")

        done += 1

    print(f"\n\n{'='*60}")
    print(f"完成! 处理 {done} 个源, 跳过 {skipped} 个")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
