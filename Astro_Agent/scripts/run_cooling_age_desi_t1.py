#!/usr/bin/env python3
"""
对 desi_match.csv 中所有 Tier1 源运行冷却年龄分析
===================================================
Tier1 源已被认证为星团成员，不需要轨道回溯。
直接分析 t_cool + t_MS vs t_cluster 的关系。
"""
import os
import sys
import warnings
import time

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CSV_PATH = os.path.join(PROJECT_ROOT, '未回溯t1234', 'desi_match.csv')
RESULTS_BASE = os.path.join(PROJECT_ROOT, '未回溯t1234', 'cooling_age_results')

import pandas as pd
import numpy as np
from astro_toolbox.cooling_age import (
    run_cooling_age_analysis, _load_wd_model, get_gaia_photometry
)


def main():
    os.makedirs(RESULTS_BASE, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    t1 = df[df['tier'] == 'Tier1'].copy()
    t1 = t1.drop_duplicates(subset='source_id', keep='first')

    print(f"Tier1 源总数: {len(t1)}")
    print(f"输出目录: {RESULTS_BASE}")
    print()

    # 预加载模型
    _load_wd_model()

    reports = []
    total = len(t1)

    for i, (_, row) in enumerate(t1.iterrows()):
        sid = str(int(row['source_id']))
        ra = float(row['ra'])
        dec = float(row['dec'])
        cluster = str(row['cluster'])
        bp_rp = float(row['bp_rp'])
        gmag = float(row['phot_g_mean_mag'])
        plx = float(row['parallax'])

        dir_name = f"{cluster}_Gaia_{sid}"
        out_dir = os.path.join(RESULTS_BASE, dir_name)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"[{i+1}/{total}] {cluster}  Gaia {sid}")
        print(f"  RA={ra:.6f}  DEC={dec:.6f}")
        print(f"  G={gmag:.3f}  BP-RP={bp_rp:.3f}  Plx={plx:.3f} mas")
        print(f"{'='*60}")

        # 跳过已完成的
        existing = os.path.join(out_dir, 'cooling_age_analysis.txt')
        if os.path.exists(existing):
            print("  已完成, 跳过")
            with open(existing) as f:
                text = f.read()
            status = 'UNKNOWN'
            for line in text.split('\n'):
                if line.startswith('状态:'):
                    status = line.split(':')[1].strip()
                    break
            reports.append({
                'source_id': sid, 'cluster': cluster, 'status': status,
                'ra': ra, 'dec': dec, 'bp_rp_csv': bp_rp, 'gmag_csv': gmag,
                'plx_csv': plx
            })
            continue

        # 直接用 CSV 中的 Gaia 数据构造测光字典，避免重复查询
        if plx > 0:
            dist_pc = 1000.0 / plx
            M_G = gmag - 5.0 * np.log10(dist_pc / 10.0)
            gaia_phot = {
                'source_id': float(row['source_id']),
                'ra': ra, 'dec': dec,
                'Gmag': gmag,
                'BPmag': gmag - (gmag - (gmag + bp_rp)) / 2 + bp_rp / 2,  # rough
                'RPmag': gmag - (gmag - (gmag + bp_rp)) / 2 - bp_rp / 2,  # rough
                'Plx': plx, 'e_Plx': 0.1,
                'RUWE': np.nan,
                'dist_pc': dist_pc,
                'M_G': M_G,
                'BP_RP': bp_rp,
            }
            # BPmag = Gmag + BC_BP, RPmag = Gmag + BC_RP
            # but BP_RP = BPmag - RPmag is what matters
            # Actually we need proper BPmag and RPmag for the report
            # Let's query Gaia for exact values
            gaia_phot_real = get_gaia_photometry(ra, dec)
            if gaia_phot_real is not None:
                gaia_phot = gaia_phot_real
            else:
                # 用 CSV 数据
                gaia_phot['BPmag'] = gmag + bp_rp / 2  # approximate
                gaia_phot['RPmag'] = gmag - bp_rp / 2
        else:
            gaia_phot = None

        try:
            report = run_cooling_age_analysis(
                ra, dec,
                cluster_name=cluster,
                output_dir=out_dir,
                gaia_phot=gaia_phot)
            report['source_id'] = sid
            report['bp_rp_csv'] = bp_rp
            report['gmag_csv'] = gmag
            report['plx_csv'] = plx
            reports.append(report)
        except Exception as e:
            print(f"  出错: {e}")
            import traceback
            traceback.print_exc()
            reports.append({
                'ra': ra, 'dec': dec, 'cluster': cluster,
                'source_id': sid, 'status': 'ERROR', 'error': str(e),
                'bp_rp_csv': bp_rp, 'gmag_csv': gmag, 'plx_csv': plx
            })

        # 限速: 避免 Vizier 查询太快
        time.sleep(0.3)

    # ---- 汇总 ----
    _print_detailed_summary(reports)

    # 保存汇总 CSV
    _save_summary_csv(reports)


def _print_detailed_summary(reports):
    print(f"\n\n{'='*70}")
    print("Tier1 冷却年龄分析汇总")
    print(f"{'='*70}")

    merger = [r for r in reports if r.get('status') == 'MERGER_CANDIDATE']
    marginal = [r for r in reports if r.get('status') == 'MARGINAL_MERGER']
    consistent = [r for r in reports if r.get('status') == 'CONSISTENT']
    failed = [r for r in reports if r.get('status') in ('FAILED', 'NO_CLUSTER_AGE', 'ERROR', 'UNKNOWN')]
    wd_total = len(merger) + len(marginal) + len(consistent)

    print(f"\n总源数: {len(reports)}")
    print(f"成功分析 (在WD冷却轨迹上): {wd_total}")
    print(f"  并合候选体: {len(merger)}")
    print(f"  边缘候选: {len(marginal)}")
    print(f"  单星一致: {len(consistent)}")
    print(f"失败/超出范围 (非WD?): {len(failed)}")

    if merger:
        print(f"\n--- 并合候选体 ({len(merger)}) ---")
        print(f"{'Cluster':15s} {'Source_ID':22s} {'M_WD':>6s} {'Teff':>6s} "
              f"{'t_cool':>8s} {'t_MS':>8s} {'t_total':>8s} {'t_clust':>8s} {'ratio':>6s}")
        for r in sorted(merger, key=lambda x: x.get('cluster', '')):
            w = r.get('wd_params', {})
            m = r.get('merger', {})
            print(f"{r.get('cluster',''):15s} {r.get('source_id',''):22s} "
                  f"{w.get('mass',0):6.3f} {w.get('teff',0):6.0f} "
                  f"{m.get('t_cool_gyr',0)*1e3:8.1f} {m.get('t_ms_gyr',0)*1e3:8.1f} "
                  f"{m.get('t_total_single_gyr',0)*1e3:8.1f} {m.get('t_cluster_gyr',0)*1e3:8.1f} "
                  f"{m.get('ratio',0):6.1f}x")

    if consistent:
        print(f"\n--- 与单星演化一致 ({len(consistent)}) ---")
        for r in consistent:
            w = r.get('wd_params', {})
            m = r.get('merger', {})
            print(f"  {r.get('cluster',''):15s} {r.get('source_id',''):22s} "
                  f"M_WD={w.get('mass',0):.3f}  "
                  f"t_cool+t_MS={m.get('t_total_single_gyr',0)*1e3:.0f} Myr  "
                  f"t_cluster={m.get('t_cluster_gyr',0)*1e3:.0f} Myr")


def _save_summary_csv(reports):
    rows = []
    for r in reports:
        row = {
            'source_id': r.get('source_id', ''),
            'cluster': r.get('cluster', ''),
            'ra': r.get('ra', ''),
            'dec': r.get('dec', ''),
            'status': r.get('status', ''),
        }
        if 'gaia' in r:
            g = r['gaia']
            row['Gmag'] = f"{g.get('Gmag', np.nan):.4f}"
            row['BP_RP'] = f"{g.get('BP_RP', np.nan):.4f}"
            row['M_G'] = f"{g.get('M_G', np.nan):.4f}"
            row['Plx'] = f"{g.get('Plx', np.nan):.4f}"
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
            row['mass_flag'] = m.get('mass_flag', '')
        rows.append(row)

    summary_path = os.path.join(RESULTS_BASE, 'cooling_age_summary_t1.csv')
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"\n汇总 CSV 已保存: {summary_path}")


if __name__ == '__main__':
    main()
