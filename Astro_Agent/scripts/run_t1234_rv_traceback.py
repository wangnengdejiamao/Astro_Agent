#!/usr/bin/env python3
"""
未回溯 Tier1234 样本 — DESI 光谱 + RV 拟合 + 轨道回溯
=====================================================
读取 未回溯t1234/desi_match.csv，对每个源:
  1. 查询 DESI 光谱 + SDSS 光谱 + 多波段测光
  2. SED 拟合 (黑体 + 双星模版)
  3. CCF RV 径向速度测量
  4. 轨道回溯到 Hunt+2023 星团
  5. 汇总结果到 CSV

用法:
    python scripts/run_t1234_rv_traceback.py
"""
import os
import sys
import time
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
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# 输入 CSV
CSV_PATH = os.path.join(PROJECT_ROOT, '未回溯t1234', 'desi_match.csv')

# 输出目录
OUTPUT_BASE = os.path.join(PROJECT_ROOT, '未回溯t1234', 'results')

# 汇总表路径
SUMMARY_CSV = os.path.join(PROJECT_ROOT, '未回溯t1234', 'rv_traceback_summary.csv')


def main():
    from test_toolbox import AstroQueryAll

    df = pd.read_csv(CSV_PATH)
    print(f"输入: {CSV_PATH}")
    print(f"共 {len(df)} 个源\n")

    os.makedirs(OUTPUT_BASE, exist_ok=True)

    # 汇总结果列表
    summary_rows = []

    total = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        ra = float(row['ra'])
        dec = float(row['dec'])
        source_id = str(int(float(row['source_id'])))
        cluster = str(row['cluster']) if pd.notna(row.get('cluster')) else ''
        tier = str(row['tier']) if pd.notna(row.get('tier')) else ''
        orbit_confirmed = row.get('orbit_confirmed', False)

        # 目录名: 星团_Gaia_sourceID
        dir_name = f"{cluster}_Gaia_{source_id}" if cluster else f"Gaia_{source_id}"
        dir_name = dir_name.replace(' ', '_').replace('/', '_')
        out_dir = os.path.join(OUTPUT_BASE, dir_name)

        source_label = f"[{tier}] {cluster} Gaia {source_id}"

        # 检查是否已完成 (有 summary.txt 且有 rv_analysis.txt)
        if (os.path.exists(os.path.join(out_dir, 'summary.txt'))
                and os.path.exists(os.path.join(out_dir, 'rv_analysis.txt'))):
            print(f"[{i+1}/{total}] {source_label}: 已完成, 跳过")
            # 从已有结果中提取汇总信息
            summary_row = _extract_existing_summary(
                out_dir, source_id, ra, dec, cluster, tier, orbit_confirmed)
            if summary_row:
                summary_rows.append(summary_row)
            continue

        # 如果已有 summary 但没有 rv_analysis → 补跑 RV + 回溯
        if os.path.exists(os.path.join(out_dir, 'summary.txt')):
            print(f"\n[{i+1}/{total}] {source_label}: 补跑 RV + 回溯")
            t0 = time.time()
            try:
                _rerun_rv_traceback(out_dir, ra, dec)
            except Exception as e:
                print(f"  补跑失败: {e}")
                import traceback
                traceback.print_exc()
            dt = time.time() - t0
            # 简单汇总
            summary_row = _extract_existing_summary(
                out_dir, source_id, ra, dec, cluster, tier, orbit_confirmed)
            if summary_row:
                summary_row['elapsed_s'] = round(dt, 1)
                summary_rows.append(summary_row)
            _save_summary_csv(summary_rows)
            continue

        print(f"\n\n{'#'*70}")
        print(f"  [{i+1}/{total}] {source_label}")
        print(f"  RA={ra:.6f}  DEC={dec:.6f}")
        print(f"  输出: {out_dir}")
        print(f"{'#'*70}")

        t0 = time.time()

        # 运行全套查询 + 分析
        querier = AstroQueryAll(ra, dec, output_dir=out_dir)
        querier.source_label = source_label
        try:
            querier.query_all()
            querier.save_and_plot_all()
        except Exception as e:
            print(f"  处理失败: {e}")
            import traceback
            traceback.print_exc()

        dt = time.time() - t0

        # 提取关键结果用于汇总
        summary_row = _extract_summary(
            querier, source_id, ra, dec, cluster, tier, orbit_confirmed, dt)
        summary_rows.append(summary_row)

        # 每处理一个源就追加保存汇总 (防止中断丢失)
        _save_summary_csv(summary_rows)

    # 最终保存
    _save_summary_csv(summary_rows)

    print(f"\n\n{'='*70}")
    print(f"  全部完成! 共 {total} 个源")
    print(f"  结果目录: {OUTPUT_BASE}")
    print(f"  汇总表: {SUMMARY_CSV}")
    print(f"{'='*70}")


def _rerun_rv_traceback(out_dir, ra, dec):
    """从已保存的光谱 CSV 恢复数据，只补跑 RV + 轨道回溯"""
    import pandas as pd

    results = {}

    # 恢复 DESI 光谱
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

    # 恢复 SDSS 光谱
    sdss_csv = os.path.join(out_dir, 'sdss_spectrum.csv')
    if os.path.exists(sdss_csv):
        df = pd.read_csv(sdss_csv)
        if len(df) > 0:
            results['SDSS_spectrum'] = {
                'wavelength': df['wavelength_A'].values,
                'flux': df['flux'].values,
                'error': df['error'].values,
                'z': 0.0,
            }

    # 恢复 HST 光谱
    hst_csv = os.path.join(out_dir, 'hst_spectrum.csv')
    if os.path.exists(hst_csv):
        df = pd.read_csv(hst_csv)
        if len(df) > 0:
            results['HST_spectrum'] = {
                'wavelength': df['wavelength_A'].values,
                'flux': df['flux'].values,
                'error': df['error'].values,
            }

    if not results:
        print("  无光谱数据, 跳过 RV")
        return

    print(f"  可用光谱: {list(results.keys())}")

    # RV 分析
    from astro_toolbox.rv_fitting import run_rv_analysis
    rv_report = run_rv_analysis(results, output_dir=out_dir, ra=ra, dec=dec)

    # 轨道回溯
    if rv_report and rv_report.get('best_rv') is not None:
        from astro_toolbox.orbit_traceback import run_traceback_analysis
        run_traceback_analysis(results, rv_report,
                               output_dir=out_dir, ra=ra, dec=dec)


def _extract_summary(querier, source_id, ra, dec, cluster, tier,
                     orbit_confirmed, elapsed):
    """从 AstroQueryAll 结果中提取汇总行"""
    results = querier.results
    row = {
        'source_id': source_id,
        'ra': ra,
        'dec': dec,
        'cluster': cluster,
        'tier': tier,
        'orbit_confirmed_input': orbit_confirmed,
        'elapsed_s': round(elapsed, 1),
    }

    # DESI 光谱
    desi = results.get('DESI')
    row['has_desi_spec'] = desi is not None and 'spectrum' in (desi or {})

    # SDSS 光谱
    sdss = results.get('SDSS_spectrum')
    row['has_sdss_spec'] = sdss is not None

    # SED 拟合
    sed = results.get('SED')
    if sed and hasattr(sed, 'fit_result') and sed.fit_result:
        row['sed_teff'] = round(sed.fit_result['Teff'], 0)
        row['sed_teff_err'] = round(sed.fit_result['Teff_err'], 0)
        row['sed_chi2'] = round(sed.fit_result['chi2_red'], 2)
    else:
        row['sed_teff'] = np.nan
        row['sed_teff_err'] = np.nan
        row['sed_chi2'] = np.nan

    # 双星 SED 拟合
    bsed = results.get('Binary_SED')
    if bsed and hasattr(bsed, 'fit_result') and bsed.fit_result:
        fr = bsed.fit_result
        row['binary_wd_teff'] = fr['wd_teff']
        row['binary_wd_logg'] = fr['wd_logg']
        row['binary_mdwarf_type'] = fr['mdwarf_type']
        row['binary_chi2'] = round(fr['chi2_red'], 2)
    else:
        row['binary_wd_teff'] = np.nan
        row['binary_wd_logg'] = np.nan
        row['binary_mdwarf_type'] = ''
        row['binary_chi2'] = np.nan

    # RV 分析
    rv = results.get('rv_analysis')
    if rv:
        row['rv_km_s'] = round(rv['best_rv'], 1) if rv.get('best_rv') is not None else np.nan
        row['rv_err_km_s'] = round(rv['best_rv_err'], 1) if rv.get('best_rv_err') is not None else np.nan
        row['rv_source'] = rv.get('best_rv_source', '')
        row['is_sb2'] = rv.get('is_sb2', False)
    else:
        row['rv_km_s'] = np.nan
        row['rv_err_km_s'] = np.nan
        row['rv_source'] = ''
        row['is_sb2'] = False

    # 轨道回溯
    tb = results.get('orbit_traceback')
    if tb and tb.get('best_match'):
        bm = tb['best_match']
        row['traceback_cluster'] = bm.get('cluster_name', '')
        row['traceback_min_sep_pc'] = round(bm.get('min_sep_pc', np.nan), 1)
        row['traceback_time_myr'] = round(bm.get('time_myr', np.nan), 1)
        row['traceback_within_tidal'] = bm.get('within_tidal', False)
    else:
        row['traceback_cluster'] = ''
        row['traceback_min_sep_pc'] = np.nan
        row['traceback_time_myr'] = np.nan
        row['traceback_within_tidal'] = False

    return row


def _extract_existing_summary(out_dir, source_id, ra, dec, cluster, tier,
                              orbit_confirmed):
    """从已完成的输出目录中提取汇总信息"""
    row = {
        'source_id': source_id,
        'ra': ra,
        'dec': dec,
        'cluster': cluster,
        'tier': tier,
        'orbit_confirmed_input': orbit_confirmed,
        'elapsed_s': 0,
    }

    # 检查各文件是否存在
    row['has_desi_spec'] = os.path.exists(os.path.join(out_dir, 'desi_spectrum.csv'))
    row['has_sdss_spec'] = os.path.exists(os.path.join(out_dir, 'sdss_spectrum.csv'))

    # 尝试从 rv_analysis.txt 提取 RV
    rv_path = os.path.join(out_dir, 'rv_analysis.txt')
    row['rv_km_s'] = np.nan
    row['rv_err_km_s'] = np.nan
    row['rv_source'] = ''
    row['is_sb2'] = False
    if os.path.exists(rv_path):
        try:
            with open(rv_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('Best RV'):
                        # "Best RV = 12.3 +/- 4.5 km/s (source)"
                        parts = line.split('=')
                        if len(parts) >= 2:
                            val = parts[1].strip()
                            if '+/-' in val:
                                rv_val, rest = val.split('+/-')
                                row['rv_km_s'] = float(rv_val.strip())
                                err_part = rest.split('km/s')[0].strip()
                                row['rv_err_km_s'] = float(err_part)
                    if 'SB2' in line.upper() and 'YES' in line.upper():
                        row['is_sb2'] = True
        except Exception:
            pass

    # 尝试从 orbit_traceback.txt 提取回溯结果
    tb_path = os.path.join(out_dir, 'orbit_traceback.txt')
    row['traceback_cluster'] = ''
    row['traceback_min_sep_pc'] = np.nan
    row['traceback_time_myr'] = np.nan
    row['traceback_within_tidal'] = False
    if os.path.exists(tb_path):
        try:
            with open(tb_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('Best match:') or line.startswith('#1'):
                        # 提取星团名
                        if ':' in line:
                            row['traceback_cluster'] = line.split(':')[1].strip().split()[0]
        except Exception:
            pass

    # SED
    row['sed_teff'] = np.nan
    row['sed_teff_err'] = np.nan
    row['sed_chi2'] = np.nan

    # Binary SED
    row['binary_wd_teff'] = np.nan
    row['binary_wd_logg'] = np.nan
    row['binary_mdwarf_type'] = ''
    row['binary_chi2'] = np.nan

    return row


def _save_summary_csv(rows):
    """保存汇总 CSV"""
    if not rows:
        return
    df = pd.DataFrame(rows)
    # 确保列顺序合理
    col_order = [
        'source_id', 'ra', 'dec', 'cluster', 'tier', 'orbit_confirmed_input',
        'has_desi_spec', 'has_sdss_spec',
        'sed_teff', 'sed_teff_err', 'sed_chi2',
        'binary_wd_teff', 'binary_wd_logg', 'binary_mdwarf_type', 'binary_chi2',
        'rv_km_s', 'rv_err_km_s', 'rv_source', 'is_sb2',
        'traceback_cluster', 'traceback_min_sep_pc', 'traceback_time_myr',
        'traceback_within_tidal',
        'elapsed_s',
    ]
    cols = [c for c in col_order if c in df.columns]
    cols += [c for c in df.columns if c not in cols]
    df = df[cols]
    df.to_csv(SUMMARY_CSV, index=False)


if __name__ == '__main__':
    main()
