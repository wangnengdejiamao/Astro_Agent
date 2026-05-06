#!/usr/bin/env python3
"""
整合全部冷却年龄分析结果 + SIMBAD 文献查询，保存为综合 CSV
==========================================================
合并两个样本:
  1. 逃逸回溯样本 (spectra_download_urls.csv) — 18 源
  2. desi_match.csv Tier1 — 64 源
去重后查询 SIMBAD，标注是否被单独研究过。
"""
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd


def load_escaped_sample():
    """加载逃逸回溯样本"""
    csv_path = os.path.join(PROJECT_ROOT, '逃逸回溯星团的双白矮星', 'spectra_download_urls.csv')
    df = pd.read_csv(csv_path, dtype={'source_id': str})
    df['source_id'] = df['source_id'].str.strip()
    df = df.drop_duplicates(subset='source_id', keep='first')
    df['sample'] = 'escaped'
    df['tier'] = 'escaped'
    df['orbit_confirmed'] = True
    return df[['source_id', 'ra', 'dec', 'cluster', 'sample', 'tier', 'orbit_confirmed']]


def load_desi_t1():
    """加载 desi_match Tier1 样本"""
    csv_path = os.path.join(PROJECT_ROOT, '未回溯t1234', 'desi_match.csv')
    df = pd.read_csv(csv_path, dtype={'source_id': str})
    df['source_id'] = df['source_id'].str.strip()
    df_t1 = df[df['tier'] == 'Tier1'].copy()
    df_t1 = df_t1.drop_duplicates(subset='source_id', keep='first')
    df_t1['sample'] = 'desi_t1'
    return df_t1[['source_id', 'ra', 'dec', 'cluster', 'sample', 'tier',
                   'orbit_confirmed', 'parallax', 'phot_g_mean_mag', 'bp_rp']]


def load_cooling_age_results():
    """加载两个冷却年龄分析结果 — 同时从分析报告文件直接读取"""
    results = {}

    # 方法1: 从汇总CSV读取
    for p in [
        os.path.join(PROJECT_ROOT, '逃逸回溯星团的双白矮星', 'toolbox_results', 'cooling_age_summary.csv'),
        os.path.join(PROJECT_ROOT, '未回溯t1234', 'cooling_age_results', 'cooling_age_summary_t1.csv'),
    ]:
        if os.path.exists(p):
            df = pd.read_csv(p, dtype={'source_id': str})
            for _, row in df.iterrows():
                sid = str(row['source_id']).strip()
                if sid not in results:
                    results[sid] = row.to_dict()

    # 方法2: 从各目录的 cooling_age_analysis.txt 补充
    for base_dir in [
        os.path.join(PROJECT_ROOT, '逃逸回溯星团的双白矮星', 'toolbox_results'),
        os.path.join(PROJECT_ROOT, '未回溯t1234', 'cooling_age_results'),
    ]:
        if not os.path.isdir(base_dir):
            continue
        for dname in os.listdir(base_dir):
            txt = os.path.join(base_dir, dname, 'cooling_age_analysis.txt')
            if not os.path.isfile(txt):
                continue
            # 从目录名提取 source_id
            parts = dname.split('_Gaia_')
            if len(parts) == 2:
                sid = parts[1].strip()
                if sid not in results:
                    # 从文件解析基本信息
                    info = _parse_cooling_report(txt)
                    if info:
                        info['source_id'] = sid
                        results[sid] = info

    return results


def _parse_cooling_report(filepath):
    """从 cooling_age_analysis.txt 解析关键数据"""
    info = {}
    try:
        with open(filepath) as f:
            text = f.read()
        for line in text.split('\n'):
            line = line.strip()
            if line.startswith('状态:'):
                info['status'] = line.split(':', 1)[1].strip()
            elif 'M_WD =' in line and 'M_WD' not in info:
                val = line.split('=')[1].strip().split()[0]
                info['M_WD'] = val
            elif 'T_eff =' in line and 'Teff' not in info:
                val = line.split('=')[1].strip().split()[0]
                info['Teff'] = val
            elif 'log g =' in line and 'logg' not in info:
                val = line.split('=')[1].strip().split()[0]
                info['logg'] = val
            elif 't_cool =' in line and 't_cool' not in info:
                val = line.split('=')[1].strip().split()[0]
                info['t_cool_Gyr'] = val
            elif 't_cool+t_MS =' in line and 't_total_Gyr' not in info:
                val = line.split('=')[1].strip().split()[0]
                info['t_total_Gyr'] = val
            elif 't_cluster   =' in line and 't_cluster_Gyr' not in info:
                val = line.split('=')[1].strip().split()[0]
                info['t_cluster_Gyr'] = val
            elif 'Δt =' in line and 'delta_Gyr' not in info:
                val = line.split('=')[1].strip().split()[0]
                info['delta_Gyr'] = val
            elif line.startswith('判据:') and 'merger_flag' not in info:
                info['merger_flag'] = line.split(':', 1)[1].strip()
            elif 'M_G =' in line and 'M_G' not in info:
                val = line.split('M_G =')[1].strip().split()[0]
                info['M_G'] = val
            elif 'BP-RP =' in line and 'BP_RP' not in info:
                val = line.split('BP-RP =')[1].strip().split()[0]
                info['BP_RP'] = val
            elif 'Plx =' in line and 'Plx' not in info:
                val = line.split('Plx =')[1].strip().split()[0]
                info['Plx'] = val
            elif 'G =' in line and 'Gmag' not in info and 'BP' in line:
                # "G = 18.181  BP = ..."
                val = line.split('G =')[1].strip().split()[0]
                info['Gmag'] = val
    except Exception:
        pass
    return info if info else None


def query_simbad_info(ra, dec):
    """查询 SIMBAD 获取对象信息和引用数"""
    from astro_toolbox import utils
    try:
        result = utils.query_simbad_references(ra, dec, max_refs=50)
        if result is None:
            return None

        info = {
            'simbad_id': result.get('main_id', ''),
            'simbad_otype': result.get('otype', ''),
            'simbad_n_refs': result.get('n_refs', 0),
        }

        # 分析引用：查找星团相关论文
        refs = result.get('references', [])
        cluster_keywords = ['cluster', 'open cluster', 'praesepe', 'coma berenices',
                          'melotte', 'NGC 2632', 'NGC 2682', 'M67', 'white dwarf',
                          'escapee', 'escape', 'runaway', 'ejected',
                          'cooling age', 'merger', 'binary']
        cluster_refs = []
        wd_refs = []

        for ref in refs:
            title = (ref.get('title', '') or '').lower()
            # 星团相关
            for kw in cluster_keywords:
                if kw.lower() in title:
                    cluster_refs.append(ref.get('bibcode', ''))
                    break
            # WD 相关
            wd_kw = ['white dwarf', 'cooling', 'WD ', 'degenerate']
            for kw in wd_kw:
                if kw.lower() in title:
                    wd_refs.append(ref.get('bibcode', ''))
                    break

        info['n_cluster_refs'] = len(set(cluster_refs))
        info['n_wd_refs'] = len(set(wd_refs))

        # 收集所有引用的年份和标题（前5个）
        ref_strs = []
        for ref in refs[:5]:
            year = ref.get('year', '?')
            title = ref.get('title', '?')
            if title and len(title) > 60:
                title = title[:57] + '...'
            ref_strs.append(f"{ref.get('bibcode','')}")

        info['top_refs'] = '; '.join(ref_strs)

        return info
    except Exception as e:
        print(f"    SIMBAD 查询出错: {e}")
        return None


def main():
    print("=" * 70)
    print("整合冷却年龄分析 + SIMBAD 文献查询")
    print("=" * 70)

    # 1. 加载两个样本
    df_escaped = load_escaped_sample()
    df_t1 = load_desi_t1()

    print(f"逃逸样本: {len(df_escaped)} 源")
    print(f"Tier1 样本: {len(df_t1)} 源")

    # 2. 合并去重
    # 优先保留 escaped 样本的信息
    escaped_ids = set(df_escaped['source_id'].astype(str).str.strip())
    t1_ids = set(df_t1['source_id'].astype(str).str.strip())

    rows_all = []
    # 先加入逃逸样本
    for _, row in df_escaped.iterrows():
        r = row.to_dict()
        r['source_id'] = str(r['source_id']).strip()
        # 检查是否也在 Tier1 中
        if r['source_id'] in t1_ids:
            r['sample'] = 'both'
        rows_all.append(r)

    # 加入 Tier1 中不在逃逸样本的
    for _, row in df_t1.iterrows():
        sid = str(row['source_id']).strip()
        if sid not in escaped_ids:
            r = row.to_dict()
            r['source_id'] = sid
            rows_all.append(r)

    df_all = pd.DataFrame(rows_all)
    print(f"合并去重后: {len(df_all)} 源")

    # 3. 加载冷却年龄结果
    cooling = load_cooling_age_results()
    print(f"冷却年龄结果: {len(cooling)} 源")

    # 4. 合并冷却年龄数据
    cool_cols = ['status', 'Gmag', 'BP_RP', 'M_G', 'Plx', 'M_WD', 'Teff', 'logg',
                 't_cool_Gyr', 't_MS_Gyr', 't_total_Gyr', 't_cluster_Gyr', 'delta_Gyr',
                 'merger_flag', 'mass_flag']
    for col in cool_cols:
        df_all[col] = ''

    for i, row in df_all.iterrows():
        sid = str(row['source_id'])
        if sid in cooling:
            c = cooling[sid]
            for col in cool_cols:
                if col in c and pd.notna(c.get(col)):
                    df_all.at[i, col] = c[col]

    # 5. 查询 SIMBAD（先检查已有缓存）
    print("\n开始查询 SIMBAD...")
    simbad_cols = ['simbad_id', 'simbad_otype', 'simbad_n_refs',
                   'n_cluster_refs', 'n_wd_refs', 'top_refs']
    for col in simbad_cols:
        df_all[col] = ''

    # 检查已有的 simbad 缓存文件
    simbad_cache = {}
    for base in [os.path.join(PROJECT_ROOT, '逃逸回溯星团的双白矮星', 'toolbox_results'),
                 os.path.join(PROJECT_ROOT, '未回溯t1234', 'results')]:
        if not os.path.isdir(base):
            continue
        for dname in os.listdir(base):
            ref_csv = os.path.join(base, dname, 'simbad_references.csv')
            ref_txt = os.path.join(base, dname, 'simbad_references.txt')
            parts = dname.split('_Gaia_')
            if len(parts) == 2:
                cached_sid = parts[1].strip()
                if os.path.exists(ref_csv):
                    try:
                        rdf = pd.read_csv(ref_csv)
                        simbad_cache[cached_sid] = {'n_refs': len(rdf), 'refs_df': rdf}
                    except:
                        pass
                if os.path.exists(ref_txt):
                    if cached_sid not in simbad_cache:
                        simbad_cache[cached_sid] = {}
                    try:
                        with open(ref_txt) as f:
                            txt = f.read()
                        for line in txt.split('\n'):
                            if 'SIMBAD ID:' in line:
                                simbad_cache[cached_sid]['simbad_id'] = line.split('SIMBAD ID:')[1].strip()
                            if '对象类型:' in line or 'Type:' in line:
                                simbad_cache[cached_sid]['otype'] = line.split(':')[-1].strip()
                    except:
                        pass

    total = len(df_all)
    for idx_count, (i, row) in enumerate(df_all.iterrows()):
        ra = float(row['ra'])
        dec = float(row['dec'])
        sid = str(row['source_id']).strip()
        cluster = row.get('cluster', '')
        print(f"  [{idx_count+1}/{total}] {cluster} Gaia {sid}  ", end='', flush=True)

        info = query_simbad_info(ra, dec)
        if info:
            for col in simbad_cols:
                if col in info:
                    df_all.at[i, col] = info[col]
            print(f"→ {info['simbad_id']}  type={info['simbad_otype']}  "
                  f"refs={info['simbad_n_refs']}  cluster_refs={info['n_cluster_refs']}  "
                  f"wd_refs={info['n_wd_refs']}")
        else:
            print("→ not found in SIMBAD")
        time.sleep(0.5)

    # 6. 添加综合备注
    df_all['note'] = ''
    for i, row in df_all.iterrows():
        notes = []
        status = str(row.get('status', ''))
        m_wd = row.get('M_WD', '')

        # WD 类型标注
        if m_wd and m_wd != '':
            try:
                m = float(m_wd)
                if m < 0.35:
                    notes.append('ELM_WD(<0.35Msun)')
                elif m < 0.45:
                    notes.append('low_mass_WD')
                elif m > 1.0:
                    notes.append('ultra_massive_WD')
                elif m > 0.8:
                    notes.append('high_mass_WD')
            except:
                pass

        # 并合判据
        if status == 'MERGER_CANDIDATE':
            delta = row.get('delta_Gyr', '')
            try:
                d = float(delta)
                if d > 10:
                    notes.append('strong_merger(delta>10Gyr)')
                elif d > 1:
                    notes.append('merger(delta>1Gyr)')
                else:
                    notes.append('merger')
            except:
                notes.append('merger')
        elif status == 'CONSISTENT':
            notes.append('single_star_ok')
        elif status == 'FAILED':
            notes.append('not_on_WD_track')

        # SIMBAD 被研究过
        n_refs = row.get('simbad_n_refs', 0)
        n_wd = row.get('n_wd_refs', 0)
        n_cl = row.get('n_cluster_refs', 0)
        try:
            if int(n_wd) > 0:
                notes.append(f'WD_studied({n_wd}papers)')
            if int(n_cl) > 0:
                notes.append(f'cluster_studied({n_cl}papers)')
            if int(n_refs) == 0:
                notes.append('no_simbad_refs')
        except:
            pass

        # 轨道回溯
        if row.get('orbit_confirmed') == True or row.get('orbit_confirmed') == 'True':
            notes.append('orbit_confirmed')

        df_all.at[i, 'note'] = '; '.join(notes)

    # 7. 整理列顺序
    col_order = [
        'source_id', 'cluster', 'ra', 'dec', 'sample', 'tier', 'orbit_confirmed',
        # Gaia 测光
        'Gmag', 'BP_RP', 'M_G', 'Plx',
        # WD 参数
        'M_WD', 'Teff', 'logg',
        # 冷却年龄分析
        't_cool_Gyr', 't_MS_Gyr', 't_total_Gyr', 't_cluster_Gyr', 'delta_Gyr',
        'merger_flag', 'mass_flag', 'status',
        # SIMBAD
        'simbad_id', 'simbad_otype', 'simbad_n_refs', 'n_cluster_refs', 'n_wd_refs',
        'top_refs',
        # 备注
        'note',
    ]
    # 只保留存在的列
    col_order = [c for c in col_order if c in df_all.columns]
    df_out = df_all[col_order].copy()

    # 8. 按星团和状态排序
    sort_order = {'MERGER_CANDIDATE': 0, 'MARGINAL_MERGER': 1, 'CONSISTENT': 2, 'FAILED': 3, '': 4}
    df_out['_sort'] = df_out['status'].map(lambda x: sort_order.get(str(x), 4))
    df_out = df_out.sort_values(['cluster', '_sort', 'source_id']).drop(columns='_sort')

    # 9. 保存
    out_path = os.path.join(PROJECT_ROOT, '未回溯t1234', 'comprehensive_cooling_age_simbad.csv')
    df_out.to_csv(out_path, index=False)
    print(f"\n综合 CSV 已保存: {out_path}")
    print(f"总源数: {len(df_out)}")

    # 10. 打印汇总
    print(f"\n{'='*70}")
    print("汇总统计")
    print(f"{'='*70}")

    wd_sources = df_out[df_out['status'].isin(['MERGER_CANDIDATE', 'MARGINAL_MERGER', 'CONSISTENT'])]
    non_wd = df_out[~df_out['status'].isin(['MERGER_CANDIDATE', 'MARGINAL_MERGER', 'CONSISTENT'])]

    print(f"\n白矮星 (在冷却轨迹上): {len(wd_sources)}")
    print(f"  并合候选体: {len(wd_sources[wd_sources['merger_flag']=='MERGER_CANDIDATE'])}")
    print(f"  单星一致: {len(wd_sources[wd_sources['merger_flag']=='CONSISTENT'])}")
    print(f"非WD/失败: {len(non_wd)}")

    print(f"\nSIMBAD 统计:")
    has_simbad = df_out[df_out['simbad_n_refs'].apply(lambda x: str(x) not in ('', '0', 'nan'))]
    print(f"  SIMBAD 有记录: {len(has_simbad)}")
    has_wd_refs = df_out[df_out['n_wd_refs'].apply(lambda x: str(x) not in ('', '0', 'nan'))]
    print(f"  有WD相关论文: {len(has_wd_refs)}")
    has_cl_refs = df_out[df_out['n_cluster_refs'].apply(lambda x: str(x) not in ('', '0', 'nan'))]
    print(f"  有星团相关论文: {len(has_cl_refs)}")

    print(f"\n按星团分布 (仅WD):")
    if len(wd_sources) > 0:
        for cluster, grp in wd_sources.groupby('cluster'):
            n_merger = len(grp[grp['merger_flag'] == 'MERGER_CANDIDATE'])
            n_ok = len(grp[grp['merger_flag'] == 'CONSISTENT'])
            print(f"  {cluster:15s}: {len(grp)} WD  ({n_merger} merger, {n_ok} consistent)")


if __name__ == '__main__':
    main()
