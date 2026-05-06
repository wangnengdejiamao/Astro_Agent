#!/usr/bin/env python3
"""
读取本地DESI DR1光谱文件

本地光谱文件路径: /data1/DESI_DR1/coadd/spectra/
文件名格式: {TARGETID}.fits

数据格式:
- HDU1: Table HDU，总星表第一个HDU中这个源的参数 (RVTAB)
- HDU2: Table HDU，总星表第二个HDU中这个源的参数 (SPTAB)
- HDU3: Table HDU，原文件第一个HDU中这个源的参数
- HDU4: Table HDU，原文件第二个HDU中这个源的参数
- HDU5: Image HDU，B波段光谱，包含三行：波长、流量、误差
- HDU6: Image HDU，R波段光谱，包含三行：波长、流量、误差
- HDU7: Image HDU，Z波段光谱，包含三行：波长、流量、误差
"""

import pandas as pd
import numpy as np
from astropy.io import fits
import os
from tqdm import tqdm
import shutil

# 本地光谱路径
LOCAL_SPECTRA_PATH = "/data1/DESI_DR1/coadd/spectra"
DUPLICATE_SPECTRA_PATH = "/data1/DESI_DR1/coadd/duplicate"
RAW_COADD_PATH = "/data1/DESI_DR1/coadd/raw"


def read_spectrum(fits_path):
    """
    读取处理后的DESI光谱文件
    
    Parameters:
    -----------
    fits_path : str
        光谱文件路径
    
    Returns:
    --------
    data : dict
        包含所有HDU数据的字典
    """
    data = {}
    
    with fits.open(fits_path) as hdul:
        # HDU1: RVTAB 参数
        if len(hdul) > 1:
            data['rvtab'] = hdul[1].data
        
        # HDU2: SPTAB 参数
        if len(hdul) > 2:
            data['sptab'] = hdul[2].data
        
        # HDU3: 原文件第一个HDU参数
        if len(hdul) > 3:
            data['orig_hdu1'] = hdul[3].data
        
        # HDU4: 原文件第二个HDU参数
        if len(hdul) > 4:
            data['orig_hdu2'] = hdul[4].data
        
        # HDU5: B波段光谱
        if len(hdul) > 5:
            data['B'] = {
                'wavelength': hdul[5].data[0],
                'flux': hdul[5].data[1],
                'error': hdul[5].data[2]
            }
        
        # HDU6: R波段光谱
        if len(hdul) > 6:
            data['R'] = {
                'wavelength': hdul[6].data[0],
                'flux': hdul[6].data[1],
                'error': hdul[6].data[2]
            }
        
        # HDU7: Z波段光谱
        if len(hdul) > 7:
            data['Z'] = {
                'wavelength': hdul[7].data[0],
                'flux': hdul[7].data[1],
                'error': hdul[7].data[2]
            }
    
    return data


def find_local_spectrum(targetid, check_duplicate=True):
    """
    在本地服务器上查找光谱文件
    
    Parameters:
    -----------
    targetid : int
        DESI TARGET ID
    check_duplicate : bool
        是否检查重复观测目录
    
    Returns:
    --------
    paths : list
        找到的文件路径列表
    """
    paths = []
    
    # 检查主目录
    main_path = os.path.join(LOCAL_SPECTRA_PATH, f"{targetid}.fits")
    if os.path.exists(main_path):
        paths.append(main_path)
    
    # 检查重复观测目录
    if check_duplicate:
        dup_path = os.path.join(DUPLICATE_SPECTRA_PATH, f"{targetid}.fits")
        if os.path.exists(dup_path):
            paths.append(dup_path)
        
        # 检查多次观测 (targetid_2.fits, targetid_3.fits, ...)
        i = 2
        while True:
            multi_path = os.path.join(DUPLICATE_SPECTRA_PATH, f"{targetid}_{i}.fits")
            if os.path.exists(multi_path):
                paths.append(multi_path)
                i += 1
            else:
                break
    
    return paths


def copy_spectra_from_local(csv_path, output_dir='./local_spectra', max_sources=None):
    """
    从本地服务器复制光谱文件到输出目录
    
    Parameters:
    -----------
    csv_path : str
        mws_gaia.csv文件路径
    output_dir : str
        输出目录
    max_sources : int
        最大处理源数
    """
    if not os.path.exists(LOCAL_SPECTRA_PATH):
        print(f"错误: 无法访问本地光谱路径: {LOCAL_SPECTRA_PATH}")
        print("请确保你正在可以访问 /data1/DESI_DR1 的机器上运行此脚本")
        return
    
    print(f"读取星表: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"共 {len(df)} 个源")
    
    if max_sources:
        df = df.head(max_sources)
        print(f"将处理前 {max_sources} 个源")
    
    os.makedirs(output_dir, exist_ok=True)
    
    found_count = 0
    not_found = []
    
    for i, row in tqdm(df.iterrows(), total=len(df), desc="复制光谱"):
        targetid = int(row['TARGETID'])
        
        # 查找本地文件
        paths = find_local_spectrum(targetid)
        
        if paths:
            found_count += 1
            # 复制文件
            for j, src_path in enumerate(paths):
                if j == 0:
                    dst_path = os.path.join(output_dir, f"{targetid}.fits")
                else:
                    dst_path = os.path.join(output_dir, f"{targetid}_{j}.fits")
                
                if not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)
        else:
            not_found.append(targetid)
    
    print(f"\n处理完成!")
    print(f"找到并复制: {found_count}/{len(df)}")
    print(f"未找到: {len(not_found)}")
    
    if not_found:
        # 保存未找到的TARGETID列表
        with open(f"{output_dir}/not_found.txt", 'w') as f:
            for tid in not_found:
                f.write(f"{tid}\n")
        print(f"未找到的TARGETID列表已保存至: {output_dir}/not_found.txt")


def extract_spectrum_data(csv_path, output_dir='./spectrum_data', max_sources=None):
    """
    从本地光谱文件提取数据并保存为易读格式
    
    Parameters:
    -----------
    csv_path : str
        mws_gaia.csv文件路径
    output_dir : str
        输出目录
    max_sources : int
        最大处理源数
    """
    if not os.path.exists(LOCAL_SPECTRA_PATH):
        print(f"错误: 无法访问本地光谱路径: {LOCAL_SPECTRA_PATH}")
        return
    
    print(f"读取星表: {csv_path}")
    df = pd.read_csv(csv_path)
    
    if max_sources:
        df = df.head(max_sources)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i, row in tqdm(df.iterrows(), total=len(df), desc="提取光谱数据"):
        targetid = int(row['TARGETID'])
        
        # 查找本地文件
        paths = find_local_spectrum(targetid)
        
        if not paths:
            continue
        
        # 读取光谱
        spec_path = paths[0]  # 使用第一个(主观测)
        try:
            data = read_spectrum(spec_path)
            
            # 保存为numpy格式
            np.savez(
                f"{output_dir}/{targetid}_spectrum.npz",
                targetid=targetid,
                B_wavelength=data['B']['wavelength'],
                B_flux=data['B']['flux'],
                B_error=data['B']['error'],
                R_wavelength=data['R']['wavelength'],
                R_flux=data['R']['flux'],
                R_error=data['R']['error'],
                Z_wavelength=data['Z']['wavelength'],
                Z_flux=data['Z']['flux'],
                Z_error=data['Z']['error']
 )
        except Exception as e:
            print(f"处理 {targetid} 时出错: {e}")
            continue
    
    print(f"\n光谱数据已保存至: {output_dir}/")


def list_available_spectra(csv_path, sample_size=100):
    """
    检查mws_gaia.csv中有多少源在本地服务器上有光谱
    
    Parameters:
    -----------
    csv_path : str
        mws_gaia.csv文件路径
    sample_size : int
        检查的样本数(0表示检查全部)
    """
    if not os.path.exists(LOCAL_SPECTRA_PATH):
        print(f"错误: 无法访问本地光谱路径: {LOCAL_SPECTRA_PATH}")
        return
    
    print(f"读取星表: {csv_path}")
    df = pd.read_csv(csv_path)
    
    if sample_size > 0:
        df = df.head(sample_size)
        print(f"检查前 {sample_size} 个源...")
    else:
        print(f"检查全部 {len(df)} 个源...")
    
    found = 0
    has_duplicate = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        targetid = int(row['TARGETID'])
        paths = find_local_spectrum(targetid, check_duplicate=True)
        
        if paths:
            found += 1
            if len(paths) > 1:
                has_duplicate += 1
    
    print(f"\n结果:")
    print(f"  总检查数: {len(df)}")
    print(f"  找到光谱: {found} ({100*found/len(df):.1f}%)")
    print(f"  有重复观测: {has_duplicate}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='读取本地DESI DR1光谱文件')
    parser.add_argument('--csv', default='mws_gaia.csv', help='输入CSV文件路径')
    parser.add_argument('--output', default='./local_spectra', help='输出目录')
    parser.add_argument('--max-sources', type=int, default=None, help='最大处理源数')
    parser.add_argument('--check', action='store_true', help='只检查可用光谱数量')
    parser.add_argument('--check-sample', type=int, default=1000, help='检查的样本数')
    parser.add_argument('--extract', action='store_true', help='提取光谱数据为numpy格式')
    
    args = parser.parse_args()
    
    if args.check:
        list_available_spectra(args.csv, args.check_sample)
    elif args.extract:
        extract_spectrum_data(args.csv, args.output, args.max_sources)
    else:
        copy_spectra_from_local(args.csv, args.output, args.max_sources)
