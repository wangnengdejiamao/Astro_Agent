#!/usr/bin/env python3
"""
DESI光谱获取快速开始脚本

这个脚本演示如何：
1. 从mws_gaia.csv读取源信息
2. 下载光谱数据
3. 读取和可视化光谱

建议先运行此脚本测试环境是否配置正确。
"""

import pandas as pd
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import os
import sys

# 检查依赖
try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("请安装依赖: pip install requests tqdm")
    sys.exit(1)

# 导入下载模块
from download_desi_spectra import (
    get_healpix_url, 
    download_file, 
    extract_spectrum_from_coadd,
    save_spectrum_to_fits,
    read_extracted_spectrum
)


def main():
    print("=" * 60)
    print("DESI DR1 光谱获取 - 快速开始")
    print("=" * 60)
    
    # 1. 读取星表
    print("\n[1/5] 读取星表...")
    csv_path = 'mws_gaia.csv'
    if not os.path.exists(csv_path):
        print(f"错误: 找不到文件 {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    print(f"  星表共包含 {len(df):,} 个源")
    print(f"  列名: {', '.join(df.columns[:5])}...")
    
    # 2. 选择前10个源作为示例
    print("\n[2/5] 选择示例源 (前10个)...")
    sample_df = df.head(10)
    print("  示例源信息:")
    for i, row in sample_df.iterrows():
        print(f"    {i+1}. TARGETID: {row['TARGETID']}, "
              f"RA: {row['RA']:.4f}, DEC: {row['DEC']:.4f}, "
              f"HEALPIX: {row['HEALPIX']}")
    
    # 3. 创建输出目录
    print("\n[3/5] 准备下载...")
    output_dir = './sample_spectra'
    cache_dir = f'{output_dir}/coadd_cache'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    print(f"  输出目录: {output_dir}/")
    
    # 4. 下载并处理光谱
    print("\n[4/5] 下载并处理光谱...")
    downloaded_coadds = {}
    success_count = 0
    
    for i, row in sample_df.iterrows():
        targetid = int(row['TARGETID'])
        survey = row['SURVEY']
        program = row['PROGRAM']
        healpix = int(row['HEALPIX'])
        
        output_path = f"{output_dir}/{targetid}.fits"
        
        # 获取coadd文件URL
        coadd_url = get_healpix_url(survey, program, healpix)
        coadd_cache_path = f"{cache_dir}/coadd-{survey}-{program}-{healpix}.fits"
        
        print(f"\n  处理源 {i+1}/10: TARGETID={targetid}")
        print(f"    URL: {coadd_url}")
        
        # 下载coadd文件(如果尚未下载)
        if healpix not in downloaded_coadds:
            if not os.path.exists(coadd_cache_path):
                print(f"    下载coadd文件...")
                success = download_file(coadd_url, coadd_cache_path)
                if not success:
                    print(f"    下载失败，跳过")
                    continue
            downloaded_coadds[healpix] = coadd_cache_path
        
        coadd_path = downloaded_coadds[healpix]
        
        # 从coadd提取光谱
        print(f"    提取光谱...")
        spectrum = extract_spectrum_from_coadd(coadd_path, targetid)
        if spectrum is None:
            print(f"    提取失败，跳过")
            continue
        
        # 保存光谱
        source_info = row.to_dict()
        if save_spectrum_to_fits(spectrum, output_path, source_info):
            print(f"    成功保存: {output_path}")
            success_count += 1
        else:
            print(f"    保存失败")
    
    print(f"\n  下载完成: {success_count}/10 个源")
    
    if success_count == 0:
        print("  没有成功下载的光谱，退出")
        return
    
    # 5. 可视化第一个成功下载的光谱
    print("\n[5/5] 可视化光谱...")
    fits_files = [f for f in os.listdir(output_dir) if f.endswith('.fits')]
    
    if fits_files:
        first_file = os.path.join(output_dir, fits_files[0])
        print(f"  读取文件: {first_file}")
        
        # 读取光谱
        spectrum = read_extracted_spectrum(first_file)
        targetid = spectrum['targetid']
        
        # 绘制
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        bands = ['B', 'R', 'Z']
        colors = ['blue', 'green', 'red']
        
        for ax, band, color in zip(axes, bands, colors):
            wave = spectrum[band]['wavelength']
            flux = spectrum[band]['flux']
            error = spectrum[band]['error']
            
            ax.plot(wave, flux, color=color, linewidth=0.8, label=f'{band} band')
            ax.fill_between(wave, flux - error, flux + error, 
                          color=color, alpha=0.2)
            
            ax.set_ylabel('Flux')
            ax.set_xlabel('Wavelength (Å)')
            ax.set_title(f'{band} Band: {wave.min():.1f} - {wave.max():.1f} Å')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(f'DESI Spectrum - TARGETID: {targetid}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plot_path = f'{output_dir}/example_spectrum.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"  光谱图已保存: {plot_path}")
        plt.close()
        
        # 绘制合并光谱
        fig, ax = plt.subplots(figsize=(14, 6))
        for band, color in zip(bands, colors):
            wave = spectrum[band]['wavelength']
            flux = spectrum[band]['flux']
            ax.plot(wave, flux, color=color, linewidth=0.8, label=f'{band} band')
        
        ax.set_xlabel('Wavelength (Å)')
        ax.set_ylabel('Flux')
        ax.set_title(f'DESI Combined Spectrum - TARGETID: {targetid}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plot_path2 = f'{output_dir}/example_combined.png'
        plt.savefig(plot_path2, dpi=150, bbox_inches='tight')
        print(f"  合并光谱图已保存: {plot_path2}")
        plt.close()
    
    # 总结
    print("\n" + "=" * 60)
    print("快速开始完成!")
    print("=" * 60)
    print(f"\n输出文件:")
    print(f"  光谱目录: {output_dir}/")
    for f in os.listdir(output_dir):
        if f.endswith('.fits') or f.endswith('.png'):
            print(f"    - {f}")
    print(f"\n下一步:")
    print(f"  1. 查看生成的光谱图: {output_dir}/example_*.png")
    print(f"  2. 使用 read_extracted_spectrum() 读取 .fits 文件")
    print(f"  3. 运行完整下载: python download_desi_spectra.py --output ./all_spectra")
    print(f"\n参考文档: README_Spectra.md")


if __name__ == "__main__":
    main()
