#!/usr/bin/env python3
"""
从coadd文件提取特定TARGETID光谱并可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pandas as pd
import os


def extract_spectrum_from_coadd(coadd_path, targetid):
    """
    从coadd文件中提取特定TARGETID的光谱
    
    DESI coadd文件结构:
    - HDU0: Primary HDU
    - HDU1: FIBERMAP - 光纤映射表
    - HDU2: B_WAVELENGTH - B波段波长
    - HDU3: B_FLUX - B波段流量
    - HDU4: B_IVAR - B波段逆方差
    - HDU5: R_WAVELENGTH - R波段波长
    - HDU6: R_FLUX - R波段流量
    - HDU7: R_IVAR - R波段逆方差
    - HDU8: Z_WAVELENGTH - Z波段波长
    - HDU9: Z_FLUX - Z波段流量
    - HDU10: Z_IVAR - Z波段逆方差
    """
    print(f"正在从coadd文件提取光谱...")
    print(f"  文件: {coadd_path}")
    print(f"  TARGETID: {targetid}")
    
    with fits.open(coadd_path) as hdul:
        print(f"\n  文件结构:")
        for i, hdu in enumerate(hdul):
            print(f"    HDU{i}: {hdu.name}")
        
        # 读取FIBERMAP获取TARGETID索引
        fibermap = hdul['FIBERMAP'].data
        targetids = fibermap['TARGETID']
        
        # 找到目标索引
        idx = np.where(targetids == targetid)[0]
        if len(idx) == 0:
            print(f"\n  错误: TARGETID {targetid} 不在文件中!")
            print(f"  文件中的TARGETID示例: {targetids[:10]}")
            return None
        idx = idx[0]
        
        print(f"\n  在FIBERMAP中找到索引: {idx}")
        
        # 提取FIBERMAP中的其他信息
        source_info = {}
        for col in fibermap.columns.names:
            try:
                source_info[col] = fibermap[col][idx]
            except:
                pass
        
        # 提取光谱数据
        spectrum = {
            'targetid': targetid,
            'index_in_fibermap': idx,
            'source_info': source_info,
            'B': {
                'wavelength': hdul['B_WAVELENGTH'].data,
                'flux': hdul['B_FLUX'].data[idx],
                'ivar': hdul['B_IVAR'].data[idx],
                'error': 1.0 / np.sqrt(hdul['B_IVAR'].data[idx] + 1e-10)
            },
            'R': {
                'wavelength': hdul['R_WAVELENGTH'].data,
                'flux': hdul['R_FLUX'].data[idx],
                'ivar': hdul['R_IVAR'].data[idx],
                'error': 1.0 / np.sqrt(hdul['R_IVAR'].data[idx] + 1e-10)
            },
            'Z': {
                'wavelength': hdul['Z_WAVELENGTH'].data,
                'flux': hdul['Z_FLUX'].data[idx],
                'ivar': hdul['Z_IVAR'].data[idx],
                'error': 1.0 / np.sqrt(hdul['Z_IVAR'].data[idx] + 1e-10)
            }
        }
        
        print(f"\n  光谱提取成功!")
        print(f"  B波段: {len(spectrum['B']['wavelength'])} 像素")
        print(f"  R波段: {len(spectrum['R']['wavelength'])} 像素")
        print(f"  Z波段: {len(spectrum['Z']['wavelength'])} 像素")
        
    return spectrum


def plot_spectrum(spectrum, save_path=None, show=True, title=None):
    """
    绘制光谱图
    """
    targetid = spectrum.get('targetid', 'Unknown')
    
    # 创建图形
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), 
                             gridspec_kw={'height_ratios': [3, 1]})
    
    # ========== 主图：光谱 ==========
    ax1 = axes[0]
    
    # 绘制三个波段
    ax1.plot(spectrum['B']['wavelength'], spectrum['B']['flux'], 
             'b-', label='B band (3600-5800 Å)', linewidth=0.8, alpha=0.8)
    ax1.plot(spectrum['R']['wavelength'], spectrum['R']['flux'], 
             'g-', label='R band (5800-7600 Å)', linewidth=0.8, alpha=0.8)
    ax1.plot(spectrum['Z']['wavelength'], spectrum['Z']['flux'], 
             'r-', label='Z band (7600-9800 Å)', linewidth=0.8, alpha=0.8)
    
    # 添加误差阴影
    ax1.fill_between(spectrum['B']['wavelength'], 
                     spectrum['B']['flux'] - spectrum['B']['error'],
                     spectrum['B']['flux'] + spectrum['B']['error'],
                     color='blue', alpha=0.15)
    ax1.fill_between(spectrum['R']['wavelength'], 
                     spectrum['R']['flux'] - spectrum['R']['error'],
                     spectrum['R']['flux'] + spectrum['R']['error'],
                     color='green', alpha=0.15)
    ax1.fill_between(spectrum['Z']['wavelength'], 
                     spectrum['Z']['flux'] - spectrum['Z']['error'],
                     spectrum['Z']['flux'] + spectrum['Z']['error'],
                     color='red', alpha=0.15)
    
    ax1.set_xlabel('Wavelength (Å)', fontsize=12)
    ax1.set_ylabel('Flux (10⁻¹⁷ erg/s/cm²/Å)', fontsize=12)
    
    if title:
        ax1.set_title(title, fontsize=14)
    else:
        ax1.set_title(f'DESI Spectrum - TARGETID: {targetid}', fontsize=14)
    
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 标注一些重要的谱线
    spectral_lines = {
        4861: 'Hβ',
        6563: 'Hα',
        5890: 'Na D',
        5175: 'Mg b',
        3933: 'Ca II K',
        3968: 'Ca II H',
    }
    
    for wave, name in spectral_lines.items():
        if 3600 <= wave <= 9800:
            ax1.axvline(wave, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
            ax1.text(wave, ax1.get_ylim()[1]*0.95, name, 
                    rotation=90, fontsize=8, ha='right', va='top')
    
    # ========== 子图：信噪比 ==========
    ax2 = axes[1]
    
    # 计算每个波段的S/N
    sn_b = spectrum['B']['flux'] / spectrum['B']['error']
    sn_r = spectrum['R']['flux'] / spectrum['R']['error']
    sn_z = spectrum['Z']['flux'] / spectrum['Z']['error']
    
    ax2.plot(spectrum['B']['wavelength'], sn_b, 'b-', linewidth=0.5, alpha=0.7)
    ax2.plot(spectrum['R']['wavelength'], sn_r, 'g-', linewidth=0.5, alpha=0.7)
    ax2.plot(spectrum['Z']['wavelength'], sn_z, 'r-', linewidth=0.5, alpha=0.7)
    
    ax2.set_xlabel('Wavelength (Å)', fontsize=12)
    ax2.set_ylabel('S/N', fontsize=12)
    ax2.set_title('Signal-to-Noise Ratio', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    # 添加统计信息
    median_sn_b = np.median(sn_b)
    median_sn_r = np.median(sn_r)
    median_sn_z = np.median(sn_z)
    
    info_text = f"Median S/N: B={median_sn_b:.1f}, R={median_sn_r:.1f}, Z={median_sn_z:.1f}"
    ax2.text(0.02, 0.95, info_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n光谱图已保存: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def save_spectrum_to_fits(spectrum, output_path):
    """
    将提取的光谱保存为单独的FITS文件
    """
    from astropy.io import fits
    
    hdul = fits.HDUList()
    
    # Primary HDU
    primary_header = fits.Header()
    primary_header['TARGETID'] = spectrum['targetid']
    primary_header['COMMENT'] = 'DESI DR1 Spectrum extracted from coadd'
    
    # 添加源信息
    for key, value in spectrum.get('source_info', {}).items():
        if isinstance(value, (int, float, str)) and len(str(key)) <= 8:
            try:
                primary_header[key[:8].upper()] = value
            except:
                pass
    
    primary_hdu = fits.PrimaryHDU(header=primary_header)
    hdul.append(primary_hdu)
    
    # B波段
    b_data = np.array([spectrum['B']['wavelength'], 
                       spectrum['B']['flux'], 
                       spectrum['B']['error']])
    b_hdu = fits.ImageHDU(data=b_data, name='B_BAND')
    hdul.append(b_hdu)
    
    # R波段
    r_data = np.array([spectrum['R']['wavelength'], 
                       spectrum['R']['flux'], 
                       spectrum['R']['error']])
    r_hdu = fits.ImageHDU(data=r_data, name='R_BAND')
    hdul.append(r_hdu)
    
    # Z波段
    z_data = np.array([spectrum['Z']['wavelength'], 
                       spectrum['Z']['flux'], 
                       spectrum['Z']['error']])
    z_hdu = fits.ImageHDU(data=z_data, name='Z_BAND')
    hdul.append(z_hdu)
    
    hdul.writeto(output_path, overwrite=True)
    print(f"光谱已保存: {output_path}")


def main():
    # 配置参数
    COADD_PATH = "./coadd_cache/coadd-main-bright-4298.fits"
    TARGETID = 39628069090099224
    
    print("="*70)
    print("DESI 光谱提取和可视化")
    print("="*70)
    
    # 检查文件是否存在
    if not os.path.exists(COADD_PATH):
        print(f"错误: 找不到文件 {COADD_PATH}")
        print("请确认coadd文件已下载到正确位置")
        return
    
    # 提取光谱
    spectrum = extract_spectrum_from_coadd(COADD_PATH, TARGETID)
    
    if spectrum is None:
        return
    
    # 打印源信息
    print("\n" + "="*70)
    print("源信息")
    print("="*70)
    
    info = spectrum.get('source_info', {})
    important_cols = ['TARGETID', 'RA', 'DEC', 'FIBER', 'LOCATION', 
                      'FIBERSTATUS', 'EBV', 'FLUX_G', 'FLUX_R', 'FLUX_Z']
    
    for col in important_cols:
        if col in info:
            print(f"  {col:15s}: {info[col]}")
    
    # 打印其他信息
    print("\n  其他FIBERMAP列:")
    for col, value in info.items():
        if col not in important_cols:
            try:
                if isinstance(value, (int, float)):
                    print(f"    {col:15s}: {value}")
            except:
                pass
    
    # 绘制光谱
    print("\n" + "="*70)
    print("绘制光谱...")
    print("="*70)
    
    output_filename = f"spectrum_TARGETID_{TARGETID}.png"
    plot_spectrum(spectrum, save_path=output_filename, show=True)
    
    # 保存为单独的FITS文件（可选）
    save_choice = input("\n是否保存为单独的FITS文件? (y/n): ").strip().lower()
    if save_choice == 'y':
        fits_filename = f"spectrum_TARGETID_{TARGETID}.fits"
        save_spectrum_to_fits(spectrum, fits_filename)
    
    print("\n" + "="*70)
    print("完成!")
    print("="*70)


if __name__ == "__main__":
    main()
