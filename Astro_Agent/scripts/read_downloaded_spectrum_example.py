#!/usr/bin/env python3
"""
读取DESI光谱文件的完整示例
包含：读取单个源文件、从coadd提取、可视化、简单分析
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os


def read_spectrum(fits_path):
    """
    读取处理后的单个源光谱文件
    
    Returns:
    --------
    spectrum : dict
        {
            'targetid': int,
            'header': dict,
            'B': {'wavelength': array, 'flux': array, 'error': array},
            'R': {'wavelength': array, 'flux': array, 'error': array},
            'Z': {'wavelength': array, 'flux': array, 'error': array}
        }
    """
    with fits.open(fits_path) as hdul:
        spectrum = {
            'targetid': hdul[0].header.get('TARGETID'),
            'header': dict(hdul[0].header),
            'B': {
                'wavelength': hdul['B_BAND'].data[0],
                'flux': hdul['B_BAND'].data[1],
                'error': hdul['B_BAND'].data[2]
            },
            'R': {
                'wavelength': hdul['R_BAND'].data[0],
                'flux': hdul['R_BAND'].data[1],
                'error': hdul['R_BAND'].data[2]
            },
            'Z': {
                'wavelength': hdul['Z_BAND'].data[0],
                'flux': hdul['Z_BAND'].data[1],
                'error': hdul['Z_BAND'].data[2]
            }
        }
    return spectrum


def extract_from_coadd(coadd_path, targetid):
    """
    直接从coadd文件中提取特定TARGETID的光谱
    （无需重新下载，使用缓存的coadd文件）
    """
    with fits.open(coadd_path) as hdul:
        # 读取FIBERMAP
        fibermap = hdul['FIBERMAP'].data
        targetids = fibermap['TARGETID']
        
        # 找到目标索引
        idx = np.where(targetids == targetid)[0]
        if len(idx) == 0:
            print(f"TARGETID {targetid} 不在文件中")
            return None
        idx = idx[0]
        
        # 提取光谱
        spectrum = {
            'targetid': targetid,
            'header': {'TARGETID': targetid},
            'B': {
                'wavelength': hdul['B_WAVELENGTH'].data,
                'flux': hdul['B_FLUX'].data[idx],
                'error': 1.0 / np.sqrt(hdul['B_IVAR'].data[idx] + 1e-10)
            },
            'R': {
                'wavelength': hdul['R_WAVELENGTH'].data,
                'flux': hdul['R_FLUX'].data[idx],
                'error': 1.0 / np.sqrt(hdul['R_IVAR'].data[idx] + 1e-10)
            },
            'Z': {
                'wavelength': hdul['Z_WAVELENGTH'].data,
                'flux': hdul['Z_FLUX'].data[idx],
                'error': 1.0 / np.sqrt(hdul['Z_IVAR'].data[idx] + 1e-10)
            }
        }
    return spectrum


def plot_spectrum(spectrum, save_path=None, show=True):
    """
    可视化光谱
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), 
                             gridspec_kw={'height_ratios': [3, 1]})
    
    # 主图：光谱
    ax1 = axes[0]
    
    # 绘制三个波段
    ax1.plot(spectrum['B']['wavelength'], spectrum['B']['flux'], 
             'b-', label='B band', linewidth=0.8, alpha=0.8)
    ax1.plot(spectrum['R']['wavelength'], spectrum['R']['flux'], 
             'g-', label='R band', linewidth=0.8, alpha=0.8)
    ax1.plot(spectrum['Z']['wavelength'], spectrum['Z']['flux'], 
             'r-', label='Z band', linewidth=0.8, alpha=0.8)
    
    # 添加误差阴影
    ax1.fill_between(spectrum['B']['wavelength'], 
                     spectrum['B']['flux'] - spectrum['B']['error'],
                     spectrum['B']['flux'] + spectrum['B']['error'],
                     color='blue', alpha=0.2)
    ax1.fill_between(spectrum['R']['wavelength'], 
                     spectrum['R']['flux'] - spectrum['R']['error'],
                     spectrum['R']['flux'] + spectrum['R']['error'],
                     color='green', alpha=0.2)
    ax1.fill_between(spectrum['Z']['wavelength'], 
                     spectrum['Z']['flux'] - spectrum['Z']['error'],
                     spectrum['Z']['flux'] + spectrum['Z']['error'],
                     color='red', alpha=0.2)
    
    ax1.set_xlabel('Wavelength (Å)')
    ax1.set_ylabel('Flux')
    targetid = spectrum.get('targetid', 'Unknown')
    ax1.set_title(f'DESI Spectrum - TARGETID: {targetid}')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 子图：S/N 或 合并光谱
    ax2 = axes[1]
    
    # 计算每个波段的平均S/N
    sn_b = np.median(spectrum['B']['flux'] / (spectrum['B']['error'] + 1e-10))
    sn_r = np.median(spectrum['R']['flux'] / (spectrum['R']['error'] + 1e-10))
    sn_z = np.median(spectrum['Z']['flux'] / (spectrum['Z']['error'] + 1e-10))
    
    bands = ['B', 'R', 'Z']
    sns = [sn_b, sn_r, sn_z]
    colors = ['blue', 'green', 'red']
    
    ax2.bar(bands, sns, color=colors, alpha=0.6)
    ax2.set_ylabel('Median S/N')
    ax2.set_title('Signal-to-Noise Ratio by Band')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 在每个柱子上标注数值
    for i, (band, sn) in enumerate(zip(bands, sns)):
        ax2.text(i, sn, f'{sn:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def get_spectrum_info(spectrum):
    """
    获取光谱的基本信息
    """
    info = {
        'TARGETID': spectrum.get('targetid'),
        'Header': spectrum.get('header', {})
    }
    
    for band in ['B', 'R', 'Z']:
        wave = spectrum[band]['wavelength']
        flux = spectrum[band]['flux']
        error = spectrum[band]['error']
        
        # 计算统计信息
        sn = flux / (error + 1e-10)
        
        info[f'{band}_band'] = {
            'n_pixels': len(wave),
            'wave_range': (wave.min(), wave.max()),
            'wave_coverage': wave.max() - wave.min(),
            'median_flux': np.median(flux),
            'median_error': np.median(error),
            'median_sn': np.median(sn),
            'max_sn': np.max(sn),
        }
    
    return info


def print_spectrum_info(spectrum):
    """
    打印光谱信息
    """
    print("\n" + "="*60)
    print("光谱信息")
    print("="*60)
    
    targetid = spectrum.get('targetid', 'Unknown')
    header = spectrum.get('header', {})
    
    print(f"TARGETID: {targetid}")
    if 'RA' in header:
        print(f"RA: {header['RA']:.6f}, DEC: {header['DEC']:.6f}")
    if 'SURVEY' in header:
        print(f"Survey: {header['SURVEY']}, Program: {header.get('PROGRAM', 'N/A')}")
    if 'HEALPIX' in header:
        print(f"HEALPIX: {header['HEALPIX']}")
    
    print("\n各波段信息:")
    print("-"*60)
    
    for band in ['B', 'R', 'Z']:
        wave = spectrum[band]['wavelength']
        flux = spectrum[band]['flux']
        error = spectrum[band]['error']
        sn = flux / (error + 1e-10)
        
        print(f"\n{band} Band:")
        print(f"  波长范围: {wave.min():.1f} - {wave.max():.1f} Å")
        print(f"  像素数: {len(wave)}")
        print(f"  中值流量: {np.median(flux):.3e}")
        print(f"  中值误差: {np.median(error):.3e}")
        print(f"  中值S/N: {np.median(sn):.2f}")
    
    print("="*60)


def merge_bands(spectrum):
    """
    合并三个波段的光谱
    
    Returns:
    --------
    wave, flux, error : arrays
        合并后的波长、流量、误差数组
    """
    wave = np.concatenate([
        spectrum['B']['wavelength'],
        spectrum['R']['wavelength'],
        spectrum['Z']['wavelength']
    ])
    flux = np.concatenate([
        spectrum['B']['flux'],
        spectrum['R']['flux'],
        spectrum['Z']['flux']
    ])
    error = np.concatenate([
        spectrum['B']['error'],
        spectrum['R']['error'],
        spectrum['Z']['error']
    ])
    
    # 按波长排序
    sort_idx = np.argsort(wave)
    
    return wave[sort_idx], flux[sort_idx], error[sort_idx]


def save_to_ascii(spectrum, output_path):
    """
    将光谱保存为ASCII文本格式（便于其他软件使用）
    """
    wave, flux, error = merge_bands(spectrum)
    
    with open(output_path, 'w') as f:
        f.write(f"# DESI Spectrum - TARGETID: {spectrum.get('targetid', 'Unknown')}\n")
        f.write("# Wavelength(Angstrom) Flux Error\n")
        for w, fl, err in zip(wave, flux, error):
            f.write(f"{w:.4f} {fl:.6e} {err:.6e}\n")
    
    print(f"ASCII光谱已保存: {output_path}")


# ==================== 使用示例 ====================

if __name__ == "__main__":
    import sys
    
    # 示例1: 读取已下载的单个源光谱文件
    print("示例1: 读取单个源光谱文件")
    print("-"*60)
    
    # 假设文件已经下载好了
    spectrum_file = "39628069090099224.fits"
    
    if os.path.exists(spectrum_file):
        print(f"读取文件: {spectrum_file}")
        spec = read_spectrum(spectrum_file)
        
        # 打印光谱信息
        print_spectrum_info(spec)
        
        # 绘制光谱
        plot_spectrum(spec, save_path="spectrum_plot.png")
        
        # 合并所有波段
        wave_all, flux_all, error_all = merge_bands(spec)
        print(f"\n合并后光谱: {len(wave_all)} 个像素")
        print(f"波长范围: {wave_all.min():.1f} - {wave_all.max():.1f} Å")
        
        # 保存为ASCII格式
        save_to_ascii(spec, "spectrum.txt")
        
    else:
        print(f"文件不存在: {spectrum_file}")
        print("请等待下载完成，或从coadd缓存中提取")
    
    
    # 示例2: 直接从coadd缓存文件提取（无需等待单个文件下载）
    print("\n\n示例2: 从coadd缓存文件提取")
    print("-"*60)
    
    coadd_file = "./coadd_cache/coadd-main-bright-4298.fits"
    targetid = 39628069090099224
    
    if os.path.exists(coadd_file):
        print(f"从coadd文件提取: {coadd_file}")
        print(f"TARGETID: {targetid}")
        
        spec_from_coadd = extract_from_coadd(coadd_file, targetid)
        
        if spec_from_coadd:
            print_spectrum_info(spec_from_coadd)
            plot_spectrum(spec_from_coadd, save_path="spectrum_from_coadd.png")
            
            # 也可以保存为新的FITS文件
            # save_spectrum_to_fits(spec_from_coadd, "extracted.fits")
    else:
        print(f"coadd文件不存在: {coadd_file}")
        print("请等待下载完成")
    
    
    # 示例3: 从coadd文件中提取多个源
    print("\n\n示例3: 从coadd文件批量提取多个源")
    print("-"*60)
    
    if os.path.exists(coadd_file):
        # 先查看coadd文件中有哪些源
        with fits.open(coadd_file) as hdul:
            fibermap = hdul['FIBERMAP'].data
            targetids_in_file = fibermap['TARGETID']
            print(f"该coadd文件包含 {len(targetids_in_file)} 个源")
            print(f"前10个TARGETID: {targetids_in_file[:10]}")
            
            # 可以批量提取
            # for tid in targetids_in_file[:5]:
            #     spec = extract_from_coadd(coadd_file, tid)
            #     # 处理光谱...
