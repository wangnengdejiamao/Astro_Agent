#!/usr/bin/env python3
"""
可视化DESI光谱

可以绘制:
1. 单个光谱的B/R/Z三个波段
2. 多个光谱的对比
3. 光谱参数信息
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pandas as pd
import os


def read_spectrum(fits_path):
    """读取光谱文件"""
    data = {}
    
    with fits.open(fits_path) as hdul:
        # 尝试读取Image HDU格式 (处理后光谱)
        if len(hdul) >= 7:
            try:
                data['B'] = {
                    'wavelength': hdul[5].data[0] if len(hdul[5].data.shape) > 1 else hdul['B_BAND'].data[0],
                    'flux': hdul[5].data[1] if len(hdul[5].data.shape) > 1 else hdul['B_BAND'].data[1],
                    'error': hdul[5].data[2] if len(hdul[5].data.shape) > 1 else hdul['B_BAND'].data[2]
                }
                data['R'] = {
                    'wavelength': hdul[6].data[0] if len(hdul[6].data.shape) > 1 else hdul['R_BAND'].data[0],
                    'flux': hdul[6].data[1] if len(hdul[6].data.shape) > 1 else hdul['R_BAND'].data[1],
                    'error': hdul[6].data[1] if len(hdul[6].data.shape) > 1 else hdul['R_BAND'].data[2]
                }
                data['Z'] = {
                    'wavelength': hdul[7].data[0] if len(hdul[7].data.shape) > 1 else hdul['Z_BAND'].data[0],
                    'flux': hdul[7].data[1] if len(hdul[7].data.shape) > 1 else hdul['Z_BAND'].data[1],
                    'error': hdul[7].data[1] if len(hdul[7].data.shape) > 1 else hdul['Z_BAND'].data[2]
                }
                data['targetid'] = hdul[0].header.get('TARGETID', 'Unknown')
                return data
            except:
                pass
        
        # 尝试读取Image HDU格式 (我们的提取格式)
        try:
            data['B'] = {
                'wavelength': hdul['B_BAND'].data[0],
                'flux': hdul['B_BAND'].data[1],
                'error': hdul['B_BAND'].data[2]
            }
            data['R'] = {
                'wavelength': hdul['R_BAND'].data[0],
                'flux': hdul['R_BAND'].data[1],
                'error': hdul['R_BAND'].data[2]
            }
            data['Z'] = {
                'wavelength': hdul['Z_BAND'].data[0],
                'flux': hdul['Z_BAND'].data[1],
                'error': hdul['Z_BAND'].data[2]
            }
            data['targetid'] = hdul[0].header.get('TARGETID', 'Unknown')
            return data
        except:
            pass
    
    return None


def read_spectrum_npz(npz_path):
    """读取numpy格式的光谱"""
    data = np.load(npz_path)
    return {
        'targetid': data['targetid'],
        'B': {
            'wavelength': data['B_wavelength'],
            'flux': data['B_flux'],
            'error': data['B_error']
        },
        'R': {
            'wavelength': data['R_wavelength'],
            'flux': data['R_flux'],
            'error': data['R_error']
        },
        'Z': {
            'wavelength': data['Z_wavelength'],
            'flux': data['Z_flux'],
            'error': data['Z_error']
        }
    }


def plot_spectrum(data, save_path=None, show_error=True):
    """
    绘制光谱
    
    Parameters:
    -----------
    data : dict
        光谱数据字典
    save_path : str
        保存路径(可选)
    show_error : bool
        是否显示误差
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
    
    bands = ['B', 'R', 'Z']
    colors = ['blue', 'green', 'red']
    
    for ax, band, color in zip(axes, bands, colors):
        if band in data:
            wave = data[band]['wavelength']
            flux = data[band]['flux']
            error = data[band]['error']
            
            # 绘制流量
            ax.plot(wave, flux, color=color, linewidth=0.8, label=f'{band} band')
            
            # 绘制误差
            if show_error:
                ax.fill_between(wave, flux - error, flux + error, 
                              color=color, alpha=0.2, label='Error')
            
            ax.set_ylabel('Flux', fontsize=12)
            ax.set_xlabel('Wavelength (Å)', fontsize=12)
            ax.set_title(f'{band} Band: {wave.min():.1f} - {wave.max():.1f} Å', fontsize=12)
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
    
    targetid = data.get('targetid', 'Unknown')
    fig.suptitle(f'DESI Spectrum - TARGETID: {targetid}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"光谱图已保存: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_combined_spectrum(data, save_path=None, show_error=True):
    """
    绘制合并的光谱(B/R/Z波段连接在一起)
    
    Parameters:
    -----------
    data : dict
        光谱数据字典
    save_path : str
        保存路径(可选)
    show_error : bool
        是否显示误差
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    
    bands = ['B', 'R', 'Z']
    colors = ['blue', 'green', 'red']
    
    for band, color in zip(bands, colors):
        if band in data:
            wave = data[band]['wavelength']
            flux = data[band]['flux']
            error = data[band]['error']
            
            ax.plot(wave, flux, color=color, linewidth=0.8, label=f'{band} band')
            
            if show_error:
                ax.fill_between(wave, flux - error, flux + error, 
                              color=color, alpha=0.2)
    
    targetid = data.get('targetid', 'Unknown')
    ax.set_xlabel('Wavelength (Å)', fontsize=12)
    ax.set_ylabel('Flux', fontsize=12)
    ax.set_title(f'DESI Combined Spectrum - TARGETID: {targetid}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"合并光谱图已保存: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_multiple_spectra(fits_paths, labels=None, save_path=None):
    """
    绘制多个光谱对比
    
    Parameters:
    -----------
    fits_paths : list
        光谱文件路径列表
    labels : list
        标签列表(可选)
    save_path : str
        保存路径(可选)
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(fits_paths)))
    
    for i, path in enumerate(fits_paths):
        data = read_spectrum(path)
        if data is None:
            continue
        
        label = labels[i] if labels else f"Spectrum {i+1}"
        
        # 合并所有波段
        for band in ['B', 'R', 'Z']:
            if band in data:
                wave = data[band]['wavelength']
                flux = data[band]['flux']
                ax.plot(wave, flux, color=colors[i], linewidth=0.8, 
                       label=f"{label} ({band})", alpha=0.7)
    
    ax.set_xlabel('Wavelength (Å)', fontsize=12)
    ax.set_ylabel('Flux', fontsize=12)
    ax.set_title('DESI Spectra Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"对比图已保存: {save_path}")
    else:
        plt.show()
    
    plt.close()


def batch_plot_spectra(spectra_dir, output_dir='./spectrum_plots', max_plots=100):
    """
    批量绘制光谱
    
    Parameters:
    -----------
    spectra_dir : str
        光谱文件目录
    output_dir : str
        输出图片目录
    max_plots : int
        最大绘制数量
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有光谱文件
    fits_files = [f for f in os.listdir(spectra_dir) if f.endswith('.fits')]
    
    print(f"找到 {len(fits_files)} 个光谱文件")
    print(f"将绘制前 {min(max_plots, len(fits_files))} 个...")
    
    for i, fname in enumerate(fits_files[:max_plots]):
        fits_path = os.path.join(spectra_dir, fname)
        targetid = fname.replace('.fits', '')
        
        data = read_spectrum(fits_path)
        if data is None:
            continue
        
        save_path = os.path.join(output_dir, f"{targetid}.png")
        plot_combined_spectrum(data, save_path=save_path)
        
        if (i + 1) % 10 == 0:
            print(f"已绘制 {i + 1}/{min(max_plots, len(fits_files))}")
    
    print(f"\n所有光谱图已保存至: {output_dir}/")


def print_spectrum_info(fits_path):
    """
    打印光谱文件信息
    
    Parameters:
    -----------
    fits_path : str
        光谱文件路径
    """
    print(f"\n{'='*60}")
    print(f"文件: {fits_path}")
    print(f"{'='*60}")
    
    with fits.open(fits_path) as hdul:
        print(f"\nHDU列表:")
        for i, hdu in enumerate(hdul):
            print(f"  HDU{i}: {hdu.name} - {hdu.data.shape if hdu.data is not None else 'No data'}")
        
        print(f"\n主头信息:")
        for key in ['TARGETID', 'SURVEY', 'PROGRAM']:
            if key in hdul[0].header:
                print(f"  {key}: {hdul[0].header[key]}")
        
        # 如果有Table HDU，显示列名
        for i in [1, 2, 3, 4]:
            if len(hdul) > i and hasattr(hdul[i], 'columns'):
                print(f"\nHDU{i} ({hdul[i].name}) 列名:")
                print(f"  {', '.join(hdul[i].columns.names[:10])}...")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='可视化DESI光谱')
    parser.add_argument('input', help='输入文件或目录')
    parser.add_argument('--output', default='./spectrum_plots', help='输出目录')
    parser.add_argument('--batch', action='store_true', help='批量处理目录中的所有光谱')
    parser.add_argument('--max', type=int, default=100, help='批量处理时的最大数量')
    parser.add_argument('--combined', action='store_true', help='绘制合并光谱')
    parser.add_argument('--info', action='store_true', help='只显示文件信息')
    parser.add_argument('--compare', nargs='+', help='对比多个光谱')
    
    args = parser.parse_args()
    
    if args.info:
        print_spectrum_info(args.input)
    elif args.batch:
        batch_plot_spectra(args.input, args.output, args.max)
    elif args.compare:
        plot_multiple_spectra(args.compare, save_path=f"{args.output}/comparison.png")
    elif args.combined:
        data = read_spectrum(args.input)
        if data:
            plot_combined_spectrum(data, save_path=f"{args.output}/spectrum.png")
    else:
        data = read_spectrum(args.input)
        if data:
            plot_spectrum(data, save_path=f"{args.output}/spectrum.png")
