#!/usr/bin/env python3
"""
合并DESI B/R/Z三个波段为连续光谱

DESI的三个波段有重叠区域，这个脚本提供多种合并策略：
- simple: 直接拼接，重叠区域取平均
- weighted: 按误差加权平均重叠区域
- splice: 在重叠区域中点切换
"""

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import os


def load_individual_bands(fits_path):
    """
    从FITS文件加载B/R/Z三个波段
    
    Returns:
    --------
    bands : dict
        包含B/R/Z波段数据的字典
    """
    bands = {}
    
    with fits.open(fits_path) as hdul:
        # 尝试不同格式
        try:
            # 我们的提取格式
            for band in ['B', 'R', 'Z']:
                bands[band] = {
                    'wave': hdul[f'{band}_BAND'].data[0],
                    'flux': hdul[f'{band}_BAND'].data[1],
                    'error': hdul[f'{band}_BAND'].data[2]
                }
        except:
            # 本地服务器格式
            for i, band in enumerate(['B', 'R', 'Z'], 5):
                if len(hdul) > i:
                    bands[band] = {
                        'wave': hdul[i].data[0],
                        'flux': hdul[i].data[1],
                        'error': hdul[i].data[2]
                    }
    
    return bands


def merge_bands_simple(bands):
    """
    简单拼接：在重叠区域取平均
    
    Parameters:
    -----------
    bands : dict
        B/R/Z波段数据
    
    Returns:
    --------
    wave, flux, error : arrays
        合并后的光谱
    """
    B = bands['B']
    R = bands['R']
    Z = bands['Z']
    
    # B/R重叠区域
    br_overlap_min = max(B['wave'].min(), R['wave'].min())
    br_overlap_max = min(B['wave'].max(), R['wave'].max())
    
    # R/Z重叠区域
    rz_overlap_min = max(R['wave'].min(), Z['wave'].min())
    rz_overlap_max = min(R['wave'].max(), Z['wave'].max())
    
    # 构建合并数组
    all_wave = []
    all_flux = []
    all_error = []
    
    # B波段 (到重叠区开始)
    b_mask = B['wave'] < br_overlap_min
    all_wave.extend(B['wave'][b_mask])
    all_flux.extend(B['flux'][b_mask])
    all_error.extend(B['error'][b_mask])
    
    # B/R重叠区 - 取平均
    b_overlap_mask = (B['wave'] >= br_overlap_min) & (B['wave'] <= br_overlap_max)
    r_overlap_mask = (R['wave'] >= br_overlap_min) & (R['wave'] <= br_overlap_max)
    
    # 插值到相同网格
    wave_overlap = B['wave'][b_overlap_mask]
    flux_b_overlap = B['flux'][b_overlap_mask]
    error_b_overlap = B['error'][b_overlap_mask]
    
    flux_r_interp = np.interp(wave_overlap, R['wave'], R['flux'])
    error_r_interp = np.interp(wave_overlap, R['wave'], R['error'])
    
    # 简单平均
    flux_avg = (flux_b_overlap + flux_r_interp) / 2
    error_avg = np.sqrt(error_b_overlap**2 + error_r_interp**2) / 2
    
    all_wave.extend(wave_overlap)
    all_flux.extend(flux_avg)
    all_error.extend(error_avg)
    
    # R波段 (重叠区结束到新重叠区开始)
    r_mask = (R['wave'] > br_overlap_max) & (R['wave'] < rz_overlap_min)
    all_wave.extend(R['wave'][r_mask])
    all_flux.extend(R['flux'][r_mask])
    all_error.extend(R['error'][r_mask])
    
    # R/Z重叠区 - 取平均
    r_overlap_mask2 = (R['wave'] >= rz_overlap_min) & (R['wave'] <= rz_overlap_max)
    z_overlap_mask = (Z['wave'] >= rz_overlap_min) & (Z['wave'] <= rz_overlap_max)
    
    wave_overlap2 = R['wave'][r_overlap_mask2]
    flux_r_overlap2 = R['flux'][r_overlap_mask2]
    error_r_overlap2 = R['error'][r_overlap_mask2]
    
    flux_z_interp = np.interp(wave_overlap2, Z['wave'], Z['flux'])
    error_z_interp = np.interp(wave_overlap2, Z['wave'], Z['error'])
    
    flux_avg2 = (flux_r_overlap2 + flux_z_interp) / 2
    error_avg2 = np.sqrt(error_r_overlap2**2 + error_z_interp**2) / 2
    
    all_wave.extend(wave_overlap2)
    all_flux.extend(flux_avg2)
    all_error.extend(error_avg2)
    
    # Z波段 (重叠区结束)
    z_mask = Z['wave'] > rz_overlap_max
    all_wave.extend(Z['wave'][z_mask])
    all_flux.extend(Z['flux'][z_mask])
    all_error.extend(Z['error'][z_mask])
    
    return np.array(all_wave), np.array(all_flux), np.array(all_error)


def merge_bands_weighted(bands):
    """
    误差加权合并
    
    在重叠区域使用误差作为权重进行加权平均
    """
    B = bands['B']
    R = bands['R']
    Z = bands['Z']
    
    # 确定合并后的波长网格 (使用B的网格密度)
    wave_min = B['wave'].min()
    wave_max = Z['wave'].max()
    n_points = len(B['wave']) + len(R['wave']) + len(Z['wave'])
    wave_grid = np.linspace(wave_min, wave_max, n_points // 2)
    
    # 插值各波段到统一网格
    flux_b = np.interp(wave_grid, B['wave'], B['flux'], left=np.nan, right=np.nan)
    error_b = np.interp(wave_grid, B['wave'], B['error'], left=np.nan, right=np.nan)
    
    flux_r = np.interp(wave_grid, R['wave'], R['flux'], left=np.nan, right=np.nan)
    error_r = np.interp(wave_grid, R['wave'], R['error'], left=np.nan, right=np.nan)
    
    flux_z = np.interp(wave_grid, Z['wave'], Z['flux'], left=np.nan, right=np.nan)
    error_z = np.interp(wave_grid, Z['wave'], Z['error'], left=np.nan, right=np.nan)
    
    # 使用逆方差加权
    var_b = error_b**2
    var_r = error_r**2
    var_z = error_z**2
    
    # 合并流量: 加权平均
    weight_sum = np.zeros_like(wave_grid)
    weighted_flux_sum = np.zeros_like(wave_grid)
    weighted_var_sum = np.zeros_like(wave_grid)
    
    # B波段权重
    valid_b = ~np.isnan(flux_b) & (var_b > 0)
    weight_sum[valid_b] += 1.0 / var_b[valid_b]
    weighted_flux_sum[valid_b] += flux_b[valid_b] / var_b[valid_b]
    weighted_var_sum[valid_b] += 1.0
    
    # R波段权重
    valid_r = ~np.isnan(flux_r) & (var_r > 0)
    weight_sum[valid_r] += 1.0 / var_r[valid_r]
    weighted_flux_sum[valid_r] += flux_r[valid_r] / var_r[valid_r]
    weighted_var_sum[valid_r] += 1.0
    
    # Z波段权重
    valid_z = ~np.isnan(flux_z) & (var_z > 0)
    weight_sum[valid_z] += 1.0 / var_z[valid_z]
    weighted_flux_sum[valid_z] += flux_z[valid_z] / var_z[valid_z]
    weighted_var_sum[valid_z] += 1.0
    
    # 计算结果
    merged_flux = np.where(weight_sum > 0, weighted_flux_sum / weight_sum, np.nan)
    merged_error = np.where(weight_sum > 0, np.sqrt(1.0 / weight_sum), np.nan)
    
    # 只保留有效数据
    valid = ~np.isnan(merged_flux)
    
    return wave_grid[valid], merged_flux[valid], merged_error[valid]


def merge_bands_splice(bands):
    """
    拼接法：在重叠区域中点切换波段
    
    简单直接，不进行平均
    """
    B = bands['B']
    R = bands['R']
    Z = bands['Z']
    
    # B/R边界
    br_boundary = (B['wave'].max() + R['wave'].min()) / 2
    # R/Z边界
    rz_boundary = (R['wave'].max() + Z['wave'].min()) / 2
    
    # 选择数据
    b_mask = B['wave'] < br_boundary
    r_mask = (R['wave'] >= br_boundary) & (R['wave'] < rz_boundary)
    z_mask = Z['wave'] >= rz_boundary
    
    wave = np.concatenate([B['wave'][b_mask], R['wave'][r_mask], Z['wave'][z_mask]])
    flux = np.concatenate([B['flux'][b_mask], R['flux'][r_mask], Z['flux'][z_mask]])
    error = np.concatenate([B['error'][b_mask], R['error'][r_mask], Z['error'][z_mask]])
    
    return wave, flux, error


def save_merged_spectrum(wave, flux, error, output_path, source_info=None):
    """
    保存合并后的光谱为FITS文件
    
    Parameters:
    -----------
    wave, flux, error : arrays
        光谱数据
    output_path : str
        输出文件路径
    source_info : dict
        源信息(可选)
    """
    # 创建HDU
    hdu = fits.PrimaryHDU()
    hdu.header['NAXIS'] = 2
    hdu.header['NAXIS1'] = len(wave)
    hdu.header['NAXIS2'] = 3
    hdu.header['COMMENT'] = 'Merged DESI B/R/Z spectrum'
    hdu.header['WAVEMIN'] = wave.min()
    hdu.header['WAVEMAX'] = wave.max()
    
    if source_info:
        for key, value in source_info.items():
            if isinstance(value, (int, float)) and len(str(key)) <= 8:
                try:
                    hdu.header[key[:8].upper()] = value
                except:
                    pass
    
    # 数据表
    data = np.array([wave, flux, error])
    hdu.data = data
    
    # 保存
    hdul = fits.HDUList([hdu])
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    hdul.writeto(output_path, overwrite=True)
    
    return output_path


def plot_merged_comparison(bands, wave_merged, flux_merged, error_merged, 
                          method_name, output_path=None):
    """
    绘制合并前后的对比图
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 原始波段
    ax = axes[0]
    colors = ['blue', 'green', 'red']
    for (band_name, band_data), color in zip(bands.items(), colors):
        ax.plot(band_data['wave'], band_data['flux'], 
               color=color, alpha=0.7, linewidth=0.8, label=f'{band_name} band')
    
    ax.set_xlabel('Wavelength (Å)')
    ax.set_ylabel('Flux')
    ax.set_title('Original B/R/Z Bands')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 合并后的光谱
    ax = axes[1]
    ax.plot(wave_merged, flux_merged, 'k-', linewidth=0.8, label=f'Merged ({method_name})')
    ax.fill_between(wave_merged, flux_merged - error_merged, 
                    flux_merged + error_merged, alpha=0.2, color='gray')
    
    # 标记原波段范围
    for (band_name, band_data), color in zip(bands.items(), colors):
        ax.axvspan(band_data['wave'].min(), band_data['wave'].max(), 
                  alpha=0.1, color=color, label=f'{band_name} range')
    
    ax.set_xlabel('Wavelength (Å)')
    ax.set_ylabel('Flux')
    ax.set_title(f'Merged Spectrum - {method_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"对比图已保存: {output_path}")
    else:
        plt.show()
    
    plt.close()


def merge_spectrum_file(input_path, output_path=None, method='weighted', plot=False):
    """
    合并单个光谱文件
    
    Parameters:
    -----------
    input_path : str
        输入FITS文件路径
    output_path : str
        输出FITS文件路径 (可选)
    method : str
        合并方法: 'simple', 'weighted', 'splice'
    plot : bool
        是否生成对比图
    
    Returns:
    --------
    wave, flux, error : arrays
        合并后的光谱数据
    """
    # 加载波段
    bands = load_individual_bands(input_path)
    
    # 选择合并方法
    if method == 'simple':
        wave, flux, error = merge_bands_simple(bands)
    elif method == 'weighted':
        wave, flux, error = merge_bands_weighted(bands)
    elif method == 'splice':
        wave, flux, error = merge_bands_splice(bands)
    else:
        raise ValueError(f"未知方法: {method}")
    
    # 保存
    if output_path:
        save_merged_spectrum(wave, flux, error, output_path)
        print(f"合并光谱已保存: {output_path}")
    
    # 绘图
    if plot:
        plot_path = output_path.replace('.fits', '_comparison.png') if output_path else None
        plot_merged_comparison(bands, wave, flux, error, method, plot_path)
    
    return wave, flux, error


def batch_merge(input_dir, output_dir='./merged_spectra', method='weighted'):
    """
    批量合并光谱
    
    Parameters:
    -----------
    input_dir : str
        输入目录
    output_dir : str
        输出目录
    method : str
        合并方法
    """
    import glob
    
    os.makedirs(output_dir, exist_ok=True)
    
    fits_files = glob.glob(os.path.join(input_dir, '*.fits'))
    print(f"找到 {len(fits_files)} 个光谱文件")
    print(f"使用方法: {method}")
    
    for i, input_path in enumerate(fits_files):
        filename = os.path.basename(input_path)
        output_path = os.path.join(output_dir, filename)
        
        try:
            merge_spectrum_file(input_path, output_path, method=method)
            if (i + 1) % 10 == 0:
                print(f"已处理 {i + 1}/{len(fits_files)}")
        except Exception as e:
            print(f"处理 {filename} 时出错: {e}")
    
    print(f"\n合并完成! 输出目录: {output_dir}/")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='合并DESI B/R/Z波段光谱')
    parser.add_argument('input', help='输入文件或目录')
    parser.add_argument('--output', help='输出文件或目录')
    parser.add_argument('--method', default='weighted', 
                       choices=['simple', 'weighted', 'splice'],
                       help='合并方法 (默认: weighted)')
    parser.add_argument('--batch', action='store_true', help='批量处理目录')
    parser.add_argument('--plot', action='store_true', help='生成对比图')
    
    args = parser.parse_args()
    
    if args.batch:
        batch_merge(args.input, args.output or './merged_spectra', args.method)
    else:
        merge_spectrum_file(args.input, args.output, args.method, args.plot)
