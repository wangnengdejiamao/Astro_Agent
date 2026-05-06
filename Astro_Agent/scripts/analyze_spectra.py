#!/usr/bin/env python3
"""
DESI光谱数据分析示例

包含常见分析任务：
1. 计算信噪比(SNR)
2. 测量等值宽度(EW)
3. 归一化光谱
4. 拼接B/R/Z波段
5. 计算颜色
"""

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.interpolate import interp1d


def load_spectrum(fits_path):
    """
    加载光谱文件
    
    Returns:
    --------
    wave : array
        波长数组 (拼接B/R/Z)
    flux : array
        流量数组
    error : array
        误差数组
    info : dict
        波段信息
    """
    data = {}
    
    with fits.open(fits_path) as hdul:
        # 尝试不同的文件格式
        try:
            # 我们的提取格式
            for band in ['B', 'R', 'Z']:
                data[band] = {
                    'wave': hdul[f'{band}_BAND'].data[0],
                    'flux': hdul[f'{band}_BAND'].data[1],
                    'error': hdul[f'{band}_BAND'].data[2]
                }
            targetid = hdul[0].header.get('TARGETID', 'Unknown')
        except:
            # 本地服务器格式
            for i, band in enumerate(['B', 'R', 'Z'], 5):
                data[band] = {
                    'wave': hdul[i].data[0],
                    'flux': hdul[i].data[1],
                    'error': hdul[i].data[2]
                }
            targetid = hdul[0].header.get('TARGETID', 'Unknown')
    
    # 拼接波段
    wave = np.concatenate([data['B']['wave'], data['R']['wave'], data['Z']['wave']])
    flux = np.concatenate([data['B']['flux'], data['R']['flux'], data['Z']['flux']])
    error = np.concatenate([data['B']['error'], data['R']['error'], data['Z']['error']])
    
    info = {
        'targetid': targetid,
        'B_wave_range': (data['B']['wave'].min(), data['B']['wave'].max()),
        'R_wave_range': (data['R']['wave'].min(), data['R']['wave'].max()),
        'Z_wave_range': (data['Z']['wave'].min(), data['Z']['wave'].max()),
    }
    
    return wave, flux, error, info, data


def calculate_snr(wave, flux, error, wave_range=None):
    """
    计算光谱的信噪比
    
    Parameters:
    -----------
    wave : array
        波长数组
    flux : array
        流量数组
    error : array
        误差数组
    wave_range : tuple
        计算SNR的波长范围 (min, max)
    
    Returns:
    --------
    snr : float
        平均信噪比
    """
    if wave_range:
        mask = (wave >= wave_range[0]) & (wave <= wave_range[1])
        flux = flux[mask]
        error = error[mask]
    
    # 计算每个像素点的SNR
    snr_per_pixel = flux / error
    # 返回平均SNR
    return np.nanmedian(snr_per_pixel)


def continuum_normalize(wave, flux, error, segments=10):
    """
    简单连续谱归一化
    
    使用分位数方法估计连续谱
    
    Parameters:
    -----------
    wave : array
        波长数组
    flux : array
        流量数组
    error : array
        误差数组
    segments : int
        分段数
    
    Returns:
    --------
    flux_norm : array
        归一化后的流量
    continuum : array
        估计的连续谱
    """
    # 按波长分段
    wave_edges = np.linspace(wave.min(), wave.max(), segments + 1)
    continuum_points = []
    wave_points = []
    
    for i in range(segments):
        mask = (wave >= wave_edges[i]) & (wave < wave_edges[i+1])
        if np.sum(mask) > 0:
            # 使用上四分位数作为连续谱估计
            cont = np.percentile(flux[mask], 90)
            continuum_points.append(cont)
            wave_points.append(np.median(wave[mask]))
    
    # 插值得到连续谱
    continuum_interp = interp1d(wave_points, continuum_points, 
                                kind='linear', bounds_error=False, 
                                fill_value='extrapolate')
    continuum = continuum_interp(wave)
    
    # 归一化
    flux_norm = flux / continuum
    error_norm = error / continuum
    
    return flux_norm, error_norm, continuum


def measure_equivalent_width(wave, flux, line_center, line_width=20):
    """
    测量吸收线的等值宽度
    
    Parameters:
    -----------
    wave : array
        波长数组 (必须是归一化后的)
    flux : array
        归一化流量
    line_center : float
        谱线中心波长 (Å)
    line_width : float
        测量窗口宽度 (Å)
    
    Returns:
    --------
    ew : float
        等值宽度 (Å)
    ew_error : float
        误差
    """
    # 选择窗口
    mask = (wave >= line_center - line_width) & (wave <= line_center + line_width)
    wave_line = wave[mask]
    flux_line = flux[mask]
    
    if len(wave_line) < 2:
        return np.nan, np.nan
    
    # 计算EW = ∫(1 - F/Fc) dλ
    ew = integrate.trapezoid(1 - flux_line, wave_line)
    
    # 简单误差估计
    ew_error = np.std(1 - flux_line) * np.sqrt(len(wave_line)) * np.median(np.diff(wave_line))
    
    return ew, ew_error


def calculate_band_magnitude(wave, flux, band_center, band_width, zero_point=0):
    """
    计算某一波段的平均星等
    
    Parameters:
    -----------
    wave : array
        波长数组
    flux : array
        流量数组
    band_center : float
        波段中心波长 (Å)
    band_width : float
        波段宽度 (Å)
    zero_point : float
        零点 (用于转换为星等)
    
    Returns:
    --------
    mag : float
        星等
    """
    mask = (wave >= band_center - band_width/2) & (wave <= band_center + band_width/2)
    
    if np.sum(mask) == 0:
        return np.nan
    
    # 平均流量
    mean_flux = np.median(flux[mask])
    
    # 转换为星等 (简化)
    if mean_flux > 0:
        mag = -2.5 * np.log10(mean_flux) + zero_point
    else:
        mag = np.nan
    
    return mag


def plot_analysis(wave, flux, error, info, output_path=None):
    """
    绘制光谱分析结果
    """
    # 归一化
    flux_norm, error_norm, continuum = continuum_normalize(wave, flux, error)
    
    # 计算SNR
    snr_b = calculate_snr(wave, flux, error, info['B_wave_range'])
    snr_r = calculate_snr(wave, flux, error, info['R_wave_range'])
    snr_z = calculate_snr(wave, flux, error, info['Z_wave_range'])
    
    # 创建图
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. 原始光谱
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(wave, flux, 'k-', linewidth=0.5, alpha=0.8, label='Flux')
    ax1.plot(wave, continuum, 'r--', linewidth=1, label='Continuum')
    ax1.fill_between(wave, flux - error, flux + error, alpha=0.2, color='gray')
    ax1.axvspan(*info['B_wave_range'], alpha=0.1, color='blue', label='B band')
    ax1.axvspan(*info['R_wave_range'], alpha=0.1, color='green', label='R band')
    ax1.axvspan(*info['Z_wave_range'], alpha=0.1, color='red', label='Z band')
    ax1.set_xlabel('Wavelength (Å)')
    ax1.set_ylabel('Flux')
    ax1.set_title(f"DESI Spectrum - TARGETID: {info['targetid']}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 归一化光谱 - 全波段
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(wave, flux_norm, 'k-', linewidth=0.5, alpha=0.8)
    ax2.fill_between(wave, flux_norm - error_norm, flux_norm + error_norm, 
                     alpha=0.2, color='gray')
    ax2.axhline(y=1, color='r', linestyle='--', linewidth=1)
    ax2.set_xlabel('Wavelength (Å)')
    ax2.set_ylabel('Normalized Flux')
    ax2.set_title('Continuum Normalized Spectrum')
    ax2.set_ylim(0, 1.5)
    ax2.grid(True, alpha=0.3)
    
    # 3. 特定谱线区域 - Hα
    ax3 = fig.add_subplot(gs[2, 0])
    halpha_mask = (wave > 6500) & (wave < 6600)
    if np.sum(halpha_mask) > 0:
        ax3.plot(wave[halpha_mask], flux_norm[halpha_mask], 'k-', linewidth=1)
        ax3.axvline(x=6563, color='r', linestyle='--', label='Hα 6563Å')
        ax3.set_xlabel('Wavelength (Å)')
        ax3.set_ylabel('Normalized Flux')
        ax3.set_title('Hα Region')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. 特定谱线区域 - Mg I (5167, 5173, 5184)
    ax4 = fig.add_subplot(gs[2, 1])
    mg_mask = (wave > 5150) & (wave < 5200)
    if np.sum(mg_mask) > 0:
        ax4.plot(wave[mg_mask], flux_norm[mg_mask], 'k-', linewidth=1)
        ax4.axvline(x=5167, color='b', linestyle='--', alpha=0.5, label='Mg I')
        ax4.axvline(x=5173, color='b', linestyle='--', alpha=0.5)
        ax4.axvline(x=5184, color='b', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Wavelength (Å)')
        ax4.set_ylabel('Normalized Flux')
        ax4.set_title('Mg I Triplet Region')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # 添加信息文本
    info_text = f"SNR(B)={snr_b:.1f}, SNR(R)={snr_r:.1f}, SNR(Z)={snr_z:.1f}"
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=12, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"分析图已保存: {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    return {
        'snr_b': snr_b,
        'snr_r': snr_r,
        'snr_z': snr_z
    }


def analyze_single_spectrum(fits_path, output_dir='./analysis'):
    """
    分析单个光谱文件
    
    Parameters:
    -----------
    fits_path : str
        光谱文件路径
    output_dir : str
        输出目录
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"分析光谱: {fits_path}")
    
    # 加载光谱
    wave, flux, error, info, data = load_spectrum(fits_path)
    print(f"  TARGETID: {info['targetid']}")
    print(f"  波长范围: {wave.min():.1f} - {wave.max():.1f} Å")
    
    # 绘制分析图
    targetid = info['targetid']
    output_path = f"{output_dir}/{targetid}_analysis.png"
    results = plot_analysis(wave, flux, error, info, output_path)
    
    print(f"  SNR(B) = {results['snr_b']:.2f}")
    print(f"  SNR(R) = {results['snr_r']:.2f}")
    print(f"  SNR(Z) = {results['snr_z']:.2f}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='分析DESI光谱')
    parser.add_argument('fits_file', help='光谱FITS文件路径')
    parser.add_argument('--output', default='./analysis', help='输出目录')
    
    args = parser.parse_args()
    
    if os.path.exists(args.fits_file):
        analyze_single_spectrum(args.fits_file, args.output)
    else:
        print(f"错误: 文件不存在 {args.fits_file}")
