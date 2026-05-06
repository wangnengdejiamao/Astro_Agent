#!/usr/bin/env python3
"""
快速从coadd文件提取光谱并绘图

使用方法:
    python quick_extract.py --coadd ./coadd_cache/coadd-main-bright-4298.fits --targetid 39628069090099224
    
或者Python API:
    from quick_extract import extract_and_plot
    extract_and_plot('./coadd_cache/coadd-main-bright-4298.fits', 39628069090099224)
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import argparse
import os


def extract_spectrum(coadd_path, targetid):
    """从coadd文件提取光谱"""
    with fits.open(coadd_path) as hdul:
        fibermap = hdul['FIBERMAP'].data
        targetids = fibermap['TARGETID']
        
        idx = np.where(targetids == targetid)[0]
        if len(idx) == 0:
            return None
        idx = idx[0]
        
        spectrum = {
            'targetid': targetid,
            'ra': fibermap['TARGET_RA'][idx],
            'dec': fibermap['TARGET_DEC'][idx],
            'B': {
                'wave': hdul['B_WAVELENGTH'].data,
                'flux': hdul['B_FLUX'].data[idx],
                'err': 1/np.sqrt(hdul['B_IVAR'].data[idx] + 1e-10)
            },
            'R': {
                'wave': hdul['R_WAVELENGTH'].data,
                'flux': hdul['R_FLUX'].data[idx],
                'err': 1/np.sqrt(hdul['R_IVAR'].data[idx] + 1e-10)
            },
            'Z': {
                'wave': hdul['Z_WAVELENGTH'].data,
                'flux': hdul['Z_FLUX'].data[idx],
                'err': 1/np.sqrt(hdul['Z_IVAR'].data[idx] + 1e-10)
            }
        }
        return spectrum


def plot_spectrum(spectrum, save_path=None):
    """绘制光谱"""
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # 绘制三个波段
    ax.plot(spectrum['B']['wave'], spectrum['B']['flux'], 'b-', label='B', lw=0.8)
    ax.plot(spectrum['R']['wave'], spectrum['R']['flux'], 'g-', label='R', lw=0.8)
    ax.plot(spectrum['Z']['wave'], spectrum['Z']['flux'], 'r-', label='Z', lw=0.8)
    
    # 误差阴影
    ax.fill_between(spectrum['B']['wave'], 
                    spectrum['B']['flux']-spectrum['B']['err'],
                    spectrum['B']['flux']+spectrum['B']['err'], color='b', alpha=0.2)
    ax.fill_between(spectrum['R']['wave'], 
                    spectrum['R']['flux']-spectrum['R']['err'],
                    spectrum['R']['flux']+spectrum['R']['err'], color='g', alpha=0.2)
    ax.fill_between(spectrum['Z']['wave'], 
                    spectrum['Z']['flux']-spectrum['Z']['err'],
                    spectrum['Z']['flux']+spectrum['Z']['err'], color='r', alpha=0.2)
    
    # 谱线标注
    lines = {3933: 'Ca II K', 3968: 'Ca II H', 4861: 'Hβ', 
             5175: 'Mg b', 5890: 'Na D', 6563: 'Hα'}
    for w, name in lines.items():
        if 3600 <= w <= 9800:
            ax.axvline(w, color='gray', ls='--', alpha=0.5, lw=0.5)
            ax.text(w, ax.get_ylim()[1]*0.95, name, rotation=90, ha='right', fontsize=8)
    
    ax.set_xlabel('Wavelength (Å)')
    ax.set_ylabel('Flux (10⁻¹⁷ erg/s/cm²/Å)')
    ax.set_title(f"TARGETID: {spectrum['targetid']}, RA: {spectrum['ra']:.4f}, DEC: {spectrum['dec']:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"保存: {save_path}")
    
    plt.show()
    return fig


def extract_and_plot(coadd_path, targetid, save_path=None):
    """提取并绘制光谱"""
    print(f"提取光谱: TARGETID={targetid}")
    print(f"文件: {coadd_path}")
    
    spec = extract_spectrum(coadd_path, targetid)
    if spec is None:
        print(f"错误: TARGETID {targetid} 不在文件中")
        return None
    
    print(f"坐标: RA={spec['ra']:.6f}, DEC={spec['dec']:.6f}")
    plot_spectrum(spec, save_path)
    return spec


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='快速提取和绘制DESI光谱')
    parser.add_argument('--coadd', required=True, help='coadd文件路径')
    parser.add_argument('--targetid', type=int, required=True, help='TARGETID')
    parser.add_argument('--output', default=None, help='输出图像路径')
    
    args = parser.parse_args()
    
    extract_and_plot(args.coadd, args.targetid, args.output)
