#!/usr/bin/env python3
"""
DESI MWS 光谱一体化工具
========================
输入 RA/DEC 坐标，自动在 DESI MWS 星表中匹配并下载光谱。

特性:
  - KD-Tree 空间索引，首次构建后缓存，后续查询 <1ms
  - 并行下载 + 断点续传 + 智能缓存
  - 支持单目标和批量 CSV 模式
  - Tkinter GUI 界面
  - 可作为 Python 库 import 使用

用法:
  python desi_spectrum_tool.py                              # 启动 GUI
  python desi_spectrum_tool.py single --ra 190.3 --dec 2.6  # 单目标
  python desi_spectrum_tool.py batch targets.csv            # 批量
  python desi_spectrum_tool.py build-index                  # 预构建索引

Python API:
  from desi_spectrum_tool import DESITool
  tool = DESITool()
  result = tool.process_single(190.305, 2.596)
  summary = tool.process_batch('targets.csv')
"""

import os
import sys
import time
import pickle
import argparse
import warnings
import threading
import queue
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from astropy.io import fits
import matplotlib
matplotlib.use('Agg')  # 非交互后端，避免 GUI 线程死锁
import matplotlib.pyplot as plt
import requests
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============ 默认配置 ============
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
COMPRESSED_CSV = os.path.join(DATA_DIR, 'mws_gaia_compressed.csv')
FULL_CSV = os.path.join(DATA_DIR, 'mws_gaia.csv')
INDEX_CACHE = os.path.join(DATA_DIR, 'mws_gaia_index.pkl')
COADD_CACHE_DIR = os.path.join(PROJECT_ROOT, 'output', 'coadd_cache')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output', 'spectra_output')
BASE_URL = "https://data.desi.lbl.gov/public/dr1/spectro/redux/iron/healpix"
MATCH_RADIUS_ARCSEC = 5.0
MAX_WORKERS = 4
CHUNK_SIZE = 1024 * 1024  # 1MB


# ================================================================
#  CatalogIndex - KD-Tree 空间索引
# ================================================================

class CatalogIndex:
    """
    基于 cKDTree 的空间索引，用于在 4.5M 行 DESI MWS 星表中快速匹配 RA/DEC。

    首次运行从完整 CSV 读取 TARGETID/RA/DEC/SURVEY/PROGRAM/HEALPIX 构建索引
    并缓存为 pickle，后续运行直接加载 pickle（~2-3秒）。

    SURVEY/PROGRAM/HEALPIX 直接存入索引，匹配后无需再扫描 2GB CSV 即可下载。
    """

    INDEX_VERSION = 2  # 版本号，升级时自动重建

    def __init__(self, compressed_csv=COMPRESSED_CSV, full_csv=FULL_CSV,
                 index_cache=INDEX_CACHE, log_func=None):
        self.compressed_csv = compressed_csv
        self.full_csv = full_csv
        self.index_cache = index_cache
        self.log = log_func or print
        self.targetids = None
        self.ra = None
        self.dec = None
        self.survey = None   # numpy object array
        self.program = None  # numpy object array
        self.healpix = None  # numpy int32 array
        self.tree = None
        self._load_or_build()

    def _log(self, msg):
        self.log(msg)

    def _load_or_build(self):
        if os.path.exists(self.index_cache):
            try:
                self._load_index()
                return
            except Exception as e:
                self._log(f"索引缓存加载失败({e})，重新构建...")
                try:
                    os.remove(self.index_cache)
                except OSError:
                    pass
        self._build_index()

    def _load_index(self):
        t0 = time.time()
        self._log("加载 KD-Tree 索引缓存...")
        with open(self.index_cache, 'rb') as f:
            data = pickle.load(f)
        # 版本检查：旧索引缺少 survey/program/healpix 字段，需重建
        if data.get('version', 0) < self.INDEX_VERSION:
            raise ValueError("索引版本过旧，需重建")
        self.targetids = data['targetids']
        self.ra = data['ra']
        self.dec = data['dec']
        self.survey = data['survey']
        self.program = data['program']
        self.healpix = data['healpix']
        self.tree = data['tree']
        dt = time.time() - t0
        self._log(f"  索引加载完成: {len(self.targetids):,} 个源, 耗时 {dt:.1f}s")

    def _build_index(self):
        t0 = time.time()
        self._log(f"构建 KD-Tree 索引 (首次运行，后续将使用缓存)...")
        self._log(f"  读取星表 (仅需 TARGETID/RA/DEC/SURVEY/PROGRAM/HEALPIX)...")

        # 只读取需要的6列，比读完整 CSV 快很多
        cols = ['TARGETID', 'RA', 'DEC', 'SURVEY', 'PROGRAM', 'HEALPIX']
        df = pd.read_csv(self.full_csv, usecols=cols)

        self.targetids = df['TARGETID'].values.astype(np.int64)
        self.ra = df['RA'].values.astype(np.float64)
        self.dec = df['DEC'].values.astype(np.float64)
        self.survey = df['SURVEY'].values.astype(str)
        self.program = df['PROGRAM'].values.astype(str)
        self.healpix = df['HEALPIX'].values.astype(np.int32)

        self._log(f"  共 {len(self.targetids):,} 个源, 构建 KD-Tree...")
        xyz = self._radec_to_xyz(self.ra, self.dec)
        self.tree = cKDTree(xyz)

        self._log(f"  保存索引缓存到 {self.index_cache} ...")
        with open(self.index_cache, 'wb') as f:
            pickle.dump({
                'version': self.INDEX_VERSION,
                'targetids': self.targetids,
                'ra': self.ra,
                'dec': self.dec,
                'survey': self.survey,
                'program': self.program,
                'healpix': self.healpix,
                'tree': self.tree,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

        dt = time.time() - t0
        self._log(f"  索引构建完成, 耗时 {dt:.1f}s")

    @staticmethod
    def _radec_to_xyz(ra_deg, dec_deg):
        ra_rad = np.radians(ra_deg)
        dec_rad = np.radians(dec_deg)
        cos_dec = np.cos(dec_rad)
        x = cos_dec * np.cos(ra_rad)
        y = cos_dec * np.sin(ra_rad)
        z = np.sin(dec_rad)
        return np.column_stack([x, y, z])

    @staticmethod
    def _arcsec_to_chord(arcsec):
        """角秒 → 单位球弦距"""
        return 2.0 * np.sin(np.radians(arcsec / 3600.0) / 2.0)

    @staticmethod
    def _chord_to_arcsec(chord):
        """单位球弦距 → 角秒"""
        return np.degrees(2.0 * np.arcsin(chord / 2.0)) * 3600.0

    def query(self, ra, dec, radius_arcsec=MATCH_RADIUS_ARCSEC):
        """
        单目标查询：返回半径内最近匹配。

        Returns:
            dict 或 None: {'targetid', 'ra', 'dec', 'survey', 'program', 'healpix', 'separation_arcsec'}
        """
        xyz = self._radec_to_xyz(np.array([ra]), np.array([dec]))
        dist, idx = self.tree.query(xyz[0], k=1)
        sep_arcsec = self._chord_to_arcsec(dist)

        if sep_arcsec > radius_arcsec:
            return None

        return {
            'targetid': int(self.targetids[idx]),
            'ra': float(self.ra[idx]),
            'dec': float(self.dec[idx]),
            'survey': str(self.survey[idx]),
            'program': str(self.program[idx]),
            'healpix': int(self.healpix[idx]),
            'separation_arcsec': float(sep_arcsec),
        }

    def query_batch(self, ra_array, dec_array, radius_arcsec=MATCH_RADIUS_ARCSEC):
        """
        批量查询。

        Returns:
            list[dict or None]: 每个输入坐标的匹配结果
        """
        ra_arr = np.asarray(ra_array, dtype=np.float64)
        dec_arr = np.asarray(dec_array, dtype=np.float64)
        xyz = self._radec_to_xyz(ra_arr, dec_arr)
        dists, idxs = self.tree.query(xyz, k=1)
        sep_arcsec = self._chord_to_arcsec(dists)

        results = []
        for i in range(len(ra_arr)):
            if sep_arcsec[i] > radius_arcsec:
                results.append(None)
            else:
                j = idxs[i]
                results.append({
                    'targetid': int(self.targetids[j]),
                    'ra': float(self.ra[j]),
                    'dec': float(self.dec[j]),
                    'survey': str(self.survey[j]),
                    'program': str(self.program[j]),
                    'healpix': int(self.healpix[j]),
                    'separation_arcsec': float(sep_arcsec[i]),
                })
        return results

    def get_full_info(self, targetid_list):
        """
        从完整 CSV 中按 TARGETID 提取完整信息（SURVEY, PROGRAM, HEALPIX 等）。
        使用分块读取避免一次性加载 2GB 文件。

        Returns:
            dict: {targetid: {全部列信息}}
        """
        target_set = set(int(t) for t in targetid_list)
        result = {}

        for chunk in pd.read_csv(self.full_csv, chunksize=500000):
            matched = chunk[chunk['TARGETID'].isin(target_set)]
            for _, row in matched.iterrows():
                tid = int(row['TARGETID'])
                result[tid] = row.to_dict()
                target_set.discard(tid)
            if not target_set:
                break

        return result


# ================================================================
#  DownloadManager - 并行下载管理器
# ================================================================

class DownloadManager:
    """并行下载 coadd FITS 文件，带重试、缓存和进度回调。"""

    def __init__(self, cache_dir=COADD_CACHE_DIR, max_workers=MAX_WORKERS,
                 chunk_size=CHUNK_SIZE, max_retries=3,
                 progress_callback=None, log_func=None):
        self.cache_dir = cache_dir
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self.max_retries = max_retries
        self.progress_callback = progress_callback
        self.log = log_func or print
        # 使用代理感知的 session
        sample_url = f"{BASE_URL}/main/dark/0/0/dummy.fits"
        try:
            from astro_toolbox.utils import get_session
            self.session = get_session(sample_url)
        except ImportError:
            self.session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=max_workers, pool_maxsize=max_workers + 2,
                max_retries=0
            )
            self.session.mount('https://', adapter)
            self.session.mount('http://', adapter)
        os.makedirs(self.cache_dir, exist_ok=True)

    @staticmethod
    def get_coadd_url(survey, program, healpix):
        healpix_group = int(healpix) // 100
        filename = f"coadd-{survey}-{program}-{healpix}.fits"
        url = f"{BASE_URL}/{survey}/{program}/{healpix_group}/{healpix}/{filename}"
        return url, filename

    def _download_single(self, url, local_path):
        """下载单个文件，支持重试和原子写入。"""
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            return local_path

        partial_path = local_path + '.partial'

        for attempt in range(self.max_retries):
            try:
                resp = self.session.get(url, stream=True, timeout=(60, 600))
                resp.raise_for_status()
                total = int(resp.headers.get('content-length', 0))

                with open(partial_path, 'wb') as f:
                    downloaded = 0
                    for chunk in resp.iter_content(chunk_size=self.chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if self.progress_callback:
                                self.progress_callback(
                                    os.path.basename(local_path),
                                    downloaded, total
                                )

                os.rename(partial_path, local_path)
                return local_path

            except Exception as e:
                if os.path.exists(partial_path):
                    os.remove(partial_path)
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    self.log(f"  下载失败({e})，{wait}s后重试...")
                    time.sleep(wait)
                else:
                    self.log(f"  下载失败(已重试{self.max_retries}次): {e}")
                    return None

    def get_or_download(self, survey, program, healpix):
        """获取单个 coadd 文件（缓存或下载）。自动检测截断文件。"""
        url, filename = self.get_coadd_url(survey, program, healpix)
        local_path = os.path.join(self.cache_dir, filename)
        # 验证已缓存文件的完整性 — 必须能真正读取 B_FLUX 数据
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            try:
                with fits.open(local_path, memmap=False) as hdul:
                    # 检查是否截断: B_FLUX 行数必须等于 FIBERMAP 行数
                    n_fibers = len(hdul['FIBERMAP'].data)
                    b_shape = hdul['B_FLUX'].data.shape
                    assert b_shape[0] == n_fibers, \
                        f"B_FLUX rows {b_shape[0]} != FIBERMAP {n_fibers}"
                return local_path
            except Exception:
                self.log(f"  缓存文件损坏/截断，重新下载: {filename}")
                try:
                    os.remove(local_path)
                except OSError:
                    pass
        return self._download_single(url, local_path)

    def download_coadds(self, targets_info, cli_progress=False):
        """
        并行下载多个 coadd 文件（自动去重）。

        Args:
            targets_info: list of dict, 每个需要 'survey', 'program', 'healpix' 键
            cli_progress: 是否显示命令行进度条

        Returns:
            dict: {(survey, program, healpix) -> local_path}
        """
        # 去重
        unique = {}
        for info in targets_info:
            key = (info['SURVEY'], info['PROGRAM'], int(info['HEALPIX']))
            if key not in unique:
                url, filename = self.get_coadd_url(*key)
                local_path = os.path.join(self.cache_dir, filename)
                unique[key] = (url, local_path)

        # 检查缓存
        to_download = {}
        result = {}
        for key, (url, local_path) in unique.items():
            if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                result[key] = local_path
            else:
                to_download[key] = (url, local_path)

        if not to_download:
            self.log(f"  全部 {len(result)} 个 coadd 文件已在缓存中")
            return result

        self.log(f"  需下载 {len(to_download)} 个 coadd 文件"
                 f" (缓存命中 {len(result)} 个)")

        # 并行下载
        pbar = None
        if cli_progress:
            pbar = tqdm(total=len(to_download), desc="下载coadd", unit="file")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for key, (url, local_path) in to_download.items():
                future = executor.submit(self._download_single, url, local_path)
                futures[future] = key

            for future in as_completed(futures):
                key = futures[future]
                path = future.result()
                if path:
                    result[key] = path
                if pbar:
                    pbar.update(1)

        if pbar:
            pbar.close()

        return result


# ================================================================
#  SpectrumExtractor - 光谱提取/绘图/保存
# ================================================================

class SpectrumExtractor:
    """从 DESI coadd FITS 文件中提取、绘制、保存光谱。"""

    @staticmethod
    def extract_from_coadd(coadd_path, targetid):
        """从 coadd 文件提取特定 TARGETID 的 B/R/Z 三波段光谱。"""
        with fits.open(coadd_path, memmap=False) as hdul:
            fibermap = hdul['FIBERMAP'].data
            targetids = fibermap['TARGETID']
            idx = np.where(targetids == targetid)[0]
            if len(idx) == 0:
                return None
            idx = idx[0]

            # 验证 flux 数组维度是否与 fibermap 一致
            n_fibers = len(fibermap)
            for band_prefix in ('B', 'R', 'Z'):
                flux_ext = f'{band_prefix}_FLUX'
                if flux_ext in [h.name for h in hdul]:
                    flux_shape = hdul[flux_ext].data.shape
                    if len(flux_shape) != 2 or flux_shape[0] != n_fibers:
                        raise ValueError(
                            f"FITS 文件损坏: {flux_ext} shape={flux_shape}, "
                            f"期望 ({n_fibers}, *). 请删除缓存文件重新下载。")

            spectrum = {
                'targetid': int(targetid),
                'ra': float(fibermap['TARGET_RA'][idx]),
                'dec': float(fibermap['TARGET_DEC'][idx]),
                'fiberstatus': int(fibermap['COADD_FIBERSTATUS'][idx])
                    if 'COADD_FIBERSTATUS' in fibermap.columns.names else None,
                'B': {
                    'wavelength': hdul['B_WAVELENGTH'].data.copy(),
                    'flux': hdul['B_FLUX'].data[idx].copy(),
                    'error': 1.0 / np.sqrt(hdul['B_IVAR'].data[idx] + 1e-10),
                },
                'R': {
                    'wavelength': hdul['R_WAVELENGTH'].data.copy(),
                    'flux': hdul['R_FLUX'].data[idx].copy(),
                    'error': 1.0 / np.sqrt(hdul['R_IVAR'].data[idx] + 1e-10),
                },
                'Z': {
                    'wavelength': hdul['Z_WAVELENGTH'].data.copy(),
                    'flux': hdul['Z_FLUX'].data[idx].copy(),
                    'error': 1.0 / np.sqrt(hdul['Z_IVAR'].data[idx] + 1e-10),
                },
            }

            # 提取观测时间 (从 EXP_FIBERMAP)
            if 'EXP_FIBERMAP' in [h.name for h in hdul]:
                exp_fm = hdul['EXP_FIBERMAP'].data
                exp_mask = exp_fm['TARGETID'] == targetid
                if np.any(exp_mask):
                    mjds = exp_fm['MJD'][exp_mask]
                    nights = exp_fm['NIGHT'][exp_mask]
                    spectrum['obs_mjd'] = float(np.median(mjds))
                    spectrum['obs_mjd_all'] = mjds.tolist()
                    spectrum['obs_nights'] = sorted(set(int(n) for n in nights))

            return spectrum

    @staticmethod
    def plot_spectrum(spectrum, target_info=None, save_path=None,
                      show=False, fig=None, ax=None):
        """绘制三波段光谱 + S/N 子图。"""
        targetid = spectrum['targetid']

        if fig is None or ax is None:
            fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                                     gridspec_kw={'height_ratios': [3, 1]})
        else:
            axes = ax if hasattr(ax, '__len__') else [ax]

        ax1 = axes[0]
        colors = {'B': ('blue', 'B band (3600-5800 A)'),
                  'R': ('green', 'R band (5800-7600 A)'),
                  'Z': ('red', 'Z band (7600-9800 A)')}

        for band, (color, label) in colors.items():
            w = spectrum[band]['wavelength']
            f = spectrum[band]['flux']
            e = spectrum[band]['error']
            ax1.plot(w, f, color=color, linewidth=0.8, alpha=0.8, label=label)
            ax1.fill_between(w, f - e, f + e, color=color, alpha=0.15)

        ax1.set_xlabel('Wavelength (A)', fontsize=12)
        ax1.set_ylabel('Flux (1e-17 erg/s/cm2/A)', fontsize=12)

        title = f'DESI Spectrum - TARGETID: {targetid}'
        if target_info:
            title += f'\nRA={spectrum["ra"]:.6f}, DEC={spectrum["dec"]:.6f}'
            teff = target_info.get('TEFF')
            try:
                if teff is not None and np.isfinite(float(teff)):
                    title += f', Teff={float(teff):.0f}K'
            except (ValueError, TypeError):
                pass
        ax1.set_title(title, fontsize=12)
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)

        spectral_lines = {
            3933: 'Ca II K', 3968: 'Ca II H', 4861: 'Hb',
            5175: 'Mg b', 5890: 'Na D', 6563: 'Ha', 8500: 'Ca II IR',
        }
        ylim = ax1.get_ylim()
        for wave, name in spectral_lines.items():
            ax1.axvline(wave, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
            ax1.text(wave, ylim[1] * 0.95, name, rotation=90, fontsize=8,
                     ha='right', va='top')

        if len(axes) > 1:
            ax2 = axes[1]
            for band, (color, _) in colors.items():
                sn = spectrum[band]['flux'] / spectrum[band]['error']
                ax2.plot(spectrum[band]['wavelength'], sn, color=color,
                         linewidth=0.5, alpha=0.7)
            ax2.set_xlabel('Wavelength (A)', fontsize=12)
            ax2.set_ylabel('S/N', fontsize=12)
            ax2.set_title('Signal-to-Noise Ratio', fontsize=11)
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    @staticmethod
    def save_spectrum_fits(spectrum, target_info, output_path):
        """保存提取的光谱为 FITS 文件。"""
        hdul = fits.HDUList()

        header = fits.Header()
        header['TARGETID'] = spectrum['targetid']
        header['RA'] = spectrum['ra']
        header['DEC'] = spectrum['dec']
        if spectrum.get('fiberstatus') is not None:
            header['FIBERSTA'] = spectrum['fiberstatus']
        if target_info:
            for key in ('SURVEY', 'PROGRAM', 'VRAD', 'TEFF', 'LOGG', 'FEH'):
                val = target_info.get(key)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    header[key[:8]] = val
        header['COMMENT'] = 'Extracted by desi_spectrum_tool'
        hdul.append(fits.PrimaryHDU(header=header))

        for band in ('B', 'R', 'Z'):
            data = np.array([spectrum[band]['wavelength'],
                             spectrum[band]['flux'],
                             spectrum[band]['error']])
            hdul.append(fits.ImageHDU(data=data, name=f'{band}_BAND'))

        hdul.writeto(output_path, overwrite=True)


# ================================================================
#  DESITool - 编排器
# ================================================================

class DESITool:
    """主编排器：坐标匹配 → 下载 → 提取 → 保存。"""

    def __init__(self, output_dir=OUTPUT_DIR, cache_dir=COADD_CACHE_DIR,
                 compressed_csv=COMPRESSED_CSV, full_csv=FULL_CSV,
                 index_cache=INDEX_CACHE, max_workers=MAX_WORKERS,
                 progress_callback=None, log_func=None):
        self.output_dir = output_dir
        self.log = log_func or print
        self.progress_callback = progress_callback
        os.makedirs(output_dir, exist_ok=True)

        self.index = CatalogIndex(
            compressed_csv=compressed_csv, full_csv=full_csv,
            index_cache=index_cache, log_func=self.log)
        self.downloader = DownloadManager(
            cache_dir=cache_dir, max_workers=max_workers,
            progress_callback=progress_callback, log_func=self.log)
        self.extractor = SpectrumExtractor()

    def process_single(self, ra, dec, radius_arcsec=MATCH_RADIUS_ARCSEC,
                       show_plot=False, save_fits=True, save_png=True):
        """
        单目标处理：匹配 → 下载 → 提取 → 保存。

        Returns:
            dict: {'match': ..., 'full_info': ..., 'spectrum': ..., 'files': {...}}
                  或 None（未匹配）
        """
        self.log(f"\n{'='*60}")
        self.log(f"查询坐标: RA={ra:.6f}, DEC={dec:.6f}")
        self.log(f"{'='*60}")

        # 1. 匹配
        self.log("\n[1/3] 坐标匹配...")
        match = self.index.query(ra, dec, radius_arcsec)
        if match is None:
            self.log(f"  未找到匹配 (半径 {radius_arcsec} arcsec)")
            return None
        self.log(f"  匹配成功! TARGETID={match['targetid']}, "
                 f"距离={match['separation_arcsec']:.4f}\"")

        survey = match['survey']
        program = match['program']
        healpix = match['healpix']
        self.log(f"  SURVEY={survey}, PROGRAM={program}, HEALPIX={healpix}")

        # 2. 下载 coadd
        self.log("\n[2/3] 下载 coadd 文件...")
        coadd_path = self.downloader.get_or_download(survey, program, healpix)
        if coadd_path is None:
            self.log("  下载失败!")
            return None
        self.log(f"  coadd 文件: {coadd_path}")

        # 3. 提取光谱
        self.log("\n[3/3] 提取光谱...")
        try:
            spectrum = self.extractor.extract_from_coadd(
                coadd_path, match['targetid'])
        except (ValueError, OSError) as e:
            # FITS 文件损坏，删除重新下载一次
            self.log(f"  文件损坏({e})，重新下载...")
            try:
                os.remove(coadd_path)
            except OSError:
                pass
            coadd_path = self.downloader.get_or_download(survey, program, healpix)
            if coadd_path is None:
                self.log("  重新下载失败!")
                return None
            try:
                spectrum = self.extractor.extract_from_coadd(
                    coadd_path, match['targetid'])
            except Exception as e2:
                self.log(f"  重新提取仍失败: {e2}")
                return None
        if spectrum is None:
            self.log(f"  TARGETID {match['targetid']} 不在 coadd 文件中!")
            return None

        self.log(f"  B: {len(spectrum['B']['wavelength'])} px, "
                 f"R: {len(spectrum['R']['wavelength'])} px, "
                 f"Z: {len(spectrum['Z']['wavelength'])} px")

        # 保存
        files = {}
        tid = match['targetid']
        # 用 match 中的 survey/program 作为 info 传给 save/plot
        info = {'SURVEY': survey, 'PROGRAM': program, 'HEALPIX': healpix}
        if save_fits:
            fits_path = os.path.join(self.output_dir, f"spectrum_{tid}.fits")
            self.extractor.save_spectrum_fits(spectrum, info, fits_path)
            files['fits'] = fits_path
            self.log(f"  FITS: {fits_path}")

        if save_png:
            png_path = os.path.join(self.output_dir, f"spectrum_{tid}.png")
            self.extractor.plot_spectrum(spectrum, info,
                                        save_path=png_path, show=show_plot)
            files['png'] = png_path
            self.log(f"  PNG:  {png_path}")

        self.log(f"\n完成!")
        return {
            'match': match,
            'spectrum': spectrum,
            'files': files,
        }

    def process_batch(self, input_csv, ra_col=None, dec_col=None,
                      radius_arcsec=MATCH_RADIUS_ARCSEC,
                      show_plot=False, save_fits=True, save_png=True,
                      cli_progress=False):
        """
        批量处理：从 CSV 读取多个坐标。

        Returns:
            pd.DataFrame: 汇总表
        """
        self.log(f"\n{'='*60}")
        self.log(f"批量处理: {input_csv}")
        self.log(f"{'='*60}")

        # 读取输入 CSV
        df = pd.read_csv(input_csv)
        self.log(f"  输入: {len(df)} 个目标")

        # 自动检测列名
        ra_col = ra_col or self._detect_col(df, ['ra', 'RA', 'Ra',
                                                   'RIGHT_ASCENSION', 'right_ascension'])
        dec_col = dec_col or self._detect_col(df, ['dec', 'DEC', 'Dec', 'DE',
                                                    'DECLINATION', 'declination'])
        if not ra_col or not dec_col:
            self.log(f"  错误: 无法识别 RA/DEC 列。可用列: {list(df.columns)}")
            return None

        self.log(f"  RA列={ra_col}, DEC列={dec_col}")

        ra_arr = df[ra_col].values
        dec_arr = df[dec_col].values

        # 1. 批量匹配
        self.log("\n[1/3] 批量坐标匹配...")
        t0 = time.time()
        matches = self.index.query_batch(ra_arr, dec_arr, radius_arcsec)
        dt = time.time() - t0
        n_matched = sum(1 for m in matches if m is not None)
        self.log(f"  匹配: {n_matched}/{len(matches)}, 耗时 {dt:.3f}s")

        if n_matched == 0:
            self.log("  无匹配结果")
            return None

        # 2. 下载 coadd 文件 (survey/program/healpix 已在 match 中)
        self.log("\n[2/3] 下载 coadd 文件...")
        targets_info = [{'SURVEY': m['survey'], 'PROGRAM': m['program'],
                         'HEALPIX': m['healpix']}
                        for m in matches if m is not None]
        coadd_map = self.downloader.download_coadds(
            targets_info, cli_progress=cli_progress)

        # 3. 提取光谱
        self.log("\n[3/3] 提取光谱...")
        summary_rows = []
        pbar = None
        if cli_progress:
            pbar = tqdm(total=len(matches), desc="提取光谱", unit="target")

        for i, match in enumerate(matches):
            row = {'input_ra': ra_arr[i], 'input_dec': dec_arr[i]}

            if match is None:
                row['status'] = 'no_match'
                row['targetid'] = None
                row['separation_arcsec'] = None
                summary_rows.append(row)
                if pbar:
                    pbar.update(1)
                continue

            tid = match['targetid']
            row['targetid'] = tid
            row['separation_arcsec'] = match['separation_arcsec']

            survey = match['survey']
            program = match['program']
            healpix = match['healpix']
            row['survey'] = survey
            row['program'] = program
            row['healpix'] = healpix

            coadd_key = (survey, program, healpix)
            coadd_path = coadd_map.get(coadd_key)
            if coadd_path is None:
                row['status'] = 'download_failed'
                summary_rows.append(row)
                if pbar:
                    pbar.update(1)
                continue

            spectrum = self.extractor.extract_from_coadd(coadd_path, tid)
            if spectrum is None:
                row['status'] = 'extract_failed'
                summary_rows.append(row)
                if pbar:
                    pbar.update(1)
                continue

            if save_fits:
                fp = os.path.join(self.output_dir, f"spectrum_{tid}.fits")
                info = {'SURVEY': survey, 'PROGRAM': program, 'HEALPIX': healpix}
                self.extractor.save_spectrum_fits(spectrum, info, fp)

            if save_png:
                pp = os.path.join(self.output_dir, f"spectrum_{tid}.png")
                self.extractor.plot_spectrum(spectrum, None,
                                            save_path=pp, show=show_plot)

            row['status'] = 'ok'
            summary_rows.append(row)

            if self.progress_callback:
                self.progress_callback('batch', i + 1, len(matches))
            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()

        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(self.output_dir, 'summary.csv')
        summary_df.to_csv(summary_path, index=False)
        self.log(f"\n汇总表已保存: {summary_path}")

        ok = sum(1 for r in summary_rows if r.get('status') == 'ok')
        self.log(f"完成: 成功 {ok}/{len(matches)}")
        return summary_df

    @staticmethod
    def _detect_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None


# ================================================================
#  DESIToolGUI - Tkinter 图形界面
# ================================================================

class DESIToolGUI:
    """Tkinter GUI，支持单目标和批量模式。"""

    def __init__(self, root):
        self.root = root
        self.root.title("DESI MWS 光谱工具")
        self.root.geometry("900x720")
        self.root.minsize(800, 600)

        self.tool = None
        self._worker_thread = None
        self._msg_queue = queue.Queue()
        self._cancel_event = threading.Event()

        self._build_ui()
        self._poll_queue()

    def _build_ui(self):
        import tkinter as tk
        from tkinter import ttk

        style = ttk.Style()
        try:
            style.theme_use('aqua')
        except Exception:
            try:
                style.theme_use('clam')
            except Exception:
                pass

        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # ---- 设置区 ----
        settings_frame = ttk.LabelFrame(main, text="设置", padding=5)
        settings_frame.pack(fill=tk.X, pady=(0, 5))

        r = 0
        ttk.Label(settings_frame, text="压缩星表:").grid(row=r, column=0, sticky=tk.W)
        self.var_compressed = tk.StringVar(value=COMPRESSED_CSV)
        ttk.Entry(settings_frame, textvariable=self.var_compressed, width=50).grid(
            row=r, column=1, sticky=tk.EW, padx=3)
        ttk.Button(settings_frame, text="浏览",
                   command=lambda: self._browse_file(self.var_compressed)).grid(
            row=r, column=2)

        r = 1
        ttk.Label(settings_frame, text="完整星表:").grid(row=r, column=0, sticky=tk.W)
        self.var_full = tk.StringVar(value=FULL_CSV)
        ttk.Entry(settings_frame, textvariable=self.var_full, width=50).grid(
            row=r, column=1, sticky=tk.EW, padx=3)
        ttk.Button(settings_frame, text="浏览",
                   command=lambda: self._browse_file(self.var_full)).grid(
            row=r, column=2)

        r = 2
        ttk.Label(settings_frame, text="输出目录:").grid(row=r, column=0, sticky=tk.W)
        self.var_output = tk.StringVar(value=OUTPUT_DIR)
        ttk.Entry(settings_frame, textvariable=self.var_output, width=50).grid(
            row=r, column=1, sticky=tk.EW, padx=3)
        ttk.Button(settings_frame, text="浏览",
                   command=lambda: self._browse_dir(self.var_output)).grid(
            row=r, column=2)

        r = 3
        ttk.Label(settings_frame, text="匹配半径(arcsec):").grid(row=r, column=0, sticky=tk.W)
        self.var_radius = tk.StringVar(value="5.0")
        ttk.Entry(settings_frame, textvariable=self.var_radius, width=10).grid(
            row=r, column=1, sticky=tk.W, padx=3)

        ttk.Label(settings_frame, text="下载线程数:").grid(row=r, column=1, sticky=tk.E)
        self.var_workers = tk.StringVar(value="4")
        ttk.Entry(settings_frame, textvariable=self.var_workers, width=5).grid(
            row=r, column=2, sticky=tk.W)

        settings_frame.columnconfigure(1, weight=1)

        # ---- 标签页 ----
        self.notebook = ttk.Notebook(main)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=5)

        # == 单目标标签页 ==
        single_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(single_frame, text="  单目标  ")

        coord_frame = ttk.Frame(single_frame)
        coord_frame.pack(fill=tk.X)
        ttk.Label(coord_frame, text="RA (deg):").pack(side=tk.LEFT)
        self.var_ra = tk.StringVar()
        ttk.Entry(coord_frame, textvariable=self.var_ra, width=15).pack(
            side=tk.LEFT, padx=5)
        ttk.Label(coord_frame, text="DEC (deg):").pack(side=tk.LEFT, padx=(10, 0))
        self.var_dec = tk.StringVar()
        ttk.Entry(coord_frame, textvariable=self.var_dec, width=15).pack(
            side=tk.LEFT, padx=5)

        self.btn_single = ttk.Button(single_frame, text="开始查询",
                                     command=self._run_single)
        self.btn_single.pack(pady=10)

        # == 批量标签页 ==
        batch_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(batch_frame, text="  批量处理  ")

        csv_frame = ttk.Frame(batch_frame)
        csv_frame.pack(fill=tk.X)
        ttk.Label(csv_frame, text="输入CSV:").pack(side=tk.LEFT)
        self.var_csv = tk.StringVar()
        ttk.Entry(csv_frame, textvariable=self.var_csv, width=50).pack(
            side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(csv_frame, text="浏览",
                   command=lambda: self._browse_file(self.var_csv)).pack(side=tk.LEFT)

        col_frame = ttk.Frame(batch_frame)
        col_frame.pack(fill=tk.X, pady=5)
        ttk.Label(col_frame, text="RA列名:").pack(side=tk.LEFT)
        self.var_ra_col = tk.StringVar(value="ra")
        ttk.Entry(col_frame, textvariable=self.var_ra_col, width=10).pack(
            side=tk.LEFT, padx=5)
        ttk.Label(col_frame, text="DEC列名:").pack(side=tk.LEFT, padx=(10, 0))
        self.var_dec_col = tk.StringVar(value="dec")
        ttk.Entry(col_frame, textvariable=self.var_dec_col, width=10).pack(
            side=tk.LEFT, padx=5)
        ttk.Label(col_frame, text="(留空自动检测)").pack(side=tk.LEFT)

        self.btn_batch = ttk.Button(batch_frame, text="开始批量处理",
                                    command=self._run_batch)
        self.btn_batch.pack(pady=10)

        # ---- 进度区 ----
        progress_frame = ttk.LabelFrame(main, text="进度", padding=5)
        progress_frame.pack(fill=tk.X, pady=5)

        import tkinter as tk_core
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress_bar.pack(fill=tk.X)

        status_row = ttk.Frame(progress_frame)
        status_row.pack(fill=tk.X, pady=(3, 0))
        self.var_status = tk.StringVar(value="就绪")
        ttk.Label(status_row, textvariable=self.var_status).pack(side=tk.LEFT)
        self.btn_cancel = ttk.Button(status_row, text="取消",
                                     command=self._cancel, state=tk.DISABLED)
        self.btn_cancel.pack(side=tk.RIGHT)

        # ---- 日志区 ----
        log_frame = ttk.LabelFrame(main, text="日志", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.log_text = tk_core.Text(log_frame, height=10, wrap=tk.WORD,
                                      font=('Menlo', 11))
        scroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # ---- 底部按钮 ----
        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill=tk.X)
        ttk.Button(btn_frame, text="打开输出目录",
                   command=self._open_output).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="构建索引",
                   command=self._build_index).pack(side=tk.LEFT, padx=5)

    def _browse_file(self, var):
        from tkinter import filedialog
        path = filedialog.askopenfilename()
        if path:
            var.set(path)

    def _browse_dir(self, var):
        from tkinter import filedialog
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    def _log_gui(self, msg):
        self._msg_queue.put(('log', msg))

    def _set_status(self, msg):
        self._msg_queue.put(('status', msg))

    def _set_progress(self, value):
        self._msg_queue.put(('progress', value))

    def _poll_queue(self):
        try:
            while True:
                msg_type, data = self._msg_queue.get_nowait()
                if msg_type == 'log':
                    self.log_text.insert('end', data + '\n')
                    self.log_text.see('end')
                elif msg_type == 'status':
                    self.var_status.set(data)
                elif msg_type == 'progress':
                    self.progress_bar['value'] = data
                elif msg_type == 'done':
                    self._on_done()
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _get_tool(self):
        if self.tool is None:
            self._log_gui("初始化工具...")
            self.tool = DESITool(
                output_dir=self.var_output.get(),
                compressed_csv=self.var_compressed.get(),
                full_csv=self.var_full.get(),
                max_workers=int(self.var_workers.get()),
                log_func=self._log_gui,
            )
        return self.tool

    def _set_busy(self, busy):
        import tkinter as tk
        state = tk.DISABLED if busy else tk.NORMAL
        self.btn_single['state'] = state
        self.btn_batch['state'] = state
        self.btn_cancel['state'] = tk.NORMAL if busy else tk.DISABLED
        if busy:
            self._cancel_event.clear()

    def _cancel(self):
        self._cancel_event.set()
        self._set_status("正在取消...")

    def _on_done(self):
        self._set_busy(False)
        self._set_status("完成")

    def _run_single(self):
        try:
            ra = float(self.var_ra.get())
            dec = float(self.var_dec.get())
        except ValueError:
            self._log_gui("错误: RA 和 DEC 必须是数字")
            return

        self._set_busy(True)
        self._set_status("处理中...")
        self.progress_bar['value'] = 0
        self.log_text.delete('1.0', 'end')

        def worker():
            try:
                tool = self._get_tool()
                radius = float(self.var_radius.get())
                result = tool.process_single(ra, dec, radius_arcsec=radius,
                                             show_plot=False,
                                             save_fits=True, save_png=True)
                if result:
                    self._set_progress(100)
                    self._set_status(
                        f"成功! TARGETID={result['match']['targetid']}")
                else:
                    self._set_status("未找到匹配")
            except Exception as e:
                self._log_gui(f"\n错误: {e}")
                self._set_status(f"错误: {e}")
            finally:
                self._msg_queue.put(('done', None))

        self._worker_thread = threading.Thread(target=worker, daemon=True)
        self._worker_thread.start()

    def _run_batch(self):
        csv_path = self.var_csv.get()
        if not csv_path or not os.path.exists(csv_path):
            self._log_gui("错误: 请选择有效的 CSV 文件")
            return

        self._set_busy(True)
        self._set_status("批量处理中...")
        self.progress_bar['value'] = 0
        self.log_text.delete('1.0', 'end')

        def progress_cb(name, current, total):
            if total > 0:
                pct = current / total * 100
                self._set_progress(pct)
                self._set_status(f"处理中 {current}/{total} ...")

        def worker():
            try:
                tool = self._get_tool()
                tool.progress_callback = progress_cb
                radius = float(self.var_radius.get())
                ra_col = self.var_ra_col.get().strip() or None
                dec_col = self.var_dec_col.get().strip() or None
                summary = tool.process_batch(
                    csv_path, ra_col=ra_col, dec_col=dec_col,
                    radius_arcsec=radius, show_plot=False)
                if summary is not None:
                    ok = len(summary[summary['status'] == 'ok'])
                    self._set_progress(100)
                    self._set_status(f"完成: {ok}/{len(summary)} 成功")
                else:
                    self._set_status("处理失败")
            except Exception as e:
                self._log_gui(f"\n错误: {e}")
                self._set_status(f"错误: {e}")
            finally:
                self._msg_queue.put(('done', None))

        self._worker_thread = threading.Thread(target=worker, daemon=True)
        self._worker_thread.start()

    def _open_output(self):
        path = self.var_output.get()
        if os.path.isdir(path):
            subprocess.Popen(['open', path])

    def _build_index(self):
        self._set_busy(True)
        self._set_status("构建索引...")
        self.log_text.delete('1.0', 'end')

        def worker():
            try:
                cache = INDEX_CACHE
                if os.path.exists(cache):
                    os.remove(cache)
                self.tool = None
                self._get_tool()
                self._set_progress(100)
                self._set_status("索引构建完成!")
            except Exception as e:
                self._log_gui(f"\n错误: {e}")
                self._set_status(f"错误: {e}")
            finally:
                self._msg_queue.put(('done', None))

        self._worker_thread = threading.Thread(target=worker, daemon=True)
        self._worker_thread.start()


# ================================================================
#  CLI 入口
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description='DESI MWS 光谱一体化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                                    # 启动 GUI
  %(prog)s single --ra 190.305 --dec 2.596    # 单目标查询
  %(prog)s batch targets.csv                  # 批量处理
  %(prog)s batch targets.csv --workers 8      # 8线程批量
  %(prog)s build-index                        # 预构建索引
        """)

    subparsers = parser.add_subparsers(dest='command')

    # gui (默认)
    subparsers.add_parser('gui', help='启动 GUI 界面')

    # single
    sp_single = subparsers.add_parser('single', help='单目标查询')
    sp_single.add_argument('--ra', type=float, required=True, help='赤经 (度)')
    sp_single.add_argument('--dec', type=float, required=True, help='赤纬 (度)')
    sp_single.add_argument('--radius', type=float, default=MATCH_RADIUS_ARCSEC,
                           help=f'匹配半径 (角秒, 默认{MATCH_RADIUS_ARCSEC})')
    sp_single.add_argument('--show', action='store_true', help='显示光谱图')

    # batch
    sp_batch = subparsers.add_parser('batch', help='批量处理')
    sp_batch.add_argument('input_csv', help='输入CSV文件')
    sp_batch.add_argument('--ra-col', default=None, help='RA列名 (自动检测)')
    sp_batch.add_argument('--dec-col', default=None, help='DEC列名 (自动检测)')
    sp_batch.add_argument('--workers', type=int, default=MAX_WORKERS,
                          help=f'下载线程数 (默认{MAX_WORKERS})')
    sp_batch.add_argument('--radius', type=float, default=MATCH_RADIUS_ARCSEC,
                          help=f'匹配半径 (角秒, 默认{MATCH_RADIUS_ARCSEC})')

    # build-index
    subparsers.add_parser('build-index', help='预构建 KD-Tree 空间索引')

    # 公共参数
    parser.add_argument('--output', '-o', default=OUTPUT_DIR, help='输出目录')
    parser.add_argument('--compressed-csv', default=COMPRESSED_CSV)
    parser.add_argument('--full-csv', default=FULL_CSV)

    args = parser.parse_args()

    # 默认启动 GUI
    if args.command is None or args.command == 'gui':
        _launch_gui()
        return

    if args.command == 'build-index':
        print("构建 KD-Tree 索引...")
        if os.path.exists(INDEX_CACHE):
            os.remove(INDEX_CACHE)
        CatalogIndex(
            compressed_csv=args.compressed_csv,
            full_csv=args.full_csv,
        )
        print("完成!")
        return

    # single / batch 需要创建 DESITool
    workers = getattr(args, 'workers', MAX_WORKERS)
    tool = DESITool(
        output_dir=args.output,
        compressed_csv=args.compressed_csv,
        full_csv=args.full_csv,
        max_workers=workers,
    )

    if args.command == 'single':
        result = tool.process_single(
            args.ra, args.dec,
            radius_arcsec=args.radius,
            show_plot=args.show)
        if result is None:
            sys.exit(1)

    elif args.command == 'batch':
        summary = tool.process_batch(
            args.input_csv,
            ra_col=args.ra_col,
            dec_col=args.dec_col,
            radius_arcsec=args.radius,
            cli_progress=True)
        if summary is None:
            sys.exit(1)


def _launch_gui():
    import tkinter as tk
    root = tk.Tk()
    DESIToolGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
