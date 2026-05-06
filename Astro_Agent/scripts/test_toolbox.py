#!/usr/bin/env python3
"""
astro_toolbox 一键查询/保存/画图
=================================
输入 RA/DEC，并行查询所有波段数据，保存结果并绘图。

用法:
    python scripts/test_toolbox.py --ra 190.305 --dec 2.596
    python scripts/test_toolbox.py --csv data/matched_tier_with_desi.csv
"""
import os
import sys
import time
import argparse
import traceback
import warnings
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# 项目根目录 (scripts/ 的上级)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 输出目录
OUTPUT_BASE = os.path.join(PROJECT_ROOT, 'output', 'astro_output')


def mjd_to_date(mjd):
    """MJD 转日期字符串"""
    try:
        from astropy.time import Time
        return Time(mjd, format='mjd').iso[:10]
    except Exception:
        return f"MJD {mjd:.1f}"


class AstroQueryAll:
    """一键查询所有波段并保存/绘图"""

    def __init__(self, ra, dec, output_dir=None, enabled_modules=None):
        self.ra = ra
        self.dec = dec
        self.output_dir = output_dir or os.path.join(
            OUTPUT_BASE, f"RA{ra:.4f}_DEC{dec:.4f}")
        os.makedirs(self.output_dir, exist_ok=True)
        self.results = {}
        self.timings = {}
        self.status_callback = None  # callable(name, status, result_or_error, elapsed)
        self.enabled_modules = set(enabled_modules) if enabled_modules else None

    def _run(self, name, func):
        """运行单个查询模块"""
        if self.status_callback:
            self.status_callback(name, 'querying', None, 0)
        t0 = time.time()
        try:
            result = func()
            dt = time.time() - t0
            self.timings[name] = dt
            self.results[name] = result
            status = 'ok' if result else 'no_data'
            try:
                print(f"  [{status:>8s}] {name:15s} ({dt:.1f}s)")
            except (ValueError, OSError):
                pass
            if self.status_callback:
                self.status_callback(name, status, result, dt)
            return result
        except Exception as e:
            dt = time.time() - t0
            self.timings[name] = dt
            self.results[name] = None
            try:
                print(f"  [{'error':>8s}] {name:15s} ({dt:.1f}s) - {e}")
            except (ValueError, OSError):
                pass
            if self.status_callback:
                self.status_callback(name, 'error', str(e), dt)
            return None

    def query_all(self):
        """并行查询所有模块"""
        print(f"\n{'='*70}")
        print(f"  目标: RA={self.ra:.6f}  DEC={self.dec:.6f}")
        print(f"  输出: {self.output_dir}")
        print(f"{'='*70}\n")

        t_total = time.time()

        # 定义所有查询任务
        tasks = {
            'SDSS_spectrum':  self._query_sdss_spectrum,
            'SDSS_photometry': self._query_sdss_photometry,
            'GALAH':          self._query_galah,
            'LAMOST':         self._query_lamost,
            'DESI':           self._query_desi,
            'KOA_spectrum':   self._query_koa_spectrum,
            'SPHEREx':        self._query_spherex,
            'ZTF_lightcurve': self._query_ztf,
            'WISE_photometry': self._query_wise_phot,
            'WISE_lightcurve': self._query_wise_lc,
            'GALEX':          self._query_galex,
            '2MASS':          self._query_twomass,
            'X-ray':          self._query_xray,
            'HEASARC_Xray':   self._query_heasarc_xray,
            'Gaia_lightcurve': self._query_gaia_lc,
            'TESS':           self._query_tess,
            'Kepler/K2':      self._query_kepler,
            'HST_spectrum':   self._query_hst_spectrum,
            'HST_lightcurve': self._query_hst_lc,
            'JWST_spectrum':  self._query_jwst_spectrum,
            'JWST_lightcurve':self._query_jwst_lc,
            'SED':            self._query_sed,
            'HR_diagram':     self._query_hr,
            'Binary_SED':     self._query_binary_sed,
            'SIMBAD_refs':    self._query_simbad_refs,
        }

        # 按 enabled_modules 过滤
        if self.enabled_modules:
            tasks = {k: v for k, v in tasks.items()
                     if k in self.enabled_modules}

        print(f"并行查询 {len(tasks)} 个模块...\n")

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(self._run, name, func): name
                       for name, func in tasks.items()}
            for future in as_completed(futures):
                future.result()  # 异常已在 _run 中处理

        total_time = time.time() - t_total
        print(f"\n总耗时: {total_time:.1f}s")
        return self.results

    # ======================= 查询函数 =======================

    def _query_sdss_spectrum(self):
        from astro_toolbox import sdss
        return sdss.query_spectrum(self.ra, self.dec)

    def _query_sdss_photometry(self):
        from astro_toolbox import sdss
        return sdss.get_photometry(self.ra, self.dec)

    def _query_galah(self):
        from astro_toolbox import galah
        return galah.query_spectrum(self.ra, self.dec)

    def _query_lamost(self):
        from astro_toolbox import lamost
        return lamost.query_spectrum(self.ra, self.dec)

    def _query_desi(self):
        from astro_toolbox import desi
        return desi.query_spectrum(self.ra, self.dec,
                                   save_fits=False, save_png=False)

    def _query_koa_spectrum(self):
        from astro_toolbox import koa
        return koa.query_spectrum(self.ra, self.dec)

    def _query_spherex(self):
        from astro_toolbox import spherex
        return spherex.query_spectrum(self.ra, self.dec)

    def _query_ztf(self):
        from astro_toolbox import ztf
        return ztf.query_lightcurve(self.ra, self.dec)

    def _query_wise_phot(self):
        from astro_toolbox import wise
        return wise.get_photometry(self.ra, self.dec)

    def _query_wise_lc(self):
        from astro_toolbox import wise
        return wise.query_lightcurve(self.ra, self.dec)

    def _query_galex(self):
        from astro_toolbox import galex
        return galex.get_photometry(self.ra, self.dec)

    def _query_twomass(self):
        from astro_toolbox import twomass
        return twomass.get_photometry(self.ra, self.dec)

    def _query_xray(self):
        from astro_toolbox import xray
        return xray.query_xray(self.ra, self.dec)

    def _query_heasarc_xray(self):
        from astro_toolbox import xray
        return xray.query_heasarc_browse(self.ra, self.dec)

    def _query_gaia_lc(self):
        from astro_toolbox import gaia_lc
        return gaia_lc.query_lightcurve(self.ra, self.dec)

    def _query_tess(self):
        from astro_toolbox import tess
        return tess.query_lightcurve(self.ra, self.dec)

    def _query_kepler(self):
        from astro_toolbox import kepler
        for mission in ('Kepler', 'K2'):
            r = kepler.query_lightcurve(self.ra, self.dec, mission=mission)
            if r:
                return r
        return None

    def _query_hst_spectrum(self):
        from astro_toolbox import hst
        return hst.query_spectrum(self.ra, self.dec)

    def _query_hst_lc(self):
        from astro_toolbox import hst
        return hst.query_lightcurve(self.ra, self.dec)

    def _query_jwst_spectrum(self):
        from astro_toolbox import jwst
        return jwst.query_spectrum(self.ra, self.dec)

    def _query_jwst_lc(self):
        from astro_toolbox import jwst
        return jwst.query_lightcurve(self.ra, self.dec)

    def _query_sed(self):
        # SED 不再重复查询, 在 _save_sed 中使用已有测光结果组装
        return 'deferred'

    def _query_hr(self):
        from astro_toolbox.hr_diagram import HRDiagram
        hr = HRDiagram()
        params = hr._query_gaia_params(self.ra, self.dec)
        analysis = params.get('hr_analysis') if params else None
        return {'hr': hr, 'params': params, 'analysis': analysis}

    def _query_binary_sed(self):
        """双星 SED 拟合已移除"""
        return None

    def _query_simbad_refs(self):
        from astro_toolbox import utils
        return utils.query_simbad_references(self.ra, self.dec)

    # ======================= 保存和绘图 =======================

    def save_and_plot_all(self):
        """保存所有数据并绘图"""
        print(f"\n{'='*70}")
        print(f"  保存数据与绘图")
        print(f"{'='*70}\n")

        self._save_sdss_spectrum()
        self._save_lamost()
        self._save_desi()
        self._save_koa_spectrum()
        self._save_spherex()
        self._save_ztf()
        self._save_wise_lc()
        self._save_gaia_lc()
        self._save_tess()
        self._save_kepler()
        self._save_hst_spectrum()
        self._save_hst_lc()
        self._save_jwst_spectrum()
        self._save_jwst_lc()
        self._save_spectral_diagnostics()
        self._save_sed()
        self._save_hr()
        self._save_binary_sed()
        self._save_simbad_refs()
        self._save_rv_analysis()
        self._save_orbit_traceback()
        self._save_period_analysis()
        self._save_xray_analysis()
        self._save_combined_plots()
        self._save_summary()

        print(f"\n所有输出保存到: {self.output_dir}")

    def _save_sdss_spectrum(self):
        r = self.results.get('SDSS_spectrum')
        if not r:
            return
        from astro_toolbox import sdss
        path = os.path.join(self.output_dir, 'sdss_spectrum.png')
        sdss.plot_spectrum(r, save_path=path)
        # 保存数据
        df = pd.DataFrame({
            'wavelength_A': r['wavelength'],
            'flux': r['flux'],
            'error': r['error'],
        })
        csv_path = os.path.join(self.output_dir, 'sdss_spectrum.csv')
        df.to_csv(csv_path, index=False)
        obs = f"MJD={r['obs_mjd']} ({mjd_to_date(r['obs_mjd'])})" if 'obs_mjd' in r else ''
        print(f"  SDSS光谱: {path}  {obs}")

    def _save_desi(self):
        r = self.results.get('DESI')
        if not r or 'spectrum' not in r:
            return
        from astro_toolbox.desi import SpectrumExtractor
        sp = r['spectrum']
        # 绘图
        png_path = os.path.join(self.output_dir, 'desi_spectrum.png')
        SpectrumExtractor.plot_spectrum(sp, save_path=png_path)
        # 保存 CSV
        rows = []
        for band in ('B', 'R', 'Z'):
            w = sp[band]['wavelength']
            f = sp[band]['flux']
            e = sp[band]['error']
            for i in range(len(w)):
                rows.append({'band': band, 'wavelength_A': w[i],
                             'flux': f[i], 'error': e[i]})
        csv_path = os.path.join(self.output_dir, 'desi_spectrum.csv')
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        obs_info = ''
        if sp.get('obs_mjd'):
            obs_info = f"  MJD={sp['obs_mjd']:.1f} ({mjd_to_date(sp['obs_mjd'])})"
        if sp.get('obs_nights'):
            obs_info += f"  nights={sp['obs_nights']}"
        print(f"  DESI光谱: {png_path}{obs_info}")

    def _save_lamost(self):
        r = self.results.get('LAMOST')
        if not r:
            return
        from astro_toolbox import lamost
        if 'wavelength' in r:
            path = os.path.join(self.output_dir, 'lamost_spectrum.png')
            lamost.plot_spectrum(r, save_path=path)
        else:
            path = ''
        lamost.save_csv(r, self.output_dir)
        obs = f"  MJD={r['obs_mjd']}" if r.get('obs_mjd') else ''
        print(f"  LAMOST: {path}{obs}")

    def _save_koa_spectrum(self):
        r = self.results.get('KOA_spectrum')
        if not r:
            return
        from astro_toolbox import koa
        path = os.path.join(self.output_dir, 'koa_spectrum.png')
        koa.plot_spectrum(r, save_path=path)
        csv_path = koa.save_csv(r, self.output_dir)
        exp_path = koa.save_exposure_table(r, self.output_dir)
        report_path = koa.save_report(r, self.output_dir)
        obs = ''
        if np.isfinite(r.get('obs_mjd_min', np.nan)):
            obs = (f"  MJD={r.get('obs_mjd_min'):.5f}"
                   f"-{r.get('obs_mjd_max'):.5f}")
        saved = [p for p in (csv_path, exp_path, report_path) if p]
        print(f"  KOA/LRIS光谱: {path}  {r.get('n_files',0)} files"
              f"  arms={r.get('arms','')}{obs}  reports={len(saved)}")

    def _save_spherex(self):
        r = self.results.get('SPHEREx')
        if not r or 'wavelength' not in r:
            return
        from astro_toolbox import spherex
        path = os.path.join(self.output_dir, 'spherex_spectrum.png')
        spherex.plot_spectrum(r, save_path=path)
        df = pd.DataFrame({
            'wavelength_A': r['wavelength'],
            'flux': r['flux'],
        })
        if 'error' in r:
            df['error'] = r['error']
        csv_path = os.path.join(self.output_dir, 'spherex_spectrum.csv')
        df.to_csv(csv_path, index=False)
        print(f"  SPHEREx光谱: {path}  {r.get('n_channels', 0)} channels")

    def _save_spectral_diagnostics(self):
        try:
            from astro_toolbox.diagnostics import (
                analyze_all_spectra, save_spectral_diagnostics)
            diagnostics = analyze_all_spectra(self.results)
            if not diagnostics:
                return
            self.results['spectral_diagnostics'] = diagnostics
            csv_path, txt_path = save_spectral_diagnostics(
                diagnostics, self.output_dir)
            n_flagged = sum(bool(d.get('flags')) for d in diagnostics.values())
            print(f"  光谱诊断: {len(diagnostics)} spectra, "
                  f"{n_flagged} flagged -> {os.path.basename(txt_path)}")
        except Exception as e:
            print(f"  光谱诊断失败: {e}")


    def _save_ztf(self):
        r = self.results.get('ZTF_lightcurve')
        if not r:
            return
        from astro_toolbox import ztf
        path = os.path.join(self.output_dir, 'ztf_lightcurve.png')
        ztf.plot_lightcurve(r, save_path=path)
        # 保存各波段数据
        all_dfs = []
        for band in ('g', 'r', 'i', 'all'):
            if band in r and hasattr(r[band], 'columns'):
                df = r[band].copy()
                df['band'] = band
                all_dfs.append(df)
        if all_dfs:
            csv_path = os.path.join(self.output_dir, 'ztf_lightcurve.csv')
            pd.concat(all_dfs).to_csv(csv_path, index=False)
        obs_info = ''
        if r.get('obs_mjd_min') and r.get('obs_mjd_max'):
            obs_info = (f"  {mjd_to_date(r['obs_mjd_min'])} ~ "
                        f"{mjd_to_date(r['obs_mjd_max'])}, "
                        f"{r.get('n_epochs',0)} epochs")
        print(f"  ZTF光变: {path}{obs_info}")

    def _save_wise_lc(self):
        r = self.results.get('WISE_lightcurve')
        if not r:
            return
        from astro_toolbox import wise
        path = os.path.join(self.output_dir, 'wise_lightcurve.png')
        wise.plot_lightcurve(r, save_path=path)
        all_dfs = []
        for band in ('W1', 'W2'):
            if band in r and hasattr(r[band], 'columns'):
                df = r[band].copy()
                df['band'] = band
                all_dfs.append(df)
        if all_dfs:
            csv_path = os.path.join(self.output_dir, 'wise_lightcurve.csv')
            pd.concat(all_dfs).to_csv(csv_path, index=False)
        obs_info = ''
        if r.get('obs_mjd_min') and r.get('obs_mjd_max'):
            obs_info = (f"  {mjd_to_date(r['obs_mjd_min'])} ~ "
                        f"{mjd_to_date(r['obs_mjd_max'])}, "
                        f"{r.get('n_epochs',0)} epochs")
        print(f"  WISE光变: {path}{obs_info}")

    def _save_gaia_lc(self):
        r = self.results.get('Gaia_lightcurve')
        if not r:
            return
        from astro_toolbox import gaia_lc
        path = os.path.join(self.output_dir, 'gaia_lightcurve.png')
        gaia_lc.plot_lightcurve(r, save_path=path)
        all_dfs = []
        for band in ('G', 'BP', 'RP'):
            if band in r and hasattr(r[band], 'columns'):
                df = r[band].copy()
                df['band'] = band
                all_dfs.append(df)
        if all_dfs:
            csv_path = os.path.join(self.output_dir, 'gaia_lightcurve.csv')
            pd.concat(all_dfs).to_csv(csv_path, index=False)
        print(f"  Gaia光变: {path}")

    def _save_tess(self):
        r = self.results.get('TESS')
        if not r:
            return
        from astro_toolbox import tess
        path = os.path.join(self.output_dir, 'tess_lightcurve.png')
        tess.plot_lightcurve(r, save_path=path)
        df = pd.DataFrame({'time_BTJD': r['time'], 'flux': r['flux'],
                            'flux_err': r['flux_err']})
        csv_path = os.path.join(self.output_dir, 'tess_lightcurve.csv')
        df.to_csv(csv_path, index=False)
        print(f"  TESS光变: {path}  {r.get('n_points',0)} pts, sectors={r.get('sectors','')}")

    def _save_kepler(self):
        r = self.results.get('Kepler/K2')
        if not r:
            return
        from astro_toolbox import kepler
        path = os.path.join(self.output_dir, 'kepler_lightcurve.png')
        kepler.plot_lightcurve(r, save_path=path)
        df = pd.DataFrame({'time': r['time'], 'flux': r['flux'],
                            'flux_err': r['flux_err']})
        csv_path = os.path.join(self.output_dir, 'kepler_lightcurve.csv')
        df.to_csv(csv_path, index=False)
        print(f"  Kepler光变: {path}  {r.get('n_points',0)} pts")

    def _save_hst_spectrum(self):
        r = self.results.get('HST_spectrum')
        if not r or 'wavelength' not in r:
            return
        from astro_toolbox import hst
        path = os.path.join(self.output_dir, 'hst_spectrum.png')
        hst.plot_spectrum(r, save_path=path)
        df = pd.DataFrame({
            'wavelength_A': r['wavelength'],
            'flux': r['flux'],
            'error': r['error'],
        })
        csv_path = os.path.join(self.output_dir, 'hst_spectrum.csv')
        df.to_csv(csv_path, index=False)
        obs_info = f"  {r.get('instrument','')} obs_id={r.get('obs_id','')}"
        if r.get('obs_mjd'):
            obs_info += f"  MJD={r['obs_mjd']:.1f}"
        print(f"  HST光谱: {path}{obs_info}")

    def _save_hst_lc(self):
        r = self.results.get('HST_lightcurve')
        if not r or not r.get('filters'):
            return
        from astro_toolbox import hst
        path = os.path.join(self.output_dir, 'hst_lightcurve.png')
        hst.plot_lightcurve(r, save_path=path)
        all_dfs = []
        for filt_name, df in r['filters'].items():
            df_out = df.copy()
            df_out['filter'] = filt_name
            all_dfs.append(df_out)
        if all_dfs:
            csv_path = os.path.join(self.output_dir, 'hst_lightcurve.csv')
            pd.concat(all_dfs).to_csv(csv_path, index=False)
        print(f"  HST光变: {path}  {len(r['filters'])} filters, "
              f"{r.get('n_epochs',0)} epochs")

    def _save_jwst_spectrum(self):
        r = self.results.get('JWST_spectrum')
        if not r or 'wavelength' not in r:
            return
        from astro_toolbox import jwst
        path = os.path.join(self.output_dir, 'jwst_spectrum.png')
        jwst.plot_spectrum(r, save_path=path)
        df = pd.DataFrame({
            'wavelength_A': r['wavelength'],
            'flux': r['flux'],
            'error': r['error'],
        })
        csv_path = os.path.join(self.output_dir, 'jwst_spectrum.csv')
        df.to_csv(csv_path, index=False)
        obs_info = f"  {r.get('instrument','')} {r.get('grating','')}"
        if r.get('obs_mjd'):
            obs_info += f"  MJD={r['obs_mjd']:.1f}"
        print(f"  JWST光谱: {path}{obs_info}")

    def _save_jwst_lc(self):
        r = self.results.get('JWST_lightcurve')
        if not r or not r.get('filters'):
            return
        from astro_toolbox import jwst
        path = os.path.join(self.output_dir, 'jwst_lightcurve.png')
        jwst.plot_lightcurve(r, save_path=path)
        all_dfs = []
        for filt_name, df in r['filters'].items():
            df_out = df.copy()
            df_out['filter'] = filt_name
            all_dfs.append(df_out)
        if all_dfs:
            csv_path = os.path.join(self.output_dir, 'jwst_lightcurve.csv')
            pd.concat(all_dfs).to_csv(csv_path, index=False)
        print(f"  JWST光变: {path}  {len(r['filters'])} filters, "
              f"{r.get('n_epochs',0)} epochs")

    def _save_sed(self):
        from astro_toolbox.sed import SEDFitter
        from astro_toolbox import config
        fitter = SEDFitter(self.ra, self.dec)

        # 直接使用已查询的测光结果, 不再重复查询
        # 注意: SPHEREx 存的是光谱, 需要转成测光
        spherex_phot = None
        spherex_spec = self.results.get('SPHEREx')
        if spherex_spec and 'wavelength' in spherex_spec:
            from astro_toolbox.spherex import get_photometry
            try:
                spherex_phot = get_photometry.__wrapped__(spherex_spec) if hasattr(get_photometry, '__wrapped__') else None
            except Exception:
                spherex_phot = None
            if spherex_phot is None:
                # 手动从光谱合成测光
                import numpy as np
                wave = spherex_spec['wavelength']
                flux = spherex_spec['flux']
                error = spherex_spec.get('error', np.zeros_like(flux))
                synth_bands = {
                    'SPHEREx_1.0': (7500, 12000, 9750),
                    'SPHEREx_1.5': (12000, 17000, 14500),
                    'SPHEREx_2.0': (17000, 25000, 21000),
                    'SPHEREx_3.0': (25000, 40000, 32500),
                    'SPHEREx_4.5': (40000, 51000, 45500),
                }
                spherex_phot = {}
                c_A = 2.99792458e18
                for bname, (w_lo, w_hi, cw) in synth_bands.items():
                    mask = (wave >= w_lo) & (wave <= w_hi) & np.isfinite(flux) & (flux > 0)
                    if np.sum(mask) >= 1:
                        mf = np.mean(flux[mask])
                        me = np.sqrt(np.mean(error[mask]**2)) if np.any(error[mask] > 0) else mf * 0.1
                        f_nu = mf * cw**2 / c_A * 1e23
                        if f_nu > 0:
                            mag = -2.5 * np.log10(f_nu / 3631.0)
                            mag_err = 2.5 / np.log(10) * (me * cw**2 / c_A * 1e23) / f_nu
                            if 0 < mag < 30:
                                spherex_phot[bname] = (mag, mag_err, cw)

        fitter.load_photometry(
            self.results.get('GALEX'),           # galex photometry
            self.results.get('SDSS_photometry'), # sdss photometry
            self.results.get('2MASS'),           # 2mass photometry
            self.results.get('WISE_photometry'), # wise photometry
            spherex_phot,                        # spherex photometry (从光谱合成)
        )

        # 如果 HR 图有 Gaia 测光, 也加入
        hr_result = self.results.get('HR_diagram')
        gaia_params_for_extinction = None
        if hr_result and hr_result.get('params'):
            p = hr_result['params']
            gaia_params_for_extinction = p
            gaia_phot = {}
            for band, key in [('Gaia_G', 'Gmag'), ('Gaia_BP', 'BPmag'),
                               ('Gaia_RP', 'RPmag')]:
                if key in p:
                    wave = config.BAND_INFO[band]['wave_A']
                    gaia_phot[band] = (p[key], 0.01, wave)
            if gaia_phot:
                fitter.load_photometry(gaia_phot)

        # 补查缺失巡天 (如果某个模块未启用或查询失败)
        # SPHEREx 已在上面手动合成, 不需重复网络查询
        fitter.collect_photometry(include_spherex=False)

        if len(fitter.photometry) == 0:
            return

        # 消光改正
        fitter.apply_extinction(gaia_params=gaia_params_for_extinction)
        fitter.analyze_excesses()

        # 绘图
        path = os.path.join(self.output_dir, 'sed.png')
        fitter.plot(save_path=path)

        # 保存 CSV
        fitter.save_csv(self.output_dir)
        fitter.save_diagnostics(self.output_dir)

        # 存入 results 供 combined_plots 使用
        self.results['SED'] = fitter

        ebv_str = f"  E(B-V)={fitter.ebv:.4f}" if fitter.ebv else ""
        sed_flags = ''
        if fitter.sed_diagnostics and fitter.sed_diagnostics.get('flags'):
            sed_flags = '  flags=' + ','.join(fitter.sed_diagnostics['flags'])
        print(f"  SED: {path}  {len(fitter.photometry)} 波段{ebv_str}{sed_flags}")

    def _save_hr(self):
        r = self.results.get('HR_diagram')
        if not r or r.get('params') is None:
            return
        from astro_toolbox import hr_diagram
        hr = r['hr']
        params = r['params']
        path = os.path.join(self.output_dir, 'hr_diagram.png')
        hr._make_plot([params], save_path=path, show_background=True,
                      annotate_regions=True)
        csv_path = hr_diagram.save_csv(params, self.output_dir)
        report_path = hr_diagram.save_analysis_report(params, self.output_dir)
        info = f"  BP-RP={params['BP_RP']:.3f}  M_G={params['M_G']:.3f}"
        if 'Teff' in params:
            info += f"  Teff={params['Teff']:.0f}K"
        analysis = params.get('hr_analysis')
        if analysis:
            info += f"  region={analysis.get('region_label', analysis.get('region'))}"
            wd = analysis.get('wd_model') or {}
            if wd.get('status') == 'ok':
                info += f"  t_cool={wd.get('cooling_age_myr', np.nan):.0f} Myr"
        saved = [p for p in (csv_path, report_path) if p]
        suffix = f"  reports={len(saved)}" if saved else ""
        print(f"  HR图: {path}{info}{suffix}")

    def _save_binary_sed(self):
        pass  # 双星 SED 拟合已移除

    def _save_simbad_refs(self):
        """保存 SIMBAD 文献引用信息"""
        r = self.results.get('SIMBAD_refs')
        if not r or not isinstance(r, dict):
            return

        refs = r.get('references', [])
        main_id = r.get('main_id', '?')
        otype = r.get('otype', '')
        n_total = r.get('n_refs', 0)

        # 保存为文本报告
        lines = []
        lines.append(f"# SIMBAD Literature Report")
        lines.append(f"# Target: RA={self.ra:.6f}, DEC={self.dec:.6f}")
        lines.append(f"# SIMBAD ID: {main_id}")
        lines.append(f"# Object type: {otype}")
        lines.append(f"# Total references in SIMBAD: {n_total}")
        lines.append(f"# Showing top {len(refs)} (most recent)")
        lines.append(f"")

        if not refs:
            lines.append("No references found in SIMBAD for this object.")
            lines.append("This object may not have been studied in the literature.")
        else:
            lines.append(f"This object ({main_id}, type={otype}) has been studied "
                         f"in {n_total} publication(s).\n")

            for i, ref in enumerate(refs, 1):
                lines.append(f"{'='*70}")
                lines.append(f"[{i}] {ref.get('bibcode', '')}")
                if ref.get('title'):
                    lines.append(f"    Title:   {ref['title']}")
                if ref.get('authors'):
                    lines.append(f"    Authors: {ref['authors']}")
                if ref.get('journal'):
                    lines.append(f"    Journal: {ref['journal']} ({ref.get('year', '')})")
                if ref.get('url'):
                    lines.append(f"    ADS:     {ref['url']}")
                if ref.get('abstract'):
                    lines.append(f"    Abstract:")
                    # 自动换行摘要
                    import textwrap
                    wrapped = textwrap.fill(ref['abstract'], width=72,
                                            initial_indent='      ',
                                            subsequent_indent='      ')
                    lines.append(wrapped)
                lines.append('')

        report_path = os.path.join(self.output_dir, 'simbad_references.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')

        # 同时保存 CSV 格式便于程序读取
        if refs:
            rows = []
            for ref in refs:
                rows.append({
                    'bibcode': ref.get('bibcode', ''),
                    'year': ref.get('year', ''),
                    'title': ref.get('title', ''),
                    'authors': ref.get('authors', ''),
                    'journal': ref.get('journal', ''),
                    'url': ref.get('url', ''),
                    'has_abstract': bool(ref.get('abstract', '')),
                })
            csv_path = os.path.join(self.output_dir, 'simbad_references.csv')
            pd.DataFrame(rows).to_csv(csv_path, index=False)

        status = f"已被研究, {n_total} 篇文献" if n_total > 0 else "未找到相关文献"
        print(f"  SIMBAD文献: {report_path}  ({main_id}: {status})")

    def _save_rv_analysis(self):
        """RV 径向速度拟合 (单星 + 双线双星 CCF)"""
        try:
            from astro_toolbox.rv_fitting import run_rv_analysis
            rv = run_rv_analysis(
                self.results,
                output_dir=self.output_dir,
                ra=self.ra, dec=self.dec,
            )
            if rv is not None:
                self.results['rv_analysis'] = rv
        except Exception as e:
            print(f"  RV 分析失败: {e}")
            import traceback
            traceback.print_exc()

    def _save_orbit_traceback(self):
        """轨道回溯到 Hunt+2023 星团"""
        rv_report = self.results.get('rv_analysis')
        if rv_report is None:
            return
        try:
            from astro_toolbox.orbit_traceback import run_traceback_analysis
            tb = run_traceback_analysis(
                self.results, rv_report,
                output_dir=self.output_dir,
                ra=self.ra, dec=self.dec,
            )
            if tb is not None:
                self.results['orbit_traceback'] = tb
            self._save_rv_dynamics_flags()
        except Exception as e:
            print(f"  轨道回溯失败: {e}")
            import traceback
            traceback.print_exc()
            self._save_rv_dynamics_flags()

    def _save_period_analysis(self):
        """运行 MHAOV 周期分析 (所有光变曲线)"""
        from astro_toolbox.period_analysis import run_period_analysis, save_csv
        pa = run_period_analysis(
            self.results,
            output_dir=self.output_dir,
            ra=self.ra, dec=self.dec,
            title_prefix=getattr(self, 'source_label', '') + '  ',
        )
        if pa is not None:
            self.results['period_analysis'] = pa
            save_csv(pa, self.output_dir)

    def _save_xray_analysis(self):
        """保存 X-ray 查询表，并基于距离/SED/HR/RV/光变结果做贡献分析。"""
        try:
            from astro_toolbox import xray
            xr = self.results.get('X-ray')
            hx = self.results.get('HEASARC_Xray')

            cat_path = xray.save_csv(xr, self.output_dir)
            hea_path = xray.save_heasarc_csv(hx, self.output_dir)
            analysis = xray.analyze_xray(
                xr, hx, results=self.results, ra=self.ra, dec=self.dec)
            self.results['xray_analysis'] = analysis
            csv_path, txt_path = xray.save_analysis(analysis, self.output_dir)

            n = analysis.get('n_detections', 0)
            best = analysis.get('best_detection') or {}
            msg = f"  X-ray分析: {n} detections"
            if best:
                if np.isfinite(best.get('log_lx', np.nan)):
                    msg += f"  logLx={best['log_lx']:.2f}"
                if np.isfinite(best.get('log_fx_over_fopt_g', np.nan)):
                    msg += f"  logFx/Fopt={best['log_fx_over_fopt_g']:.2f}"
                flags = best.get('flags') or []
                if flags:
                    msg += "  flags=" + ','.join(flags[:4])
            saved = [p for p in (cat_path, hea_path, csv_path, txt_path) if p]
            if saved:
                msg += f"  reports={len(saved)}"
            print(msg)
        except Exception as e:
            print(f"  X-ray 分析失败: {e}")
            import traceback
            traceback.print_exc()

    def _save_rv_dynamics_flags(self):
        rv = self.results.get('rv_analysis')
        if rv is None:
            return
        try:
            rows = [{
                'best_rv': rv.get('best_rv'),
                'best_rv_err': rv.get('best_rv_err'),
                'best_rv_source': rv.get('best_rv_source'),
                'is_sb2': rv.get('is_sb2'),
                'rv_quality': rv.get('rv_quality', ''),
                'quality_flags': ';'.join(rv.get('quality_flags', [])),
            }]
            path = os.path.join(self.output_dir, 'rv_dynamics_flags.csv')
            pd.DataFrame(rows).to_csv(path, index=False)
            txt = os.path.join(self.output_dir, 'rv_dynamics_flags.txt')
            with open(txt, 'w', encoding='utf-8') as f:
                f.write('# RV / Dynamics Flags\n\n')
                f.write(f"RV quality: {rv.get('rv_quality', '')}\n")
                f.write("RV flags: "
                        + (', '.join(rv.get('quality_flags', [])) or 'none')
                        + '\n')
                tb = self.results.get('orbit_traceback')
                if isinstance(tb, dict) and tb.get('status'):
                    f.write(f"Orbit traceback status: {tb.get('status')}\n")
            print(f"  RV/动力学flags: {txt}")
        except Exception as e:
            print(f"  RV/动力学flags保存失败: {e}")

    def _save_combined_plots(self):
        """生成联合图: 所有光谱、所有折叠、光谱+测光"""
        from astro_toolbox.combined_plots import (
            plot_combined_spectra, plot_combined_fold,
            plot_spectra_with_photometry,
        )

        # 1. 所有光谱画一起
        try:
            path = os.path.join(self.output_dir, 'combined_spectra.png')
            plot_combined_spectra(self.results, save_path=path,
                                 ra=self.ra, dec=self.dec)
        except Exception as e:
            print(f"  联合光谱图失败: {e}")

        # 2. 所有测光折叠周期画一起
        try:
            path = os.path.join(self.output_dir, 'combined_fold.png')
            plot_combined_fold(self.results, save_path=path,
                               ra=self.ra, dec=self.dec)
        except Exception as e:
            print(f"  联合折叠图失败: {e}")

        # 3. 光谱 + 测光 (SED) 画一起
        try:
            path = os.path.join(self.output_dir, 'spectra_with_photometry.png')
            plot_spectra_with_photometry(self.results, save_path=path,
                                         ra=self.ra, dec=self.dec)
        except Exception as e:
            print(f"  光谱+测光图失败: {e}")

    def _save_summary(self):
        """保存查询汇总"""
        lines = []
        lines.append(f"# astro_toolbox 查询结果")
        lines.append(f"# RA = {self.ra:.6f}, DEC = {self.dec:.6f}")
        lines.append(f"# 查询时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        hr_result = self.results.get('HR_diagram')
        if hr_result and isinstance(hr_result, dict):
            analysis = hr_result.get('analysis')
            if analysis:
                lines.append("## Gaia HRD position analysis")
                lines.append(f"Region: {analysis.get('region_label', analysis.get('region'))}")
                lines.append("Likely type(s): "
                             + ', '.join(analysis.get('likely_types', [])))
                lines.append(f"Confidence: {analysis.get('confidence', np.nan):.2f}")
                wd = analysis.get('wd_model') or {}
                if wd.get('status') == 'ok':
                    lines.append(
                        f"WD: M={wd.get('mass_msun', np.nan):.3f} Msun, "
                        f"Teff={wd.get('teff_k', np.nan):.0f} K, "
                        f"t_cool={wd.get('cooling_age_gyr', np.nan):.4f} Gyr")
                if analysis.get('recommended_models'):
                    lines.append("Recommended models: "
                                 + '; '.join(analysis['recommended_models']))
                if analysis.get('toolbox_followup'):
                    lines.append("Toolbox follow-up: "
                                 + '; '.join(analysis['toolbox_followup'][:4]))
                lines.append("")

        spec_diag = self.results.get('spectral_diagnostics')
        if spec_diag:
            lines.append("## Spectral anomaly / emission diagnostics")
            for survey, diag in spec_diag.items():
                flags = ', '.join(diag.get('flags', [])) or 'none'
                lines.append(f"{survey}: {diag.get('likely_interpretation', '')}; flags={flags}")
            lines.append("")

        sed_result = self.results.get('SED')
        sed_diag = getattr(sed_result, 'sed_diagnostics', None)
        if sed_diag:
            lines.append("## SED diagnostics")
            lines.append(f"Interpretation: {sed_diag.get('interpretation', '')}")
            lines.append("Flags: " + (', '.join(sed_diag.get('flags', [])) or 'none'))
            lines.append("")

        pa = self.results.get('period_analysis')
        if isinstance(pa, dict) and pa.get('morphology'):
            lines.append("## Lightcurve morphology")
            for label, morph in pa['morphology'].items():
                if not morph:
                    continue
                lines.append(
                    f"{label}: {morph.get('morphology')} "
                    f"A={morph.get('amplitude_mag', np.nan):.3f}, "
                    f"dip duty={morph.get('dip_duty_cycle', np.nan):.2f}, "
                    f"asym={morph.get('asymmetry', np.nan):.2f}")
            lines.append("")

        rv = self.results.get('rv_analysis')
        if isinstance(rv, dict):
            lines.append("## RV / dynamics diagnostics")
            lines.append(f"RV quality: {rv.get('rv_quality', '')}")
            lines.append("RV flags: " + (', '.join(rv.get('quality_flags', [])) or 'none'))
            lines.append("")

        xra = self.results.get('xray_analysis')
        if isinstance(xra, dict):
            lines.append("## X-ray diagnostics")
            lines.append(f"Detections: {xra.get('n_detections', 0)}")
            lines.append(f"Distance: {xra.get('distance_pc', np.nan):.2f} pc "
                         f"({xra.get('distance_source', '')})")
            best = xra.get('best_detection') or {}
            if best:
                src = best.get('survey') or best.get('catalog') or best.get('source_key', '')
                lines.append(f"Best detection: {src}")
                if np.isfinite(best.get('log_lx', np.nan)):
                    lines.append(f"  log L_X = {best['log_lx']:.3f}")
                if np.isfinite(best.get('log_fx_over_fopt_g', np.nan)):
                    lines.append(f"  log(F_X/F_Gopt) = {best['log_fx_over_fopt_g']:.3f}")
                lines.append("  flags: " + (', '.join(best.get('flags', [])) or 'none'))
                lines.append("  interpretation: " + best.get('interpretation', ''))
            lines.append("")

        for name, result in sorted(self.results.items()):
            dt = self.timings.get(name, 0)
            if result is None:
                lines.append(f"{name}: 无数据 ({dt:.1f}s)")
            elif isinstance(result, dict):
                lines.append(f"{name}: 有数据 ({dt:.1f}s)")
                # 观测时间信息
                for k in sorted(result.keys()):
                    if 'mjd' in k.lower() or 'date' in k.lower() or 'time' in k.lower():
                        v = result[k]
                        if isinstance(v, (int, float)):
                            lines.append(f"  {k} = {v}")
                        elif isinstance(v, str):
                            lines.append(f"  {k} = {v}")
                # 关键参数
                for k in ('class', 'z', 'teff', 'rv', 'survey'):
                    if k in result:
                        lines.append(f"  {k} = {result[k]}")
            else:
                lines.append(f"{name}: 有数据 ({dt:.1f}s)")

        summary_path = os.path.join(self.output_dir, 'summary.txt')
        with open(summary_path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(f"\n  汇总: {summary_path}")


def run_batch_csv(csv_path, output_base=None):
    """
    从 CSV 文件批量运行查询, 输出目录用源名称命名。

    CSV 至少需要 ra/dec 列; 可选 best_cluster, source_id, name 列用于命名。
    """
    df = pd.read_csv(csv_path)
    output_base = output_base or OUTPUT_BASE

    # 自动检测 RA/DEC 列
    ra_col = dec_col = None
    for col in df.columns:
        cl = col.strip().lower()
        if cl in ('ra', 'right_ascension'):
            ra_col = col
        elif cl in ('dec', 'declination', 'de'):
            dec_col = col
    if ra_col is None or dec_col is None:
        print(f"错误: CSV 中找不到 RA/DEC 列。可用列: {list(df.columns)}")
        return

    print(f"\n{'='*70}")
    print(f"  批量查询: {csv_path}")
    print(f"  共 {len(df)} 个源")
    print(f"{'='*70}")

    for i, row in df.iterrows():
        ra = float(row[ra_col])
        dec = float(row[dec_col])

        # 构建源名称作为输出目录名
        parts = []
        if 'best_cluster' in df.columns and pd.notna(row['best_cluster']):
            parts.append(str(row['best_cluster']))
        if 'source_id' in df.columns and pd.notna(row['source_id']):
            sid = str(row['source_id']).replace('.', '').rstrip('0')
            # 如果 source_id 是科学计数法, 转为整数
            try:
                sid = str(int(float(row['source_id'])))
            except (ValueError, OverflowError):
                pass
            parts.append(f"Gaia_{sid}")
        if 'name' in df.columns and pd.notna(row['name']):
            parts.append(str(row['name']))

        if parts:
            dir_name = '_'.join(parts)
        else:
            dir_name = f"RA{ra:.4f}_DEC{dec:.4f}"

        # 清理目录名中的非法字符
        dir_name = dir_name.replace(' ', '_').replace('/', '_')
        out_dir = os.path.join(output_base, dir_name)

        # 设置源标签
        source_label = parts[0] if parts else f"({ra:.4f},{dec:.4f})"
        if 'tier' in df.columns and pd.notna(row['tier']):
            source_label = f"[{row['tier']}] {source_label}"

        print(f"\n\n{'#'*70}")
        print(f"  [{i+1}/{len(df)}] {source_label}")
        print(f"  RA={ra:.6f}  DEC={dec:.6f}")
        print(f"  输出: {out_dir}")
        print(f"{'#'*70}")

        querier = AstroQueryAll(ra, dec, output_dir=out_dir)
        querier.source_label = source_label
        querier.query_all()
        querier.save_and_plot_all()

    print(f"\n\n{'='*70}")
    print(f"  批量查询完成! 共 {len(df)} 个源")
    print(f"  输出目录: {output_base}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description='astro_toolbox 一键查询/保存/画图')
    parser.add_argument('--ra', type=float, default=None, help='赤经 (度)')
    parser.add_argument('--dec', type=float, default=None, help='赤纬 (度)')
    parser.add_argument('--csv', type=str, default=None,
                        help='CSV 文件路径 (批量模式, 需含 ra/dec 列)')
    parser.add_argument('--output', '-o', default=None, help='输出目录')
    args = parser.parse_args()

    if args.csv:
        run_batch_csv(args.csv, output_base=args.output)
    elif args.ra is not None and args.dec is not None:
        querier = AstroQueryAll(args.ra, args.dec, output_dir=args.output)
        querier.query_all()
        querier.save_and_plot_all()
    else:
        parser.print_help()
        print("\n示例:")
        print("  python test_toolbox.py --ra 15.793 --dec 17.960")
        print("  python test_toolbox.py --csv matched_tier_with_desi.csv")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
