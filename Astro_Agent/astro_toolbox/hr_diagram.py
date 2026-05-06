"""
Gaia 赫罗图 (HR Diagram / CMD)
================================
绘制 Gaia DR3 色-星等图 (BP-RP vs M_G)

用法:
    from astro_toolbox.hr_diagram import HRDiagram
    hr = HRDiagram()
    hr.plot_single(190.305, 2.596, save_path='hr.png')
    hr.plot_batch('targets.csv', save_path='hr_batch.png')
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import config, utils


MAIN_SEQUENCE_ANCHORS = np.array([
    [-0.35, -3.0],
    [-0.15, -0.8],
    [0.00, 0.8],
    [0.30, 2.6],
    [0.60, 4.6],
    [0.90, 5.9],
    [1.20, 7.1],
    [1.60, 8.7],
    [2.00, 10.1],
    [2.60, 11.8],
    [3.20, 13.1],
    [4.20, 15.0],
], dtype=float)

WHITE_DWARF_ANCHORS = np.array([
    [-0.55, 9.2],
    [-0.25, 10.2],
    [0.00, 11.2],
    [0.25, 12.2],
    [0.55, 13.4],
    [0.90, 14.7],
    [1.20, 15.6],
    [1.55, 16.3],
], dtype=float)

REGION_STYLES = {
    'white_dwarf_sequence': {
        'color': '#1f77b4',
        'label': 'White dwarf cooling sequence',
    },
    'main_sequence': {
        'color': '#2ca02c',
        'label': 'Main sequence',
    },
    'above_main_sequence': {
        'color': '#ff7f0e',
        'label': 'Above main sequence / unresolved binary',
    },
    'giant_branch': {
        'color': '#d62728',
        'label': 'Giant branch / red clump',
    },
    'hot_subdwarf_bhb': {
        'color': '#9467bd',
        'label': 'Hot subdwarf / blue horizontal branch',
    },
    'subdwarf_low_metallicity': {
        'color': '#8c564b',
        'label': 'Subdwarf / underluminous main sequence',
    },
    'wd_ms_composite': {
        'color': '#17becf',
        'label': 'WD+MS / accreting-binary candidate zone',
    },
    'uncertain': {
        'color': '#7f7f7f',
        'label': 'Uncertain HRD region',
    },
}


def _finite_float(value, default=np.nan):
    try:
        value = float(value)
    except (TypeError, ValueError, np.ma.MaskError):
        return default
    if np.ma.is_masked(value) or not np.isfinite(value):
        return default
    return value


def _row_float(row, col, default=np.nan):
    try:
        value = row[col]
    except (KeyError, ValueError):
        return default
    if np.ma.is_masked(value):
        return default
    return _finite_float(value, default=default)


def _interp_anchor(bp_rp, anchors):
    x = anchors[:, 0]
    y = anchors[:, 1]
    if not np.isfinite(bp_rp):
        return np.nan
    return float(np.interp(bp_rp, x, y, left=np.nan, right=np.nan))


def estimate_main_sequence_mg(bp_rp):
    """Approximate Gaia DR3 main-sequence M_G at a given BP-RP."""
    return _interp_anchor(bp_rp, MAIN_SEQUENCE_ANCHORS)


def estimate_white_dwarf_mg(bp_rp):
    """Approximate Gaia DR3 white-dwarf cooling-sequence M_G."""
    return _interp_anchor(bp_rp, WHITE_DWARF_ANCHORS)


def _spectral_hint(bp_rp):
    if not np.isfinite(bp_rp):
        return 'unknown'
    if bp_rp < -0.2:
        return 'O/B or very hot source'
    if bp_rp < 0.05:
        return 'B/A'
    if bp_rp < 0.35:
        return 'A/F'
    if bp_rp < 0.75:
        return 'F/G'
    if bp_rp < 1.15:
        return 'G/K'
    if bp_rp < 1.8:
        return 'K/M'
    return 'M dwarf / very cool star'


def _toolbox_followups(region):
    common = [
        'SED: check UV/IR excess and rough temperature',
        'SDSS/DESI/LAMOST spectra: confirm spectral class and emission lines',
        'ZTF/TESS/Gaia light curves: periods, eclipses, pulsation, outbursts',
    ]
    if region == 'white_dwarf_sequence':
        return [
            'cooling_age / wd_fitting: WD mass, log g, Teff and cooling age',
            'rv_fitting: search double-degenerate RV variability when spectra exist',
            'orbit_traceback: compare WD age with cluster age if membership exists',
        ] + common
    if region == 'wd_ms_composite':
        return [
            'SED: fit WD+MS or accretion component; look for IR excess',
            'X-ray: CV or active-binary check',
            'period_analysis: orbital period from BLS/phase-folded light curves',
        ] + common
    if region in ('main_sequence', 'above_main_sequence'):
        return [
            'period_analysis: eclipsing/ellipsoidal/spot modulation tests',
            'rv_fitting: spectroscopic-binary check',
            'SED: stellar temperature and possible companion/excess',
        ] + common
    if region == 'giant_branch':
        return [
            'SED + spectra: giant temperature, gravity, metallicity',
            'period_analysis: long-period variable or eclipsing-binary check',
            'Gaia astrometry: distance and RUWE quality check',
        ] + common
    if region == 'hot_subdwarf_bhb':
        return [
            'spectra: distinguish hot subdwarf, BHB, pre-WD, CV continuum',
            'period_analysis: compact-binary or pulsation search',
            'SED: UV excess and temperature constraint',
        ] + common
    return common


def _recommended_models(region):
    if region == 'white_dwarf_sequence':
        return [
            'Sihao Cheng WD_models / Bédard+2020 WD cooling tracks',
            'Cummings+2018 IFMR + MS lifetime for progenitor-age checks',
        ]
    if region == 'wd_ms_composite':
        return [
            'WD cooling tracks + empirical M-dwarf templates',
            'CV/accretion-disk SED or emission-line models if spectra/X-ray support it',
        ]
    if region in ('main_sequence', 'above_main_sequence',
                  'subdwarf_low_metallicity'):
        return [
            'MIST/PARSEC isochrones for stellar mass, radius and age',
            'Binary light-curve/RV models if over-luminous or periodic',
        ]
    if region == 'giant_branch':
        return [
            'MIST/PARSEC isochrones with Gaia distance',
            'Spectroscopic log g / metallicity models',
        ]
    if region == 'hot_subdwarf_bhb':
        return [
            'hot-subdwarf / BHB atmosphere grids',
            'pre-WD tracks or compact-binary models when variability is present',
        ]
    return ['SED + Gaia CMD placement; add spectra/light curves for model choice']


def _confidence_from_delta(delta, width, floor=0.45):
    if not np.isfinite(delta):
        return floor
    return float(np.clip(1.0 - abs(delta) / width, floor, 0.93))


def _add_wd_model_result(analysis, bp_rp, M_G):
    """Attach WD-model parameters when the point is on the WD sequence."""
    analysis['wd_model'] = {
        'status': 'not_run',
        'model': 'SihaoCheng/WD_models; Bedard+2020 CO-core thick-H tracks',
    }
    try:
        from .cooling_age import (
            interpolate_wd_params, compute_progenitor_lifetime,
            cummings2018_ifmr, mist_ms_lifetime,
        )
        wd = interpolate_wd_params(bp_rp, M_G)
    except Exception as exc:
        analysis['wd_model']['status'] = 'failed'
        analysis['wd_model']['error'] = str(exc)
        return analysis

    if wd is None:
        analysis['wd_model']['status'] = 'outside_grid'
        return analysis

    analysis['wd_model'].update({
        'status': 'ok',
        'mass_msun': wd.get('mass'),
        'teff_k': wd.get('teff'),
        'logteff': wd.get('logteff'),
        'logg': wd.get('logg'),
        'cooling_age_gyr': wd.get('cooling_age_gyr'),
        'cooling_age_myr': wd.get('cooling_age_gyr', np.nan) * 1e3,
        'total_age_gyr': wd.get('total_age_gyr'),
    })

    try:
        progenitor = compute_progenitor_lifetime(wd['mass'])
    except Exception:
        progenitor = None
    if progenitor:
        analysis['wd_model']['progenitor_mass_msun'] = progenitor.get(
            'm_progenitor')
        analysis['wd_model']['ms_lifetime_gyr'] = progenitor.get(
            'ms_lifetime_gyr')

    try:
        m_prog_c18 = cummings2018_ifmr(wd['mass'])
        analysis['wd_model']['cummings18_progenitor_mass_msun'] = m_prog_c18
        analysis['wd_model']['cummings18_ms_lifetime_gyr'] = (
            mist_ms_lifetime(m_prog_c18)
            if np.isfinite(m_prog_c18) else np.nan)
    except Exception:
        pass

    return analysis


def classify_hr_position(bp_rp, M_G, gaia_params=None, include_wd_model=True):
    """
    Classify a Gaia HRD position and suggest the next physical model.

    The region boundaries are intentionally conservative guide rails. They are
    meant for triage; spectra, SEDs, variability and astrometry should still be
    used for confirmation.
    """
    bp_rp = _finite_float(bp_rp)
    M_G = _finite_float(M_G)
    ms_mg = estimate_main_sequence_mg(bp_rp)
    wd_mg = estimate_white_dwarf_mg(bp_rp)
    delta_ms = M_G - ms_mg if np.isfinite(ms_mg) else np.nan
    delta_wd = M_G - wd_mg if np.isfinite(wd_mg) else np.nan
    abs_delta_wd = abs(delta_wd) if np.isfinite(delta_wd) else np.nan

    diagnostics = []
    if np.isfinite(delta_ms):
        diagnostics.append(
            f"Delta M_G vs main sequence = {delta_ms:+.2f} mag "
            "(positive means fainter)")
    if np.isfinite(delta_wd):
        diagnostics.append(
            f"Delta M_G vs WD sequence = {delta_wd:+.2f} mag")

    if not np.isfinite(bp_rp) or not np.isfinite(M_G):
        region = 'uncertain'
        likely = ['Insufficient Gaia color/parallax for HRD classification']
        confidence = 0.0
    elif (np.isfinite(abs_delta_wd) and abs_delta_wd <= 1.35
          and M_G >= 7.5 and bp_rp <= 1.75):
        region = 'white_dwarf_sequence'
        likely = ['single white dwarf', 'double degenerate if overluminous']
        confidence = _confidence_from_delta(delta_wd, 1.6)
    elif M_G >= 9.0 and bp_rp <= 1.8 and (
            not np.isfinite(delta_ms) or delta_ms >= 1.8):
        region = 'white_dwarf_sequence'
        likely = ['cool white dwarf candidate', 'subdwarf contaminant possible']
        confidence = 0.62
    elif (0.0 <= bp_rp <= 1.8 and 6.5 <= M_G <= 11.5
          and np.isfinite(delta_ms) and 0.8 <= delta_ms <= 2.6):
        region = 'wd_ms_composite'
        likely = ['WD+M/K binary', 'cataclysmic variable', 'subdwarf']
        confidence = 0.58
    elif bp_rp < 0.25 and 1.5 <= M_G <= 7.2:
        region = 'hot_subdwarf_bhb'
        likely = ['hot subdwarf', 'blue horizontal branch', 'pre-WD']
        confidence = 0.58
    elif ((bp_rp > 0.65 and M_G < 3.2 and
           (not np.isfinite(delta_ms) or delta_ms < -1.5))
          or (bp_rp > 1.15 and M_G < 1.6)):
        region = 'giant_branch'
        likely = ['red giant branch', 'red clump / horizontal branch']
        confidence = 0.70
    elif np.isfinite(delta_ms) and -0.85 <= delta_ms <= 0.95:
        region = 'main_sequence'
        likely = [f'{_spectral_hint(bp_rp)} main-sequence star']
        confidence = _confidence_from_delta(delta_ms, 1.2, floor=0.50)
    elif np.isfinite(delta_ms) and -1.65 <= delta_ms < -0.85:
        region = 'above_main_sequence'
        likely = ['unresolved binary / multiple system', 'young or evolved star']
        confidence = _confidence_from_delta(delta_ms + 1.0, 0.9, floor=0.45)
    elif np.isfinite(delta_ms) and delta_ms > 0.95:
        region = 'subdwarf_low_metallicity'
        likely = ['subdwarf / metal-poor star', 'bad parallax or extinction issue']
        confidence = 0.52
    else:
        region = 'uncertain'
        likely = ['ambiguous HRD position; use spectra, SED and variability']
        confidence = 0.35

    style = REGION_STYLES.get(region, REGION_STYLES['uncertain'])
    analysis = {
        'bp_rp': bp_rp,
        'M_G': M_G,
        'region': region,
        'region_label': style['label'],
        'likely_types': likely,
        'confidence': confidence,
        'main_sequence_M_G': ms_mg,
        'delta_mag_from_main_sequence': delta_ms,
        'white_dwarf_sequence_M_G': wd_mg,
        'delta_mag_from_white_dwarf_sequence': delta_wd,
        'diagnostics': diagnostics,
        'recommended_models': _recommended_models(region),
        'toolbox_followup': _toolbox_followups(region),
    }

    if gaia_params:
        for key in ('source_id', 'Gmag', 'BPmag', 'RPmag', 'Plx', 'e_Plx',
                    'dist_pc', 'RUWE', 'Teff', 'logg'):
            if key in gaia_params:
                analysis[f'gaia_{key}'] = gaia_params[key]

    if include_wd_model and region == 'white_dwarf_sequence':
        _add_wd_model_result(analysis, bp_rp, M_G)

    return analysis


def format_hr_analysis(analysis):
    if not analysis:
        return []
    lines = []
    lines.append("HRD position analysis")
    lines.append("-" * 40)
    lines.append(f"Region: {analysis.get('region_label', 'unknown')}")
    lines.append(f"Confidence: {analysis.get('confidence', np.nan):.2f}")
    lines.append("Likely type(s): " + ', '.join(analysis.get('likely_types', [])))
    lines.append(f"BP-RP = {analysis.get('bp_rp', np.nan):.4f}")
    lines.append(f"M_G   = {analysis.get('M_G', np.nan):.4f}")
    if np.isfinite(analysis.get('delta_mag_from_main_sequence', np.nan)):
        lines.append(
            "Delta M_G from main sequence = "
            f"{analysis['delta_mag_from_main_sequence']:+.3f} mag")
    if np.isfinite(analysis.get('delta_mag_from_white_dwarf_sequence', np.nan)):
        lines.append(
            "Delta M_G from WD sequence   = "
            f"{analysis['delta_mag_from_white_dwarf_sequence']:+.3f} mag")

    wd = analysis.get('wd_model')
    if wd:
        lines.append("")
        lines.append("White-dwarf model")
        lines.append(f"  Model: {wd.get('model')}")
        lines.append(f"  Status: {wd.get('status')}")
        if wd.get('status') == 'ok':
            lines.append(f"  M_WD = {wd.get('mass_msun', np.nan):.4f} Msun")
            lines.append(f"  Teff = {wd.get('teff_k', np.nan):.0f} K")
            lines.append(f"  log g = {wd.get('logg', np.nan):.4f}")
            lines.append(
                f"  Cooling age = {wd.get('cooling_age_gyr', np.nan):.4f} Gyr "
                f"({wd.get('cooling_age_myr', np.nan):.1f} Myr)")
            if np.isfinite(wd.get('progenitor_mass_msun', np.nan)):
                lines.append(
                    "  Progenitor (WD_models IFMR): "
                    f"{wd['progenitor_mass_msun']:.3f} Msun, "
                    f"t_MS={wd.get('ms_lifetime_gyr', np.nan):.3f} Gyr")
        elif wd.get('error'):
            lines.append(f"  Error: {wd.get('error')}")

    if analysis.get('recommended_models'):
        lines.append("")
        lines.append("Recommended model path")
        for item in analysis['recommended_models']:
            lines.append(f"  - {item}")

    if analysis.get('toolbox_followup'):
        lines.append("")
        lines.append("Toolbox follow-up")
        for item in analysis['toolbox_followup']:
            lines.append(f"  - {item}")

    if analysis.get('diagnostics'):
        lines.append("")
        lines.append("Diagnostics")
        for item in analysis['diagnostics']:
            lines.append(f"  - {item}")
    return lines


def save_analysis_report(params, output_dir, filename='hr_diagram_analysis.txt'):
    """Save HRD classification and model suggestions."""
    if not params or output_dir is None:
        return None
    analysis = params.get('hr_analysis') or params.get('analysis')
    if not analysis:
        return None
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    lines = []
    lines.append("=" * 60)
    lines.append("Gaia HRD Classification Report")
    lines.append("=" * 60)
    if 'ra' in params and 'dec' in params:
        lines.append(f"RA={params['ra']:.6f}  DEC={params['dec']:.6f}")
    if 'source_id' in params:
        lines.append(f"Gaia source_id={params['source_id']}")
    lines.append("")
    lines.extend(format_hr_analysis(analysis))
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    return path


class HRDiagram:
    """Gaia DR3 赫罗图绘制器"""

    def __init__(self):
        self._bg_data = None

    def _query_gaia_params(self, ra, dec, radius_arcsec=config.SEARCH_RADIUS_ARCSEC):
        """查询单个目标的 Gaia DR3 参数"""
        tbl = utils.query_vizier('I/355/gaiadr3', ra, dec, radius_arcsec,
                                 columns=['Source', 'RA_ICRS', 'DE_ICRS',
                                          'Gmag', 'BPmag', 'RPmag',
                                          'Plx', 'e_Plx', 'RUWE', 'Teff',
                                          'logg'])
        if tbl is None or len(tbl) == 0:
            return None
        row = tbl[0]
        try:
            gmag = _row_float(row, 'Gmag')
            bp = _row_float(row, 'BPmag')
            rp = _row_float(row, 'RPmag')
            plx = _row_float(row, 'Plx')
            if not np.isfinite(gmag + bp + rp + plx) or plx <= 0:
                return None
            bp_rp = bp - rp
            dist_pc = 1000.0 / plx
            M_G = gmag + 5 * np.log10(plx / 1000.0) + 5
            result = {'ra': ra, 'dec': dec, 'Gmag': gmag, 'BPmag': bp,
                      'RPmag': rp, 'BP_RP': bp_rp, 'M_G': M_G,
                      'Plx': plx, 'dist_pc': dist_pc}
            try:
                result['source_id'] = str(row['Source']).strip()
            except (KeyError, ValueError):
                pass
            for out_key, col in [('gaia_ra', 'RA_ICRS'), ('gaia_dec', 'DE_ICRS'),
                                 ('e_Plx', 'e_Plx'), ('RUWE', 'RUWE'),
                                 ('Teff', 'Teff'), ('logg', 'logg')]:
                val = _row_float(row, col)
                if np.isfinite(val):
                    result[out_key] = val
            result['hr_analysis'] = classify_hr_position(
                bp_rp, M_G, gaia_params=result)
            return result
        except (ValueError, KeyError, np.ma.MaskError):
            return None

    def _query_gaia_batch(self, ra_arr, dec_arr, radius_arcsec=config.SEARCH_RADIUS_ARCSEC):
        """批量查询 Gaia 参数"""
        results = []
        for ra, dec in zip(ra_arr, dec_arr):
            r = self._query_gaia_params(ra, dec, radius_arcsec)
            if r is not None:
                results.append(r)
        return results

    def load_background(self, n_sample=50000):
        """
        加载 Gaia 背景恒星用于密度图。
        优先从本地 CSV 文件加载 (TAP_1632913590914.csv),
        回退到 Gaia TAP ADQL 查询。
        """
        if self._bg_data is not None:
            return self._bg_data

        # 优先: 本地 CSV 文件
        import os
        csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                'data', 'TAP_1632913590914.csv')
        if os.path.exists(csv_path):
            try:
                import pandas as pd
                df = pd.read_csv(csv_path)
                mask = np.isfinite(df['bp_rp']) & np.isfinite(df['absmag'])
                df = df[mask]
                if n_sample and len(df) > n_sample:
                    df = df.sample(n=n_sample, random_state=42)
                self._bg_data = {
                    'BP_RP': np.array(df['bp_rp'], dtype=float),
                    'M_G': np.array(df['absmag'], dtype=float),
                }
                print(f"  背景恒星加载 (本地): {len(self._bg_data['BP_RP'])} 个")
                return self._bg_data
            except Exception as e:
                print(f"  本地背景文件加载失败: {e}")

        try:
            from astroquery.gaia import Gaia
            if config.GAIA_TOKEN:
                try:
                    Gaia.login(token=config.GAIA_TOKEN)
                except Exception:
                    pass
            query = f"""
            SELECT TOP {n_sample}
                phot_g_mean_mag, bp_rp,
                phot_g_mean_mag + 5*LOG10(parallax/1000) + 5 AS abs_g
            FROM gaiadr3.gaia_source
            WHERE parallax > 1 AND parallax_over_error > 10
                AND phot_g_mean_mag IS NOT NULL
                AND bp_rp IS NOT NULL
                AND bp_rp BETWEEN -0.5 AND 5.0
            ORDER BY random_index
            """
            job = Gaia.launch_job(query)
            tbl = job.get_results()
            self._bg_data = {
                'BP_RP': np.array(tbl['bp_rp'], dtype=float),
                'M_G': np.array(tbl['abs_g'], dtype=float),
            }
            print(f"  背景恒星加载: {len(self._bg_data['BP_RP'])} 个")
        except Exception as e:
            print(f"  背景恒星加载失败: {e}")
            self._bg_data = None
        return self._bg_data

    def plot_single(self, ra, dec, save_path=None, show_background=True,
                    annotate_regions=True):
        """在 HR 图上标注单个目标"""
        params = self._query_gaia_params(ra, dec)
        if params is None:
            print(f"无法获取 Gaia 参数: RA={ra}, DEC={dec}")
            return None
        return self._make_plot([params], save_path, show_background,
                               annotate_regions=annotate_regions)

    def plot_batch(self, input_csv=None, ra_arr=None, dec_arr=None,
                   ra_col=None, dec_col=None, save_path=None,
                   show_background=True, annotate_regions=True):
        """在 HR 图上标注多个目标"""
        if input_csv is not None:
            df = pd.read_csv(input_csv)
            ra_candidates = ['ra', 'RA', 'Ra', 'RIGHT_ASCENSION']
            dec_candidates = ['dec', 'DEC', 'Dec', 'DE', 'DECLINATION']
            if ra_col is None:
                for c in ra_candidates:
                    if c in df.columns:
                        ra_col = c
                        break
            if dec_col is None:
                for c in dec_candidates:
                    if c in df.columns:
                        dec_col = c
                        break
            ra_arr = df[ra_col].values
            dec_arr = df[dec_col].values

        if ra_arr is None or dec_arr is None:
            print("需要提供坐标")
            return None

        print(f"批量查询 Gaia 参数: {len(ra_arr)} 个目标")
        targets = self._query_gaia_batch(ra_arr, dec_arr)
        print(f"  成功获取: {len(targets)} 个")
        if len(targets) == 0:
            return None
        return self._make_plot(targets, save_path, show_background,
                               annotate_regions=annotate_regions)

    def plot_from_gaia_data(self, gmag, bp_rp, parallax, save_path=None,
                            show_background=True, labels=None,
                            annotate_regions=True):
        """直接用已有的 Gaia 数据画 HR 图 (不需要再查询)"""
        gmag = np.asarray(gmag, dtype=float)
        bp_rp = np.asarray(bp_rp, dtype=float)
        parallax = np.asarray(parallax, dtype=float)
        mask = (parallax > 0) & np.isfinite(gmag) & np.isfinite(bp_rp)
        M_G = gmag[mask] + 5 * np.log10(parallax[mask] / 1000.0) + 5

        targets = []
        for i in range(len(M_G)):
            t = {'BP_RP': bp_rp[mask][i], 'M_G': M_G[i]}
            if labels is not None:
                t['label'] = labels[i] if i < len(labels) else ''
            t['hr_analysis'] = classify_hr_position(t['BP_RP'], t['M_G'],
                                                     gaia_params=t)
            targets.append(t)
        return self._make_plot(targets, save_path, show_background,
                               annotate_regions=annotate_regions)

    def _draw_region_guides(self, ax):
        """Draw approximate HRD regions behind the target points."""
        x_ms = MAIN_SEQUENCE_ANCHORS[:, 0]
        y_ms = MAIN_SEQUENCE_ANCHORS[:, 1]
        ax.fill_between(x_ms, y_ms - 0.8, y_ms + 0.9,
                        color=REGION_STYLES['main_sequence']['color'],
                        alpha=0.10, zorder=2,
                        label='Main-sequence band')
        ax.plot(x_ms, y_ms, color=REGION_STYLES['main_sequence']['color'],
                lw=1.2, alpha=0.65, zorder=3)

        x_wd = WHITE_DWARF_ANCHORS[:, 0]
        y_wd = WHITE_DWARF_ANCHORS[:, 1]
        ax.fill_between(x_wd, y_wd - 1.1, y_wd + 1.2,
                        color=REGION_STYLES['white_dwarf_sequence']['color'],
                        alpha=0.12, zorder=2,
                        label='WD cooling band')
        ax.plot(x_wd, y_wd,
                color=REGION_STYLES['white_dwarf_sequence']['color'],
                lw=1.2, alpha=0.75, zorder=3)

        ax.axvspan(0.65, 4.5, ymin=0.73, ymax=1.0,
                   color=REGION_STYLES['giant_branch']['color'],
                   alpha=0.06, zorder=2, label='Giant / red-clump zone')
        ax.axvspan(-0.5, 0.25, ymin=0.44, ymax=0.78,
                   color=REGION_STYLES['hot_subdwarf_bhb']['color'],
                   alpha=0.07, zorder=2, label='Hot subdwarf / BHB zone')

        ax.text(0.8, 5.3, 'Main sequence', fontsize=8,
                color=REGION_STYLES['main_sequence']['color'],
                rotation=58, ha='center', va='center')
        ax.text(0.35, 13.0, 'White dwarfs', fontsize=8,
                color=REGION_STYLES['white_dwarf_sequence']['color'],
                rotation=70, ha='center', va='center')
        ax.text(2.4, 0.2, 'Giants', fontsize=8,
                color=REGION_STYLES['giant_branch']['color'],
                ha='center')
        ax.text(-0.25, 4.7, 'sdB/BHB', fontsize=8,
                color=REGION_STYLES['hot_subdwarf_bhb']['color'],
                ha='left')

    def _make_plot(self, targets, save_path=None, show_background=True,
                   annotate_regions=True):
        """核心绘图"""
        fig, ax = plt.subplots(figsize=(8, 10))

        # 背景密度图
        if show_background:
            bg = self.load_background()
            if bg is not None:
                ax.hist2d(bg['BP_RP'], bg['M_G'], bins=200,
                          cmap='Greys', norm=matplotlib.colors.LogNorm(),
                          alpha=0.6, zorder=1)

        if annotate_regions:
            self._draw_region_guides(ax)

        # 目标点
        bp_rp_arr = np.array([t['BP_RP'] for t in targets])
        mg_arr = np.array([t['M_G'] for t in targets])
        color_arr = []
        for t in targets:
            analysis = t.get('hr_analysis') or classify_hr_position(
                t['BP_RP'], t['M_G'], gaia_params=t)
            t['hr_analysis'] = analysis
            color_arr.append(REGION_STYLES.get(
                analysis.get('region'), REGION_STYLES['uncertain'])['color'])

        ax.scatter(bp_rp_arr, mg_arr, c=color_arr, s=70, edgecolors='black',
                   linewidths=0.6, zorder=10, label=f'Targets ({len(targets)})')

        # 标注区域与 Teff/冷却年龄
        for t in targets:
            text_parts = []
            if t.get('label'):
                text_parts.append(str(t['label']))
            analysis = t.get('hr_analysis') or {}
            if analysis:
                text_parts.append(analysis.get('region_label', 'HRD region'))
                wd = analysis.get('wd_model', {})
                if wd.get('status') == 'ok':
                    text_parts.append(
                        f"t_cool={wd.get('cooling_age_myr', np.nan):.0f} Myr")
                    text_parts.append(
                        f"M={wd.get('mass_msun', np.nan):.2f} Msun")
            elif 'Teff' in t:
                text_parts.append(f"{t['Teff']:.0f}K")
            if text_parts:
                color = REGION_STYLES.get(
                    analysis.get('region'), REGION_STYLES['uncertain'])['color']
                ax.annotate('\n'.join(text_parts[:3]),
                            (t['BP_RP'], t['M_G']),
                            xytext=(8, 8), textcoords='offset points',
                            fontsize=7, ha='left', va='bottom', color=color,
                            arrowprops=dict(arrowstyle='-', lw=0.5,
                                            color=color, alpha=0.8))

        ax.set_xlabel('BP - RP (mag)', fontsize=13)
        ax.set_ylabel('$M_G$ (mag)', fontsize=13)
        ax.set_title('Gaia DR3 HR Diagram', fontsize=14)
        ax.invert_yaxis()
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(16, -4)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)

        fig.tight_layout()
        utils.save_and_close(fig, save_path)
        return fig


def quick_hr(ra, dec, save_path=None, annotate_regions=True):
    """一键画 HR 图"""
    hr = HRDiagram()
    return hr.plot_single(ra, dec, save_path=save_path,
                          annotate_regions=annotate_regions)


def save_csv(params, output_dir):
    """保存 Gaia HR 图参数为 CSV"""
    if params is None:
        return None
    row = {}
    for k, v in params.items():
        if isinstance(v, (int, float, str, bool, np.integer, np.floating)):
            row[k] = v
    analysis = params.get('hr_analysis') or params.get('analysis')
    if analysis:
        for key in ('region', 'region_label', 'confidence',
                    'delta_mag_from_main_sequence',
                    'delta_mag_from_white_dwarf_sequence'):
            row[f'hr_{key}'] = analysis.get(key)
        row['hr_likely_types'] = '; '.join(analysis.get('likely_types', []))
        row['hr_recommended_models'] = '; '.join(
            analysis.get('recommended_models', []))
        wd = analysis.get('wd_model') or {}
        for key in ('status', 'model', 'mass_msun', 'teff_k', 'logg',
                    'cooling_age_gyr', 'cooling_age_myr',
                    'progenitor_mass_msun', 'ms_lifetime_gyr'):
            if key in wd:
                row[f'wd_{key}'] = wd.get(key)
    df = pd.DataFrame([row]) if row else utils.keyvalue_to_dataframe(params)
    return utils.write_csv(df, output_dir, 'hr_diagram_params.csv')
