"""
TESS 光变曲线
=============
使用 lightkurve 包查询和下载 TESS 光变曲线

用法:
    from astro_toolbox.tess import query_lightcurve, plot_lightcurve
    lc = query_lightcurve(190.305, 2.596)
    plot_lightcurve(lc, save_path='tess_lc.png')
"""
import numpy as np
from . import config, utils


def query_lightcurve(ra, dec, author='SPOC'):
    """
    查询 TESS 光变曲线。

    Args:
        author: 'SPOC' (2-min cadence) 或 'TESS-SPOC' 或 'QLP'

    Returns:
        dict: {'time': array, 'flux': array, 'flux_err': array,
               'sector': list, 'author': str}
        或 None
    """
    import lightkurve as lk
    c = f"{ra} {dec}"
    search = lk.search_lightcurve(c, mission='TESS', author=author)
    if search is None or len(search) == 0:
        return None

    # 下载并拼接所有 sector
    lc_collection = search.download_all()
    if lc_collection is None or len(lc_collection) == 0:
        return None

    lc = lc_collection.stitch()

    time = lc.time.value
    flux = lc.flux.value
    flux_err = lc.flux_err.value if hasattr(lc.flux_err, 'value') else np.zeros_like(flux)

    sectors = list(set([r.meta.get('SECTOR', '?') for r in lc_collection]))

    return {
        'survey': 'TESS',
        'ra': ra, 'dec': dec,
        'time': time,      # BTJD
        'flux': flux,       # 归一化流量
        'flux_err': flux_err,
        'sectors': sectors,
        'author': author,
        'n_points': len(time),
        'obs_time_min': float(np.nanmin(time)),
        'obs_time_max': float(np.nanmax(time)),
        'time_system': 'BTJD',
    }


def plot_lightcurve(result, save_path=None):
    """绘制 TESS 光变曲线"""
    if result is None:
        return None
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.scatter(result['time'], result['flux'], s=0.5, c='black', alpha=0.5)
    ax.set_xlabel('Time (BTJD)')
    ax.set_ylabel('Normalized Flux')
    ax.set_title(f"TESS Light Curve  RA={result['ra']:.4f} DEC={result['dec']:.4f}  "
                 f"Sectors={result['sectors']}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    utils.save_and_close(fig, save_path)
    return fig


def _as_lightkurve_lc(result, flatten=True, flatten_window_length=401,
                      remove_outliers=True):
    """Build a Lightkurve LightCurve from the toolbox TESS dict."""
    if result is None:
        return None
    import lightkurve as lk
    time = np.asarray(result.get('time'), dtype=float)
    flux = np.asarray(result.get('flux'), dtype=float)
    ferr = np.asarray(result.get('flux_err'), dtype=float)
    if ferr.shape != flux.shape:
        ferr = np.full_like(flux, np.nan)
    valid = np.isfinite(time) & np.isfinite(flux) & (flux > 0)
    if np.sum(valid) < 20:
        return None
    lc = lk.LightCurve(time=time[valid], flux=flux[valid],
                       flux_err=ferr[valid])
    try:
        lc = lc.remove_nans()
    except Exception:
        pass
    try:
        lc = lc.normalize(unit='unscaled')
    except Exception:
        pass
    if flatten:
        try:
            # Lightkurve's flatten removes sector-scale thermal/systematic
            # trends that otherwise dominate the LS periodogram.  Keep the
            # window odd and long enough not to erase hour-scale WD variability.
            win = int(flatten_window_length)
            if win % 2 == 0:
                win += 1
            lc = lc.flatten(window_length=max(win, 101))
        except Exception:
            pass
    if remove_outliers:
        try:
            lc = lc.remove_outliers(sigma_upper=5.0, sigma_lower=np.inf)
        except Exception:
            pass
    return lc


def analyze_period_lightkurve(result, output_dir=None,
                              minimum_period_day=None,
                              maximum_period_day=None,
                              oversample_factor=10,
                              flatten_window_length=401,
                              search_hours=True):
    """
    Lightkurve-native Lomb-Scargle analysis for TESS.

    Saves a periodogram plus P and 2P folded plots using
    ``lc.fold(period).scatter()``.  The 2P plot is intentionally written
    because compact/eclipsing WD systems often return the strongest harmonic.
    """
    lc = _as_lightkurve_lc(
        result, flatten=True,
        flatten_window_length=flatten_window_length,
        remove_outliers=True)
    if lc is None:
        return None

    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import astropy.units as u

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    t = np.asarray(lc.time.value, dtype=float)
    span = float(np.nanmax(t) - np.nanmin(t))
    dt = np.diff(np.sort(t))
    dt = dt[np.isfinite(dt) & (dt > 0)]
    cadence = float(np.nanmedian(dt)) if len(dt) else 2.0 / 1440.0
    if minimum_period_day is None:
        # User-requested hard floor: 5 minutes.
        minimum_period_day = max(5.0 / 1440.0, 2.2 * cadence)
    if maximum_period_day is None:
        if search_hours:
            maximum_period_day = min(max(span / 5.0, minimum_period_day * 20.0), 2.0)
        else:
            maximum_period_day = min(max(span / 2.0, minimum_period_day * 10.0), 30.0)
    if maximum_period_day <= minimum_period_day:
        maximum_period_day = minimum_period_day * 20.0

    try:
        pg = lc.to_periodogram(
            method='lombscargle',
            minimum_period=minimum_period_day * u.day,
            maximum_period=maximum_period_day * u.day,
            oversample_factor=int(oversample_factor),
            normalization='amplitude',
        )
    except Exception as exc:
        print(f"  TESS Lightkurve 周期分析失败: {exc}")
        return None

    period = pg.period_at_max_power
    period_day = float(period.to_value(u.day))
    frequency = pg.frequency_at_max_power
    max_power = float(np.nanmax(np.asarray(pg.power.value
                                           if hasattr(pg.power, 'value')
                                           else pg.power, dtype=float)))
    boundary_alias_warning = (
        period_day <= minimum_period_day * 1.05
        or period_day >= maximum_period_day / 1.05
    )

    files = {}
    if output_dir:
        fig, ax = plt.subplots(figsize=(8.0, 4.2))
        try:
            pg.plot(ax=ax, view='period', scale='log')
        except Exception:
            pday = np.asarray(pg.period.to_value(u.day), dtype=float)
            power = np.asarray(pg.power.value if hasattr(pg.power, 'value') else pg.power, dtype=float)
            ax.plot(pday, power, color='black', lw=0.8)
            ax.set_xscale('log')
            ax.set_xlabel('Period (d)')
            ax.set_ylabel('Lomb-Scargle power')
        ax.axvline(period_day, color='crimson', ls='--', lw=1.0,
                   label=f'P={period_day:.6f} d')
        ax.set_title('TESS Lightkurve Lomb-Scargle periodogram')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25, which='both')
        fig.tight_layout()
        path = os.path.join(output_dir, 'tess_lightkurve_periodogram.png')
        utils.save_and_close(fig, path)
        files['periodogram'] = path

        for factor, suffix in [(1.0, 'P'), (2.0, '2P')]:
            fold_period = factor * period
            fig, ax = plt.subplots(figsize=(7.2, 4.4))
            try:
                lc.fold(fold_period).scatter(
                    ax=ax,
                    s=14,
                    alpha=0.58,
                    color='black',
                    label=fr'Period = {fold_period.to_value(u.day):.5f} d')
            except TypeError:
                lc.fold(fold_period).scatter(
                    ax=ax,
                    label=fr'Period = {fold_period.to_value(u.day):.5f} d')
            ax.set_title(f'TESS folded light curve ({suffix})')
            ax.legend(fontsize=9)
            ax.grid(alpha=0.25)
            fig.tight_layout()
            path = os.path.join(output_dir, f'tess_lightkurve_fold_{suffix}.png')
            utils.save_and_close(fig, path)
            files[f'fold_{suffix}'] = path

        summary_path = os.path.join(output_dir, 'tess_lightkurve_period.csv')
        pd.DataFrame([{
            'method': 'lightkurve_lombscargle',
            'period_day': period_day,
            'period_hour': period_day * 24.0,
            'period_min': period_day * 24.0 * 60.0,
            'two_period_day': 2.0 * period_day,
            'two_period_hour': 2.0 * period_day * 24.0,
            'frequency_per_day': float(frequency.to_value(1 / u.day)),
            'max_power': max_power,
            'boundary_alias_warning': bool(boundary_alias_warning),
            'minimum_period_day': minimum_period_day,
            'maximum_period_day': maximum_period_day,
            'oversample_factor': int(oversample_factor),
            'flatten_window_length': int(flatten_window_length),
            'search_hours': bool(search_hours),
            'n_points': int(len(lc.time)),
        }]).to_csv(summary_path, index=False)
        files['summary_csv'] = summary_path

    return {
        'method': 'lightkurve_lombscargle',
        'best_period': period_day,
        'best_period_day': period_day,
        'best_period_hour': period_day * 24.0,
        'best_period_min': period_day * 24.0 * 60.0,
        'two_period_day': 2.0 * period_day,
        'fap': 0.0,
        'power': max_power,
        'boundary_alias_warning': bool(boundary_alias_warning),
        'n_points': int(len(lc.time)),
        'minimum_period_day': float(minimum_period_day),
        'maximum_period_day': float(maximum_period_day),
        'oversample_factor': int(oversample_factor),
        'flatten_window_length': int(flatten_window_length),
        'search_hours': bool(search_hours),
        'files': files,
    }


def save_csv(result, output_dir):
    """保存 TESS 光变曲线为 CSV"""
    import pandas as pd
    if result is None:
        return None
    df = pd.DataFrame({
        'time_BTJD': result['time'],
        'flux': result['flux'],
        'flux_err': result['flux_err'],
    })
    return utils.write_csv(df, output_dir, 'tess_lightcurve.csv')
