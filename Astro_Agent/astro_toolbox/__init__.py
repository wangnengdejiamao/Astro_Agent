"""
astro_toolbox - 多波段天文光谱/光变曲线/SED/HR图 工具箱
=========================================================
覆盖 X射线 → 紫外 → 光学 → 红外全波段。

用法:
    from astro_toolbox import sdss, ztf, wise, sed, hr_diagram
    spec = sdss.query_spectrum(190.3, 2.6)
    lc   = ztf.query_lightcurve(190.3, 2.6)
"""

from . import config, utils, six_dim, diagnostics, koa, koa_batch
