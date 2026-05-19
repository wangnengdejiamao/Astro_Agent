"""
DESI DR1 光谱 (封装现有 desi_spectrum_tool)
=============================================
波长覆盖: 3600-9800 A (B+R+Z)

用法:
    from astro_toolbox.desi import query_spectrum
    result = query_spectrum(190.305, 2.596)
"""
import sys
import os
from . import config

# desi_spectrum_tool 位于 desi_tool/ 目录
_desi_tool_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'desi_tool'
)
if _desi_tool_dir not in sys.path:
    sys.path.insert(0, _desi_tool_dir)


def query_spectrum(ra, dec, radius_arcsec=config.SEARCH_RADIUS_ARCSEC,
                   save_fits=True, save_png=True):
    """
    通过 DESI MWS 匹配并提取光谱。

    Returns:
        dict: {'match', 'spectrum', 'files'} 或 None
    """
    from desi_spectrum_tool import DESITool
    tool = DESITool(log_func=print)
    return tool.process_single(ra, dec, radius_arcsec=radius_arcsec,
                               save_fits=save_fits, save_png=save_png)


# 重导出, 方便其他脚本使用
try:
    from desi_spectrum_tool import DESITool, SpectrumExtractor
except ImportError:
    pass
