"""全局配置常量"""
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)

# 缓存和输出
CACHE_DIR = os.path.join(PARENT_DIR, 'output', 'astro_cache')
OUTPUT_DIR = os.path.join(PARENT_DIR, 'output', 'astro_output')

# 网络
TIMEOUT = 600          # seconds (read timeout, 适应代理/大文件下载)
CONNECT_TIMEOUT = 60   # connect timeout
MAX_WORKERS = 4
MAX_RETRIES = 5        # download retry attempts
CHUNK_SIZE = 1024 * 1024  # 1MB

# 代理设置 (Clash 本地代理)
# 优先读取环境变量, 其次使用 Clash 默认端口
PROXY_URL = os.environ.get('ASTRO_PROXY',
                           os.environ.get('https_proxy',
                                          'http://127.0.0.1:7890'))
# 机场订阅地址 (仅用于提示, 不在代码中自动订阅)
PROXY_SUBSCRIBE_URL = None

# NASA ADS API Token (用于获取文献摘要)
# 只从环境变量读取；utils.py 还会在需要时回退读取 ~/.ads/dev_key。
ADS_TOKEN = os.environ.get('ADS_DEV_KEY', '')

# Gaia Archive Token (用于 DataLink 等认证访问)
GAIA_TOKEN = os.environ.get('GAIA_TOKEN', '')

# LAMOST DR12 / pylamost / My Data Disk requests.
# 凭据只从环境变量读取，避免把 token/邮箱提交到 GitHub。
LAMOST_DR = os.environ.get('LAMOST_DR', 'dr12')
LAMOST_TOKEN = os.environ.get('LAMOST_TOKEN', '')
LAMOST_FTP_SERVER = os.environ.get('LAMOST_FTP_SERVER', 'ftp3.lamost.org')
LAMOST_FTP_USER = os.environ.get('LAMOST_FTP_USER', '')
LAMOST_FTP_PASSWORD = os.environ.get('LAMOST_FTP_PASSWORD', LAMOST_TOKEN)
LAMOST_FTP_MAX_DEPTH = int(os.environ.get('LAMOST_FTP_MAX_DEPTH', '5'))
LAMOST_FTP_MANIFEST = os.environ.get(
    'LAMOST_FTP_MANIFEST',
    os.path.join(CACHE_DIR, 'lamost_ftp_manifest.csv'))

# 本地三维尘埃图。优先用于 SED 消光改正；不存在时自动回退到旧的 IRSA/SFD。
_PACKAGE_BAYESTAR2019_PATH = os.path.join(
    SCRIPT_DIR, 'data', 'bayestar', 'bayestar2019.h5')
_LEGACY_BAYESTAR2019_PATH = '/Users/ljm/dustmaps_data/bayestar/bayestar2019.h5'
BAYESTAR2019_PATH = os.environ.get(
    'BAYESTAR2019_PATH',
    _PACKAGE_BAYESTAR2019_PATH
    if os.path.exists(_PACKAGE_BAYESTAR2019_PATH)
    else _LEGACY_BAYESTAR2019_PATH)
BAYESTAR_MAX_SAMPLES = int(os.environ.get('BAYESTAR_MAX_SAMPLES', '1'))
BAYESTAR_QUERY_MODE = os.environ.get('BAYESTAR_QUERY_MODE', 'best')
BAYESTAR_FALLBACK_DISTANCE_PC = float(
    os.environ.get('BAYESTAR_FALLBACK_DISTANCE_PC', '10000'))
# dustmaps Bayestar2019 返回 Green+2019 reddening 单位。这里作为 E(B-V)
# 等效值使用；如需严格匹配某一滤光片系统，可用环境变量覆盖比例。
BAYESTAR_TO_EBV = float(os.environ.get('BAYESTAR_TO_EBV', '1.0'))

# KOA / Keck local extracted spectra. 这里主要用于已经从 KOA 下载并用
# PypeIt/其它流程提取过的一维 FITS。在线 KOA 查询由 koa.py 的 pykoa
# 可选入口处理。
KOA_LOGIN_URL = 'https://koa.ipac.caltech.edu/cgi-bin/KOA/nph-KOAlogin'
KOA_LOCAL_ROOT = os.environ.get('KOA_LOCAL_ROOT',
                                '/Users/ljm/Desktop/DWD/speutrem')
KOA_SEARCH_RADIUS_ARCSEC = float(
    os.environ.get('KOA_SEARCH_RADIUS_ARCSEC', '8.0'))
KOA_RESAMPLE_STEP_A = float(os.environ.get('KOA_RESAMPLE_STEP_A', '1.0'))

# 需要走代理的美国数据库域名
PROXY_DOMAINS = [
    'data.desi.lbl.gov',
    'irsa.ipac.caltech.edu',
    'mast.stsci.edu',
    'archive.stsci.edu',
    'skyserver.sdss.org',
    'dr18.sdss.org',
    'gea.esac.esa.int',       # Gaia (欧洲, 有时也慢)
    'api.adsabs.harvard.edu', # NASA ADS API
]

# 不走代理的域名 (国内可直连)
NO_PROXY_DOMAINS = [
    'vizier.cds.unistra.fr',
    'vizier.china-vo.org',    # 中国 Vizier 镜像
    'simbad.u-strasbg.fr',
    'simbad.cds.unistra.fr',  # 新 SIMBAD 域名 (astroquery >=0.4.7)
]

# 默认搜索参数
SEARCH_RADIUS_ARCSEC = 3.0

# 各巡天波段参考波长 (Angstrom) 和零点 (AB mag → Jy)
BAND_INFO = {
    # X-ray (keV → Angstrom: lambda = 12398.4 / E_keV)
    'ROSAT_PSPC':     {'wave_A': 12.4,   'label': 'ROSAT 0.1-2.4 keV'},
    # UV
    'GALEX_FUV':      {'wave_A': 1528.0, 'label': 'GALEX FUV', 'zero_Jy': 3631.0},
    'GALEX_NUV':      {'wave_A': 2271.0, 'label': 'GALEX NUV', 'zero_Jy': 3631.0},
    # Optical
    'SDSS_u':         {'wave_A': 3543.0, 'label': 'SDSS u', 'zero_Jy': 3631.0},
    'SDSS_g':         {'wave_A': 4770.0, 'label': 'SDSS g', 'zero_Jy': 3631.0},
    'SDSS_r':         {'wave_A': 6231.0, 'label': 'SDSS r', 'zero_Jy': 3631.0},
    'SDSS_i':         {'wave_A': 7625.0, 'label': 'SDSS i', 'zero_Jy': 3631.0},
    'SDSS_z':         {'wave_A': 9134.0, 'label': 'SDSS z', 'zero_Jy': 3631.0},
    'Gaia_G':         {'wave_A': 6230.0, 'label': 'Gaia G',  'zero_Jy': 3228.75},
    'Gaia_BP':        {'wave_A': 5110.0, 'label': 'Gaia BP', 'zero_Jy': 3552.01},
    'Gaia_RP':        {'wave_A': 7770.0, 'label': 'Gaia RP', 'zero_Jy': 2554.95},
    # NIR
    '2MASS_J':        {'wave_A': 12350.0,'label': '2MASS J', 'zero_Jy': 1594.0},
    '2MASS_H':        {'wave_A': 16620.0,'label': '2MASS H', 'zero_Jy': 1024.0},
    '2MASS_Ks':       {'wave_A': 21590.0,'label': '2MASS Ks','zero_Jy': 666.7},
    # MIR
    'WISE_W1':        {'wave_A': 33526.0,'label': 'WISE W1', 'zero_Jy': 309.54},
    'WISE_W2':        {'wave_A': 46028.0,'label': 'WISE W2', 'zero_Jy': 171.79},
    'WISE_W3':        {'wave_A':115608.0,'label': 'WISE W3', 'zero_Jy': 31.674},
    'WISE_W4':        {'wave_A':220883.0,'label': 'WISE W4', 'zero_Jy': 8.363},
    # SPHEREx synth bands (0.75-5.0 um, QR2: 6 detectors D1-D6)
    'SPHEREx_1.0':    {'wave_A':  9750.0,'label': 'SPHEREx D1', 'zero_Jy': 3631.0},
    'SPHEREx_1.5':    {'wave_A': 14500.0,'label': 'SPHEREx D2', 'zero_Jy': 3631.0},
    'SPHEREx_2.0':    {'wave_A': 21000.0,'label': 'SPHEREx D3', 'zero_Jy': 3631.0},
    'SPHEREx_3.0':    {'wave_A': 32500.0,'label': 'SPHEREx D4/5','zero_Jy': 3631.0},
    'SPHEREx_4.5':    {'wave_A': 45500.0,'label': 'SPHEREx D6', 'zero_Jy': 3631.0},
    # HST common filters
    'HST_F555W':      {'wave_A':  5308.0,'label': 'HST F555W',  'zero_Jy': 3631.0},
    'HST_F606W':      {'wave_A':  5887.0,'label': 'HST F606W',  'zero_Jy': 3631.0},
    'HST_F814W':      {'wave_A':  8029.0,'label': 'HST F814W',  'zero_Jy': 3631.0},
    'HST_F275W':      {'wave_A':  2710.0,'label': 'HST F275W',  'zero_Jy': 3631.0},
    'HST_F336W':      {'wave_A':  3355.0,'label': 'HST F336W',  'zero_Jy': 3631.0},
    'HST_F438W':      {'wave_A':  4326.0,'label': 'HST F438W',  'zero_Jy': 3631.0},
    'HST_F110W':      {'wave_A': 11534.0,'label': 'HST F110W',  'zero_Jy': 3631.0},
    'HST_F160W':      {'wave_A': 15369.0,'label': 'HST F160W',  'zero_Jy': 3631.0},
    # JWST common filters
    'JWST_F070W':     {'wave_A':  7040.0,'label': 'JWST F070W', 'zero_Jy': 3631.0},
    'JWST_F090W':     {'wave_A':  9020.0,'label': 'JWST F090W', 'zero_Jy': 3631.0},
    'JWST_F115W':     {'wave_A': 11540.0,'label': 'JWST F115W', 'zero_Jy': 3631.0},
    'JWST_F150W':     {'wave_A': 15010.0,'label': 'JWST F150W', 'zero_Jy': 3631.0},
    'JWST_F200W':     {'wave_A': 19886.0,'label': 'JWST F200W', 'zero_Jy': 3631.0},
    'JWST_F277W':     {'wave_A': 27620.0,'label': 'JWST F277W', 'zero_Jy': 3631.0},
    'JWST_F356W':     {'wave_A': 35680.0,'label': 'JWST F356W', 'zero_Jy': 3631.0},
    'JWST_F444W':     {'wave_A': 44043.0,'label': 'JWST F444W', 'zero_Jy': 3631.0},
}
