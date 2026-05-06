# astro_toolbox

面向白矮星、双星和多波段巡天研究的 Python 工具箱。它把光谱、测光、光变曲线、SED、HR 图、视向速度、冷却年龄和轨道回溯放在一套统一接口里，也内置了 KOA/Keck LRIS 的下载与一维谱提取流程。

这个 README 的目标不是“介绍模块名”，而是让你直接知道：

1. 这个工具箱怎么装、怎么启动。
2. 每个模块的入口函数是什么。
3. 输出文件会落在哪里。
4. KOA 从元数据到 raw FITS 到一维谱要怎么跑。
5. 批处理 CSV 时应该看哪几个状态表。

---

## 1. 目录结构

```text
astro_toolbox/
├── __init__.py
├── config.py
├── utils.py
├── gui.py
├── diagnostics.py
├── combined_plots.py
├── koa.py
├── koa_batch.py
├── sdss.py
├── desi.py
├── galah.py
├── lamost.py
├── hst.py
├── jwst.py
├── spherex.py
├── ztf.py
├── wise.py
├── gaia_lc.py
├── tess.py
├── kepler.py
├── galex.py
├── twomass.py
├── xray.py
├── sed.py
├── hr_diagram.py
├── wd_fitting.py
├── cooling_age.py
├── period_analysis.py
├── rv_fitting.py
├── rv_correction.py
├── orbit_traceback.py
└── six_dim.py
```

---

## 2. 安装与环境

建议在项目父目录运行：

```bash
cd /Users/ljm/Desktop/csst/desi匹配
python -m pip install numpy pandas scipy matplotlib astropy astroquery requests
python -m pip install lightkurve galpy dustmaps
```

如果你要跑 KOA 的 LRIS 自动归约，还需要：

```bash
python -m pip install pypeit
```

如果你要在线查询 KOA 元数据，当前本机建议优先使用：

```bash
/opt/local/bin/python
```

因为之前默认 `python` 环境里 `bs4/xml` 解析器有兼容问题。

---

## 3. 快速开始

### 3.1 启动 GUI

```bash
cd /Users/ljm/Desktop/csst/desi匹配
python -m astro_toolbox.gui
```

GUI 支持：

- 单目标输入 `RA/DEC`
- CSV 批量处理
- 勾选模块组合查询
- 实时日志
- 结果图预览

### 3.2 Python 里直接调用

```python
from astro_toolbox import sdss, ztf, sed

ra, dec = 190.305, 2.596

spec = sdss.query_spectrum(ra, dec)
lc = ztf.query_lightcurve(ra, dec)

fitter = sed.SEDFitter(ra, dec)
fitter.collect_photometry()
fitter.apply_extinction()
fitter.plot('output/sed.png')
```

---

## 4. 统一输入输出约定

### 4.1 坐标输入

绝大多数模块都支持：

- `ra`, `dec`：十进制度
- `radius_arcsec`：搜索半径，默认通常来自 `config.SEARCH_RADIUS_ARCSEC`

### 4.2 常见返回格式

光谱模块通常返回：

```python
{
    'wavelength': ...,
    'flux': ...,
    'error': ...,
    'survey': ...,
    'instrument': ...,
    ...
}
```

光变曲线模块通常返回：

```python
{
    'g': {'time': ..., 'mag': ..., 'magerr': ...},
    'r': {...},
    ...
}
```

测光模块通常返回字典，键名是波段名，值里带 `mag / mag_err / wave_A / flux` 等字段。

### 4.3 默认缓存和输出目录

在 [config.py](/Users/ljm/Desktop/csst/desi匹配/astro_toolbox/config.py) 里：

- `CACHE_DIR = /Users/ljm/Desktop/csst/desi匹配/output/astro_cache`
- `OUTPUT_DIR = /Users/ljm/Desktop/csst/desi匹配/output/astro_output`

你之前看到很大的 `output/coadd_cache`、`output/astro_cache` 也正是这一套工作流写出来的缓存。

---

## 5. 基础模块教程

### 5.1 `config.py`

作用：

- 控制缓存目录、输出目录
- 控制代理、超时、重试
- 从环境变量或本地私有文件读取 ADS/Gaia/LAMOST token
- 定义 KOA 本地一维谱根目录
- 定义所有常用波段的参考波长和零点

你最常改的配置通常是：

- `KOA_LOCAL_ROOT`
- `PROXY_URL`
- `BAYESTAR2019_PATH`
- `CACHE_DIR`
- `OUTPUT_DIR`

敏感凭据不要写进 `config.py` 或提交到 GitHub。建议在本机 shell 配置里设置：

```bash
export ADS_DEV_KEY="..."
export GAIA_TOKEN="..."
export LAMOST_TOKEN="..."
export LAMOST_FTP_USER="..."
export LAMOST_FTP_PASSWORD="..."
```

也可以参考 `.env.example` 建一个本地 `.env`；`.gitignore` 已经排除了 `.env`。

### 5.2 `utils.py`

这是所有模块的底座。最常用函数：

- `get_session(url=None)`：自动判断是否走代理
- `query_vizier(...)`：VizieR 查询
- `query_simbad(...)`
- `query_simbad_references(...)`
- `download_file(...)`
- `mag_to_flux_jy(...)`
- `mag_to_flux_cgs(...)`
- `write_csv(...)`
- `spectrum_to_dataframe(...)`
- `photometry_to_dataframe(...)`
- `lightcurve_to_dataframe(...)`
- `mhaov(...)`：AoV 周期搜索

最短示例：

```python
from astro_toolbox import utils

tbl = utils.query_vizier('I/355/gaiadr3', 190.305, 2.596, radius_arcsec=3)
```

### 5.3 `gui.py`

入口：

```bash
python -m astro_toolbox.gui
```

GUI 里当前内置模块映射包括：

- `SDSS_spectrum`
- `GALAH`
- `LAMOST`
- `DESI`
- `KOA_spectrum`
- `SPHEREx`
- `ZTF_lightcurve`
- `WISE_lightcurve`
- `Gaia_lightcurve`
- `TESS`
- `Kepler/K2`
- `HST_spectrum`
- `HST_lightcurve`
- `JWST_spectrum`
- `JWST_lightcurve`
- `SDSS_photometry`
- `GALEX`
- `2MASS`
- `WISE_photometry`
- `X-ray`
- `HEASARC_Xray`
- `SED`
- `HR_diagram`
- `Binary_SED`
- `SIMBAD_refs`

### 5.4 `diagnostics.py`

用途：

- 自动检查谱线异常
- 标记发射线、低信噪比、非恒星特征
- 做 SED 诊断
- 做 RV 标志检查

常用函数：

- `analyze_spectrum(wave, flux, err=None, survey='', metadata=None)`
- `analyze_all_spectra(results)`
- `save_spectral_diagnostics(...)`
- `analyze_sed(flux_data)`
- `evaluate_rv_flags(rv_report)`

### 5.5 `combined_plots.py`

统一出多波段合成图：

- `plot_combined_spectra(results, save_path=None, ra=None, dec=None)`
- `plot_combined_fold(results, save_path=None, ra=None, dec=None)`
- `plot_spectra_with_photometry(results, save_path=None, ra=None, dec=None)`

---

## 6. 光谱模块教程

### 6.1 `sdss.py`

函数：

- `query_spectrum(ra, dec, radius_arcsec=...)`
- `get_photometry(ra, dec, radius_arcsec=...)`
- `plot_spectrum(result, save_path=None)`
- `save_spectrum_csv(result, output_dir)`
- `save_photometry_csv(result, output_dir)`

示例：

```python
from astro_toolbox import sdss

res = sdss.query_spectrum(190.305, 2.596)
sdss.plot_spectrum(res, 'sdss_spectrum.png')
sdss.save_spectrum_csv(res, 'output')
```

### 6.2 `desi.py`

函数：

- `query_spectrum(ra, dec, radius_arcsec=...)`

说明：

- 返回 DESI B/R/Z 光谱
- 常见输出是 FITS + PNG
- 适合和 `rv_fitting.py`、`combined_plots.py` 配套

示例：

```python
from astro_toolbox import desi

res = desi.query_spectrum(190.305, 2.596)
```

### 6.3 `galah.py`

函数：

- `query_spectrum(ra, dec, radius_arcsec=...)`
- `save_csv(result, output_dir)`

说明：

- 主要返回 GALAH DR4 参数表信息
- 不是本地直接下载完整高分辨率光谱的主入口

### 6.4 `lamost.py`

函数：

- `query_spectrum(ra, dec, radius_arcsec=...)`
- `plot_spectrum(result, save_path=None)`
- `save_csv(result, output_dir)`

示例：

```python
from astro_toolbox import lamost

res = lamost.query_spectrum(ra, dec)
lamost.plot_spectrum(res, 'lamost.png')
lamost.save_csv(res, 'output')
```

### 6.5 `hst.py`

函数：

- `query_spectrum(...)`
- `query_lightcurve(...)`
- `plot_spectrum(...)`
- `plot_lightcurve(...)`
- `save_spectrum_csv(...)`
- `save_lightcurve_csv(...)`

适合：

- HST COS/STIS/FOS 光谱
- HST 多历元测光

### 6.6 `jwst.py`

函数：

- `query_spectrum(...)`
- `query_lightcurve(...)`
- `plot_spectrum(...)`
- `plot_lightcurve(...)`
- `save_spectrum_csv(...)`
- `save_lightcurve_csv(...)`

适合：

- JWST NIRSpec / NIRISS / MIRI 光谱
- JWST 成像历元信息

### 6.7 `spherex.py`

函数：

- `query_spectrum(...)`
- `get_photometry(...)`
- `plot_spectrum(...)`
- `save_spectrum_csv(...)`
- `save_photometry_csv(...)`

说明：

- 从 cutout 组装低分辨率近红外光谱
- 自动做背景扣除

### 6.8 `koa.py`

这是 KOA/Keck LRIS 的核心模块。

最重要函数：

- `query_spectrum(...)`
- `prepare_koa_download(...)`
- `download_koa_files(...)`
- `setup_pypeit_lris(...)`
- `run_pypeit_reduction(...)`
- `download_and_extract_spectrum(...)`
- `plot_spectrum(...)`
- `save_csv(...)`
- `save_exposure_table(...)`
- `save_report(...)`

#### 6.8.1 只读取本地已经抽好的 1D 谱

```bash
cd /Users/ljm/Desktop/csst/desi匹配
python -m astro_toolbox.koa \
  --target ZTFJ035352.96+431525.16 \
  --local-root /Users/ljm/Desktop/DWD/speutrem \
  --output-dir /tmp/koa_test \
  --no-download
```

#### 6.8.2 在线查询 KOA 元数据，但先不下载

```bash
cd /Users/ljm/Desktop/csst/desi匹配
/opt/local/bin/python -m astro_toolbox.koa \
  --target ZTFJ035352.96+431525.16 \
  --work-dir /tmp/koa_query_only \
  --row-limit 5 \
  --no-download
```

#### 6.8.3 下载 raw FITS 并自动尝试抽一维

```bash
cd /Users/ljm/Desktop/csst/desi匹配
/opt/local/bin/python -m astro_toolbox.koa \
  --target ZTFJ035352.96+431525.16 \
  --work-dir /tmp/koa_download \
  --row-limit 5 \
  --no-calibfile \
  --auto-pypeit \
  --pypeit-setup-only
```

#### 6.8.4 KOA 输出文件

常见输出：

- `koa_spectrum.csv`
- `koa_spectrum.png`
- `koa_exposures.csv`
- `koa_spectrum_report.txt`
- `spec1d_*.fits`
- `spec2d_*.fits`

---

## 7. KOA 批处理教程

### 7.1 `koa_batch.py` 的用途

用于把一个 CSV 星表批量变成：

1. KOA 元数据表
2. raw FITS 下载目录
3. raw PNG 快视图
4. PypeIt setup
5. 一维谱 CSV/PNG/FITS

### 7.2 最常用命令

#### 只查询元数据，不下载

```bash
cd /Users/ljm/Desktop/csst/desi匹配
/opt/local/bin/python -m astro_toolbox.koa_batch \
  /Users/ljm/Desktop/csst/desi匹配/DWD_new/DWD_combined_clean.csv \
  --output-root /Users/ljm/Desktop/csst/desi匹配/DWD_new/koa_batch \
  --metadata-only
```

#### 根据已有 `*_selected.tbl` 下载 raw FITS

```bash
cd /Users/ljm/Desktop/csst/desi匹配
/opt/local/bin/python -m astro_toolbox.koa_batch \
  /Users/ljm/Desktop/csst/desi匹配/DWD_new/DWD_combined_clean.csv \
  --output-root /Users/ljm/Desktop/csst/desi匹配/DWD_new/koa_batch \
  --download-existing-metadata
```

#### 只下载最长科学曝光，减少下载量

```bash
cd /Users/ljm/Desktop/csst/desi匹配
/opt/local/bin/python -m astro_toolbox.koa_batch \
  /Users/ljm/Desktop/csst/desi匹配/DWD_new/DWD_combined_clean.csv \
  --output-root /Users/ljm/Desktop/csst/desi匹配/DWD_new/koa_batch \
  --download-existing-metadata \
  --fast-download \
  --fast-prefer-arm red
```

#### 对已有目录抽取一维谱

```bash
cd /Users/ljm/Desktop/csst/desi匹配
python -m astro_toolbox.koa_batch \
  /Users/ljm/Desktop/csst/desi匹配/DWD_new/DWD_combined_clean.csv \
  --output-root /Users/ljm/Desktop/csst/desi匹配/DWD_new/koa_batch \
  --reduce-existing
```

#### 走快提取路径

```bash
cd /Users/ljm/Desktop/csst/desi匹配
python -m astro_toolbox.koa_batch \
  /Users/ljm/Desktop/csst/desi匹配/DWD_new/DWD_combined_clean.csv \
  --output-root /Users/ljm/Desktop/csst/desi匹配/DWD_new/koa_batch \
  --reduce-existing \
  --fast-pypeit \
  --fast-prefer-arm red
```

#### 对某几个目标单独跑

```bash
cd /Users/ljm/Desktop/csst/desi匹配
python -m astro_toolbox.koa_batch \
  /Users/ljm/Desktop/csst/desi匹配/DWD_new/DWD_combined_clean.csv \
  --output-root /Users/ljm/Desktop/csst/desi匹配/DWD_new/koa_batch \
  --reduce-existing \
  --reduce-target ZTFJ035352.96+431525.16 \
  --reduce-target ZTFJ181311.11+425150.45
```

### 7.3 `koa_batch` 产物文件怎么读

以 `/Users/ljm/Desktop/csst/desi匹配/DWD_new/koa_batch` 为例：

- [koa_lris_observed_targets.csv](</Users/ljm/Desktop/csst/desi匹配/DWD_new/koa_batch/koa_lris_observed_targets.csv>)
  - 已知被 KOA/LRIS 观测过的目标级列表
  - 适合先和大星表做预匹配

- [koa_lris_observed_exposures.csv](</Users/ljm/Desktop/csst/desi匹配/DWD_new/koa_batch/koa_lris_observed_exposures.csv>)
  - 每一条 exposure 的观测信息
  - 适合看日期、光栅、狭缝、PI、项目号

- [koa_batch_summary.csv](</Users/ljm/Desktop/csst/desi匹配/DWD_new/koa_batch/koa_batch_summary.csv>)
  - 批量查询/下载的摘要
  - 字段：`n_metadata_rows / n_selected_rows / n_downloaded_files / target_dir`

- [koa_file_manifest.csv](</Users/ljm/Desktop/csst/desi匹配/DWD_new/koa_batch/koa_file_manifest.csv>)
  - 本地已下载文件清单
  - 最重要字段：`fits_path / png_path`

- `koa_reduction_summary.csv`
  - 一维谱提取和 reduction 状态表
  - 最重要字段：
    - `setup_status`
    - `spectrum_status`
    - `spectrum_csv`
    - `spectrum_png`
    - `spectrum_report`
    - `spectrum_exposures`

### 7.4 单目标目录结构

每个目标目录通常长这样：

```text
ZTFJxxxx/
├── metadata/
├── download/
│   └── lris/
│       ├── lev0/
│       └── calib/
├── png/
├── pypeit_setup/
└── spectrum/
```

其中：

- `metadata/`：KOA 查询结果表
- `download/`：原始 FITS
- `png/`：raw 快视图
- `pypeit_setup/`：给 PypeIt 的配置和标准化 raw
- `spectrum/`：最终一维谱结果

### 7.5 KOA 的几个现实限制

- KOA 对 LRIS 通常给的是 raw `lev0`，不是现成一维谱
- 真正科学级的一维流量定标谱，通常要靠 PypeIt
- 如果目录里已经有 `spec1d_*.fits`，工具箱会优先直接标准化
- 如果只有 `spec2d_*.fits`，工具箱会尝试 `_extract_1d_from_spec2d()` 做回退提取
- 如果只有 raw 而没有校准帧，`fast_pypeit` 也可能失败

---

## 8. 光变曲线模块教程

### 8.1 `ztf.py`

函数：

- `get_web_url(...)`
- `query_lightcurve(...)`
- `plot_lightcurve(...)`
- `save_csv(...)`

示例：

```python
from astro_toolbox import ztf

res = ztf.query_lightcurve(190.305, 2.596)
ztf.plot_lightcurve(res, 'ztf.png')
ztf.save_csv(res, 'output')
```

### 8.2 `wise.py`

函数：

- `get_photometry(...)`
- `query_lightcurve(...)`
- `plot_lightcurve(...)`
- `save_photometry_csv(...)`
- `save_lightcurve_csv(...)`

### 8.3 `gaia_lc.py`

函数：

- `query_lightcurve(...)`
- `plot_lightcurve(...)`
- `save_csv(...)`

### 8.4 `tess.py`

函数：

- `query_lightcurve(ra, dec, author='SPOC')`
- `plot_lightcurve(...)`
- `save_csv(...)`

### 8.5 `kepler.py`

函数：

- `query_lightcurve(ra, dec, mission='Kepler')`
- `plot_lightcurve(...)`
- `save_csv(...)`

---

## 9. 测光与巡天模块教程

### 9.1 `galex.py`

- `get_photometry(...)`
- `save_csv(...)`

### 9.2 `twomass.py`

- `get_photometry(...)`
- `save_csv(...)`

### 9.3 `xray.py`

函数：

- `query_rosat(...)`
- `query_xmm(...)`
- `query_chandra(...)`
- `query_erosita(...)`
- `query_xray(...)`
- `query_heasarc_browse(...)`
- `analyze_xray(...)`
- `save_analysis(...)`
- `save_csv(...)`
- `save_heasarc_csv(...)`

示例：

```python
from astro_toolbox.xray import query_xray, analyze_xray

res = query_xray(190.305, 2.596)
analysis = analyze_xray(res, ra=190.305, dec=2.596)
```

---

## 10. 物理分析模块教程

### 10.1 `sed.py`

常用对象和函数：

- `query_gaia_distance(...)`
- `query_bayestar_ebv(...)`
- `query_ebv(...)`
- `query_extinction(...)`
- `SEDFitter`

最短工作流：

```python
from astro_toolbox.sed import SEDFitter

fitter = SEDFitter(190.305, 2.596)
fitter.collect_photometry()
fitter.apply_extinction()
fitter.plot('sed.png')
```

### 10.2 `hr_diagram.py`

常用对象和函数：

- `HRDiagram`
- `classify_hr_position(...)`
- `quick_hr(...)`
- `save_analysis_report(...)`
- `save_csv(...)`

最短示例：

```python
from astro_toolbox.hr_diagram import HRDiagram

hr = HRDiagram()
res = hr.plot_single(190.305, 2.596, save_path='hr.png')
```

### 10.3 `wd_fitting.py`

常用拟合函数：

- `fit_single_wd(wave, flux, err=None, line_only=False)`
- `fit_sed(photometry, parallax_mas)`
- `fit_dwd(wave, flux, err=None, single_result=None)`

用途：

- DA/DB/DC 等光谱分类
- Koester 网格拟合
- 双白矮星复合拟合

### 10.4 `cooling_age.py`

函数：

- `get_gaia_photometry(...)`
- `run_cooling_age_analysis(...)`
- `run_batch_cooling_age(...)`
- `save_csv(...)`

### 10.5 `period_analysis.py`

函数：

- `analyze_folded_morphology(...)`
- `plot_combined_fold(...)`
- `run_period_analysis(...)`
- `save_csv(...)`

### 10.6 `rv_fitting.py`

函数：

- `run_rv_analysis(results, output_dir=None, ra=None, dec=None)`
- `save_csv(...)`

### 10.7 `rv_correction.py`

函数：

- `fit_line_core(...)`
- `run_rv_correction(...)`
- `plot_line_fits(...)`
- `save_csv(...)`

---

## 11. 动力学模块教程

### 11.1 `orbit_traceback.py`

函数：

- `get_gaia_astrometry(...)`
- `run_traceback_analysis(...)`
- `save_csv(...)`

示例：

```python
from astro_toolbox.orbit_traceback import run_traceback_analysis

res = run_traceback_analysis(results, rv_report, output_dir='traceback_out')
```

### 11.2 `six_dim.py`

主要是六维成员分析和出图模块。常见入口是它的一组绘图函数：

- `plot_spectrum(...)`
- `plot_ztf(...)`
- `plot_hrd(...)`
- `plot_rv_info(...)`
- `plot_sed(...)`
- `plot_total_hrd(...)`

适合把光谱、光变、HR 图、RV、SED 拼成一份综合判据图。

---

## 12. 典型工作流

### 工作流 A：单目标多波段检查

1. `sdss/desi/lamost/koa` 查光谱
2. `ztf/wise/gaia_lc/tess` 查光变
3. `galex/twomass/wise` 查测光
4. `sed.py` 做 SED
5. `hr_diagram.py` 看 Gaia CMD 位置
6. `period_analysis.py` 跑周期
7. `rv_fitting.py` / `rv_correction.py` 跑 RV

### 工作流 B：KOA 大表匹配

1. 用 `koa_lris_observed_targets.csv` 先和你的源表做坐标/名称匹配
2. 对匹配上的源查看 `koa_lris_observed_exposures.csv`
3. 用 `koa_batch.py --download-existing-metadata` 下载 raw
4. 用 `koa_batch.py --reduce-existing` 生成一维谱
5. 结果最终看 `koa_reduction_summary.csv`

### 工作流 C：白矮星年龄和母星团分析

1. `hr_diagram.py` 定位
2. `wd_fitting.py` 求 `Teff/logg`
3. `cooling_age.py` 算冷却年龄
4. `orbit_traceback.py` 回溯轨道
5. `six_dim.py` 合成图

---

## 13. 常见问题

### Q1. 为什么 KOA 下载后没有直接的一维谱？

因为 LRIS 在 KOA 常见产品是 raw `lev0`。一维谱要么来自你已有的 `spec1d`，要么需要工具箱配合 PypeIt 去做提取。

### Q2. `koa_batch_summary.csv` 和真实文件数不一致怎么办？

以文件系统和 `koa_file_manifest.csv`、`koa_reduction_summary.csv` 为准。`koa_batch_summary.csv` 更偏向查询/下载阶段的摘要。

### Q3. `spec1d` 和 `spec2d` 的区别？

- `spec1d`：已经抽出的一维谱，最理想
- `spec2d`：二维校准后谱图，工具箱可以尝试回退提取一维

### Q4. 什么时候用 `fast-pypeit`？

当你只想先拿到一条“够看”的一维谱，而不是完整科学级归约时，用它最快。

### Q5. 哪些目录最占空间？

通常是：

- `output/coadd_cache`
- `output/astro_cache`
- `koa_batch/*/download`
- `koa_batch/*/pypeit_setup`

---

## 14. 你当前 KOA 结果目录

你当前 KOA 批处理输出根目录是：

`/Users/ljm/Desktop/csst/desi匹配/DWD_new/koa_batch`

你当前工具箱源码目录是：

`/Users/ljm/Desktop/csst/desi匹配/astro_toolbox`

建议你日常主要看这几个文件：

- [README.md](/Users/ljm/Desktop/csst/desi匹配/astro_toolbox/README.md)
- [koa_batch.py](/Users/ljm/Desktop/csst/desi匹配/astro_toolbox/koa_batch.py)
- [koa.py](/Users/ljm/Desktop/csst/desi匹配/astro_toolbox/koa.py)
- [koa_lris_observed_targets.csv](</Users/ljm/Desktop/csst/desi匹配/DWD_new/koa_batch/koa_lris_observed_targets.csv>)
- [koa_lris_observed_exposures.csv](</Users/ljm/Desktop/csst/desi匹配/DWD_new/koa_batch/koa_lris_observed_exposures.csv>)
- [koa_file_manifest.csv](</Users/ljm/Desktop/csst/desi匹配/DWD_new/koa_batch/koa_file_manifest.csv>)
- [koa_batch_summary.csv](</Users/ljm/Desktop/csst/desi匹配/DWD_new/koa_batch/koa_batch_summary.csv>)

---

## 15. 更新记录

- `2026-04-26`：重写 README，加入 KOA 批处理、模块教程、输出文件说明和常见工作流。
