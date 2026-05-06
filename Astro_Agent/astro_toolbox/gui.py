"""
统一 Tkinter GUI — 多波段天文工具箱
=====================================
一键查询所有波段数据，实时显示进度，内嵌 matplotlib 图表预览。
支持模块选择、CSV 批量处理。

用法:
    python -m astro_toolbox.gui
"""
import os
import sys
import threading
import queue
import subprocess

import matplotlib
matplotlib.use('TkAgg')

import tkinter as tk
from tkinter import ttk, filedialog

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.image as mpimg

# 模块列表: (key, category, description)
MODULE_LIST = [
    ('SDSS_spectrum',   'Spectra',    'SDSS DR18 optical spectrum'),
    ('GALAH',           'Spectra',    'GALAH DR4 info'),
    ('LAMOST',          'Spectra',    'LAMOST DR8 optical spectrum'),
    ('DESI',            'Spectra',    'DESI DR1 B/R/Z spectrum'),
    ('KOA_spectrum',    'Spectra',    'KOA/Keck local LRIS spectrum'),
    ('SPHEREx',         'Spectra',    'SPHEREx low-resolution spectrum'),
    ('ZTF_lightcurve',  'LightCurve', 'ZTF DR23 g/r/i'),
    ('WISE_lightcurve', 'LightCurve', 'NEOWISE W1/W2'),
    ('Gaia_lightcurve', 'LightCurve', 'Gaia DR3 epoch phot'),
    ('TESS',            'LightCurve', 'TESS SPOC'),
    ('Kepler/K2',       'LightCurve', 'Kepler or K2'),
    ('HST_spectrum',    'Spectra',    'HST COS/STIS spectrum'),
    ('HST_lightcurve',  'LightCurve', 'HST multi-epoch phot'),
    ('JWST_spectrum',   'Spectra',    'JWST NIRSpec/MIRI spectrum'),
    ('JWST_lightcurve', 'LightCurve', 'JWST multi-epoch phot'),
    ('SDSS_photometry', 'Photometry', 'SDSS ugriz'),
    ('GALEX',           'Photometry', 'GALEX FUV/NUV'),
    ('2MASS',           'Photometry', '2MASS JHKs'),
    ('WISE_photometry', 'Photometry', 'AllWISE W1-W4'),
    ('X-ray',           'X-ray',      'ROSAT/XMM/Chandra/eROSITA'),
    ('HEASARC_Xray',    'X-ray',      'HEASARC Browse (Swift/NuSTAR/...)'),
    ('SED',             'Analysis',   'Multi-band SED + BB fit'),
    ('HR_diagram',      'Analysis',   'Gaia HR diagram + region/age'),
    ('Binary_SED',      'Analysis',   'WD+M-dwarf binary SED fit'),
    ('SIMBAD_refs',     'Analysis',   'SIMBAD literature refs'),
]

# module key -> plot filename mapping
PLOT_MAP = {
    'SDSS_spectrum':   'sdss_spectrum',
    'LAMOST':          'lamost_spectrum',
    'DESI':            'desi_spectrum',
    'KOA_spectrum':    'koa_spectrum',
    'SPHEREx':         'spherex_spectrum',
    'ZTF_lightcurve':  'ztf_lightcurve',
    'WISE_lightcurve': 'wise_lightcurve',
    'Gaia_lightcurve': 'gaia_lightcurve',
    'TESS':            'tess_lightcurve',
    'Kepler/K2':       'kepler_lightcurve',
    'HST_spectrum':    'hst_spectrum',
    'HST_lightcurve':  'hst_lightcurve',
    'JWST_spectrum':   'jwst_spectrum',
    'JWST_lightcurve': 'jwst_lightcurve',
    'SED':             'sed',
    'HR_diagram':      'hr_diagram',
    'Binary_SED':      'binary_sed',
}


class AstroToolboxGUI:
    """多波段天文工具箱 — 统一 GUI"""

    def __init__(self, root):
        self.root = root
        self._msg_queue = queue.Queue()
        self._cancel_event = threading.Event()
        self._plot_paths = {}
        self._plot_names = []
        self._current_plot_idx = -1
        self._completed_count = 0
        self._output_dir = None
        self._csv_path = None
        self._batch_mode = False
        self._build_ui()
        self._poll_queue()

    # ================================================================
    #  UI 构建
    # ================================================================

    def _build_ui(self):
        style = ttk.Style()
        for theme in ('aqua', 'clam', 'default'):
            try:
                style.theme_use(theme)
                break
            except Exception:
                continue

        main = ttk.Frame(self.root, padding=8)
        main.pack(fill=tk.BOTH, expand=True)

        self._build_coord_bar(main)
        self._build_module_selector(main)

        # 中间区域: 左=状态表, 右=图表预览
        paned = ttk.PanedWindow(main, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, pady=5)

        left = ttk.Frame(paned)
        right = ttk.Frame(paned)
        paned.add(left, weight=2)
        paned.add(right, weight=3)

        self._build_status_table(left)
        self._build_plot_panel(right)

        self._build_log_area(main)
        self._build_progress_bar(main)

    def _build_coord_bar(self, parent):
        frame = ttk.LabelFrame(parent, text="Target / Input", padding=5)
        frame.pack(fill=tk.X, pady=(0, 5))

        # 第一行: RA/DEC 单目标
        row1 = ttk.Frame(frame)
        row1.pack(fill=tk.X, pady=(0, 3))

        ttk.Label(row1, text="RA (deg):").pack(side=tk.LEFT)
        self.var_ra = tk.StringVar(value="190.305")
        ttk.Entry(row1, textvariable=self.var_ra, width=14).pack(
            side=tk.LEFT, padx=3)

        ttk.Label(row1, text="DEC (deg):").pack(side=tk.LEFT, padx=(10, 0))
        self.var_dec = tk.StringVar(value="2.596")
        ttk.Entry(row1, textvariable=self.var_dec, width=14).pack(
            side=tk.LEFT, padx=3)

        ttk.Label(row1, text='Radius ("):').pack(side=tk.LEFT, padx=(10, 0))
        self.var_radius = tk.StringVar(value="5.0")
        ttk.Entry(row1, textvariable=self.var_radius, width=6).pack(
            side=tk.LEFT, padx=3)

        self.btn_query = ttk.Button(row1, text="一键查询",
                                    command=self._start_query)
        self.btn_query.pack(side=tk.LEFT, padx=(20, 5))

        self.btn_cancel = ttk.Button(row1, text="取消",
                                     command=self._cancel_query,
                                     state=tk.DISABLED)
        self.btn_cancel.pack(side=tk.LEFT, padx=3)

        self.btn_open = ttk.Button(row1, text="打开输出目录",
                                   command=self._open_output_dir)
        self.btn_open.pack(side=tk.RIGHT, padx=3)

        # 第二行: CSV 批量输入
        row2 = ttk.Frame(frame)
        row2.pack(fill=tk.X)

        ttk.Label(row2, text="CSV 批量:").pack(side=tk.LEFT)

        self.var_csv_label = tk.StringVar(value="未选择文件")
        ttk.Label(row2, textvariable=self.var_csv_label,
                  foreground='gray', width=30).pack(side=tk.LEFT, padx=5)

        self.btn_browse_csv = ttk.Button(row2, text="选择CSV文件...",
                                         command=self._browse_csv)
        self.btn_browse_csv.pack(side=tk.LEFT, padx=3)

        ttk.Label(row2, text="RA列:").pack(side=tk.LEFT, padx=(8, 0))
        self.var_ra_col = tk.StringVar()
        self.combo_ra_col = ttk.Combobox(row2, textvariable=self.var_ra_col,
                                         width=14, state='disabled')
        self.combo_ra_col.pack(side=tk.LEFT, padx=2)

        ttk.Label(row2, text="DEC列:").pack(side=tk.LEFT, padx=(5, 0))
        self.var_dec_col = tk.StringVar()
        self.combo_dec_col = ttk.Combobox(row2, textvariable=self.var_dec_col,
                                          width=14, state='disabled')
        self.combo_dec_col.pack(side=tk.LEFT, padx=2)

        self.btn_batch = ttk.Button(row2, text="批量查询",
                                    command=self._start_batch_query,
                                    state=tk.DISABLED)
        self.btn_batch.pack(side=tk.LEFT, padx=5)

        self.btn_clear_csv = ttk.Button(row2, text="清除",
                                        command=self._clear_csv,
                                        state=tk.DISABLED)
        self.btn_clear_csv.pack(side=tk.LEFT, padx=3)

    def _build_module_selector(self, parent):
        """模块选择区: 复选框网格 + 全选/取消全选"""
        frame = ttk.LabelFrame(parent, text="Select Modules (勾选要查询的数据)",
                               padding=5)
        frame.pack(fill=tk.X, pady=(0, 5))

        # 全选/取消按钮
        btn_row = ttk.Frame(frame)
        btn_row.pack(fill=tk.X, pady=(0, 3))
        ttk.Button(btn_row, text="全选",
                   command=self._select_all_modules).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_row, text="取消全选",
                   command=self._deselect_all_modules).pack(side=tk.LEFT, padx=3)

        # 按类别分组
        categories = {}
        for key, category, desc in MODULE_LIST:
            categories.setdefault(category, []).append((key, desc))

        self._module_vars = {}  # key -> BooleanVar

        grid = ttk.Frame(frame)
        grid.pack(fill=tk.X)

        col = 0
        for cat_name, items in categories.items():
            cat_frame = ttk.LabelFrame(grid, text=cat_name, padding=3)
            cat_frame.grid(row=0, column=col, sticky='nsew', padx=3, pady=2)
            grid.columnconfigure(col, weight=1)

            for i, (key, desc) in enumerate(items):
                var = tk.BooleanVar(value=True)
                self._module_vars[key] = var
                cb = ttk.Checkbutton(cat_frame, text=key, variable=var)
                cb.pack(anchor='w')

            col += 1

    def _select_all_modules(self):
        for var in self._module_vars.values():
            var.set(True)

    def _deselect_all_modules(self):
        for var in self._module_vars.values():
            var.set(False)

    def _get_enabled_modules(self):
        """返回用户勾选的模块列表"""
        return [key for key, var in self._module_vars.items() if var.get()]

    def _browse_csv(self):
        path = filedialog.askopenfilename(
            title="选择 CSV 文件 (需含 RA/DEC 列)",
            filetypes=[('CSV files', '*.csv'), ('All files', '*.*')],
        )
        if not path:
            return
        self._csv_path = path
        basename = os.path.basename(path)

        # 只读第一行获取列名 (不读全部数据, 避免卡顿)
        try:
            with open(path, 'r') as f:
                header_line = f.readline().strip()
            import csv
            reader = csv.reader([header_line])
            columns = next(reader)
            columns = [c.strip() for c in columns]
        except Exception as e:
            self.var_csv_label.set(f"{basename} (读取失败)")
            self._log(f"CSV 读取错误: {e}")
            return

        # 统计行数 (快速逐行计数, 不加载到内存)
        try:
            with open(path, 'r') as f:
                n_total = sum(1 for _ in f) - 1  # 减去表头
            self.var_csv_label.set(f"{basename} ({n_total} rows)")
        except Exception:
            self.var_csv_label.set(basename)

        # 设置列选择下拉框
        self.combo_ra_col['values'] = columns
        self.combo_dec_col['values'] = columns
        self.combo_ra_col['state'] = 'readonly'
        self.combo_dec_col['state'] = 'readonly'

        # 自动猜测 RA/DEC 列
        ra_guess = dec_guess = ''
        for c in columns:
            cl = c.lower().strip()
            if cl in ('ra', 'right_ascension', 'ra_deg'):
                ra_guess = c
            elif cl in ('dec', 'declination', 'de', 'dec_deg'):
                dec_guess = c
        self.var_ra_col.set(ra_guess)
        self.var_dec_col.set(dec_guess)

        self.btn_batch['state'] = tk.NORMAL
        self.btn_clear_csv['state'] = tk.NORMAL

    def _clear_csv(self):
        self._csv_path = None
        self.var_csv_label.set("未选择文件")
        self.var_ra_col.set('')
        self.var_dec_col.set('')
        self.combo_ra_col['values'] = []
        self.combo_dec_col['values'] = []
        self.combo_ra_col['state'] = 'disabled'
        self.combo_dec_col['state'] = 'disabled'
        self.btn_batch['state'] = tk.DISABLED
        self.btn_clear_csv['state'] = tk.DISABLED

    def _build_status_table(self, parent):
        frame = ttk.LabelFrame(parent, text="Query Status", padding=5)
        frame.pack(fill=tk.BOTH, expand=True)

        columns = ('module', 'category', 'status', 'time', 'info')
        self.tree = ttk.Treeview(frame, columns=columns, show='headings',
                                 selectmode='browse', height=16)
        self.tree.heading('module', text='Module')
        self.tree.heading('category', text='Type')
        self.tree.heading('status', text='Status')
        self.tree.heading('time', text='Time')
        self.tree.heading('info', text='Info')

        self.tree.column('module', width=120, minwidth=90)
        self.tree.column('category', width=75, anchor='center', minwidth=60)
        self.tree.column('status', width=65, anchor='center', minwidth=50)
        self.tree.column('time', width=50, anchor='center', minwidth=40)
        self.tree.column('info', width=220, minwidth=100)

        self.tree.tag_configure('ok', foreground='#228B22')
        self.tree.tag_configure('error', foreground='#DC143C')
        self.tree.tag_configure('no_data', foreground='#808080')
        self.tree.tag_configure('querying', foreground='#4169E1')
        self.tree.tag_configure('pending', foreground='#A0A0A0')

        scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL,
                               command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(fill=tk.BOTH, expand=True)

        # 预填充
        self._tree_iids = {}
        for name, category, desc in MODULE_LIST:
            iid = self.tree.insert(
                '', tk.END, text=name,
                values=(name, category, '--', '--', desc),
                tags=('pending',))
            self._tree_iids[name] = iid

        self.tree.bind('<<TreeviewSelect>>', self._on_tree_select)

    def _build_plot_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="Plot Preview", padding=5)
        frame.pack(fill=tk.BOTH, expand=True)

        # matplotlib canvas
        self.fig = Figure(figsize=(6, 4.5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 导航
        nav = ttk.Frame(frame)
        nav.pack(fill=tk.X, pady=3)
        ttk.Button(nav, text="<< Prev", command=self._prev_plot).pack(
            side=tk.LEFT, padx=3)
        ttk.Button(nav, text="Next >>", command=self._next_plot).pack(
            side=tk.LEFT, padx=3)
        self.var_plot_label = tk.StringVar(value="No plot yet")
        ttk.Label(nav, textvariable=self.var_plot_label).pack(
            side=tk.LEFT, padx=10)
        ttk.Button(nav, text="Open Full Size",
                   command=self._open_current_plot).pack(side=tk.RIGHT, padx=3)

    def _build_log_area(self, parent):
        frame = ttk.LabelFrame(parent, text="Log", padding=5)
        frame.pack(fill=tk.BOTH, expand=False, pady=3)

        self.log_text = tk.Text(frame, height=6, wrap=tk.WORD,
                                font=('Menlo', 11))
        scroll = ttk.Scrollbar(frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _build_progress_bar(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=(3, 0))

        self.progress = ttk.Progressbar(frame, mode='determinate', maximum=100)
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        self.var_progress_text = tk.StringVar(value="Ready")
        ttk.Label(frame, textvariable=self.var_progress_text).pack(
            side=tk.LEFT)

        self.var_output_label = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.var_output_label,
                  foreground='gray').pack(side=tk.RIGHT)

    # ================================================================
    #  消息队列轮询
    # ================================================================

    def _poll_queue(self):
        try:
            while True:
                msg_type, data = self._msg_queue.get_nowait()
                if msg_type == 'log':
                    self.log_text.insert('end', str(data) + '\n')
                    self.log_text.see('end')
                elif msg_type == 'module_status':
                    name, status, result, elapsed = data
                    self._update_tree_row(name, status, elapsed, result)
                elif msg_type == 'batch_progress':
                    idx, total, label = data
                    self.var_progress_text.set(
                        f"Batch: {idx}/{total}  {label}")
                elif msg_type == 'reset_table':
                    self._reset_status_table()
                    self._completed_count = 0
                    self.progress['value'] = 0
                elif msg_type == 'done':
                    self._on_all_done(data)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _log(self, msg):
        self._msg_queue.put(('log', msg))

    # ================================================================
    #  查询编排
    # ================================================================

    def _start_query(self):
        try:
            ra = float(self.var_ra.get())
            dec = float(self.var_dec.get())
        except ValueError:
            self._msg_queue.put(('log', 'Error: RA and DEC must be numbers'))
            return

        enabled = self._get_enabled_modules()
        if not enabled:
            self._msg_queue.put(('log', 'Error: 请至少选择一个模块'))
            return

        self._batch_mode = False
        self._set_busy(True)
        self._reset_status_table()
        self.log_text.delete('1.0', 'end')
        self._plot_paths = {}
        self._plot_names = []
        self._current_plot_idx = -1
        self._completed_count = 0
        self.progress['value'] = 0

        self._log(f"开始查询: RA={ra:.6f}, DEC={dec:.6f}"
                  f"  ({len(enabled)}/{len(MODULE_LIST)} modules)")

        self._cancel_event.clear()
        t = threading.Thread(target=self._query_worker,
                             args=(ra, dec, enabled), daemon=True)
        t.start()

    def _start_batch_query(self):
        """从 CSV 文件批量查询"""
        if not self._csv_path:
            return

        ra_col = self.var_ra_col.get().strip()
        dec_col = self.var_dec_col.get().strip()
        if not ra_col or not dec_col:
            self._msg_queue.put(('log', 'Error: 请选择 RA 和 DEC 列'))
            return

        enabled = self._get_enabled_modules()
        if not enabled:
            self._msg_queue.put(('log', 'Error: 请至少选择一个模块'))
            return

        self._batch_mode = True
        self._set_busy(True)
        self.log_text.delete('1.0', 'end')
        self._plot_paths = {}
        self._plot_names = []
        self._current_plot_idx = -1
        self._completed_count = 0
        self.progress['value'] = 0

        self._log(f"批量查询: {self._csv_path}  RA={ra_col}  DEC={dec_col}")

        self._cancel_event.clear()
        t = threading.Thread(target=self._batch_worker,
                             args=(self._csv_path, enabled, ra_col, dec_col),
                             daemon=True)
        t.start()

    def _cancel_query(self):
        self._cancel_event.set()
        self._log("取消请求已发送，等待运行中的查询结束...")
        self.var_progress_text.set("Cancelling...")

    def _query_worker(self, ra, dec, enabled_modules):
        """后台线程: 运行 AstroQueryAll"""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        scripts_dir = os.path.join(project_root, 'scripts')
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from test_toolbox import AstroQueryAll

        output_dir = os.path.join(project_root, 'output', 'astro_output',
                                  f'RA{ra:.4f}_DEC{dec:.4f}')
        self._output_dir = output_dir

        querier = AstroQueryAll(ra, dec, output_dir=output_dir,
                                enabled_modules=enabled_modules)
        querier.status_callback = self._on_module_callback

        try:
            querier.query_all()

            if self._cancel_event.is_set():
                self._log("已取消，跳过保存")
                self._msg_queue.put(('done', output_dir))
                return

            self._log("正在保存数据和生成图表...")
            querier.save_and_plot_all()
            self._msg_queue.put(('done', output_dir))
        except Exception as e:
            self._log(f"Error: {e}")
            self._msg_queue.put(('done', output_dir))

    def _batch_worker(self, csv_path, enabled_modules, ra_col, dec_col):
        """后台线程: CSV 批量查询"""
        import pandas as pd

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        scripts_dir = os.path.join(project_root, 'scripts')
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from test_toolbox import AstroQueryAll

        output_base = os.path.join(project_root, 'output', 'astro_output')

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            self._log(f"CSV 读取失败: {e}")
            self._msg_queue.put(('done', output_base))
            return

        if ra_col not in df.columns or dec_col not in df.columns:
            self._log(f"列名不存在: RA={ra_col}, DEC={dec_col}, "
                      f"可用: {list(df.columns)}")
            self._msg_queue.put(('done', output_base))
            return

        total = len(df)
        self._log(f"共 {total} 个源, 开始批量查询...")

        last_output_dir = output_base
        for i, row in df.iterrows():
            if self._cancel_event.is_set():
                self._log(f"已取消, 完成 {i}/{total}")
                break

            ra = float(row[ra_col])
            dec = float(row[dec_col])

            # 构建源名称
            parts = []
            if 'best_cluster' in df.columns and pd.notna(row.get('best_cluster')):
                parts.append(str(row['best_cluster']))
            if 'source_id' in df.columns and pd.notna(row.get('source_id')):
                try:
                    sid = str(int(float(row['source_id'])))
                except (ValueError, OverflowError):
                    sid = str(row['source_id'])
                parts.append(f"Gaia_{sid}")
            if 'name' in df.columns and pd.notna(row.get('name')):
                parts.append(str(row['name']))

            dir_name = '_'.join(parts) if parts else f"RA{ra:.4f}_DEC{dec:.4f}"
            dir_name = dir_name.replace(' ', '_').replace('/', '_')
            out_dir = os.path.join(output_base, dir_name)
            last_output_dir = out_dir

            source_label = parts[0] if parts else f"({ra:.4f},{dec:.4f})"
            self._msg_queue.put(('batch_progress', (i + 1, total, source_label)))
            self._log(f"\n--- [{i+1}/{total}] {source_label} "
                      f"RA={ra:.4f} DEC={dec:.4f} ---")

            self._msg_queue.put(('reset_table', None))

            querier = AstroQueryAll(ra, dec, output_dir=out_dir,
                                    enabled_modules=enabled_modules)
            querier.status_callback = self._on_module_callback
            try:
                querier.query_all()
                if not self._cancel_event.is_set():
                    querier.save_and_plot_all()
            except Exception as e:
                self._log(f"  源 {source_label} 处理失败: {e}")

        self._output_dir = last_output_dir
        self._msg_queue.put(('done', last_output_dir))

    def _on_module_callback(self, name, status, result, elapsed):
        """从工作线程调用 — 通过消息队列传递到主线程"""
        self._msg_queue.put(('module_status', (name, status, result, elapsed)))

    # ================================================================
    #  状态表更新
    # ================================================================

    def _reset_status_table(self):
        for name, category, desc in MODULE_LIST:
            iid = self._tree_iids.get(name)
            if iid:
                self.tree.item(iid,
                               values=(name, category, '--', '--', desc),
                               tags=('pending',))

    def _update_tree_row(self, name, status, elapsed, result):
        iid = self._tree_iids.get(name)
        if not iid:
            return

        status_map = {
            'querying': '...',
            'ok': 'OK',
            'no_data': 'No Data',
            'error': 'ERROR',
        }
        status_str = status_map.get(status, status)
        time_str = f"{elapsed:.1f}s" if elapsed > 0 else "..."
        info_str = self._extract_info(name, status, result)

        old_vals = self.tree.item(iid, 'values')
        self.tree.item(iid,
                       values=(old_vals[0], old_vals[1],
                               status_str, time_str, info_str),
                       tags=(status,))

        if status in ('ok', 'no_data', 'error'):
            self._completed_count += 1
            enabled = self._get_enabled_modules()
            total = len(enabled) if enabled else len(MODULE_LIST)
            self.progress['value'] = self._completed_count / total * 100
            if not self._batch_mode:
                self.var_progress_text.set(
                    f"{self._completed_count}/{total} complete")

    def _extract_info(self, name, status, result):
        """提取简短信息"""
        if status == 'querying':
            return 'querying...'
        if status != 'ok' or result is None:
            if status == 'error' and isinstance(result, str):
                return result[:50]
            return ''

        try:
            if name == 'SDSS_spectrum':
                s = f"class={result.get('class','')}"
                if result.get('obs_mjd'):
                    s += f"  MJD={result['obs_mjd']}"
                return s
            elif name == 'SDSS_photometry':
                return f"{len(result)} bands"
            elif name == 'GALAH':
                return f"sobject_id={result.get('sobject_id','?')}"
            elif name == 'LAMOST':
                s = f"class={result.get('class','')}"
                if result.get('rv') is not None:
                    s += f" RV={result['rv']:.1f}"
                return s
            elif name == 'DESI':
                m = result.get('match', {})
                sp = result.get('spectrum', {})
                s = f"TID={m.get('targetid','?')}"
                if sp.get('obs_nights'):
                    s += f"  nights={sp['obs_nights']}"
                return s
            elif name == 'KOA_spectrum':
                return (f"{result.get('n_files',0)} files "
                        f"{result.get('arms','')} "
                        f"MJD={result.get('obs_mjd',0):.2f}")
            elif name == 'SPHEREx':
                return f"{result.get('n_channels',0)} channels"
            elif name == 'ZTF_lightcurve':
                bands = [b for b in ('g', 'r', 'i') if b in result]
                n = result.get('n_epochs', 0)
                return f"{bands} {n} epochs"
            elif name == 'WISE_lightcurve':
                return f"{result.get('n_epochs',0)} epochs"
            elif name == 'Gaia_lightcurve':
                bands = [b for b in ('G', 'BP', 'RP') if b in result]
                return f"bands={bands}"
            elif name == 'TESS':
                return f"{result.get('n_points',0)} pts sectors={result.get('sectors','')}"
            elif name == 'Kepler/K2':
                return f"{result.get('n_points',0)} pts"
            elif name == 'HST_spectrum':
                s = f"{result.get('instrument','')}"
                if result.get('obs_id'):
                    s += f"  {result['obs_id']}"
                return s
            elif name == 'HST_lightcurve':
                n_filt = len(result.get('filters', {}))
                return f"{n_filt} filters, {result.get('n_epochs',0)} epochs"
            elif name == 'JWST_spectrum':
                s = f"{result.get('instrument','')}"
                if result.get('grating'):
                    s += f" {result['grating']}"
                return s
            elif name == 'JWST_lightcurve':
                n_filt = len(result.get('filters', {}))
                return f"{n_filt} filters, {result.get('n_epochs',0)} epochs"
            elif name in ('GALEX', '2MASS', 'WISE_photometry'):
                return f"{len(result)} bands" if isinstance(result, dict) else ''
            elif name == 'X-ray':
                hits = [k for k in result if result[k]] if isinstance(result, dict) else []
                return ', '.join(hits) if hits else 'no detection'
            elif name == 'HEASARC_Xray':
                if isinstance(result, dict):
                    cats = list(result.keys())
                    return ', '.join(cats) if cats else 'no detection'
                return ''
            elif name == 'SED':
                n = len(getattr(result, 'flux_data', {}))
                return f"{n} bands"
            elif name == 'HR_diagram':
                p = result.get('params') if isinstance(result, dict) else None
                if p:
                    s = f"BP-RP={p.get('BP_RP',0):.2f} M_G={p.get('M_G',0):.2f}"
                    analysis = p.get('hr_analysis') or result.get('analysis')
                    if analysis:
                        s += f" {analysis.get('region_label', analysis.get('region'))}"
                        wd = analysis.get('wd_model') or {}
                        if wd.get('status') == 'ok':
                            s += f" tcool={wd.get('cooling_age_myr',0):.0f}Myr"
                    if 'Teff' in p:
                        s += f" {p['Teff']:.0f}K"
                    return s
            elif name == 'Binary_SED':
                if hasattr(result, 'fit_result') and result.fit_result:
                    fr = result.fit_result
                    return (f"WD:{fr['wd_teff']}K "
                            f"M:{fr['mdwarf_type']} "
                            f"chi2={fr['chi2_red']:.1f}")
                return ''
            elif name == 'SIMBAD_refs':
                mid = result.get('main_id', '?')
                n = result.get('n_refs', 0)
                otype = result.get('otype', '')
                if n > 0:
                    return f"{mid} ({otype}) {n} refs"
                return f"{mid} ({otype}) no refs"
        except Exception:
            pass
        return ''

    # ================================================================
    #  完成处理 + 图表显示
    # ================================================================

    def _on_all_done(self, output_dir):
        self._set_busy(False)
        if output_dir and os.path.isdir(output_dir):
            self.var_output_label.set(f"Output: {output_dir}")
            self._scan_plots(output_dir)
            if self._plot_names:
                self._show_plot_at_index(0)
            self._log(f"完成! {len(self._plot_names)} plots saved to {output_dir}")
            self.var_progress_text.set(
                f"Done - {len(self._plot_names)} plots")
        else:
            self.var_progress_text.set("Done")

    def _scan_plots(self, output_dir):
        self._plot_paths = {}
        self._plot_names = []
        for f in sorted(os.listdir(output_dir)):
            if f.endswith('.png'):
                name = f[:-4]
                self._plot_paths[name] = os.path.join(output_dir, f)
                self._plot_names.append(name)

    def _show_plot_at_index(self, idx):
        if not self._plot_names:
            return
        idx = max(0, min(idx, len(self._plot_names) - 1))
        self._current_plot_idx = idx
        name = self._plot_names[idx]
        path = self._plot_paths[name]
        self.var_plot_label.set(
            f"{name} ({idx + 1}/{len(self._plot_names)})")

        self.fig.clear()
        ax = self.fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        try:
            img = mpimg.imread(path)
            ax.imshow(img, aspect='auto')
        except Exception as e:
            ax.text(0.5, 0.5, f"Cannot load:\n{e}",
                    ha='center', va='center', transform=ax.transAxes)
        self.canvas.draw()

    def _next_plot(self):
        self._show_plot_at_index(self._current_plot_idx + 1)

    def _prev_plot(self):
        self._show_plot_at_index(self._current_plot_idx - 1)

    def _on_tree_select(self, event):
        selection = self.tree.selection()
        if not selection:
            return
        iid = selection[0]
        name = self.tree.item(iid, 'text')
        plot_name = PLOT_MAP.get(name)
        if plot_name and plot_name in self._plot_names:
            idx = self._plot_names.index(plot_name)
            self._show_plot_at_index(idx)

    def _open_current_plot(self):
        if 0 <= self._current_plot_idx < len(self._plot_names):
            name = self._plot_names[self._current_plot_idx]
            path = self._plot_paths[name]
            subprocess.Popen(['open', path])

    def _open_output_dir(self):
        d = self._output_dir
        if d and os.path.isdir(d):
            subprocess.Popen(['open', d])

    # ================================================================
    #  工具
    # ================================================================

    def _set_busy(self, busy):
        state = tk.DISABLED if busy else tk.NORMAL
        self.btn_query['state'] = state
        self.btn_batch['state'] = (tk.NORMAL if not busy and self._csv_path
                                   else tk.DISABLED)
        self.btn_cancel['state'] = tk.NORMAL if busy else tk.DISABLED
        self.btn_browse_csv['state'] = state


def main():
    root = tk.Tk()
    root.title("Astro Toolbox - Multi-Band Astronomy Toolkit")
    root.geometry("1300x900")
    root.minsize(1100, 800)
    AstroToolboxGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
