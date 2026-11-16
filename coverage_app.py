"""Capture check app: python coverage_app.py

Start screen: "New build from photos" (a step-by-step wizard: Photos -> Compare -> Align ->
Coverage -> Mesh -> Score) or "Open an existing build" (a list of runs with cameras.csv under
results/ and the runs folder). Settings, the Metashape interpreter and the watchdog limits are
under Advanced; the raw log is under "Show details".

The logic lives in src/bd3d_quality (builds, progress, pipeline_steps, coverage,
metashape_locator, evaluation). Long work runs in worker threads that never touch Tk: they post
callables to a queue that the main thread drains with root.after(), so every UI update and
error box happens on the main thread (required on macOS).
"""
import json
import os
import queue
import sys
import threading
import time
import tkinter as tk
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, font as tkfont, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

try:
    import bd3d_quality  # noqa: F401
except ImportError:  # not installed: use the source tree next to this file
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from bd3d_quality import builds as bl
from bd3d_quality import metashape_locator as ml
from bd3d_quality import pipeline_steps as ps
from bd3d_quality import quality_model as qm
from bd3d_quality.coverage import BAND_NAMES, analyze, compare, load_cameras
from bd3d_quality.progress import Progress, StageTracker

REPO = Path(__file__).resolve().parent
DEFAULT_RUNS = REPO / 'runs' / 'app'
QUALITY_MODELS = REPO / 'models' / 'quality-attention'   # experimental, gitignored
STEPS = ['Photos', 'Compare', 'Align', 'Coverage', 'Mesh', 'Score']

# look
BG, SURFACE, BORDER = '#ffffff', '#f3f4f6', '#d9dce1'
TEXT, MUTED = '#1d1d1f', '#6e6e73'
ACCENT, ACCENT_DARK, ACCENT_SOFT = '#0a66d8', '#0853ad', '#dbe8fb'
OK, WARN, ERROR = '#2e7d32', '#a35c00', '#c62828'
RED, GREY = '#e05a5a', '#c8c8c8'
EXP_BG, EXP_BORDER, EXP_TAG = '#fffaf0', '#e3c27a', '#8a5a00'   # experimental box, never like a result


def setup_style(root):
    family = tkfont.nametofont('TkDefaultFont').actual('family')
    fonts = {
        'body': tkfont.Font(family=family, size=15),
        'small': tkfont.Font(family=family, size=13),
        'bold': tkfont.Font(family=family, size=15, weight='bold'),
        'big': tkfont.Font(family=family, size=18),
        'h2': tkfont.Font(family=family, size=19, weight='bold'),
        'h1': tkfont.Font(family=family, size=25, weight='bold'),
        'huge': tkfont.Font(family=family, size=20, weight='bold'),
        'mono': tkfont.Font(family='Menlo', size=11),
    }
    root.configure(bg=BG)
    s = ttk.Style(root)
    s.theme_use('clam')
    s.configure('.', background=BG, foreground=TEXT, font=fonts['body'], bordercolor=BORDER,
                lightcolor=BG, darkcolor=BG, focuscolor=ACCENT, troughcolor=SURFACE)
    s.configure('TFrame', background=BG)
    s.configure('Card.TFrame', background=SURFACE)
    s.configure('Exp.TFrame', background=EXP_BG)
    for name, fg, f in [('TLabel', TEXT, 'body'), ('Muted', MUTED, 'body'), ('Small', MUTED, 'small'),
                        ('H1', TEXT, 'h1'), ('H2', TEXT, 'h2'), ('Big', TEXT, 'big'), ('Bold', TEXT, 'bold'),
                        ('Ok', OK, 'body'), ('Warn', WARN, 'body'), ('Error', ERROR, 'body'),
                        ('StepNow', ACCENT, 'bold'), ('StepDone', OK, 'body'), ('StepTodo', MUTED, 'body')]:
        style = name if name == 'TLabel' else f'{name}.TLabel'
        s.configure(style, background=BG, foreground=fg, font=fonts[f])
        s.configure('Card.' + style, background=SURFACE, foreground=fg, font=fonts[f])
        s.configure('Exp.' + style, background=EXP_BG, foreground=fg, font=fonts[f])
    s.configure('Exp.Tag.TLabel', background=EXP_BG, foreground=EXP_TAG, font=fonts['small'])
    s.configure('TButton', background=SURFACE, foreground=TEXT, bordercolor=BORDER, lightcolor=SURFACE,
                darkcolor=SURFACE, padding=(16, 8), font=fonts['body'])
    s.map('TButton', background=[('disabled', BG), ('pressed', '#e2e5ea'), ('active', '#e9ebef')],
          foreground=[('disabled', '#b0b3b8')])
    s.configure('Accent.TButton', background=ACCENT, foreground='white', bordercolor=ACCENT,
                lightcolor=ACCENT, darkcolor=ACCENT)
    s.map('Accent.TButton', background=[('disabled', '#a9c4ec'), ('pressed', ACCENT_DARK), ('active', ACCENT_DARK)],
          foreground=[('disabled', 'white')], bordercolor=[('disabled', '#a9c4ec')],
          lightcolor=[('disabled', '#a9c4ec')], darkcolor=[('disabled', '#a9c4ec')])
    s.configure('Huge.TButton', padding=(36, 34), font=fonts['huge'], background=SURFACE)
    s.configure('Start.TButton', padding=(48, 16), font=fonts['huge'])
    s.configure('Link.TButton', background=BG, foreground=ACCENT, bordercolor=BG, lightcolor=BG, darkcolor=BG,
                padding=(4, 2), font=fonts['body'])
    s.map('Link.TButton', background=[('active', BG), ('pressed', BG)],
          foreground=[('disabled', '#b0b3b8'), ('active', ACCENT_DARK)])
    s.configure('Treeview', background='white', fieldbackground='white', foreground=TEXT, rowheight=34,
                font=fonts['body'], bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER)
    s.map('Treeview', background=[('selected', ACCENT_SOFT)], foreground=[('selected', TEXT)])
    s.configure('Treeview.Heading', background=SURFACE, foreground=MUTED, font=fonts['small'], relief='flat',
                padding=(8, 6), bordercolor=BORDER, lightcolor=SURFACE, darkcolor=SURFACE)
    s.map('Treeview.Heading', background=[('active', '#e9ebef')])
    s.configure('Horizontal.TProgressbar', background=ACCENT, troughcolor=SURFACE, bordercolor=SURFACE,
                lightcolor=ACCENT, darkcolor=ACCENT, thickness=16)
    s.configure('TCombobox', padding=6, arrowsize=16)
    root.option_add('*TCombobox*Listbox.font', fonts['body'])
    return fonts


# ---------------------------------------------------------------- plain-language text

def plain_outcome(state, detail):
    """One plain sentence for a step outcome from pipeline_steps.run_outcome."""
    if state == 'cancelled':
        return 'Cancelled. Nothing was deleted.'
    d = detail or ''
    if 'license' in d.lower():
        return 'Metashape could not get a license. Check the VPN or the license server, then try again.'
    if d.startswith('watchdog stop'):
        return ('Stopped by the safety limits to protect this computer '
                f'({d.split(":", 1)[1].strip().replace("_", " ")}).')
    if 'alignment failed' in d:
        return 'Too few cameras could be placed for a usable alignment.'
    return 'Something went wrong. Open "Show details" below for the log.'


def score_lines(row, ref_name):
    """(headline, numbers, explanation, warning or '') for a v3 score row."""
    y = row['label_v3']
    headline = f'Score vs {ref_name}:  {y:.4f}   (lower is better)'
    numbers = f'Missing {row["r2c_rmse_norm"]:.4f}  ·  Extra {row["c2r_rmse_norm"]:.4f}'
    explanation = (f'The typical distance between the two surfaces, as a fraction of {ref_name}\'s size '
                   f'(its bounding-box diagonal): {y:.4f} is about {y * 100:.2g} % of the object. '
                   f'"Missing" measures parts of {ref_name} this mesh lacks or gets wrong; "extra" measures '
                   f'surface this mesh has that {ref_name} does not.')
    warning = ''
    if row.get('qc_v3') != 'pass' or row.get('scale_check') == 'disagree':
        warning = ('The two meshes lined up less well than usual '
                   f'(fit check: {row.get("qc_v3")}, scale check: {row.get("scale_check")}); treat the number with care.')
    return headline, numbers, explanation, warning


def aligned_count(run_dir):
    labels, aligned, _, _ = load_cameras(Path(run_dir) / 'cameras.csv')
    return int(aligned.sum()), len(labels)


def date_text(d):
    return d.astimezone().strftime('%-d %b %Y') if d else ''


# ---------------------------------------------------------------- figures

def draw_coverage(fig, cov, run_name, ref_name=None):
    """Top view and azimuth x height map, with a legend (single build or vs a reference)."""
    fig.clear()
    ax, cm = fig.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1.6]})
    ref = cov.reference
    Q, W = cov.positions, cov.views
    allQ = Q if ref is None else np.vstack([Q, ref.positions])
    r = np.linalg.norm(allQ[:, :2], axis=1).max()
    if ref is not None:
        ax.scatter(ref.positions[:, 0], ref.positions[:, 1], s=40, facecolors='none', edgecolors='0.6',
                   linewidths=0.8, zorder=1, label=f'{ref_name} ({len(ref.labels)} cameras)')
    sc = ax.scatter(Q[:, 0], Q[:, 1], c=cov.elevation, cmap='viridis', s=24, zorder=3,
                    label=f'{run_name} ({len(cov.labels)} cameras)')
    ax.quiver(Q[:, 0], Q[:, 1], W[:, 0], W[:, 1], color='0.5', width=0.003, scale=12, zorder=2)
    ax.plot(0, 0, '+', color='k', ms=12)
    ax.annotate('0°', (r * 1.08, 0), ha='left', va='center', color='0.4')
    ax.set_title('Cameras seen from above', fontsize=12)
    if cov.failed:
        shown = ', '.join(cov.failed[:3]) + (' …' if len(cov.failed) > 3 else '')
        ax.plot([], [], 'x', color='tab:red', label=f'{len(cov.failed)} not placed: {shown}')
    ax.legend(loc='upper left', fontsize=9, frameon=False, bbox_to_anchor=(0, -0.02))
    ax.set_aspect('equal')
    ax.set_xlim(-r * 1.25, r * 1.25)
    ax.set_ylim(-r * 1.25, r * 1.25)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(sc, ax=ax, shrink=0.7, label='height (°)')

    az, el, n = cov.az_edges, cov.el_edges, cov.counts
    vmax = max(n.max(), 4)
    cm.pcolormesh(az, el, n, cmap='Blues', vmin=0, vmax=vmax, edgecolors='white', linewidth=1)
    for i in range(n.shape[0]):
        for j in range(n.shape[1]):
            colour = RED if cov.missing[i, j] else GREY if cov.never[i, j] else None
            if colour:
                cm.add_patch(Rectangle((az[j], el[i]), az[j + 1] - az[j], el[i + 1] - el[i],
                                       facecolor=colour, edgecolor='white'))
            text = f'{n[i, j]}' if ref is None else f'{n[i, j]}/{ref.counts[i, j]}'
            cm.text((az[j] + az[j + 1]) / 2, (el[i] + el[i + 1]) / 2, text, ha='center', va='center',
                    fontsize=10 if ref is None else 8,
                    color='white' if cov.missing[i, j] or (colour is None and n[i, j] > 0.6 * vmax) else 'k')
    cm.scatter(cov.azimuth, cov.elevation, s=5, color='k', alpha=0.35)
    for name, lo, hi in zip(BAND_NAMES.get(len(n), []), el[:-1], el[1:]):
        cm.text(362, (lo + hi) / 2, name, va='center', fontsize=10, color='0.3')
    cm.set_xlim(0, 360)
    cm.set_ylim(el[0], el[-1])
    cm.set_xticks(az[::2])
    cm.set_ylabel('height (°)')
    dot = Line2D([], [], marker='o', ls='', color='k', alpha=0.35, ms=4, label='one camera')
    if ref is None:
        cm.set_xlabel('around the object (°, 0 = first camera)')
        cm.set_title('Where the cameras are', fontsize=12)
        handles = [Patch(facecolor=RED, label=f'gap: fewer than {cov.min_cameras} cameras'),
                   Patch(facecolor='#6baed6', label='number = cameras in the cell'), dot]
    else:
        cm.set_xlabel(f'around the object (°, as in {ref_name})')
        cm.set_title(f'Cameras: this build / {ref_name}', fontsize=12)
        handles = [Patch(facecolor=RED, label=f'missing: {ref_name} has {cov.min_cameras}+, this build fewer'),
                   Patch(facecolor=GREY, label=f'{ref_name} has few here too'), dot]
    cm.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False, fontsize=9)
    fig.tight_layout()


def draw_score(fig, row, run_name, ref_name):
    fig.clear()
    ax = fig.subplots()
    names = ['Score', 'Missing', 'Extra']
    values = [row['label_v3'], row['r2c_rmse_norm'], row['c2r_rmse_norm']]
    bars = ax.barh(names[::-1], values[::-1], color=['#9ecae1', '#9ecae1', ACCENT], height=0.55)  # Score on top
    top = max(values) * 1.35 or 1
    for b, v in zip(bars, values[::-1]):
        ax.text(b.get_width() + top * 0.01, b.get_y() + b.get_height() / 2, f'{v:.4f}', va='center', fontsize=12)
    ax.set_xlim(0, top)
    ax.set_title(f'{run_name} vs {ref_name}', fontsize=13, loc='left')
    ax.set_xlabel(f'fraction of {ref_name}\'s size (lower is better)')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=12)
    fig.tight_layout()


# ---------------------------------------------------------------- state

@dataclass
class Wizard:
    folder: bl.PhotoFolder = None
    photos: dict = None              # {SHA-256: label}
    checked: bool = False            # fingerprints, compatible builds and cache lookup done
    name: str = ''
    compatible: list = field(default_factory=list)   # [(Build, shared)]
    compare: bl.Build = None
    cache: Path = None
    cache_reason: str = ''
    paths: ps.CapturePaths = None
    align_run: Path = None
    mesh_run: Path = None
    status: dict = field(default_factory=lambda: dict.fromkeys(STEPS, 'todo'))
    outcome: dict = field(default_factory=dict)      # step -> plain sentence of the last outcome
    score: dict = None
    step: str = 'Photos'


class Task:
    def __init__(self, kind, photo_count):
        self.kind, self.wp, self.cancelled = kind, None, False
        self.started = time.monotonic()
        self.tracker = StageTracker(photo_count)
        self.view = None

    def cancel(self):
        self.cancelled = True
        if self.wp is not None:
            self.wp.cancel()


class ProgressView(ttk.Frame):
    """Stage in plain words, a bar (filled where the log counts items), elapsed time and Cancel."""

    def __init__(self, master, app, task):
        super().__init__(master)
        self.app, self.task = app, task
        self.stage = ttk.Label(self, text=task.tracker.current.stage + '…', style='H2.TLabel')
        self.stage.pack(anchor='w')
        self.bar = ttk.Progressbar(self, mode='indeterminate', length=640)
        self.bar.pack(anchor='w', pady=(14, 6), fill='x')
        row = ttk.Frame(self)
        row.pack(fill='x')
        self.detail = ttk.Label(row, text='', style='Muted.TLabel')
        self.detail.pack(side='left')
        self.elapsed = ttk.Label(row, text='0:00', style='Muted.TLabel')
        self.elapsed.pack(side='right')
        self.cancel_button = ttk.Button(self, text='Cancel', command=self.cancel)
        self.cancel_button.pack(anchor='e', pady=(18, 0))
        self.indeterminate = False
        task.view = self
        self.update_progress(task.tracker.current)
        self.tick()

    def update_progress(self, p):
        if not self.winfo_exists():
            return
        self.stage.config(text=p.stage + '…')
        self.detail.config(text=p.detail)
        if p.fraction is None:
            if not self.indeterminate:
                self.bar.config(mode='indeterminate', value=0)
                self.bar.start(12)
                self.indeterminate = True
        else:
            if self.indeterminate:
                self.bar.stop()
                self.indeterminate = False
            self.bar.config(mode='determinate', value=100 * p.fraction)

    def tick(self):
        if not self.winfo_exists() or self.app.task is not self.task:
            return
        s = int(time.monotonic() - self.task.started)
        self.elapsed.config(text=f'{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}' if s >= 3600 else f'{s // 60}:{s % 60:02d}')
        self.after(1000, self.tick)

    def cancel(self):
        self.cancel_button.config(text='Cancelling…', state='disabled')
        self.app.cancel_task()


# ---------------------------------------------------------------- app

class CaptureApp:
    def __init__(self, root):
        self.root = root
        self.fonts = setup_style(root)
        self.q = queue.Queue()
        self.task = None
        self.python, self.python_text, self.detecting = None, 'Looking for Metashape…', True
        self.runs_folder = Path(ml.load_setting('runs_folder') or DEFAULT_RUNS)
        self.builds = None
        self.w = None
        self.gen = 0                    # bumps when a screen's background work becomes stale
        self.figures = {}
        self.thumbs = []
        self.next_button = self.back_button = None
        self.advanced = None
        root.title('Capture check')
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.report_callback_exception = self.callback_error
        for key in ('<Return>', '<KP_Enter>'):
            root.bind(key, lambda e: self.press(self.next_button))
        root.bind('<Escape>', lambda e: self.press(self.back_button))

        top = ttk.Frame(root, padding=(20, 12, 20, 0))
        top.pack(fill='x')
        ttk.Label(top, text='Capture check', style='Bold.TLabel').pack(side='left')
        ttk.Button(top, text='⚙  Advanced', style='Link.TButton', command=self.open_advanced).pack(side='right')
        ttk.Separator(root).pack(fill='x', pady=(10, 0))
        self.body = ttk.Frame(root)
        self.body.pack(fill='both', expand=True)

        self.details = ttk.Frame(root, padding=(20, 0, 20, 10))
        self.details_button = ttk.Button(self.details, text='▸  Show details', style='Link.TButton',
                                         command=self.toggle_details)
        self.details_button.pack(anchor='w')
        self.log_box = ScrolledText(self.details, height=12, state='disabled', font=self.fonts['mono'],
                                    relief='flat', bg='#fafafa', highlightthickness=1, highlightbackground=BORDER)
        self.details_open = False

        self.root.after(100, self.pump)
        self.detect_python()
        self.show_start()

    # ------------------------------------------------------------ plumbing

    def post(self, fn, *args):
        """Called from any thread; fn(*args) runs on the main thread."""
        self.q.put((fn, args))

    def pump(self):
        try:
            while True:
                fn, args = self.q.get_nowait()
                try:
                    fn(*args)
                except Exception as exc:
                    self.log(traceback.format_exc().rstrip())
                    self.error('Unexpected error', exc)
        except queue.Empty:
            pass
        self.root.after(100, self.pump)

    def background(self, job):
        """Run job() in a daemon thread; exceptions are logged and shown on the main thread."""
        def body():
            try:
                job()
            except Exception as exc:
                tb = traceback.format_exc().rstrip()
                self.post(self.log, tb)
                self.post(self.error, 'Something went wrong', exc)
        threading.Thread(target=body, daemon=True).start()

    def log(self, text):
        self.log_box.configure(state='normal')
        self.log_box.insert('end', text + '\n')
        if int(self.log_box.index('end-1c').split('.')[0]) > 8000:
            self.log_box.delete('1.0', '2000.0')
        self.log_box.see('end')
        self.log_box.configure(state='disabled')

    def error(self, title, exc_or_text):
        text = exc_or_text if isinstance(exc_or_text, str) else f'{type(exc_or_text).__name__}: {exc_or_text}'
        self.log(f'ERROR: {title}: {text}')
        messagebox.showerror(title, text, parent=self.root)

    def callback_error(self, exc_type, exc, tb):
        self.log(''.join(traceback.format_exception(exc_type, exc, tb)).rstrip())
        self.error('Unexpected error', exc)

    def press(self, button):
        """Enter = Next, Esc = Back, only when that button exists and is enabled."""
        if button is not None and button.winfo_exists() and button.instate(['!disabled']):
            button.invoke()

    def toggle_details(self):
        self.details_open = not self.details_open
        if self.details_open:
            self.log_box.pack(fill='both', expand=True, pady=(4, 0))
            self.details_button.config(text='▾  Hide details')
        else:
            self.log_box.pack_forget()
            self.details_button.config(text='▸  Show details')

    def clear(self, details=True):
        self.gen += 1
        for child in self.body.winfo_children():
            child.destroy()
        for fig in self.figures.values():
            fig.clear()
        self.figures = {}
        self.next_button = self.back_button = None
        if details:
            self.details.pack(fill='x', side='bottom', before=self.body)
        else:
            self.details.pack_forget()

    def label(self, master, text, style='TLabel', wrap=900, **pack):
        lab = ttk.Label(master, text=text, style=style, wraplength=wrap, justify='left')
        lab.pack(anchor='w', **pack)
        return lab

    def figure(self, master, key, size=(11, 4.6)):
        fig = Figure(figsize=size, facecolor=BG)
        canvas = FigureCanvasTkAgg(fig, master=master)
        canvas.get_tk_widget().configure(bg=BG, highlightthickness=0)
        self.figures[key] = fig
        return fig, canvas

    def save_image(self, fig, name):
        start = Path.home() / 'Desktop'
        path = filedialog.asksaveasfilename(parent=self.root, title='Save image', defaultextension='.png',
                                            initialfile=f'{name}.png', filetypes=[('PNG image', '*.png')],
                                            initialdir=str(start if start.is_dir() else Path.home()))
        if not path:
            return
        try:
            fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
        except Exception as exc:
            return self.error('Could not save the image', exc)
        self.log(f'saved image {path}')

    # ------------------------------------------------------------ Metashape interpreter

    def detect_python(self):
        self.detecting = True
        self.python_text = 'Looking for Metashape…'

        def on_try(source, python, ok, detail):
            self.post(self.log, f'Metashape Python: {"OK " if ok else "no "} {python}  [{source}] {detail}')

        def job():
            try:
                search = ml.find_metashape_python(on_try=on_try)
            except Exception as exc:
                search = ml.Search(tried=[('search', '', f'{type(exc).__name__}: {exc}')])
            self.post(self.python_detected, search)
        threading.Thread(target=job, daemon=True).start()
        self.refresh_advanced()

    def python_detected(self, search):
        self.detecting = False
        if search.found:
            self.use_python(search.found.python, search.found.version, search.found.source)
        else:
            self.python = None
            self.python_text = 'Not found. Choose the Python that can import Metashape with Change…'
            self.log(f'Metashape Python: none of {len(search.tried)} candidates can import Metashape')
            self.metashape_changed()

    def use_python(self, python, version, source):
        self.python = python
        self.python_text = f'Metashape {version} (found via {source})'
        try:
            ml.save_choice(python)
        except OSError as exc:
            self.log(f'could not save the Metashape Python choice: {exc}')
        self.metashape_changed()

    def metashape_changed(self):
        self.refresh_advanced()
        if self.w is not None and self.task is None and self.w.step in ('Align', 'Mesh') \
                and self.body.winfo_children():
            self.show_step(self.w.step)

    def metashape_note(self):
        """'' when Metashape can be used, else one plain line."""
        if self.python:
            return ''
        if self.detecting:
            return 'Looking for Metashape on this Mac…'
        return 'Metashape was not found. Choose it under ⚙ Advanced.'

    # ------------------------------------------------------------ Advanced

    def open_advanced(self):
        if self.advanced is not None and self.advanced.winfo_exists():
            self.advanced.lift()
            return
        win = self.advanced = tk.Toplevel(self.root, bg=BG)
        win.title('Advanced')
        win.transient(self.root)
        frame = ttk.Frame(win, padding=24)
        frame.pack(fill='both', expand=True)
        win.bind('<Escape>', lambda e: win.destroy())

        def section(title):
            ttk.Label(frame, text=title, style='Bold.TLabel').pack(anchor='w', pady=(14, 4))

        section('Metashape')
        self.adv_python = ttk.Label(frame, style='Small.TLabel', wraplength=640)
        self.adv_python.pack(anchor='w')
        self.adv_python_status = ttk.Label(frame, wraplength=640)
        self.adv_python_status.pack(anchor='w', pady=(2, 6))
        row = ttk.Frame(frame)
        row.pack(anchor='w')
        ttk.Button(row, text='Change…', command=self.change_python).pack(side='left')
        ttk.Button(row, text='Search again', command=self.detect_python).pack(side='left', padx=8)
        self.label(frame, 'Align and Build mesh run Metashape on this Mac and need a Metashape license '
                          '(on the network: check the VPN).', 'Small.TLabel', 640, pady=(6, 0))

        section('Where new builds go')
        self.adv_runs = ttk.Label(frame, style='Small.TLabel', wraplength=640)
        self.adv_runs.pack(anchor='w')
        row = ttk.Frame(frame)
        row.pack(anchor='w', pady=(4, 0))
        ttk.Button(row, text='Change…', command=self.change_runs_folder).pack(side='left')
        ttk.Button(row, text='Use the default', command=lambda: self.set_runs_folder(DEFAULT_RUNS)).pack(side='left', padx=8)

        section('Mesh settings (as the pilot builds ref50 and ref145b)')
        self.label(frame, ps.settings_text(), 'Small.TLabel', 640)
        section('Safety limits (resource watchdog)')
        self.label(frame, f'Align: {ps.ALIGN_LIMITS.text()}\nMesh:  {ps.MESH_LIMITS.text()}', 'Small.TLabel', 640)
        ttk.Button(frame, text='Close', command=win.destroy).pack(anchor='e', pady=(20, 0))
        self.refresh_advanced()

    def refresh_advanced(self):
        if self.advanced is None or not self.advanced.winfo_exists():
            return
        self.adv_python.config(text=self.python or '(none)')
        self.adv_python_status.config(text=self.python_text,
                                      style='Ok.TLabel' if self.python else 'Muted.TLabel' if self.detecting else 'Error.TLabel')
        self.adv_runs.config(text=str(self.runs_folder))

    def change_runs_folder(self):
        folder = filedialog.askdirectory(parent=self.advanced, title='Where new builds go',
                                         initialdir=str(self.runs_folder if self.runs_folder.exists() else REPO))
        if folder:
            self.set_runs_folder(Path(folder))

    def set_runs_folder(self, folder):
        if folder != DEFAULT_RUNS:
            ok, resolved, reason = ps.inspect_folder(folder)
            self.log(f'runs folder: tried {resolved} -> {reason}')
            if not ok:
                return self.error('Cannot use this folder', f'{resolved}\n{reason}')
            folder = resolved
        self.runs_folder = Path(folder)
        self.builds = None
        try:
            ml.save_setting('runs_folder', str(folder))
        except OSError as exc:
            self.log(f'could not save the runs folder: {exc}')
        self.log(f'new builds go to {folder}')
        if self.w is not None and self.w.folder is not None and self.w.status['Align'] == 'todo':
            self.w.name = ps.capture_name(self.w.folder.root, self.runs_folder)
        self.refresh_advanced()

    def change_python(self):
        """Type or browse to an interpreter, test it, and use and save it if it works."""
        win = tk.Toplevel(self.advanced or self.root, bg=BG)
        win.title('Metashape Python')
        win.transient(self.advanced or self.root)
        frame = ttk.Frame(win, padding=20)
        frame.pack(fill='both', expand=True)
        self.label(frame, 'Python interpreter that can import Metashape (for example a venv\'s bin/python):',
                   wrap=600, pady=(0, 6))
        var = tk.StringVar(value=self.python or '')
        entry = ttk.Entry(frame, textvariable=var, width=70, font=self.fonts['small'])
        entry.pack(fill='x')
        note = ttk.Label(frame, text='', style='Muted.TLabel', wraplength=600)
        note.pack(anchor='w', pady=6)
        buttons = ttk.Frame(frame)
        buttons.pack(fill='x')

        def browse():
            path = filedialog.askopenfilename(parent=win, title='Python interpreter',
                                              initialdir=str(Path(var.get()).parent if var.get() else Path.home()))
            if path:
                var.set(path)

        def done(python, ok, detail):
            if not win.winfo_exists():
                return
            test.config(state='normal')
            self.log(f'Metashape Python: {"OK " if ok else "no "} {python}  [chosen by you] {detail}')
            if ok:
                self.use_python(python, detail, 'your choice')
                win.destroy()
            else:
                note.config(text=f'This one does not work: {detail}', style='Error.TLabel')

        def try_it():
            python = os.path.expanduser(var.get().strip())
            if not python:
                return
            test.config(state='disabled')
            note.config(text='Testing…', style='Muted.TLabel')
            threading.Thread(target=lambda: self.post(done, python, *ml.probe(python)), daemon=True).start()

        ttk.Button(buttons, text='Browse…', command=browse).pack(side='left')
        test = ttk.Button(buttons, text='Test and use', style='Accent.TButton', command=try_it)
        test.pack(side='right')
        ttk.Button(buttons, text='Cancel', command=win.destroy).pack(side='right', padx=8)
        win.bind('<Return>', lambda e: try_it())
        win.bind('<Escape>', lambda e: win.destroy())
        entry.focus_set()

    # ------------------------------------------------------------ start

    def show_start(self):
        self.w = None
        self.clear(details=False)
        page = ttk.Frame(self.body)
        page.place(relx=0.5, rely=0.45, anchor='center')
        ttk.Label(page, text='What would you like to do?', style='H1.TLabel').pack(pady=(0, 36))
        row = ttk.Frame(page)
        row.pack()
        ttk.Button(row, text='New build\nfrom photos', style='Huge.TButton', width=16,
                   command=self.new_build).pack(side='left', padx=16)
        ttk.Button(row, text='Open an\nexisting build', style='Huge.TButton', width=16,
                   command=self.show_open).pack(side='left', padx=16)

    # ------------------------------------------------------------ wizard frame

    def new_build(self):
        self.w = Wizard()
        self.show_step('Photos')

    def show_step(self, step):
        self.w.step = step
        {'Photos': self.step_photos, 'Compare': self.step_compare, 'Align': self.step_align,
         'Coverage': self.step_coverage, 'Mesh': self.step_mesh, 'Score': self.step_score}[step]()

    def wizard_page(self, step, title, subtitle=''):
        """Header (Start link + step indicator), content frame, and nav frame; returns (content, nav)."""
        self.clear()
        bar = ttk.Frame(self.body, padding=(20, 12, 20, 0))
        bar.pack(fill='x')
        home = ttk.Button(bar, text='←  Start', style='Link.TButton', command=self.show_start)
        home.pack(side='left')
        if self.task is not None:
            home.config(state='disabled')
        steps = ttk.Frame(bar)
        steps.pack(side='left', expand=True)
        for i, s in enumerate(STEPS):
            state = self.w.status[s]
            if s == step:
                text, style = f'●  {s}', 'StepNow.TLabel'
            elif state == 'done':
                text, style = f'✓  {s}', 'StepDone.TLabel'
            elif state == 'skipped':
                text, style = f'–  {s}', 'StepTodo.TLabel'
            elif state in ('failed', 'cancelled'):
                text, style = f'✗  {s}', 'Warn.TLabel'
            else:
                text, style = f'○  {s}', 'StepTodo.TLabel'
            if i:
                ttk.Label(steps, text='—', style='StepTodo.TLabel').pack(side='left', padx=8)
            ttk.Label(steps, text=text, style=style).pack(side='left')
        ttk.Frame(bar, width=70).pack(side='right')        # balances the Start link

        nav = ttk.Frame(self.body, padding=(32, 0, 32, 16))
        nav.pack(fill='x', side='bottom')
        content = ttk.Frame(self.body, padding=(48, 28, 48, 12))
        content.pack(fill='both', expand=True)
        ttk.Label(content, text=title, style='H1.TLabel').pack(anchor='w')
        if subtitle:
            self.label(content, subtitle, 'Muted.TLabel', 1000, pady=(6, 0))
        return content, nav

    def nav_buttons(self, nav, back=None, next_=None, next_text='Next  →', next_enabled=True, extra=()):
        busy = self.task is not None
        if back is not None:
            self.back_button = ttk.Button(nav, text='←  Back', command=back)
            self.back_button.pack(side='left')
            if busy:
                self.back_button.config(state='disabled')
        if next_ is not None:
            self.next_button = ttk.Button(nav, text=next_text, style='Accent.TButton', command=next_)
            self.next_button.pack(side='right')
            if busy or not next_enabled:
                self.next_button.config(state='disabled')
        for text, command in extra:
            b = ttk.Button(nav, text=text, command=command)
            b.pack(side='right', padx=8)
            if busy:
                b.config(state='disabled')

    # ------------------------------------------------------------ 1 Photos

    def step_photos(self):
        w = self.w
        content, nav = self.wizard_page('Photos', 'Choose the photo folder',
                                        'All photos of one object, in one folder.')
        row = ttk.Frame(content)
        row.pack(anchor='w', pady=(22, 10))
        ttk.Button(row, text='Choose folder…' if w.folder is None else 'Choose another folder…',
                   style='Accent.TButton' if w.folder is None else 'TButton',
                   command=self.choose_photos).pack(side='left')
        self.photos_box = ttk.Frame(content)
        self.photos_box.pack(fill='both', expand=True)
        self.nav_buttons(nav, back=self.show_start, next_=lambda: self.leave_photos(),
                         next_enabled=w.folder is not None and w.folder.ok and w.checked)
        if w.folder is not None:
            self.render_photos()

    def choose_photos(self):
        w = self.w
        if w.status['Align'] != 'todo':
            if not messagebox.askyesno('Start a new build?', 'This starts a new build with other photos. '
                                       'The current build stays saved.', parent=self.root):
                return
        folder = filedialog.askdirectory(parent=self.root, title='Photo folder', mustexist=True,
                                         initialdir=str(w.folder.root.parent if w.folder else Path.home()))
        if not folder:
            return
        try:
            pf = bl.scan_photo_folder(folder)
        except ValueError as exc:
            self.log(f'photo folder {folder}: {exc}')
            return self.error('Cannot use this folder', str(exc))
        self.w = w = Wizard(folder=pf)
        w.name = ps.capture_name(pf.root, self.runs_folder)
        self.log(f'photo folder {pf.root}: {pf.summary}; ' + ' '.join(pf.warnings))
        self.step_photos()
        if pf.ok:
            self.check_photos(w)

    def check_photos(self, w):
        """Background: thumbnails, fingerprints, compatible builds and the image-quality cache."""
        roots = [REPO / 'results', self.runs_folder]

        def stale():
            return self.w is not w

        def job():
            self.load_thumbs(w)
            last = [0.0]

            def progress(done, total):
                if time.monotonic() - last[0] > 0.1 or done == total:
                    last[0] = time.monotonic()
                    self.post(self.photos_progress, w, f'Checking photos… {done} of {total}')
            photos = bl.photo_fingerprints(w.folder, progress)
            if stale():
                return
            self.post(self.photos_progress, w, 'Looking for earlier builds of the same object…')
            builds = bl.find_builds(roots)
            compatible = bl.compatible_builds(photos, builds)
            self.post(self.photos_checked, w, photos, builds, compatible)
        self.background(job)

    def add_thumb(self, w, im):
        if self.w is not w or w.step != 'Photos' or not hasattr(self, 'thumb_row') or not self.thumb_row.winfo_exists():
            return
        from PIL import ImageTk
        photo = ImageTk.PhotoImage(im)
        self.thumbs.append(photo)
        tk.Label(self.thumb_row, image=photo, bg=BG, bd=0).pack(side='left', padx=(0, 8))

    def photos_progress(self, w, text):
        if self.w is w and w.step == 'Photos' and hasattr(self, 'check_label') and self.check_label.winfo_exists():
            self.check_label.config(text=text)

    def photos_checked(self, w, photos, builds, compatible):
        if self.w is not w:
            return
        w.photos, w.compatible, w.checked = photos, compatible, True
        self.builds = builds
        self.log(f'photos checked: {len(photos)} distinct; {len(compatible)} earlier builds share photos: '
                 + ', '.join(f'{b.name} ({n})' for b, n in compatible))
        if len(photos) < len(w.folder.photos):
            self.log(f'note: {len(w.folder.photos) - len(photos)} photos are exact duplicates of others')
        if w.step == 'Photos':
            self.photos_progress(w, 'Photos checked.')
            if self.next_button is not None and self.next_button.winfo_exists():
                self.next_button.config(state='normal')

    def render_photos(self):
        w, box = self.w, self.photos_box
        pf = w.folder
        line = ttk.Frame(box)
        line.pack(anchor='w', fill='x')
        ttk.Label(line, text=('✓  ' if pf.ok else '✗  ') + pf.summary, style='H2.TLabel' if pf.ok else 'Error.TLabel'
                  ).pack(side='left')
        if pf.ok:
            ttk.Label(line, text=f'Build name: {w.name}', style='Muted.TLabel').pack(side='left', padx=(28, 0))
        self.label(box, f'Folder: {pf.root.name}', 'Muted.TLabel', pady=(4, 0))
        self.thumb_row = ttk.Frame(box)
        self.thumb_row.pack(anchor='w', pady=14)
        self.thumbs = []
        for text in pf.warnings:
            if pf.ok or 'No supported' not in text:
                self.label(box, '⚠  ' + text, 'Warn.TLabel', pady=2)
            else:
                self.label(box, text, 'Error.TLabel', pady=2)
        if pf.ok:
            self.check_label = self.label(box, 'Photos checked.' if w.checked else 'Checking photos…', 'Muted.TLabel',
                                          pady=(12, 0))
            if w.checked:                      # came back with Back: reload the thumbnails only
                self.background(lambda: self.load_thumbs(w))

    def load_thumbs(self, w, count=7):
        """Worker thread: small previews of the first photos (PhotoImage is made on the main thread)."""
        from PIL import Image
        for rel in w.folder.photos[:count]:
            if self.w is not w:
                return
            try:
                with Image.open(w.folder.root / rel) as im:
                    im.draft('RGB', (240, 240))         # fast JPEG decode at reduced size
                    im = im.convert('RGB')
                    im.thumbnail((120, 120))
                self.post(self.add_thumb, w, im)
            except Exception as exc:
                self.post(self.log, f'preview of {rel}: {exc}')

    def leave_photos(self):
        self.w.status['Photos'] = 'done'
        self.show_step('Compare')

    # ------------------------------------------------------------ 2 Compare

    def step_compare(self):
        w = self.w
        content, nav = self.wizard_page('Compare', 'Compare with an earlier build?',
                                        'Earlier builds of the same object (they share photos with this set). '
                                        'Comparing shows what this capture misses and lets you score its mesh. '
                                        'Optional.')
        self.nav_buttons(nav, back=self.compare_back, next_=lambda: self.leave_compare(self.compare_pick()),
                         next_enabled=False, extra=[('Skip', lambda: self.leave_compare(None))])
        if not w.compatible:
            self.label(content, 'No earlier builds share photos with this set.', 'Big.TLabel', pady=(30, 0))
            self.next_button.config(state='disabled')
            return
        tree = self.build_table(content, [('name', 'Name', 260), ('group', 'Location', 200), ('photos', 'Photos', 90),
                                          ('shared', 'Shared', 90), ('mesh', 'Mesh', 70), ('date', 'Date', 130)])
        self.compare_rows = {}
        for b, n in w.compatible:
            iid = tree.insert('', 'end', values=(b.name, b.group, b.photo_count, n, '✓' if b.has_mesh else '–',
                                                 date_text(b.date)))
            self.compare_rows[iid] = b
            if w.compare is not None and b.key == w.compare.key:
                tree.selection_set(iid)
                tree.see(iid)
        tree.rows = self.compare_rows
        self.compare_tree = tree
        self.compare_note = self.label(content, '', 'Warn.TLabel', pady=(10, 0))

        def picked(_=None):
            b = self.compare_pick()
            if self.next_button is not None and self.task is None:
                self.next_button.config(state='normal' if b else 'disabled')
            self.compare_note.config(text='' if b is None or b.has_mesh else
                                     f'⚠  {b.name} has no mesh, so the score step will not be available.')
        tree.bind('<<TreeviewSelect>>', picked)
        tree.bind('<Double-1>', lambda e: self.press(self.next_button))
        picked()
        tree.focus_set()

    def compare_pick(self):
        if not hasattr(self, 'compare_tree') or not self.compare_tree.winfo_exists():
            return None
        sel = self.compare_tree.selection()
        return self.compare_rows.get(sel[0]) if sel else None

    def compare_back(self):
        if self.w.status['Align'] != 'todo':
            if not messagebox.askyesno('Start a new build?', 'Going back to the photos starts a new build. '
                                       'The current build stays saved.', parent=self.root):
                return
            folder = self.w.folder
            self.w = Wizard(folder=folder, name=ps.capture_name(folder.root, self.runs_folder))
            self.check_photos(self.w)
        self.show_step('Photos')

    def leave_compare(self, build):
        w = self.w
        w.compare = build
        w.status['Compare'] = 'done' if build else 'skipped'
        if w.status['Score'] in ('todo', 'skipped', 'done'):
            w.status['Score'], w.score = ('todo' if build else 'skipped'), None
        self.log(f'compare with: {build.name} ({build.path})' if build else 'compare: skipped')
        if w.status['Align'] == 'todo':
            w.cache, w.cache_reason = bl.find_usable_quality_cache(w.photos, build, [REPO / 'results'])
            self.log(f'image quality cache: {w.cache or "none"} ({w.cache_reason})')
        self.show_step('Align')

    # ------------------------------------------------------------ 3 Align

    def step_align(self):
        w = self.w
        content, nav = self.wizard_page('Align', 'Find camera positions',
                                        'Uses Metashape on this Mac (license needed). Usually a few minutes.')
        state = w.status['Align']
        self.nav_buttons(nav, back=lambda: self.show_step('Compare'), next_=self.leave_align,
                         next_enabled=state == 'done' or (state == 'failed' and self.has_cameras(w.align_run)))
        box = ttk.Frame(content, padding=(0, 30, 0, 0))
        box.pack(fill='both', expand=True)
        if self.task is not None and self.task.kind == 'align':
            ProgressView(box, self, self.task).pack(fill='x')
            return
        if state == 'done':
            placed, total = aligned_count(w.align_run)
            self.label(box, f'✓  Cameras placed for {placed} of {total} photos.', 'H2.TLabel')
            if placed < total:
                self.label(box, f'{total - placed} photos could not be placed; the next screen shows where they were.',
                           'Muted.TLabel', pady=(6, 0))
            return
        if state in ('failed', 'cancelled'):
            self.label(box, ('✗  ' if state == 'failed' else '') + w.outcome.get('Align', ''),
                       'Error.TLabel' if state == 'failed' else 'Warn.TLabel', pady=(0, 16))
            if state == 'failed' and self.has_cameras(w.align_run):
                self.label(box, 'Some cameras were placed: Next shows what there is.', 'Muted.TLabel', pady=(0, 16))
        note = self.metashape_note()
        start = ttk.Button(box, text='Start' if state == 'todo' else 'Try again', style='Start.TButton',
                           command=self.run_align)
        start.pack(anchor='w')
        if note:
            start.config(state='disabled')
            self.label(box, note, 'Error.TLabel' if not self.detecting else 'Muted.TLabel', pady=(14, 0))
        if w.cache:
            self.label(box, 'Image quality: reused from an earlier build (skips one step).', 'Muted.TLabel', pady=(18, 0))
        else:
            self.label(box, 'Image quality: will be measured (no earlier measurement covers all these photos).',
                       'Muted.TLabel', pady=(18, 0))

    @staticmethod
    def has_cameras(run_dir):
        return run_dir is not None and (Path(run_dir) / 'cameras.csv').is_file()

    def start_task(self, kind, photo_count, job, on_done):
        """job(task) runs in a worker and returns (state, detail); on_done(state, detail) runs on the main thread."""
        task = self.task = Task(kind, photo_count)
        self.log(f'--- {kind}: started')

        def body():
            try:
                state, detail = job(task)
            except Exception as exc:
                self.post(self.log, traceback.format_exc().rstrip())
                state, detail = 'failed', f'{type(exc).__name__}: {exc}'
            if task.cancelled:
                state, detail = 'cancelled', 'by user'
            self.post(self.task_done, task, state, detail, on_done)
        threading.Thread(target=body, daemon=True).start()
        return task

    def task_done(self, task, state, detail, on_done):
        if self.task is task:
            self.task = None
        self.log(f'--- {task.kind}: {state}' + (f' ({detail})' if detail else ''))
        on_done(state, detail)

    def task_stage(self, task, stage):
        task.tracker.current = Progress(stage)
        self.log(stage)
        if task.view is not None:
            task.view.update_progress(task.tracker.current)

    def task_line(self, task, line):
        self.log(line)
        p = task.tracker.feed(line)
        if p is not None and task.view is not None:
            task.view.update_progress(p)

    def cancel_task(self):
        task = self.task
        if task is None:
            return
        pid = task.wp.proc.pid if task.wp is not None and task.wp.proc is not None else None
        self.log(f'cancel requested: stopping the watchdog process group (pid {pid}); nothing is deleted')
        task.cancel()

    def watched(self, task, cmd, run_dir):
        """Worker thread: run a watched command and return (state, detail) from disk."""
        wp = ps.WatchedProcess(cmd, run_dir, lambda line: self.post(self.task_line, task, line))
        task.wp = wp
        if task.cancelled:
            return 'cancelled', 'by user'
        self.post(self.log, 'command: ' + ' '.join(cmd))
        wp.run()
        state, detail = ps.run_outcome(run_dir)
        return ('cancelled', 'by user') if wp.cancelled else (state, detail)

    def run_align(self):
        w, python = self.w, self.python
        self.runs_folder.mkdir(parents=True, exist_ok=True)
        if w.paths is None or ps.run_name_taken(self.runs_folder, f'{w.name}-align'):
            old = w.name
            w.name = ps.capture_name(w.folder.root, self.runs_folder)
            if w.paths is not None:
                self.log(f'{old}-align already exists (earlier attempt, kept); using {w.name}-align')
        w.paths = ps.CapturePaths(self.runs_folder, w.name)
        paths, photos, cache = w.paths, w.folder.root, w.cache

        def job(task):
            self.post(self.task_stage, task, 'Checking the Metashape license')
            ok, detail = ps.license_ok(python)
            self.post(self.log, f'license: {detail}')
            if not ok:
                return 'failed', 'license: ' + detail
            if task.cancelled:
                return 'cancelled', 'by user'
            self.post(self.task_stage, task, 'Preparing the photo list')
            manifest = ps.alignment_manifest(photos, paths.name)
            ps.write_manifest(paths.align_manifest, manifest)
            self.post(self.log, f'manifest {paths.align_manifest}: {manifest["image_count"]} images')
            use = None
            if cache is not None:
                usable, why = ps.cache_coverage(cache, manifest)
                use = cache if usable else None
                self.post(self.log, f'image quality: {why}; ' + (f'reusing {cache}' if usable else
                                                                  'not used, analyzeImages will run'))
            run_cmd = ps.watched_command(python, paths.align_manifest, photos, paths.align_run, ps.ALIGN_LIMITS,
                                         image_quality_cache=use)
            return self.watched(task, run_cmd, paths.align_run)

        w.status['Align'] = 'running'
        self.log(f'build "{w.name}": photos {photos}, runs in {self.runs_folder}, Metashape Python {python}')
        self.start_task('align', len(w.folder.photos), job, lambda s, d: self.align_done(w, s, d))
        self.step_align()

    def align_done(self, w, state, detail):
        source = ps.image_quality_source(w.paths.align_run)
        if source:
            self.log(f'image quality source: {source}')
        w.status['Align'] = state if state in ('done', 'cancelled') else 'failed'
        w.outcome['Align'] = plain_outcome(state, detail)
        if self.has_cameras(w.paths.align_run):
            w.align_run = w.paths.align_run
        if self.w is w and w.step == 'Align':
            self.step_align()
        if state == 'failed':
            self.error('Alignment did not finish', plain_outcome(state, detail))

    def leave_align(self):
        self.show_step('Coverage')

    # ------------------------------------------------------------ 4 Coverage

    def coverage_for(self, run_dir, ref):
        """(Coverage, note): vs ref when it shares 3+ placed cameras, else the build alone."""
        cams = Path(run_dir) / 'cameras.csv'
        if ref is None:
            return analyze(cams), ''
        try:
            return compare(cams, ref.path / 'cameras.csv'), ''
        except ValueError as exc:
            if 'share fewer than 3' not in str(exc):
                raise
            self.log(f'coverage: {exc}; showing the build alone')
            return analyze(cams), f'(Could not line up with {ref.name}: too few shared cameras; shown alone.)'

    def step_coverage(self):
        w = self.w
        content, nav = self.wizard_page('Coverage', 'Camera coverage')
        self.nav_buttons(nav, back=lambda: self.show_step('Align'), next_=self.leave_coverage)
        self.coverage_panel(content, w.align_run, w.name, w.compare)
        w.status['Coverage'] = 'done'

    def coverage_panel(self, master, run_dir, name, ref):
        """Figure, the one-line summary and Save image… (shared by the wizard and the result screen)."""
        try:
            cov, note = self.coverage_for(run_dir, ref)
        except Exception as exc:
            self.log(traceback.format_exc().rstrip())
            self.label(master, f'The coverage could not be computed: {exc}', 'Error.TLabel', pady=(20, 0))
            return
        if Path(run_dir, 'cameras.csv').is_file():
            self.prediction_box(master, Path(run_dir) / 'cameras.csv', name)
        bottom = ttk.Frame(master)
        bottom.pack(fill='x', side='bottom', pady=(6, 0))
        text = cov.summary + (f' {note}' if note else '')
        ttk.Label(bottom, text=text, style='Big.TLabel', wraplength=900, justify='left').pack(side='left', anchor='w')
        fig, canvas = self.figure(master, 'coverage')
        ref_name = ref.name if cov.reference is not None else None
        draw_coverage(fig, cov, name, ref_name)
        canvas.draw()
        ttk.Button(bottom, text='Save image…', command=lambda: self.save_image(fig, f'{name}-coverage')
                   ).pack(side='right', anchor='n')
        canvas.get_tk_widget().pack(fill='both', expand=True, pady=(8, 0))
        self.log(f'coverage of {name}' + (f' vs {ref_name}' if ref_name else '') + f': {text}')

    def prediction_box(self, master, cameras, name):
        """EXPERIMENTAL predicted quality from the aligned cameras alone (no mesh or reference needed).
        Its own bordered box under the coverage summary: a number with its spread across the 5 seed
        models and the training caveat, never a verdict and never styled like a measured score."""
        outer = tk.Frame(master, bg=EXP_BG, highlightthickness=1, highlightbackground=EXP_BORDER)
        outer.pack(fill='x', side='bottom', pady=(10, 0))
        box = ttk.Frame(outer, style='Exp.TFrame', padding=(14, 8))
        box.pack(fill='x')
        top = ttk.Frame(box, style='Exp.TFrame')
        top.pack(fill='x')
        ttk.Label(top, text='EXPERIMENTAL', style='Exp.Tag.TLabel').pack(side='left', anchor='n', padx=(0, 12))
        line = ttk.Label(top, text='Predicted quality: working…', style='Exp.Muted.TLabel', wraplength=900,
                         justify='left')
        line.pack(side='left', anchor='w')
        gen = self.gen

        def show(p):
            if self.gen != gen or not box.winfo_exists():
                return
            line.config(text=p.text, style='Exp.TLabel')
            caveat = p.caption
            if p.in_training:
                caveat += '. This build was one of the training runs, so this is not a test of the model.'
            ttk.Label(box, text=caveat, style='Exp.Small.TLabel', wraplength=1000, justify='left'
                      ).pack(anchor='w', pady=(2, 0))
            seeds = ', '.join(f'seed {s} {v:.4f}' for s, v in sorted(p.per_seed.items()))
            self.log(f'experimental prediction for {name} ({p.cameras} cameras; {seeds}); '
                     f'{p.caption}' + (f'; training run {p.in_training}' if p.in_training else ''))

        def unavailable(why):
            if self.gen == gen and box.winfo_exists():
                line.config(text=f'Predicted quality: not available ({why}).')

        def job():
            try:
                p = qm.predict_quality(cameras, QUALITY_MODELS)
            except qm.ModelsMissing as exc:
                self.post(self.log, f'experimental prediction: {exc}')
                self.post(unavailable, 'the experimental models are not on this computer')
                return
            except ValueError as exc:
                self.post(self.log, f'experimental prediction: {exc}')
                self.post(unavailable, str(exc))
                return
            self.post(show, p)
        self.background(job)

    def leave_coverage(self):
        self.show_step('Mesh')

    # ------------------------------------------------------------ 5 Mesh

    def score_possible(self):
        w = self.w
        return w.compare is not None and w.compare.has_mesh

    def step_mesh(self):
        w = self.w
        n = len(w.folder.photos)
        content, nav = self.wizard_page('Mesh', 'Build a 3D mesh (optional)',
                                        f'Uses Metashape (license needed). Takes {bl.mesh_time_estimate(n)} '
                                        f'for {n} photos.')
        state = w.status['Mesh']
        to_score = w.compare is not None
        self.nav_buttons(nav, back=lambda: self.show_step('Coverage'), next_=self.leave_mesh,
                         next_text='Next  →' if to_score else 'Finish', next_enabled=state in ('done', 'skipped'))
        box = ttk.Frame(content, padding=(0, 30, 0, 0))
        box.pack(fill='both', expand=True)
        if self.task is not None and self.task.kind == 'mesh':
            ProgressView(box, self, self.task).pack(fill='x')
            return
        if state == 'done':
            self.label(box, '✓  Mesh built.', 'H2.TLabel')
            if to_score and not w.compare.has_mesh:
                self.label(box, f'{w.compare.name} has no mesh, so there is nothing to score against.',
                           'Muted.TLabel', pady=(6, 0))
            return
        if state in ('failed', 'cancelled'):
            self.label(box, ('✗  ' if state == 'failed' else '') + w.outcome.get('Mesh', ''),
                       'Error.TLabel' if state == 'failed' else 'Warn.TLabel', pady=(0, 16))
        if state == 'skipped':
            self.label(box, 'Skipped. You can still build the mesh now.', 'Muted.TLabel', pady=(0, 16))
        row = ttk.Frame(box)
        row.pack(anchor='w')
        build = ttk.Button(row, text='Build mesh' if state in ('todo', 'skipped') else 'Try again',
                           style='Start.TButton', command=self.run_mesh)
        build.pack(side='left')
        if state != 'skipped':
            ttk.Button(row, text='Skip', command=self.skip_mesh).pack(side='left', padx=16)
        reason = self.metashape_note()
        if not reason and not (w.align_run and ps.is_alignment_checkpoint(w.align_run)):
            reason = 'The mesh needs a successful alignment.'
        if reason:
            build.config(state='disabled')
            self.label(box, reason, 'Error.TLabel' if not self.detecting else 'Muted.TLabel', pady=(14, 0))

    def skip_mesh(self):
        w = self.w
        w.status['Mesh'] = 'skipped'
        if w.compare is not None:
            w.status['Score'] = 'skipped'
        self.log('mesh: skipped')
        self.leave_mesh()

    def leave_mesh(self):
        if self.w.compare is not None:
            self.show_step('Score')
        else:
            self.finish()

    def run_mesh(self):
        w, python, photos = self.w, self.python, self.w.folder.root
        align_run = w.align_run
        mesh_run = self.runs_folder / ps.unused_run_name(self.runs_folder, w.paths.mesh_run.name)
        if mesh_run != w.paths.mesh_run:
            self.log(f'{w.paths.mesh_run.name} already exists (earlier attempt, kept); using {mesh_run.name}')
        manifest_path = w.paths.manifests / f'{mesh_run.name}.json'

        def job(task):
            self.post(self.task_stage, task, 'Checking the Metashape license')
            ok, detail = ps.license_ok(python)
            self.post(self.log, f'license: {detail}')
            if not ok:
                return 'failed', 'license: ' + detail
            if task.cancelled:
                return 'cancelled', 'by user'
            align = json.loads((align_run / 'manifest.json').read_text())
            ps.write_manifest(manifest_path, ps.mesh_manifest(align))
            self.post(self.log, f'mesh manifest {manifest_path}, seeded from {align_run}')
            cmd = ps.watched_command(python, manifest_path, photos, mesh_run, ps.MESH_LIMITS, seed=align_run)
            return self.watched(task, cmd, mesh_run)

        w.status['Mesh'] = 'running'
        if w.compare is not None and w.status['Score'] == 'skipped':
            w.status['Score'] = 'todo'
        self.start_task('mesh', len(w.folder.photos), job, lambda s, d: self.mesh_done(w, mesh_run, s, d))
        self.step_mesh()

    def mesh_done(self, w, mesh_run, state, detail):
        ok = state == 'done' and (mesh_run / 'raw-mesh.obj').is_file()
        w.status['Mesh'] = 'done' if ok else 'cancelled' if state == 'cancelled' else 'failed'
        w.outcome['Mesh'] = plain_outcome(state, detail)
        if ok:
            w.mesh_run, w.score = mesh_run, None
            self.builds = None
        if self.w is w and w.step == 'Mesh':
            self.step_mesh()
        if w.status['Mesh'] == 'failed':
            self.error('The mesh did not finish', w.outcome['Mesh'])

    # ------------------------------------------------------------ 6 Score

    def step_score(self):
        w = self.w
        content, nav = self.wizard_page('Score', 'Score the mesh')
        self.nav_buttons(nav, back=lambda: self.show_step('Mesh'), next_=self.finish, next_text='Finish')
        box = ttk.Frame(content, padding=(0, 20, 0, 0))
        box.pack(fill='both', expand=True)
        if w.compare is None:
            self.label(box, 'Skipped: no earlier build was chosen to compare with.', 'Big.TLabel')
        elif w.mesh_run is None:
            self.label(box, 'Skipped: no mesh was built.', 'Big.TLabel')
            w.status['Score'] = 'skipped'
        elif not w.compare.has_mesh:
            self.label(box, f'Skipped: {w.compare.name} has no mesh.', 'Big.TLabel')
            w.status['Score'] = 'skipped'
        else:
            self.score_panel(box, w.mesh_run, w.name, w.compare, cached=w.score,
                             on_row=lambda row: self.wizard_scored(w, row), show_figure=True)

    def wizard_scored(self, w, row):
        w.score, w.status['Score'] = row, 'done'

    def score_panel(self, box, run_dir, name, ref, cached=None, on_row=None, show_figure=False):
        """'Scoring…', then the score lines (+ figure) and Save image…; the score runs in a worker."""
        holder = ttk.Frame(box)
        holder.pack(fill='both', expand=True)
        waiting = self.label(holder, 'Scoring… (a few seconds)', 'Muted.TLabel')
        gen = self.gen

        def show(row):
            if self.gen != gen or not holder.winfo_exists():
                return
            waiting.destroy()
            if on_row:
                on_row(row)
            headline, numbers, explanation, warning = score_lines(row, ref.name)
            top = ttk.Frame(holder)
            top.pack(fill='x')
            ttk.Label(top, text=headline, style='H2.TLabel').pack(side='left')
            fig = Figure(figsize=(8, 2.6), facecolor=BG)
            draw_score(fig, row, name, ref.name)
            ttk.Button(top, text='Save image…', command=lambda: self.save_score_image(row, name, ref.name)
                       ).pack(side='right')
            self.label(holder, numbers, 'Big.TLabel', pady=(4, 0))
            self.label(holder, explanation, 'Muted.TLabel', 1000, pady=(8, 0))
            if warning:
                self.label(holder, '⚠  ' + warning, 'Warn.TLabel', 1000, pady=(8, 0))
            if show_figure:
                self.figures['score'] = fig
                canvas = FigureCanvasTkAgg(fig, master=holder)
                canvas.get_tk_widget().configure(bg=BG, highlightthickness=0)
                canvas.draw()
                canvas.get_tk_widget().pack(fill='both', expand=True, pady=(12, 0))
            self.log(f'score of {name} vs {ref.name}: {ps.score_text(row)}')

        def failed(exc):
            if self.gen == gen and holder.winfo_exists():
                waiting.config(text=f'The score could not be computed: {exc}', style='Error.TLabel')

        if cached is not None:
            return show(cached)

        def job():
            try:
                row = ps.score_run(run_dir, ref.mesh)
            except Exception as exc:
                self.post(self.log, traceback.format_exc().rstrip())
                self.post(failed, exc)
                return
            self.post(show, row)
        self.log(f'scoring {run_dir} against {ref.mesh} (evaluator v3) …')
        self.background(job)

    def save_score_image(self, row, name, ref_name):
        fig = Figure(figsize=(8, 2.6), facecolor='white')
        draw_score(fig, row, name, ref_name)
        self.save_image(fig, f'{name}-score-vs-{ref_name}')

    def finish(self):
        self.log(f'build "{self.w.name}" finished' if self.w else 'finished')
        self.builds = None
        self.show_start()

    # ------------------------------------------------------------ Open an existing build

    def build_table(self, master, columns, height=12):
        frame = ttk.Frame(master)
        frame.pack(fill='both', expand=True, pady=(18, 0))
        tree = ttk.Treeview(frame, columns=[c for c, _, _ in columns], show='headings', height=height,
                            selectmode='browse')
        for key, title, width in columns:
            anchor = 'w' if key in ('name', 'group', 'date') else 'center'
            tree.heading(key, text=title, anchor=anchor, command=lambda k=key: self.sort_table(tree, k))
            tree.column(key, width=width, anchor=anchor, stretch=key == 'name')
        scroll = ttk.Scrollbar(frame, orient='vertical', command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        tree.pack(side='left', fill='both', expand=True)
        scroll.pack(side='right', fill='y')
        return tree

    def sort_table(self, tree, key):
        """Click a column heading to sort by it; click again to reverse."""
        reverse = getattr(tree, 'sorted_by', None) == key
        tree.sorted_by = None if reverse else key
        rows = getattr(tree, 'rows', {})

        def value(iid):
            if key == 'date':
                b = rows.get(iid)
                return b.date.timestamp() if b is not None and b.date else 0
            v = tree.set(iid, key)
            try:
                return float(v)
            except ValueError:
                return v.lower()
        for i, iid in enumerate(sorted(tree.get_children(''), key=value, reverse=reverse)):
            tree.move(iid, '', i)

    def show_open(self):
        self.w = None
        self.clear()
        bar = ttk.Frame(self.body, padding=(20, 12, 20, 0))
        bar.pack(fill='x')
        self.back_button = ttk.Button(bar, text='←  Start', style='Link.TButton', command=self.show_start)
        self.back_button.pack(side='left')
        content = ttk.Frame(self.body, padding=(48, 20, 48, 12))
        content.pack(fill='both', expand=True)
        head = ttk.Frame(content)
        head.pack(fill='x')
        ttk.Label(head, text='Open an existing build', style='H1.TLabel').pack(side='left')
        ttk.Button(head, text='Refresh', command=lambda: (setattr(self, 'builds', None), self.show_open())
                   ).pack(side='right')
        self.label(content, 'Builds in the results folder and in your builds folder. Double-click to open.',
                   'Muted.TLabel', pady=(6, 0))
        tree = self.build_table(content, [('name', 'Name', 360), ('group', 'Location', 220), ('photos', 'Photos', 90),
                                          ('mesh', 'Mesh', 70), ('date', 'Date', 140)], height=14)
        nav = ttk.Frame(self.body, padding=(32, 0, 32, 16))
        nav.pack(fill='x', side='bottom', before=content)
        self.next_button = ttk.Button(nav, text='Open', style='Accent.TButton', state='disabled',
                                      command=lambda: self.open_pick(tree))
        self.next_button.pack(side='right')
        tree.bind('<<TreeviewSelect>>', lambda e: self.next_button.config(state='normal' if tree.selection() else 'disabled'))
        tree.bind('<Double-1>', lambda e: self.open_pick(tree))
        self.open_tree = tree
        if self.builds is None:
            waiting = self.label(content, 'Looking for builds…', 'Muted.TLabel', pady=(10, 0))
            roots = [REPO / 'results', self.runs_folder]
            gen = self.gen

            def found(builds):
                self.builds = builds
                if self.gen == gen:
                    waiting.destroy()
                    self.fill_open(tree)

            self.background(lambda: self.post(found, bl.find_builds(roots)))
        else:
            self.fill_open(tree)
        tree.focus_set()

    def fill_open(self, tree):
        self.open_rows = {}
        for b in self.builds:
            iid = tree.insert('', 'end', values=(b.name, b.group, b.photo_count, '✓' if b.has_mesh else '–',
                                                 date_text(b.date)))
            self.open_rows[iid] = b
        tree.rows = self.open_rows
        if not self.builds:
            tree.insert('', 'end', values=('No builds found', '', '', '', ''))
        else:
            first = tree.get_children('')[0]
            tree.selection_set(first)
            tree.focus(first)

    def open_pick(self, tree):
        sel = tree.selection()
        build = self.open_rows.get(sel[0]) if sel else None
        if build is not None:
            self.show_result(build)

    def show_result(self, build, ref='default'):
        self.w = None
        self.clear()
        builds = self.builds or []
        if ref == 'default':
            ref = bl.default_reference(build, builds)
        bar = ttk.Frame(self.body, padding=(20, 12, 20, 0))
        bar.pack(fill='x')
        self.back_button = ttk.Button(bar, text='←  All builds', style='Link.TButton', command=self.show_open)
        self.back_button.pack(side='left')
        content = ttk.Frame(self.body, padding=(48, 14, 48, 12))
        content.pack(fill='both', expand=True)
        head = ttk.Frame(content)
        head.pack(fill='x')
        ttk.Label(head, text=build.name, style='H1.TLabel').pack(side='left')
        ttk.Label(head, text=f'{build.group}  ·  {build.photo_count} photos  ·  mesh {"✓" if build.has_mesh else "–"}'
                             f'  ·  {date_text(build.date)}', style='Muted.TLabel').pack(side='left', padx=20)

        options = [(None, 'Nothing (this build alone)')]
        options += [(b, f'{b.name}  ({b.group}, shares {n} photos{", mesh" if b.has_mesh else ""})')
                    for b, n in bl.compatible_builds(build.photos, builds, exclude=build)]
        row = ttk.Frame(content)
        row.pack(fill='x', pady=(12, 0))
        ttk.Label(row, text='Compare with:').pack(side='left')
        names = [t for _, t in options]
        var = tk.StringVar(value=next((t for b, t in options if b is ref), names[0]))
        box = ttk.Combobox(row, textvariable=var, values=names, state='readonly', width=56, font=self.fonts['body'])
        box.pack(side='left', padx=10)
        box.bind('<<ComboboxSelected>>',
                 lambda e: self.show_result(build, next(b for b, t in options if t == var.get())))
        if len(options) == 1:
            box.config(state='disabled')
            ttk.Label(row, text='(no other build shares photos with this one)', style='Small.TLabel').pack(side='left')

        score_card = ttk.Frame(content, style='Card.TFrame', padding=16)
        score_card.pack(fill='x', side='bottom', pady=(10, 0))
        cov_box = ttk.Frame(content)
        cov_box.pack(fill='both', expand=True, pady=(8, 0))
        self.coverage_panel(cov_box, build.path, build.name, ref)
        if ref is None:
            ttk.Label(score_card, text='No score: choose a build to compare with.', style='Card.Muted.TLabel'
                      ).pack(anchor='w')
        elif not build.has_mesh or not ref.has_mesh:
            missing = build.name if not build.has_mesh else ref.name
            ttk.Label(score_card, text=f'No score: {missing} has no mesh.', style='Card.Muted.TLabel').pack(anchor='w')
        else:
            self.result_score(score_card, build, ref)

    def result_score(self, card, build, ref):
        """Compact score block on the card: headline, numbers, explanation, Save image…"""
        waiting = ttk.Label(card, text='Scoring… (a few seconds)', style='Card.Muted.TLabel')
        waiting.pack(anchor='w')
        gen = self.gen

        def show(row):
            if self.gen != gen or not card.winfo_exists():
                return
            waiting.destroy()
            headline, numbers, explanation, warning = score_lines(row, ref.name)
            top = ttk.Frame(card, style='Card.TFrame')
            top.pack(fill='x')
            ttk.Label(top, text=headline, style='Card.H2.TLabel').pack(side='left')
            ttk.Label(top, text=numbers, style='Card.Big.TLabel').pack(side='left', padx=24)
            ttk.Button(top, text='Save image…', command=lambda: self.save_score_image(row, build.name, ref.name)
                       ).pack(side='right')
            ttk.Label(card, text=explanation, style='Card.Muted.TLabel', wraplength=1050, justify='left'
                      ).pack(anchor='w', pady=(6, 0))
            if warning:
                ttk.Label(card, text='⚠  ' + warning, style='Card.Warn.TLabel', wraplength=1050, justify='left'
                          ).pack(anchor='w', pady=(4, 0))
            self.log(f'score of {build.name} vs {ref.name}: {ps.score_text(row)}')

        def failed(exc):
            if self.gen == gen and card.winfo_exists():
                waiting.config(text=f'The score could not be computed: {exc}', style='Card.Error.TLabel')

        def job():
            try:
                row = ps.score_run(build.mesh, ref.mesh)
            except Exception as exc:
                self.post(self.log, traceback.format_exc().rstrip())
                self.post(failed, exc)
                return
            self.post(show, row)
        self.log(f'scoring {build.mesh} against {ref.mesh} (evaluator v3) …')
        self.background(job)

    # ------------------------------------------------------------ close

    def close(self):
        if self.task is not None:
            if not messagebox.askyesno('Still working', 'A step is running. Cancel it and quit? Nothing is deleted.',
                                       parent=self.root):
                return
            self.task.cancel()
        self.root.destroy()


def main():
    root = tk.Tk()
    root.geometry('1280x900')
    root.minsize(1000, 720)
    CaptureApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
