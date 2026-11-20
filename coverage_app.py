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
import re
import sys
import threading
import time
import tkinter as tk
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, font as tkfont, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import matplotlib
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle

try:
    import bd3d_quality  # noqa: F401
except ImportError:  # not installed: use the source tree next to this file
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from bd3d_quality import builds as bl
from bd3d_quality import metashape_locator as ml
from bd3d_quality import pipeline_steps as ps
from bd3d_quality import quality_model as qm
from bd3d_quality import theme as th
from bd3d_quality import viewer_export as vx
from bd3d_quality.coverage import BAND_NAMES, analyze, compare, load_cameras
from bd3d_quality.progress import Progress, StageTracker

REPO = Path(__file__).resolve().parent
DEFAULT_RUNS = REPO / 'runs' / 'app'
QUALITY_MODELS = REPO / 'models' / 'quality-attention'   # experimental, gitignored
STEPS = ['Photos', 'Compare', 'Align', 'Coverage', 'Mesh', 'Score']
STEP_NAMES = {'Compare': 'Reference'}   # shown name; the key stays

# look: every value comes from viewer/tokens.css (bd3d_quality.theme), shared with the web viewer
BG, SURFACE, BORDER, PANEL = th.PAGE, th.CARD, th.LINE, th.PANEL
TEXT, MUTED, INK = th.TEXT, th.TEXT_2, th.INK
ACCENT, ACCENT_DARK, ACCENT_SOFT = th.INK, '#2a2a28', th.NEUTRAL_SOFT
OK, WARN, ERROR = th.GREEN, th.AMBER, th.RED
RED, GREY = th.RED, '#dcdad4'
EXP_BG, EXP_BORDER, EXP_TAG = th.AMBER_SOFT, th.AMBER_SOFT, th.AMBER   # experimental: caution, never like a result
matplotlib.rcParams.update({'font.family': ['System Font', 'Helvetica Neue', 'DejaVu Sans'], 'font.size': 10,
                            'text.color': TEXT, 'axes.labelcolor': MUTED, 'xtick.color': MUTED,
                            'ytick.color': MUTED, 'axes.edgecolor': BORDER, 'axes.titlecolor': INK})


def setup_style(root):
    family = tkfont.nametofont('TkDefaultFont').actual('family')

    def font(size, weight='normal'):
        return tkfont.Font(family=family, size=size, weight=weight)
    fonts = {'body': font(14), 'small': font(12), 'bold': font(14, 'bold'), 'big': font(16),
             'h2': font(19, 'bold'), 'h1': font(26, 'bold'), 'stat': font(34, 'bold'), 'tag': font(11, 'bold'),
             'mono': tkfont.Font(family='Menlo', size=11)}
    root.configure(bg=BG)
    s = ttk.Style(root)
    s.theme_use('clam')
    s.configure('.', background=BG, foreground=TEXT, font=fonts['body'], bordercolor=BORDER,
                lightcolor=BG, darkcolor=BG, focuscolor=INK, troughcolor=PANEL)
    for name, bg in (('TFrame', BG), ('In.TFrame', SURFACE), ('OnPanel.TFrame', PANEL), ('Exp.TFrame', EXP_BG)):
        s.configure(name, background=bg)
    labels = [('TLabel', TEXT, 'body'), ('Muted', MUTED, 'body'), ('Small', MUTED, 'small'), ('H1', INK, 'h1'),
              ('H2', INK, 'h2'), ('Big', TEXT, 'big'), ('Bold', INK, 'bold'), ('Stat', INK, 'stat'),
              ('Ok', OK, 'body'), ('Warn', WARN, 'small'), ('Error', ERROR, 'body'),
              ('StepNow', INK, 'bold'), ('StepDone', TEXT, 'body'), ('StepTodo', MUTED, 'body'), ('StepBad', ERROR, 'body')]
    for prefix, bg in (('', BG), ('Card.', SURFACE), ('Panel.', PANEL), ('Exp.', EXP_BG)):
        for name, fg, f in labels:
            s.configure(prefix + (name if name == 'TLabel' else f'{name}.TLabel'), background=bg, foreground=fg,
                        font=fonts[f])
    s.configure('Exp.Tag.TLabel', background=EXP_BG, foreground=EXP_TAG, font=fonts['tag'])

    th.install(s, root)                       # Card/Panel frames, Primary pill, Text buttons, Tag labels
    for host in ('', 'Card.'):
        s.configure(f'{host}Primary.TButton', font=fonts['bold'])
        s.configure(f'{host}Text.TButton', font=fonts['body'])
    s.layout('Panel.Text.TButton', th.TEXT_LAYOUT)
    s.configure('Panel.Text.TButton', background=PANEL, foreground=INK, bordercolor=PANEL, lightcolor=PANEL,
                darkcolor=PANEL, padding=(0, 6), relief='flat', font=fonts['body'], width=0, anchor='w')
    s.map('Panel.Text.TButton', background=[('active', PANEL), ('pressed', PANEL)],
          foreground=[('disabled', '#b9b7b1'), ('active', MUTED)])
    s.configure('TButton', background=SURFACE, foreground=INK, bordercolor=BORDER, lightcolor=SURFACE,
                darkcolor=SURFACE, padding=(14, 7), font=fonts['body'])
    s.map('TButton', background=[('active', ACCENT_SOFT), ('pressed', ACCENT_SOFT)], foreground=[('disabled', '#b9b7b1')])
    s.configure('Horizontal.TProgressbar', background=INK, troughcolor='#dddbd5', bordercolor='#dddbd5',
                lightcolor=INK, darkcolor=INK, thickness=6)
    s.configure('TCombobox', padding=6, arrowsize=14, fieldbackground=SURFACE, background=SURFACE,
                bordercolor=BORDER, lightcolor=SURFACE, darkcolor=SURFACE, arrowcolor=MUTED)
    s.map('TCombobox', fieldbackground=[('readonly', SURFACE)], background=[('readonly', SURFACE)])
    s.configure('TEntry', fieldbackground=SURFACE, bordercolor=BORDER, lightcolor=SURFACE, darkcolor=SURFACE, padding=6)
    s.configure('Vertical.TScrollbar', background=BG, troughcolor=BG, bordercolor=BG, arrowcolor=MUTED,
                lightcolor=BG, darkcolor=BG)
    root.option_add('*TCombobox*Listbox.font', fonts['body'])
    return fonts


# ---------------------------------------------------------------- plain-language text

def plain_outcome(state, detail):
    """One short sentence for a step outcome from pipeline_steps.run_outcome."""
    if state == 'cancelled':
        return 'Cancelled. Nothing was deleted.'
    d = detail or ''
    if 'license' in d.lower():
        return 'No Metashape license. Check the VPN or license server.'
    if d.startswith('watchdog stop'):
        return f'Stopped by the resource limits ({d.split(":", 1)[1].strip().replace("_", " ")}).'
    if 'alignment failed' in d:
        return 'Too few cameras aligned.'
    return 'Failed. See the log.'


def quality(label):
    """(text, tag kind) for a v3 label: the viewer's bands (under 1.2% good, under 2% fair)."""
    return ('Good', 'good') if label < 0.012 else ('Fair', 'caution') if label < 0.02 else ('Poor', 'bad')


def score_rows(row):
    """[(name, value, tag kind or None)] for a v3 score row; distances are fractions of the reference diagonal."""
    y = row['label_v3']
    qc, scale = row.get('qc_v3'), row.get('scale_check')
    return [('Geometric error', f'{y * 100:.2f}%', quality(y)),
            ('Missing RMSE', f'{row["r2c_rmse_norm"] * 100:.2f}%', None),
            ('Extra RMSE', f'{row["c2r_rmse_norm"] * 100:.2f}%', None),
            ('Registration QC', None, (str(qc), 'good' if qc == 'pass' else 'bad')),
            ('Scale check', None, (str(scale), 'good' if scale == 'agree' else 'caution'))]


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def aligned_count(run_dir):
    labels, aligned, _, _ = load_cameras(Path(run_dir) / 'cameras.csv')
    return int(aligned.sum()), len(labels)


def date_text(d):
    return d.astimezone().strftime('%-d %b %Y') if d else ''


# ---------------------------------------------------------------- figures

def camera_states(cov):
    """Positions (top view, x/y) of used, removed and failed cameras; same three states as the viewer.
    Removed and failed are placed where the reference had them; failed with no reference has no place."""
    used = cov.positions[:, :2]
    ref = cov.reference
    if ref is None:
        return used, np.empty((0, 2)), np.empty((0, 2)), len(cov.failed)
    have, failed = set(cov.labels), set(cov.failed)
    rem = [i for i, l in enumerate(ref.labels) if l not in have and l not in failed]
    bad = [i for i, l in enumerate(ref.labels) if l in failed]
    return used, ref.positions[rem, :2], ref.positions[bad, :2], len(cov.failed) - len(bad)


def draw_coverage(fig, cov, run_name, ref_name=None):
    """Top view and azimuth x height map (single build or vs a reference). Colour only means state:
    grey = camera used, red = problem (failed to align, removed, too few cameras in a cell)."""
    fig.clear()
    fig.set_facecolor(PANEL)
    ax, cm = fig.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1.6]})
    for a in (ax, cm):
        a.set_facecolor(PANEL)
        for sp in a.spines.values():
            sp.set_visible(False)
    ref = cov.reference
    used, removed, failed, unplaced = camera_states(cov)
    allp = np.vstack([used, removed, failed])
    r = np.linalg.norm(allp, axis=1).max()
    ax.add_patch(plt_circle(r * 1.06))
    if len(removed):
        ax.scatter(removed[:, 0], removed[:, 1], s=30, facecolors='none', edgecolors=RED, linewidths=1,
                   linestyles=(0, (2, 1.5)), zorder=2, label=f'removed for this build ({len(removed)})')
    ax.scatter(used[:, 0], used[:, 1], s=16, color=th.CAMERA, zorder=3, label=f'used ({len(used)})')
    n_failed = len(failed) + unplaced
    if len(failed):
        ax.scatter(failed[:, 0], failed[:, 1], s=34, color=RED, zorder=4, label=f'failed to align ({n_failed})')
    elif n_failed:
        ax.scatter([], [], s=34, color=RED, label=f'failed to align ({n_failed})')
    ax.plot(0, 0, '+', color=MUTED, ms=10)
    ax.annotate('0°', (r * 1.12, 0), ha='left', va='center', color=MUTED)
    ax.set_title('Cameras from above', fontsize=12, loc='left')
    ax.legend(loc='upper left', fontsize=9, frameon=False, bbox_to_anchor=(0, -0.02))
    ax.set_aspect('equal')
    ax.set_xlim(-r * 1.3, r * 1.3)
    ax.set_ylim(-r * 1.3, r * 1.3)
    ax.set_xticks([])
    ax.set_yticks([])

    az, el, n = cov.az_edges, cov.el_edges, cov.counts
    for i in range(n.shape[0]):
        for j in range(n.shape[1]):
            miss, never = cov.missing[i, j], cov.never[i, j]
            face = RED if miss else GREY if never else 'white'
            cm.add_patch(Rectangle((az[j], el[i]), az[j + 1] - az[j], el[i + 1] - el[i],
                                   facecolor=face, edgecolor=PANEL, linewidth=2))
            text = f'{n[i, j]}' if ref is None else f'{n[i, j]}/{ref.counts[i, j]}'
            cm.text((az[j] + az[j + 1]) / 2, (el[i] + el[i + 1]) / 2, text, ha='center', va='center',
                    fontsize=10 if ref is None else 8.5, color='white' if miss else TEXT)
    for name, lo, hi in zip(BAND_NAMES.get(len(n), []), el[:-1], el[1:]):
        cm.text(363, (lo + hi) / 2, name, va='center', fontsize=10, color=MUTED)
    cm.set_xlim(0, 360)
    cm.set_ylim(el[0], el[-1])
    cm.set_xticks(az[::2])
    cm.tick_params(length=0)
    cm.set_ylabel('height (°)')
    if ref is None:
        cm.set_xlabel('around the object (°, 0 = first camera)')
        cm.set_title('Cameras per cell', fontsize=12, loc='left')
        handles = [Patch(facecolor=RED, label=f'fewer than {cov.min_cameras} cameras'),
                   Patch(facecolor='white', edgecolor=BORDER, label='number = cameras in the cell')]
    else:
        cm.set_xlabel(f'around the object (°, as in {ref_name})')
        cm.set_title('Cameras per cell: this build / reference', fontsize=12, loc='left')
        handles = [Patch(facecolor=RED, label=f'reference has {cov.min_cameras}+, this build fewer'),
                   Patch(facecolor=GREY, label='reference has few here too'),
                   Patch(facecolor='white', edgecolor=BORDER, label='covered')]
    cm.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False, fontsize=9)
    fig.tight_layout()


def plt_circle(r):
    from matplotlib.patches import Circle
    return Circle((0, 0), r, facecolor='white', edgecolor='none', zorder=0)


def draw_score(fig, row, run_name, ref_name, bg=PANEL):
    """Geometric error, missing and extra RMSE as bars, with the good / fair bands marked."""
    fig.clear()
    fig.set_facecolor(bg)
    ax = fig.subplots()
    ax.set_facecolor(bg)
    names = ['Geometric error', 'Missing RMSE', 'Extra RMSE']
    values = [100 * row['label_v3'], 100 * row['r2c_rmse_norm'], 100 * row['c2r_rmse_norm']]
    colors = [INK, th.MESH, th.MESH]
    bars = ax.barh(names[::-1], values[::-1], color=colors[::-1], height=0.5)
    top = max(max(values) * 1.3, 2.4)
    for x, text in ((1.2, 'good < 1.2%'), (2.0, 'fair < 2%')):
        ax.plot([x, x], [-0.6, 2.4], color=MUTED, lw=0.8, ls=(0, (3, 3)))
        ax.text(x, 2.55, text, ha='center', va='bottom', fontsize=9, color=MUTED)
    for b, v in zip(bars, values[::-1]):
        ax.text(b.get_width() + top * 0.012, b.get_y() + b.get_height() / 2, f'{v:.2f}%', va='center', fontsize=11)
    ax.set_xlim(0, top)
    ax.set_ylim(-0.6, 2.9)
    ax.set_xlabel(f'% of the reference size ({ref_name} bounding-box diagonal) · lower is better')
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0, labelsize=11)
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
    """Stage, a thin bar (filled where the log counts items), elapsed time and Cancel, on a grey panel."""

    def __init__(self, master, app, task):
        super().__init__(master, style='OnPanel.TFrame')
        self.app, self.task = app, task
        self.stage = ttk.Label(self, text=task.tracker.current.stage, style='Panel.H2.TLabel')
        self.stage.pack(anchor='w')
        self.bar = ttk.Progressbar(self, mode='indeterminate', length=560)
        self.bar.pack(anchor='w', pady=(16, 8), fill='x')
        row = ttk.Frame(self, style='OnPanel.TFrame')
        row.pack(fill='x')
        self.detail = ttk.Label(row, text='', style='Panel.Muted.TLabel')
        self.detail.pack(side='left')
        self.elapsed = ttk.Label(row, text='0:00', style='Panel.Muted.TLabel')
        self.elapsed.pack(side='right')
        self.cancel_button = ttk.Button(self, text='Cancel', style='Panel.Text.TButton', command=self.cancel)
        self.cancel_button.pack(anchor='w', pady=(18, 0))
        self.indeterminate = False
        task.view = self
        self.update_progress(task.tracker.current)
        self.tick()

    def update_progress(self, p):
        if not self.winfo_exists():
            return
        self.stage.config(text=p.stage)
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

        top = ttk.Frame(root, padding=(40, 14, 34, 14))
        top.pack(fill='x')
        ttk.Label(top, text='BD3D', style='Bold.TLabel').pack(side='left')
        ttk.Label(top, text='Capture check', style='Muted.TLabel').pack(side='left', padx=(10, 0))
        ttk.Button(top, text='Advanced', style='Text.TButton', command=self.open_advanced).pack(side='right')
        tk.Frame(root, height=1, bg=BORDER).pack(fill='x')
        self.body = ttk.Frame(root)
        self.body.pack(fill='both', expand=True)

        self.details = ttk.Frame(root, padding=(34, 0, 40, 10))
        self.details_button = ttk.Button(self.details, text='Log', style='Text.TButton',
                                         command=self.toggle_details)
        self.details_button.pack(anchor='w')
        self.log_box = ScrolledText(self.details, height=12, state='disabled', font=self.fonts['mono'],
                                    relief='flat', bg=SURFACE, fg=TEXT, highlightthickness=1, highlightbackground=BORDER)
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
            self.details_button.config(text='Hide log')
        else:
            self.log_box.pack_forget()
            self.details_button.config(text='Log')

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

    def figure(self, master, key, size=(11, 4.6), bg=PANEL):
        fig = Figure(figsize=size, facecolor=bg)
        canvas = FigureCanvasTkAgg(fig, master=master)
        canvas.get_tk_widget().configure(bg=bg, highlightthickness=0)
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
        outer = ttk.Frame(win, padding=24)
        outer.pack(fill='both', expand=True)
        frame = ttk.Frame(outer, style='Card.TFrame', padding=28)
        frame.pack(fill='both', expand=True)
        win.bind('<Escape>', lambda e: win.destroy())

        def section(title, first=False):
            ttk.Label(frame, text=title, style='Card.Bold.TLabel').pack(anchor='w', pady=(0 if first else 18, 4))

        section('Metashape', first=True)
        self.adv_python = ttk.Label(frame, style='Card.Small.TLabel', wraplength=600)
        self.adv_python.pack(anchor='w')
        self.adv_python_status = ttk.Label(frame, style='Card.TLabel', wraplength=600)
        self.adv_python_status.pack(anchor='w', pady=(2, 2))
        row = ttk.Frame(frame, style='In.TFrame')
        row.pack(anchor='w')
        ttk.Button(row, text='Change…', style='Card.Text.TButton', command=self.change_python).pack(side='left')
        ttk.Button(row, text='Search again', style='Card.Text.TButton', command=self.detect_python
                   ).pack(side='left', padx=12)
        self.label(frame, 'Align and Mesh need a Metashape license (on the network: VPN).', 'Card.Small.TLabel', 600)

        section('Builds folder')
        self.adv_runs = ttk.Label(frame, style='Card.Small.TLabel', wraplength=600)
        self.adv_runs.pack(anchor='w')
        row = ttk.Frame(frame, style='In.TFrame')
        row.pack(anchor='w')
        ttk.Button(row, text='Change…', style='Card.Text.TButton', command=self.change_runs_folder).pack(side='left')
        ttk.Button(row, text='Use the default', style='Card.Text.TButton',
                   command=lambda: self.set_runs_folder(DEFAULT_RUNS)).pack(side='left', padx=12)

        section('Reconstruction settings (as ref50 and ref145b)')
        self.label(frame, ps.settings_text(), 'Card.Small.TLabel', 600)
        section('Resource limits')
        self.label(frame, f'Align  {ps.ALIGN_LIMITS.text()}\nMesh   {ps.MESH_LIMITS.text()}', 'Card.Small.TLabel', 600)
        ttk.Button(outer, text='Done', style='Primary.TButton', command=win.destroy).pack(anchor='e', pady=(16, 0))
        self.refresh_advanced()

    def refresh_advanced(self):
        if self.advanced is None or not self.advanced.winfo_exists():
            return
        self.adv_python.config(text=self.python or '(none)')
        self.adv_python_status.config(text=self.python_text, style='Card.Ok.TLabel' if self.python else
                                      'Card.Muted.TLabel' if self.detecting else 'Card.Error.TLabel')
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
        self.label(frame, 'Python that can import Metashape (for example a venv\'s bin/python):',
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

        ttk.Button(buttons, text='Browse…', style='Text.TButton', command=browse).pack(side='left')
        test = ttk.Button(buttons, text='Test and use', style='Primary.TButton', command=try_it)
        test.pack(side='right')
        ttk.Button(buttons, text='Cancel', style='Text.TButton', command=win.destroy).pack(side='right', padx=12)
        win.bind('<Return>', lambda e: try_it())
        win.bind('<Escape>', lambda e: win.destroy())
        entry.focus_set()

    # ------------------------------------------------------------ start

    def show_start(self):
        self.w = None
        self.clear(details=False)
        page = ttk.Frame(self.body)
        page.place(relx=0.5, rely=0.42, anchor='center')
        for title, text, button, style, command in (
                ('New build', 'Align photos, check camera coverage,\nbuild and score a mesh.', 'New build',
                 'Card.Primary.TButton', self.new_build),
                ('Builds', 'Open an earlier build: coverage,\nscore and 3D view.', 'Open builds',
                 'Card.Text.TButton', self.show_open)):
            card = ttk.Frame(page, style='Card.TFrame', padding=32)
            card.pack(side='left', padx=12, fill='y')
            ttk.Label(card, text=title, style='Card.H2.TLabel').pack(anchor='w')
            ttk.Label(card, text=text, style='Card.Muted.TLabel', justify='left').pack(anchor='w', pady=(8, 28))
            ttk.Button(card, text=button, style=style, command=command).pack(anchor='w')

    # ------------------------------------------------------------ wizard frame

    def new_build(self):
        self.w = Wizard()
        self.show_step('Photos')

    def show_step(self, step):
        self.w.step = step
        {'Photos': self.step_photos, 'Compare': self.step_compare, 'Align': self.step_align,
         'Coverage': self.step_coverage, 'Mesh': self.step_mesh, 'Score': self.step_score}[step]()

    def card(self, master, width=340, **pack):
        """A white card split in two: left = text column (returned frame), right = grey visual panel."""
        card = ttk.Frame(master, style='Card.TFrame', padding=28)
        card.pack(fill='both', expand=True, **pack)
        left = ttk.Frame(card, style='In.TFrame', width=width)
        left.pack(side='left', fill='y', padx=(0, 28))
        left.pack_propagate(False)
        right = ttk.Frame(card, style='Panel.TFrame', padding=20)
        right.pack(side='left', fill='both', expand=True)
        return left, right

    def empty(self, panel, text):
        """Centered grey note in an empty visual panel."""
        ttk.Label(panel, text=text, style='Panel.Muted.TLabel').place(relx=0.5, rely=0.5, anchor='center')

    def stat(self, panel, value, caption, kind=None):
        """One large number with a caption, centered in a visual panel."""
        box = ttk.Frame(panel, style='OnPanel.TFrame')
        box.place(relx=0.5, rely=0.5, anchor='center')
        ttk.Label(box, text=value, style='Panel.Stat.TLabel').pack()
        ttk.Label(box, text=caption, style='Panel.Muted.TLabel').pack(pady=(4, 0))
        return box

    def actions(self, left):
        """Bottom row of the text column, for the step's buttons."""
        row = ttk.Frame(left, style='In.TFrame')
        row.pack(side='bottom', anchor='w', fill='x')
        return row

    def wizard_page(self, step, title, intro=''):
        """Step bar, one card (left text column, right visual panel) and the nav row; returns (left, right, nav)."""
        self.clear()
        bar = ttk.Frame(self.body, padding=(40, 18, 40, 0))
        bar.pack(fill='x')
        home = ttk.Button(bar, text='←  Start', style='Text.TButton', command=self.show_start)
        home.pack(side='left')
        if self.task is not None:
            home.config(state='disabled')
        steps = ttk.Frame(bar)
        steps.pack(side='right')
        for i, s in enumerate(STEPS):
            state = self.w.status[s]
            name = f'{i + 1}  {STEP_NAMES.get(s, s)}'
            if s == step:
                text, style = name, 'StepNow.TLabel'
            elif state == 'done':
                text, style = '✓  ' + STEP_NAMES.get(s, s), 'StepDone.TLabel'
            elif state in ('failed', 'cancelled'):
                text, style = '✕  ' + STEP_NAMES.get(s, s), 'StepBad.TLabel'
            else:
                text, style = name, 'StepTodo.TLabel'
            ttk.Label(steps, text=text, style=style).pack(side='left', padx=(22 if i else 0, 0))

        nav = ttk.Frame(self.body, padding=(40, 0, 40, 22))
        nav.pack(fill='x', side='bottom')
        content = ttk.Frame(self.body, padding=(40, 16, 40, 16))
        content.pack(fill='both', expand=True)
        left, right = self.card(content)
        ttk.Label(left, text=title, style='Card.H2.TLabel').pack(anchor='w')
        if intro:
            self.label(left, intro, 'Card.Muted.TLabel', 320, pady=(6, 0))
        return left, right, nav

    def nav_buttons(self, nav, back=None, next_=None, next_text='Next', next_enabled=True, extra=(),
                    next_primary=True):
        """Back (text) on the left; Next on the right: the black pill when it is the step's main action."""
        busy = self.task is not None
        if back is not None:
            self.back_button = ttk.Button(nav, text='←  Back', style='Text.TButton', command=back)
            self.back_button.pack(side='left')
            if busy:
                self.back_button.config(state='disabled')
        if next_ is not None:
            self.next_button = ttk.Button(nav, text=next_text if next_primary else f'{next_text}  →',
                                          style='Primary.TButton' if next_primary else 'Text.TButton', command=next_)
            self.next_button.pack(side='right')
            if busy or not next_enabled:
                self.next_button.config(state='disabled')
        for text, command in extra:
            b = ttk.Button(nav, text=text, style='Text.TButton', command=command)
            b.pack(side='right', padx=16)
            if busy:
                b.config(state='disabled')

    # ------------------------------------------------------------ 1 Photos

    def step_photos(self):
        w = self.w
        has = w.folder is not None
        left, right, nav = self.wizard_page('Photos', 'Photos', 'One folder, one object.')
        self.nav_buttons(nav, back=self.show_start, next_=lambda: self.leave_photos(), next_primary=has,
                         next_enabled=has and w.folder.ok and w.checked)
        row = self.actions(left)
        ttk.Button(row, text='Choose folder…' if not has else 'Choose another folder…',
                   style='Card.Primary.TButton' if not has else 'Card.Text.TButton',
                   command=self.choose_photos).pack(side='left')
        self.photos_box = ttk.Frame(left, style='In.TFrame')
        self.photos_box.pack(fill='x', pady=(16, 0))
        self.thumb_row = ttk.Frame(right, style='OnPanel.TFrame')
        self.thumb_row.pack(anchor='nw')
        self.thumbs = []
        if has:
            self.render_photos()
        else:
            self.empty(right, 'No folder chosen.')

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
            return self.error('This folder cannot be used', str(exc))
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
                    self.post(self.photos_progress, w, f'Checking {done} / {total}')
            photos = bl.photo_fingerprints(w.folder, progress)
            if stale():
                return
            self.post(self.photos_progress, w, 'Finding earlier builds…')
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
        i = len(self.thumbs) - 1
        tk.Label(self.thumb_row, image=photo, bg=PANEL, bd=0).grid(row=i // 5, column=i % 5, padx=(0, 10), pady=(0, 10))

    def photos_progress(self, w, text, done=False):
        if self.w is w and w.step == 'Photos' and hasattr(self, 'check_label') and self.check_label.winfo_exists():
            self.check_label.config(text=text, style='Card.Tag.good.TLabel' if done else 'Card.Tag.neutral.TLabel')

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
            self.photos_progress(w, 'Checked', done=True)
            if self.next_button is not None and self.next_button.winfo_exists():
                self.next_button.config(state='normal', style='Primary.TButton', text='Next')

    def render_photos(self):
        w, box = self.w, self.photos_box
        pf = w.folder
        tags = ttk.Frame(box, style='In.TFrame')
        tags.pack(anchor='w')
        self.tag(tags, f'{len(pf.photos)} photos', '' if pf.ok else 'bad')
        if pf.ok:
            self.check_label = self.tag(tags, 'Checked' if w.checked else 'Checking…', 'good' if w.checked else '')
        rows = [('Folder', pf.root.name)] + ([('Build', w.name)] if pf.ok else [])
        grid = ttk.Frame(box, style='In.TFrame')
        grid.pack(anchor='w', fill='x', pady=(14, 0))
        for r, (k, v) in enumerate(rows):
            ttk.Label(grid, text=k, style='Card.Muted.TLabel').grid(row=r, column=0, sticky='w', pady=2)
            ttk.Label(grid, text=v, style='Card.TLabel').grid(row=r, column=1, sticky='w', padx=(16, 0), pady=2)
        for text in pf.warnings:
            self.label(box, text, 'Card.Warn.TLabel' if pf.ok else 'Card.Error.TLabel', 320, pady=(8, 0))
        if pf.ok and w.checked:                # came back with Back: reload the thumbnails only
            self.background(lambda: self.load_thumbs(w))

    def load_thumbs(self, w, count=15):
        """Worker thread: small previews of the first photos (PhotoImage is made on the main thread)."""
        from PIL import Image
        for rel in w.folder.photos[:count]:
            if self.w is not w:
                return
            try:
                with Image.open(w.folder.root / rel) as im:
                    im.draft('RGB', (240, 240))         # fast JPEG decode at reduced size
                    im = im.convert('RGB')
                    im.thumbnail((132, 132))
                self.post(self.add_thumb, w, im)
            except Exception as exc:
                self.post(self.log, f'preview of {rel}: {exc}')

    def leave_photos(self):
        self.w.status['Photos'] = 'done'
        self.show_step('Compare')

    # ------------------------------------------------------------ 2 Compare

    def step_compare(self):
        w = self.w
        left, right, nav = self.wizard_page('Compare', 'Reference',
                                            'Optional. An earlier build of the same object: coverage is measured '
                                            'against it, and the mesh is scored against its mesh.')
        self.nav_buttons(nav, back=self.compare_back, next_=lambda: self.leave_compare(self.compare_pick()),
                         next_enabled=False, extra=[('Skip', lambda: self.leave_compare(None))])
        self.compare_note = self.label(left, '', 'Card.Warn.TLabel', 320, pady=(14, 0))
        self.compare_selected = None
        if not w.compatible:
            self.empty(right, 'No earlier build shares photos with this set.')
            return

        def picked(b):
            self.compare_selected = b
            if self.next_button is not None and self.task is None:
                self.next_button.config(state='normal')
            self.compare_note.config(text='' if b.has_mesh else f'{b.name} has no mesh: no score.')
        selected = next((b for b, _ in w.compatible if w.compare is not None and b.key == w.compare.key), None)
        shared = {b.key: n for b, n in w.compatible}
        self.build_rows(right, [b for b, _ in w.compatible], [
            ('Build', 0, lambda b: b.name, lambda b: b.name.lower()),
            ('Location', 190, lambda b: b.group, lambda b: b.group.lower()),
            ('Photos', 80, lambda b: str(b.photo_count), lambda b: b.photo_count),
            ('Shared', 80, lambda b: str(shared[b.key]), lambda b: shared[b.key]),
            ('Mesh', 100, lambda b: ('tag', 'Mesh', 'good') if b.has_mesh else ('tag', 'No mesh', ''),
             lambda b: b.has_mesh),
            ('Date', 120, lambda b: date_text(b.date), lambda b: b.date.timestamp() if b.date else 0)],
            on_open=lambda b: (picked(b), self.press(self.next_button)), on_select=picked, selected=selected,
            bg=PANEL)
        if selected is not None:
            picked(selected)

    def compare_pick(self):
        return getattr(self, 'compare_selected', None)

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
        left, right, nav = self.wizard_page('Align', 'Align', 'Camera positions from the photos (Metashape).')
        state = w.status['Align']
        usable = state == 'done' or (state == 'failed' and self.has_cameras(w.align_run))
        self.nav_buttons(nav, back=lambda: self.show_step('Compare'), next_=self.leave_align,
                         next_enabled=usable, next_primary=usable)
        status = ttk.Frame(left, style='In.TFrame')
        status.pack(anchor='w', fill='x', pady=(16, 0))
        if self.task is not None and self.task.kind == 'align':
            self.tag(status, 'Running')
            ProgressView(right, self, self.task).pack(fill='x', anchor='n')
            return
        if state == 'done' or (state == 'failed' and self.has_cameras(w.align_run)):
            placed, total = aligned_count(w.align_run)
            tags = ttk.Frame(status, style='In.TFrame')
            tags.pack(anchor='w')
            self.tag(tags, f'{placed} / {total} aligned', 'good' if placed == total else 'caution')
            if placed < total:
                self.tag(tags, f'{total - placed} failed to align', 'bad')
            self.stat(right, f'{placed}', f'of {total} cameras aligned')
        else:
            self.empty(right, 'Not started.' if state == 'todo' else 'No cameras.')
        if state in ('failed', 'cancelled'):
            self.label(status, w.outcome.get('Align', ''), 'Card.Error.TLabel' if state == 'failed'
                       else 'Card.Warn.TLabel', 320, pady=(10, 0))
        source = 'reused from an earlier build' if w.cache else 'measured during alignment'
        self.label(status, f'Image quality: {source}.', 'Card.Small.TLabel', 320, pady=(10, 0))
        if state == 'done':
            return
        row = self.actions(left)
        start = ttk.Button(row, text='Start alignment' if state == 'todo' else 'Try again',
                           style='Card.Primary.TButton', command=self.run_align)
        start.pack(side='left')
        note = self.metashape_note()
        if note:
            start.config(state='disabled')
            self.label(left, note, 'Card.Error.TLabel' if not self.detecting else 'Card.Muted.TLabel', 320,
                       side='bottom', pady=(0, 12))

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
        left, right, nav = self.wizard_page('Coverage', 'Coverage', 'Cameras by direction around the object.')
        self.nav_buttons(nav, back=lambda: self.show_step('Align'), next_=self.leave_coverage)
        self.coverage_panel(left, right, w.align_run, w.name, w.compare)
        w.status['Coverage'] = 'done'

    def tag(self, master, text, kind='', card=True):
        """Small pill status tag; kind is good, bad, caution or '' (neutral)."""
        lab = ttk.Label(master, text=text, style=f'{"Card." if card else ""}Tag.{kind or "neutral"}.TLabel',
                        font=self.fonts['tag'])
        lab.pack(side='left', padx=(0, 6))
        return lab

    def coverage_panel(self, left, right, run_dir, name, ref, primary=None):
        """Fill a card: left = tags, the gap summary, actions; right = the coverage chart.
        primary = text makes the 3D view the screen's black pill; otherwise it is a text button."""
        try:
            cov, note = self.coverage_for(run_dir, ref)
        except Exception as exc:
            self.log(traceback.format_exc().rstrip())
            self.label(left, f'Coverage could not be computed: {exc}', 'Card.Error.TLabel', 320, pady=(14, 0))
            return
        tags = ttk.Frame(left, style='In.TFrame')
        tags.pack(anchor='w', pady=(16, 0))
        self.tag(tags, f'{len(cov.labels)} cameras')
        if cov.failed:
            self.tag(tags, f'{len(cov.failed)} failed to align', 'bad')
        gaps = int(cov.missing.sum())
        self.tag(tags, f'{gaps} gap{"s" if gaps != 1 else ""}' if gaps else 'No gaps', 'bad' if gaps else 'good')
        text = re.sub(r' \(\d+ of \d+ cameras failed to align\)', '', cov.summary) + (f' {note}' if note else '')
        self.label(left, text, 'Card.TLabel', 320, pady=(12, 0))
        if cov.reference is not None and ref is not None and not primary:   # the result screen has a picker
            self.label(left, f'Reference: {ref.name}', 'Card.Small.TLabel', 320, pady=(6, 0))

        row = self.actions(left)
        open_3d = lambda: self.export_3d(run_dir, self.mesh_of(ref, run_dir), True)
        ttk.Button(row, text=primary or '3D view', style='Card.Primary.TButton' if primary else 'Card.Text.TButton',
                   command=open_3d).pack(side='left')
        fig, canvas = self.figure(right, 'coverage')
        ttk.Button(row, text='Save image…', style='Card.Text.TButton',
                   command=lambda: self.save_image(fig, f'{name}-coverage')).pack(side='left', padx=(14, 0))
        if Path(run_dir, 'cameras.csv').is_file():
            self.prediction_box(left, Path(run_dir) / 'cameras.csv', name)
        ref_name = ref.name if cov.reference is not None else None
        draw_coverage(fig, cov, name, ref_name)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        self.log(f'coverage of {name}' + (f' vs {ref_name}' if ref_name else '') + f': {cov.summary}'
                 + (f' {note}' if note else ''))

    def prediction_box(self, master, cameras, name):
        """EXPERIMENTAL predicted quality from the aligned cameras alone (no mesh or reference needed).
        A caution-coloured box above the actions: a number with its spread across the 5 seed models and
        the training caveat, never a verdict and never styled like a measured score."""
        box = ttk.Frame(master, style='Exp.TFrame', padding=(12, 8))
        box.pack(fill='x', side='bottom', pady=(0, 16))
        ttk.Label(box, text='EXPERIMENTAL', style='Exp.Tag.TLabel').pack(anchor='w')
        line = ttk.Label(box, text='Predicted quality: working…', style='Exp.Small.TLabel', wraplength=290,
                         justify='left')
        line.pack(anchor='w', pady=(2, 0))
        gen = self.gen

        def show(p):
            if self.gen != gen or not box.winfo_exists():
                return
            line.config(text=p.text.replace('Predicted quality: ', 'Predicted '), style='Exp.Bold.TLabel', font=self.fonts['small'])
            caveat = re.sub(r'^Experimental:\s*', '', p.caption)
            caveat = caveat[:1].upper() + caveat[1:]
            if p.in_training:
                caveat += '. Training run: not a test.'
            ttk.Label(box, text=caveat, style='Exp.Small.TLabel', wraplength=290, justify='left'
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
                self.post(unavailable, 'models not on this computer')
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
        left, right, nav = self.wizard_page('Mesh', 'Mesh', f'Dense surface (Metashape). Optional. '
                                                            f'{bl.mesh_time_estimate(n).capitalize()} for {n} photos.')
        state = w.status['Mesh']
        to_score = w.compare is not None
        ready = state in ('done', 'skipped')
        self.nav_buttons(nav, back=lambda: self.show_step('Coverage'), next_=self.leave_mesh,
                         next_text='Next' if to_score else 'Finish', next_enabled=ready, next_primary=ready)
        status = ttk.Frame(left, style='In.TFrame')
        status.pack(anchor='w', fill='x', pady=(16, 0))
        if self.task is not None and self.task.kind == 'mesh':
            self.tag(status, 'Running')
            ProgressView(right, self, self.task).pack(fill='x', anchor='n')
            return
        if state == 'done':
            self.tag(status, 'Mesh built', 'good')
            if to_score and not w.compare.has_mesh:
                self.label(left, f'{w.compare.name} has no mesh: no score.', 'Card.Muted.TLabel', 320, pady=(10, 0))
            faces = (_read_json(w.mesh_run / 'metrics-mesh_generation.json').get('metrics') or {}).get('mesh_face_count')
            self.stat(right, f'{faces:,}' if faces else '✓', 'mesh faces' if faces else 'mesh built')
            ttk.Button(self.actions(left), text='3D view', style='Card.Text.TButton',
                       command=lambda: self.export_3d(w.mesh_run, self.mesh_of(w.compare, w.mesh_run), True)
                       ).pack(side='left')
            return
        self.empty(right, 'Skipped.' if state == 'skipped' else 'Not built.')
        if state in ('failed', 'cancelled'):
            self.label(status, w.outcome.get('Mesh', ''), 'Card.Error.TLabel' if state == 'failed'
                       else 'Card.Warn.TLabel', 320)
        row = self.actions(left)
        build = ttk.Button(row, text='Build mesh' if state in ('todo', 'skipped') else 'Try again',
                           style='Card.Primary.TButton', command=self.run_mesh)
        build.pack(side='left')
        if state != 'skipped':
            ttk.Button(row, text='Skip', style='Card.Text.TButton', command=self.skip_mesh).pack(side='left', padx=14)
        reason = self.metashape_note()
        if not reason and not (w.align_run and ps.is_alignment_checkpoint(w.align_run)):
            reason = 'The mesh needs a complete alignment.'
        if reason:
            build.config(state='disabled')
            self.label(left, reason, 'Card.Error.TLabel' if not self.detecting else 'Card.Muted.TLabel', 320,
                       side='bottom', pady=(0, 12))

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
            self.export_3d(mesh_run, self.mesh_of(w.compare, mesh_run))   # ready for "3D view"
        if self.w is w and w.step == 'Mesh':
            self.step_mesh()
        if w.status['Mesh'] == 'failed':
            self.error('The mesh did not finish', w.outcome['Mesh'])

    # ------------------------------------------------------------ 3D view (viewer/index.html)

    @staticmethod
    def mesh_of(ref, run_dir):
        """The reference build's mesh folder, when both it and run_dir have a mesh; else None (view alone)."""
        if ref is None or not ref.has_mesh or not (Path(run_dir) / 'raw-mesh.obj').is_file():
            return None
        return ref.mesh

    def export_3d(self, run_dir, ref_dir=None, open_after=False):
        """Write the web viewer's data for run_dir in a worker (against ref_dir's mesh when given),
        then open it in the browser if asked. Without a reference the build is shown on its own."""
        run_dir = Path(run_dir)
        what = f'{run_dir.name}' + (f' vs {Path(ref_dir).name}' if ref_dir else '')
        self.log(f'3D view: preparing {what} …')

        def job():
            try:
                out = vx.export_build(run_dir, ref_dir, log=lambda t: self.post(self.log, t))
            except Exception as exc:
                self.post(self.log, f'3D view: export of {what} failed: {exc}')
                if open_after:
                    self.post(self.error, 'The 3D view could not be prepared', exc)
                return
            if open_after:
                self.post(self.log, f'3D view: {vx.open_viewer(out.name)}')
        self.background(job)

    # ------------------------------------------------------------ 6 Score

    def step_score(self):
        w = self.w
        ref = w.compare.name if w.compare is not None else 'the reference'
        left, right, nav = self.wizard_page('Score', 'Score', f'Surface distance to {ref} (evaluator v3).')
        self.nav_buttons(nav, back=lambda: self.show_step('Mesh'), next_=self.finish, next_text='Finish')
        why = ('No reference was chosen.' if w.compare is None else 'No mesh was built.' if w.mesh_run is None
               else f'{w.compare.name} has no mesh.' if not w.compare.has_mesh else '')
        if why:
            self.label(left, why, 'Card.Muted.TLabel', 320, pady=(16, 0))
            self.empty(right, 'No score.')
            if w.compare is not None:
                w.status['Score'] = 'skipped'
            return
        self.score_panel(left, right, w.mesh_run, w.name, w.compare, cached=w.score,
                         on_row=lambda row: self.wizard_scored(w, row))

    def wizard_scored(self, w, row):
        w.score, w.status['Score'] = row, 'done'

    def metric_rows(self, master, rows, pady=(16, 0)):
        """Labeled values (name left, value or tag right), as in the viewer's inspector."""
        grid = ttk.Frame(master, style='In.TFrame')
        grid.pack(fill='x', pady=pady)
        grid.columnconfigure(1, weight=1)
        for r, (name, value, tag) in enumerate(rows):
            ttk.Label(grid, text=name, style='Card.Muted.TLabel').grid(row=r, column=0, sticky='w', pady=4)
            cell = ttk.Frame(grid, style='In.TFrame')
            cell.grid(row=r, column=1, sticky='e', pady=4)
            if tag:
                self.tag(cell, *tag).pack_configure(side='right', padx=(6, 0))
            if value:
                ttk.Label(cell, text=value, style='Card.Bold.TLabel').pack(side='right')
        return grid

    def score_panel(self, left, right, run_dir, name, ref, cached=None, on_row=None):
        """Score rows in the text column and the bar chart in the panel; the score runs in a worker."""
        waiting = self.label(left, 'Scoring… (a few seconds)', 'Card.Muted.TLabel', 320, pady=(16, 0))
        gen = self.gen

        def show(row):
            if self.gen != gen or not left.winfo_exists():
                return
            waiting.destroy()
            if on_row:
                on_row(row)
            self.metric_rows(left, score_rows(row))
            self.label(left, 'Fractions of the reference size (bounding-box diagonal). Lower is better.',
                       'Card.Small.TLabel', 320, pady=(10, 0))
            if row.get('qc_v3') != 'pass' or row.get('scale_check') == 'disagree':
                self.label(left, 'The meshes registered less well than usual; treat the numbers with care.',
                           'Card.Warn.TLabel', 320, pady=(10, 0))
            ttk.Button(self.actions(left), text='Save image…', style='Card.Text.TButton',
                       command=lambda: self.save_score_image(row, name, ref.name)).pack(side='left')
            fig, canvas = self.figure(right, 'score', size=(8, 3))
            draw_score(fig, row, name, ref.name)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            self.log(f'score of {name} vs {ref.name}: {ps.score_text(row)}')

        def failed(exc):
            if self.gen == gen and left.winfo_exists():
                waiting.config(text=f'The score could not be computed: {exc}', style='Card.Error.TLabel')

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
        fig = Figure(figsize=(8, 3), facecolor='white')
        draw_score(fig, row, name, ref_name, bg='white')
        self.save_image(fig, f'{name}-score-vs-{ref_name}')

    def finish(self):
        self.log(f'build "{self.w.name}" finished' if self.w else 'finished')
        self.builds = None
        self.show_start()

    # ------------------------------------------------------------ Open an existing build

    def build_rows(self, master, builds, columns, on_open, on_select=None, selected=None, bg=SURFACE):
        """A clean table of builds: a header and one white row per build, sortable by header.
        columns: [(title, width or 0 to stretch, cell(build), sort_key(build))]; a cell is text,
        ('tag', text, kind) or ('button', text, command). Click selects (or opens without on_select);
        double-click opens."""
        outer = tk.Frame(master, bg=bg)
        outer.pack(fill='both', expand=True)
        header = tk.Frame(outer, bg=bg)
        header.pack(fill='x', padx=(4, 18))
        canvas = tk.Canvas(outer, bg=bg, highlightthickness=0, bd=0)
        scroll = ttk.Scrollbar(outer, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)
        inner = tk.Frame(canvas, bg=bg)
        win = canvas.create_window(0, 0, window=inner, anchor='nw')
        inner.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>', lambda e: canvas.itemconfigure(win, width=e.width))

        def wheel(e):
            canvas.yview_scroll(-1 * (e.delta if abs(e.delta) < 20 else e.delta // 120), 'units')
        canvas.bind('<Enter>', lambda e: canvas.bind_all('<MouseWheel>', wheel))
        canvas.bind('<Leave>', lambda e: canvas.unbind_all('<MouseWheel>'))

        def layout(frame):
            for c, (_, width, _, _) in enumerate(columns):      # fixed widths include the 14px gap
                frame.columnconfigure(c + 1, minsize=width + 14 if width else 0, weight=0 if width else 1)
        layout(header)
        state = {'selected': selected, 'order': list(builds), 'key': None, 'marks': {}}

        def select(b):
            state['selected'] = b
            for key, mark in state['marks'].items():
                mark.config(bg=INK if b is not None and key == b.key else SURFACE)
            if on_select:
                on_select(b)

        def render():
            for child in inner.winfo_children():
                child.destroy()
            state['marks'] = {}
            for b in state['order']:
                row = tk.Frame(inner, bg=SURFACE)
                row.pack(fill='x', pady=(0, 1))
                layout(row)
                mark = tk.Frame(row, bg=SURFACE, width=3)
                mark.grid(row=0, column=0, sticky='ns')
                state['marks'][b.key] = mark
                clickable = [row]
                for c, (_, _, cell, _) in enumerate(columns):
                    value = cell(b)
                    if isinstance(value, tuple) and value[0] == 'tag':
                        w = ttk.Label(row, text=value[1], style=f'Card.Tag.{value[2] or "neutral"}.TLabel',
                                      font=self.fonts['tag'])
                    elif isinstance(value, tuple) and value[0] == 'button':
                        w = ttk.Button(row, text=value[1], style='Card.Text.TButton', command=value[2])
                    else:
                        w = tk.Label(row, text=value, bg=SURFACE, fg=INK if c == 0 else TEXT, anchor='w',
                                     font=self.fonts['bold' if c == 0 else 'body'])
                        clickable.append(w)
                    w.grid(row=0, column=c + 1, sticky='w', padx=(14, 0), pady=10)
                for w in clickable:
                    w.bind('<Button-1>', lambda e, b=b: select(b) if on_select else on_open(b))
                    w.bind('<Double-1>', lambda e, b=b: on_open(b))
            select(state['selected'])

        def sort(i):
            key = columns[i][3]
            reverse = state['key'] == i
            state['key'] = None if reverse else i
            state['order'].sort(key=key, reverse=reverse)
            render()
        for c, (title, _, _, key) in enumerate(columns):
            h = tk.Label(header, text=title, bg=bg, fg=MUTED, font=self.fonts['small'], anchor='w', cursor='hand2')
            h.grid(row=0, column=c + 1, sticky='w', padx=(14, 0), pady=(0, 8))
            h.bind('<Button-1>', lambda e, c=c: sort(c))
        render()
        return outer

    def show_open(self):
        self.w = None
        self.clear()
        bar = ttk.Frame(self.body, padding=(40, 18, 40, 0))
        bar.pack(fill='x')
        self.back_button = ttk.Button(bar, text='←  Start', style='Text.TButton', command=self.show_start)
        self.back_button.pack(side='left')
        ttk.Button(bar, text='Refresh', style='Text.TButton',
                   command=lambda: (setattr(self, 'builds', None), self.show_open())).pack(side='right')
        content = ttk.Frame(self.body, padding=(40, 8, 40, 16))
        content.pack(fill='both', expand=True)
        ttk.Label(content, text='Builds', style='H1.TLabel').pack(anchor='w')
        self.open_count = self.label(content, '', 'Muted.TLabel', pady=(4, 16))
        self.open_card = ttk.Frame(content, style='Card.TFrame', padding=(16, 20, 16, 12))
        self.open_card.pack(fill='both', expand=True)
        nav = ttk.Frame(self.body, padding=(40, 0, 40, 22))
        nav.pack(fill='x', side='bottom', before=content)
        self.open_selected = None
        self.next_button = ttk.Button(nav, text='Open', style='Primary.TButton', state='disabled',
                                      command=lambda: self.open_pick())
        self.next_button.pack(side='right')
        if self.builds is None:
            waiting = ttk.Label(self.open_card, text='Looking for builds…', style='Card.Muted.TLabel')
            waiting.pack(anchor='w', padx=14)
            roots = [REPO / 'results', self.runs_folder]
            gen = self.gen

            def found(builds):
                self.builds = builds
                if self.gen == gen:
                    waiting.destroy()
                    self.fill_open()

            self.background(lambda: self.post(found, bl.find_builds(roots)))
        else:
            self.fill_open()

    def fill_open(self):
        builds = self.builds or []
        self.open_count.config(text=f'{len(builds)} builds in results/ and {self.runs_folder.name}/. '
                                    'Double-click to open.')
        if not builds:
            ttk.Label(self.open_card, text='No builds found.', style='Card.Muted.TLabel').pack(anchor='w', padx=14)
            return

        def picked(b):
            self.open_selected = b
            if self.next_button is not None and self.next_button.winfo_exists():
                self.next_button.config(state='normal')
        self.build_rows(self.open_card, builds, [
            ('Build', 0, lambda b: b.name, lambda b: b.name.lower()),
            ('Location', 220, lambda b: b.group, lambda b: b.group.lower()),
            ('Photos', 80, lambda b: str(b.photo_count), lambda b: b.photo_count),
            ('Date', 120, lambda b: date_text(b.date), lambda b: b.date.timestamp() if b.date else 0),
            ('Status', 130, lambda b: ('tag', 'Mesh', 'good') if b.has_mesh else ('tag', 'Cameras only', ''),
             lambda b: b.has_mesh),
            ('', 90, lambda b: ('button', '3D view', lambda: self.export_3d(b.mesh or b.path, None, True)),
             lambda b: 0)],
            on_open=self.show_result, on_select=picked, selected=builds[0])

    def open_pick(self):
        if self.open_selected is not None:
            self.show_result(self.open_selected)

    def show_result(self, build, ref='default'):
        self.w = None
        self.clear()
        builds = self.builds or []
        if ref == 'default':
            ref = bl.default_reference(build, builds)
        bar = ttk.Frame(self.body, padding=(40, 18, 40, 0))
        bar.pack(fill='x')
        self.back_button = ttk.Button(bar, text='←  Builds', style='Text.TButton', command=self.show_open)
        self.back_button.pack(side='left')
        content = ttk.Frame(self.body, padding=(40, 8, 40, 16))
        content.pack(fill='both', expand=True)
        head = ttk.Frame(content)
        head.pack(fill='x')
        ttk.Label(head, text=build.name, style='H1.TLabel').pack(side='left')
        tags = ttk.Frame(head)
        tags.pack(side='left', padx=(16, 0), pady=(6, 0))
        self.tag(tags, 'Mesh' if build.has_mesh else 'Cameras only', 'good' if build.has_mesh else '', card=False)
        self.label(content, f'{build.group}  ·  {build.photo_count} photos  ·  {date_text(build.date)}',
                   'Muted.TLabel', pady=(4, 0))

        options = [(None, 'None (this build alone)')]
        options += [(b, f'{b.name}  ({b.group} · {n} shared photos{" · mesh" if b.has_mesh else ""})')
                    for b, n in bl.compatible_builds(build.photos, builds, exclude=build)]
        row = ttk.Frame(content)
        row.pack(fill='x', pady=(14, 16))
        ttk.Label(row, text='Reference', style='Muted.TLabel').pack(side='left')
        names = [t for _, t in options]
        var = tk.StringVar(value=next((t for b, t in options if b is ref), names[0]))
        box = ttk.Combobox(row, textvariable=var, values=names, state='readonly', width=58, font=self.fonts['body'])
        box.pack(side='left', padx=12)
        box.bind('<<ComboboxSelected>>',
                 lambda e: self.show_result(build, next(b for b, t in options if t == var.get())))
        if len(options) == 1:
            box.config(state='disabled')
            ttk.Label(row, text='No other build shares photos with this one.', style='Small.TLabel').pack(side='left')

        score_card = ttk.Frame(content, style='Card.TFrame', padding=(28, 18))
        score_card.pack(fill='x', side='bottom', pady=(16, 0))
        left, right = self.card(content)
        ttk.Label(left, text='Coverage', style='Card.H2.TLabel').pack(anchor='w')
        self.coverage_panel(left, right, build.path, build.name, ref, primary='Open 3D view')
        if ref is None:
            ttk.Label(score_card, text='No score: choose a reference build.', style='Card.Muted.TLabel').pack(anchor='w')
        elif not build.has_mesh or not ref.has_mesh:
            missing = build.name if not build.has_mesh else ref.name
            ttk.Label(score_card, text=f'No score: {missing} has no mesh.', style='Card.Muted.TLabel').pack(anchor='w')
        else:
            self.result_score(score_card, build, ref)

    def result_score(self, card, build, ref):
        """One row on the score card: the v3 numbers as labeled values, tags, and Save image…"""
        waiting = ttk.Label(card, text='Scoring… (a few seconds)', style='Card.Muted.TLabel')
        waiting.pack(anchor='w')
        gen = self.gen

        def show(row):
            if self.gen != gen or not card.winfo_exists():
                return
            waiting.destroy()
            line = ttk.Frame(card, style='In.TFrame')
            line.pack(fill='x')
            ttk.Label(line, text='Score', style='Card.H2.TLabel').pack(side='left', padx=(0, 28))
            for name, value, tag in score_rows(row):
                cell = ttk.Frame(line, style='In.TFrame')
                cell.pack(side='left', padx=(0, 28))
                ttk.Label(cell, text=name, style='Card.Small.TLabel').pack(anchor='w')
                val = ttk.Frame(cell, style='In.TFrame')
                val.pack(anchor='w')
                if value:
                    ttk.Label(val, text=value, style='Card.Bold.TLabel').pack(side='left', padx=(0, 6))
                if tag:
                    self.tag(val, *tag)
            ttk.Button(line, text='Save image…', style='Card.Text.TButton',
                       command=lambda: self.save_score_image(row, build.name, ref.name)).pack(side='right')
            note = f'vs {ref.name} · fractions of the reference size (bounding-box diagonal) · lower is better'
            if row.get('qc_v3') != 'pass' or row.get('scale_check') == 'disagree':
                note += ' · registered less well than usual: treat with care'
            ttk.Label(card, text=note, style='Card.Small.TLabel').pack(anchor='w', pady=(8, 0))
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
