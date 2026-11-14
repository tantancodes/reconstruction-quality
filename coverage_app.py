"""Camera coverage viewer for one aligned run: python coverage_app.py

Open a run folder containing cameras.csv to see a top-down camera layout and an
azimuth x elevation coverage map (red = fewer than 2 cameras). Measured coverage only.

Optionally compare to a reference run: the run is moved into the reference frame with the
v3 camera similarity fit, red = the reference covers the cell (2+ cameras) but this run does
not, grey = the reference never covered it either. Default: side40_s1 against ref145b.
"""
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

try:
    from bd3d_quality.coverage import BAND_NAMES, analyze, compare, default_reference, default_run
except ImportError:  # not installed: use the source tree next to this file
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
    from bd3d_quality.coverage import BAND_NAMES, analyze, compare, default_reference, default_run

RED, GREY = '#e05a5a', '#c8c8c8'


class CoverageApp:
    def __init__(self, root):
        self.root = root
        self.run, self.reference = None, None
        root.title('Camera coverage')
        root.report_callback_exception = lambda *exc: messagebox.showerror('Error', str(exc[1]))
        bar = tk.Frame(root)
        bar.pack(fill='x', padx=8, pady=6)
        tk.Button(bar, text='Open run…', command=self.choose_run).pack(side='left')
        tk.Button(bar, text='Compare to reference…', command=self.choose_reference).pack(side='left', padx=4)
        tk.Button(bar, text='No reference', command=lambda: self.load(self.run, None)).pack(side='left')
        self.path_label = tk.Label(bar, anchor='w')
        self.path_label.pack(side='left', padx=8)
        self.fig = Figure(figsize=(12, 5.5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self.summary = tk.Label(root, anchor='w', justify='left', wraplength=1100, font=('TkDefaultFont', 13))
        self.summary.pack(fill='x', padx=8, pady=6)
        if (default_run() / 'cameras.csv').exists():
            ref = default_reference() if (default_reference() / 'cameras.csv').exists() else None
            self.load(default_run(), ref)

    def ask_folder(self, title):
        start = default_run().parent if default_run().parent.exists() else Path.cwd()
        folder = filedialog.askdirectory(title=title, initialdir=start)
        return Path(folder) if folder else None

    def choose_run(self):
        folder = self.ask_folder('Choose a run folder with cameras.csv')
        if folder:
            self.load(folder, self.reference)

    def choose_reference(self):
        if self.run is None:
            messagebox.showerror('No run', 'Open a run first.')
            return
        folder = self.ask_folder('Choose a reference run folder with cameras.csv')
        if folder:
            self.load(self.run, folder)

    def load(self, folder, reference):
        if folder is None:
            return
        try:
            for f in [folder] + ([reference] if reference else []):
                if not (f / 'cameras.csv').exists():
                    raise FileNotFoundError(f'No cameras.csv in {f}')
            cov = (compare(folder / 'cameras.csv', reference / 'cameras.csv') if reference
                   else analyze(folder / 'cameras.csv'))
            self.draw(cov, folder, reference)
        except Exception as exc:  # show any failure, keep the window and the previous view alive
            messagebox.showerror('Could not load run', f'{type(exc).__name__}: {exc}')
            return
        self.run, self.reference = folder, reference
        self.path_label.config(text=str(folder) + (f'   vs reference {reference.name}' if reference else ''))
        self.summary.config(text=cov.summary)

    def draw(self, cov, folder, reference):
        self.fig.clear()
        ax, cm = self.fig.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1.6]})
        ref = cov.reference

        # 1. top-down layout (in the reference frame when comparing)
        Q, W = cov.positions, cov.views
        allQ = Q if ref is None else np.vstack([Q, ref.positions])
        r = np.linalg.norm(allQ[:, :2], axis=1).max()
        if ref is not None:
            ax.scatter(ref.positions[:, 0], ref.positions[:, 1], s=40, facecolors='none', edgecolors='0.6',
                       linewidths=0.8, zorder=1, label=f'reference {reference.name} ({len(ref.labels)})')
        sc = ax.scatter(Q[:, 0], Q[:, 1], c=cov.elevation, cmap='viridis', s=22, zorder=3,
                        label=f'{folder.name} ({len(cov.labels)})' if ref is not None else None)
        ax.quiver(Q[:, 0], Q[:, 1], W[:, 0], W[:, 1], color='0.5', width=0.003, scale=12, zorder=2)
        ax.plot(0, 0, '+', color='k', ms=12)
        ax.annotate('0°', (r * 1.08, 0), ha='left', va='center', color='0.4')
        if ref is None:
            ax.set_title(f'{len(cov.labels)} aligned cameras, top-down')
        else:
            shared, resid = cov.fit
            ax.set_title(f'top-down, reference frame\nfit on {shared} shared cameras, residual {resid:.2%}',
                         fontsize=10)
        if cov.failed:
            shown = ', '.join(cov.failed[:3]) + (' …' if len(cov.failed) > 3 else '')
            ax.plot([], [], 'x', color='tab:red', label=f'{len(cov.failed)} failed (no pose): {shown}')
        if ref is not None or cov.failed:
            ax.legend(loc='upper left', fontsize=8, frameon=False, bbox_to_anchor=(0, -0.02))
        ax.set_aspect('equal')
        ax.set_xlim(-r * 1.25, r * 1.25)
        ax.set_ylim(-r * 1.25, r * 1.25)
        ax.set_xticks([])
        ax.set_yticks([])
        self.fig.colorbar(sc, ax=ax, shrink=0.7, label='elevation (°)')

        # 2. azimuth x elevation coverage map
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
                        fontsize=9 if ref is None else 8,
                        color='white' if cov.missing[i, j] or (colour is None and n[i, j] > 0.6 * vmax) else 'k')
        cm.scatter(cov.azimuth, cov.elevation, s=4, color='k', alpha=0.35)
        for name, lo, hi in zip(BAND_NAMES.get(len(n), []), el[:-1], el[1:]):
            cm.text(361, (lo + hi) / 2, name, va='center', fontsize=9, color='0.3')
        cm.set_xlim(0, 360)
        cm.set_ylim(el[0], el[-1])
        cm.set_xticks(az[::2])
        if ref is None:
            cm.set_xlabel('azimuth (°, 0 = first aligned camera)')
            cm.set_title(f'coverage (red: fewer than {cov.min_cameras} cameras)')
        else:
            cm.set_xlabel(f'azimuth (°, reference frame, 0 = first aligned camera of {reference.name})')
            cm.set_title(f'cameras run/reference (red: reference has {cov.min_cameras}+, run fewer; '
                         'grey: reference fewer too)', fontsize=10)
        cm.set_ylabel('elevation (°)')
        self.fig.tight_layout()
        self.canvas.draw()


def main():
    root = tk.Tk()
    CoverageApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
