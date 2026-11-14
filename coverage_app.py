"""Camera coverage viewer for one aligned run: python coverage_app.py

Open a run folder containing cameras.csv to see a top-down camera layout and an
azimuth x elevation coverage map (red = fewer than 2 cameras). Measured coverage only.
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
    from bd3d_quality.coverage import analyze, default_run
except ImportError:  # not installed: use the source tree next to this file
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
    from bd3d_quality.coverage import analyze, default_run


class CoverageApp:
    def __init__(self, root):
        self.root = root
        root.title('Camera coverage')
        root.report_callback_exception = lambda *exc: messagebox.showerror('Error', str(exc[1]))
        bar = tk.Frame(root)
        bar.pack(fill='x', padx=8, pady=6)
        tk.Button(bar, text='Open run…', command=self.choose).pack(side='left')
        self.path_label = tk.Label(bar, anchor='w')
        self.path_label.pack(side='left', padx=8)
        self.fig = Figure(figsize=(12, 5.5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self.summary = tk.Label(root, anchor='w', justify='left', wraplength=1100, font=('TkDefaultFont', 13))
        self.summary.pack(fill='x', padx=8, pady=6)
        if (default_run() / 'cameras.csv').exists():
            self.load(default_run())

    def choose(self):
        start = default_run().parent if default_run().parent.exists() else Path.cwd()
        folder = filedialog.askdirectory(title='Choose a run folder with cameras.csv', initialdir=start)
        if folder:
            self.load(Path(folder))

    def load(self, folder):
        try:
            csv_path = folder / 'cameras.csv'
            if not csv_path.exists():
                raise FileNotFoundError(f'No cameras.csv in {folder}')
            cov = analyze(csv_path)
            self.draw(cov)
        except Exception as exc:  # show any failure, keep the window alive
            messagebox.showerror('Could not load run', f'{type(exc).__name__}: {exc}')
            return
        self.path_label.config(text=str(folder))
        self.summary.config(text=cov.summary)

    def draw(self, cov):
        self.fig.clear()
        ax, cm = self.fig.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1.6]})

        # 1. top-down layout in the up frame
        Q, W = cov.positions, cov.views
        r = np.linalg.norm(Q[:, :2], axis=1).max()
        band = np.clip(np.digitize(cov.elevation, cov.el_edges[1:-1]), 0, len(cov.el_edges) - 2)
        sc = ax.scatter(Q[:, 0], Q[:, 1], c=cov.elevation, cmap='viridis', s=22, zorder=3)
        ax.quiver(Q[:, 0], Q[:, 1], W[:, 0], W[:, 1], color='0.5', width=0.003, scale=12, zorder=2)
        ax.plot(0, 0, '+', color='k', ms=12)
        ax.annotate('0°', (r * 1.08, 0), ha='left', va='center', color='0.4')
        ax.set_title(f'{len(cov.labels)} aligned cameras, top-down')
        if cov.failed:
            shown = ', '.join(cov.failed[:3]) + (' …' if len(cov.failed) > 3 else '')
            ax.plot([], [], 'x', color='tab:red', label=f'{len(cov.failed)} failed (no pose): {shown}')
            ax.legend(loc='upper left', fontsize=8, frameon=False, bbox_to_anchor=(0, -0.02))
        ax.set_aspect('equal')
        ax.set_xlim(-r * 1.25, r * 1.25)
        ax.set_ylim(-r * 1.25, r * 1.25)
        ax.set_xticks([])
        ax.set_yticks([])
        self.fig.colorbar(sc, ax=ax, shrink=0.7, label='elevation (°)')

        # 2. azimuth x elevation coverage map
        az, el, n = cov.az_edges, cov.el_edges, cov.counts
        cm.pcolormesh(az, el, n, cmap='Blues', vmin=0, vmax=max(n.max(), 4), edgecolors='white', linewidth=1)
        for i in range(n.shape[0]):
            for j in range(n.shape[1]):
                if n[i, j] < cov.min_cameras:
                    cm.add_patch(Rectangle((az[j], el[i]), az[j + 1] - az[j], el[i + 1] - el[i],
                                           facecolor='#e05a5a', edgecolor='white'))
                cm.text((az[j] + az[j + 1]) / 2, (el[i] + el[i + 1]) / 2, n[i, j], ha='center', va='center',
                        fontsize=9, color='white' if n[i, j] < cov.min_cameras or n[i, j] > 4 else 'k')
        cm.scatter(cov.azimuth, cov.elevation, s=4, color='k', alpha=0.35)
        names = {1: ['all'], 2: ['low', 'high'], 3: ['low', 'mid', 'high']}.get(len(n), [])
        for name, lo, hi in zip(names, el[:-1], el[1:]):
            cm.text(361, (lo + hi) / 2, name, va='center', fontsize=9, color='0.3')
        cm.set_xlim(0, 360)
        cm.set_ylim(el[0], el[-1])
        cm.set_xticks(az[::2])
        cm.set_xlabel('azimuth (°, 0 = first aligned camera)')
        cm.set_ylabel('elevation (°)')
        cm.set_title(f'coverage (red: fewer than {cov.min_cameras} cameras)')
        self.fig.tight_layout()
        self.canvas.draw()


def main():
    root = tk.Tk()
    CoverageApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
