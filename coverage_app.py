"""Capture check and pipeline app: python coverage_app.py

New capture: photo folder + output folder, then
  1. Align (alignment only, existing pipeline under the resource watchdog; needs Metashape)
  2. Coverage view (top-down cameras and azimuth x elevation map; vs reference if chosen)
  3. Build mesh (optional; continues the alignment run to raw-mesh.obj; needs Metashape)
  4. Score vs reference (optional; v3 label, missing and extra RMSE)
Open existing run: coverage, plus the score when both runs have raw-mesh.obj.

Long steps run in a worker thread. The worker never touches Tk: it posts callables to a queue
that the main thread drains with root.after(), so every UI update and error box happens on the
main thread (required on macOS).
"""
import json
import os
import queue
import sys
import threading
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

try:
    import bd3d_quality  # noqa: F401
except ImportError:  # not installed: use the source tree next to this file
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from bd3d_quality import metashape_locator as ml
from bd3d_quality import pipeline_steps as ps
from bd3d_quality.coverage import BAND_NAMES, analyze, compare, default_reference, default_run

RED, GREY = '#e05a5a', '#c8c8c8'
STEPS = [('align', '1. Align (needs Metashape)'), ('coverage', '2. Coverage view'),
         ('mesh', '3. Build mesh (optional, needs Metashape)'), ('score', '4. Score vs reference (optional)')]
STATUS_COLOURS = {'not started': '#777777', 'running': '#1f6fd1', 'done': '#2a8a2a', 'failed': '#c0392b',
                  'cancelled': '#b8860b', 'not available': '#aaaaaa'}


class CoverageApp:
    def __init__(self, root):
        self.root = root
        self.q = queue.Queue()
        self.worker = None        # running step key, or None
        self.proc = None          # WatchedProcess of the running step (for Cancel)
        self.photos = self.paths = None
        self.run = self.mesh_run = self.reference = None
        root.title('Capture coverage and pipeline')
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.report_callback_exception = self.callback_error
        self.metashape_ok = False
        self.build_ui()
        self.root.after(100, self.pump)
        self.detect_python()
        if (default_run() / 'cameras.csv').exists():
            if (default_reference() / 'cameras.csv').exists():
                self.reference = default_reference()
            self.open_run(default_run())

    # ------------------------------------------------------------ layout

    def build_ui(self):
        top = tk.Frame(self.root)
        top.pack(fill='x', padx=8, pady=(6, 0))
        tk.Button(top, text='New capture…', command=self.new_capture).pack(side='left')
        tk.Button(top, text='Open existing run…', command=self.choose_run).pack(side='left', padx=4)
        tk.Button(top, text='Reference…', command=self.choose_reference).pack(side='left', padx=(12, 0))
        tk.Button(top, text='No reference', command=self.clear_reference).pack(side='left', padx=4)
        self.where = tk.Label(top, anchor='w')
        self.where.pack(side='left', padx=8, fill='x', expand=True)

        info = tk.Frame(self.root)
        info.pack(fill='x', padx=8, pady=4)
        tk.Label(info, text='Steps 1 and 3 run Metashape and need a local Metashape license.',
                 fg='#c0392b', font=('TkDefaultFont', 12, 'bold')).grid(row=0, column=0, columnspan=2, sticky='w')
        tk.Label(info, text='Metashape Python:').grid(row=1, column=0, sticky='w')
        row = tk.Frame(info)
        row.grid(row=1, column=1, sticky='w')
        self.py_var = tk.StringVar(value='')
        tk.Entry(row, textvariable=self.py_var, width=70, state='readonly').pack(side='left')
        self.change_button = tk.Button(row, text='Change…', command=self.change_python)
        self.change_button.pack(side='left', padx=4)
        self.ms_label = tk.Label(row, text='searching for a Python that can import Metashape…', fg='#777')
        self.ms_label.pack(side='left', padx=4)
        tk.Label(info, text=f'Mesh settings (as pilot ref50 / ref145b): {ps.settings_text()}',
                 fg='#444').grid(row=2, column=0, columnspan=2, sticky='w')
        tk.Label(info, text=f'Watchdog limits: align {ps.ALIGN_LIMITS.text()}; mesh {ps.MESH_LIMITS.text()}',
                 fg='#444').grid(row=3, column=0, columnspan=2, sticky='w')

        mid = tk.Frame(self.root)
        mid.pack(fill='both', expand=True, padx=8)
        steps = tk.Frame(mid)
        steps.pack(side='left', fill='y', padx=(0, 8), pady=4)
        self.status, self.buttons = {}, {}
        actions = {'align': self.run_align, 'coverage': self.show_coverage, 'mesh': self.run_mesh,
                   'score': self.run_score}
        for i, (key, text) in enumerate(STEPS):
            tk.Label(steps, text=text, anchor='w').grid(row=2 * i, column=0, columnspan=2, sticky='w', pady=(8, 0))
            self.status[key] = tk.Label(steps, text='not started', width=13, anchor='w')
            self.status[key].grid(row=2 * i + 1, column=0, sticky='w')
            self.buttons[key] = tk.Button(steps, text='Run' if key != 'coverage' else 'Show',
                                          command=actions[key], state='disabled')
            self.buttons[key].grid(row=2 * i + 1, column=1, sticky='e')
        self.cancel_button = tk.Button(steps, text='Cancel running step', command=self.cancel, state='disabled')
        self.cancel_button.grid(row=2 * len(STEPS), column=0, columnspan=2, sticky='we', pady=(16, 0))
        for key in self.status:
            self.set_status(key, 'not started')

        self.fig = Figure(figsize=(11, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=mid)
        self.canvas.get_tk_widget().pack(side='left', fill='both', expand=True)

        self.summary = tk.Label(self.root, anchor='w', justify='left', wraplength=1250, font=('TkDefaultFont', 13))
        self.summary.pack(fill='x', padx=8)
        self.score_label = tk.Label(self.root, anchor='w', justify='left', wraplength=1250, fg='#1f4e8a')
        self.score_label.pack(fill='x', padx=8)
        self.log_box = ScrolledText(self.root, height=10, state='disabled', font=('Menlo', 10))
        self.log_box.pack(fill='both', padx=8, pady=(4, 8))

    # ------------------------------------------------------------ main-thread plumbing

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
                    self.error('Unexpected error', exc)
        except queue.Empty:
            pass
        self.root.after(100, self.pump)

    def log(self, text):
        self.log_box.configure(state='normal')
        self.log_box.insert('end', text + '\n')
        if int(self.log_box.index('end-1c').split('.')[0]) > 5000:
            self.log_box.delete('1.0', '1000.0')
        self.log_box.see('end')
        self.log_box.configure(state='disabled')

    def error(self, title, exc_or_text):
        text = exc_or_text if isinstance(exc_or_text, str) else f'{type(exc_or_text).__name__}: {exc_or_text}'
        self.log(f'ERROR: {title}: {text}')
        messagebox.showerror(title, text)

    def callback_error(self, exc_type, exc, tb):
        self.log(''.join(traceback.format_exception(exc_type, exc, tb)).rstrip())
        self.error('Unexpected error', exc)

    def set_status(self, key, state, detail=''):
        self.status[key].config(text=state, fg=STATUS_COLOURS.get(state, 'black'))
        if detail:
            self.log(f'{dict(STEPS)[key]}: {state} ({detail})')

    def refresh_buttons(self):
        busy = self.worker is not None
        can = {
            'align': self.metashape_ok and self.paths is not None
                     and self.status['align'].cget('text') in ('not started', 'failed', 'cancelled'),
            'coverage': self.run is not None and (self.run / 'cameras.csv').exists(),
            'mesh': self.metashape_ok and self.run is not None and ps.is_alignment_checkpoint(self.run)
                    and self.mesh_run is None,
            'score': self.mesh_run is not None and self.reference is not None
                     and (self.reference / 'raw-mesh.obj').exists(),
        }
        for key, ok in can.items():
            self.buttons[key].config(state='normal' if ok and not busy else 'disabled')
        self.cancel_button.config(state='normal' if self.proc is not None else 'disabled')

    def start(self, key, job):
        """Run job() in a worker thread; it reports back only through self.post()."""
        if self.worker is not None:
            messagebox.showinfo('Busy', f'{dict(STEPS)[self.worker]} is still running.')
            return
        self.worker = key
        self.set_status(key, 'running')
        self.log(f'--- {dict(STEPS)[key]}: started')
        self.refresh_buttons()

        def body():
            try:
                job()
            except Exception as exc:
                self.post(self.finish, key, 'failed', f'{type(exc).__name__}: {exc}', True)
        threading.Thread(target=body, daemon=True).start()

    def finish(self, key, state, detail='', show_error=False):
        self.worker, self.proc = None, None
        self.set_status(key, state, detail or state)
        if show_error:
            self.error(f'{dict(STEPS)[key]} {state}', detail)
        self.refresh_buttons()

    def check_folder(self, what, path, need=()):
        ok, resolved, reason = ps.inspect_folder(path, need)
        self.log(f'{what}: tried {resolved} -> {reason}')
        if not ok:
            self.error(f'Cannot use {what}', f'{resolved}\n{reason}')
        return resolved if ok else None

    def ask_folder(self, title, start=None):
        start = start or (default_run().parent if default_run().parent.exists() else Path.home())
        folder = filedialog.askdirectory(title=title, initialdir=str(start), mustexist=True)
        return Path(folder) if folder else None

    # ------------------------------------------------------------ Metashape interpreter

    def detect_python(self):
        """Search for a Python that imports Metashape in a background thread (see metashape_locator)."""
        self.change_button.config(state='disabled')

        def on_try(source, python, ok, detail):
            self.post(self.log, f'Metashape Python: {"OK " if ok else "no "} {python}  [{source}] {detail}')

        def job():
            try:
                search = ml.find_metashape_python(on_try=on_try)
            except Exception as exc:
                search = ml.Search(tried=[('search', '', f'{type(exc).__name__}: {exc}')])
            self.post(self.python_detected, search)
        threading.Thread(target=job, daemon=True).start()

    def python_detected(self, search):
        self.change_button.config(state='normal')
        if search.found:
            self.use_python(search.found.python, search.found.version, search.found.source)
        else:
            self.set_python(None)
            self.log(f'Metashape Python: none of {len(search.tried)} candidates can import Metashape; '
                     'Align and Build mesh stay disabled. Use Change… or set BD3D_METASHAPE_PYTHON.')

    def use_python(self, python, version, source):
        self.set_python(python, f'Metashape {version} (found via {source})')
        try:
            ml.save_choice(python)
        except OSError as exc:
            self.log(f'could not save the Metashape Python choice: {exc}')

    def set_python(self, python, text=None):
        self.metashape_ok = python is not None
        self.py_var.set(python or '')
        if python:
            self.ms_label.config(text=text, fg='#2a8a2a')
        else:
            self.ms_label.config(text='No Python that can import Metashape was found: Align and Build mesh '
                                      'are disabled. Use Change… to pick one.', fg='#c0392b')
        self.refresh_buttons()

    def change_python(self):
        """Small dialog: type or browse to an interpreter, test it, and use and save it if it works."""
        win = tk.Toplevel(self.root)
        win.title('Metashape Python')
        win.transient(self.root)
        tk.Label(win, text='Python interpreter that can import Metashape (e.g. a venv\'s bin/python;\n'
                           'the one inside MetashapePro.app can be typed or pasted here):',
                 justify='left').pack(anchor='w', padx=10, pady=(10, 4))
        var = tk.StringVar(value=self.py_var.get())
        entry = tk.Entry(win, textvariable=var, width=80)
        entry.pack(padx=10, fill='x')
        note = tk.Label(win, text='', fg='#777', justify='left')
        note.pack(anchor='w', padx=10, pady=4)
        buttons = tk.Frame(win)
        buttons.pack(fill='x', padx=10, pady=(0, 10))

        def browse():
            path = filedialog.askopenfilename(parent=win, title='Python interpreter',
                                              initialdir=str(Path(var.get()).parent if var.get() else Path.home()))
            if path:
                var.set(path)

        def done(python, ok, detail):
            if not win.winfo_exists():
                return
            test.config(state='normal')
            if ok:
                self.log(f'Metashape Python: OK  {python}  [chosen by you] {detail}')
                self.use_python(python, detail, 'your choice')
                win.destroy()
            else:
                note.config(text=f'Does not work: {detail}', fg='#c0392b')
                self.log(f'Metashape Python: no  {python}  [chosen by you] {detail}')

        def try_it():
            python = os.path.expanduser(var.get().strip())
            if not python:
                return
            test.config(state='disabled')
            note.config(text='testing: import Metashape …', fg='#777')
            threading.Thread(target=lambda: self.post(done, python, *ml.probe(python)), daemon=True).start()

        tk.Button(buttons, text='Browse…', command=browse).pack(side='left')
        test = tk.Button(buttons, text='Test and use', command=try_it)
        test.pack(side='right')
        tk.Button(buttons, text='Cancel', command=win.destroy).pack(side='right', padx=4)
        entry.focus_set()

    # ------------------------------------------------------------ entry points

    def new_capture(self):
        if self.worker:
            return messagebox.showinfo('Busy', 'Wait for the running step or cancel it first.')
        photos = self.ask_folder('Photo folder of the new capture', Path.home())
        photos = photos and self.check_folder('photo folder', photos)
        if not photos:
            return
        output = self.ask_folder('Output folder for the runs (new runs are created inside it)', photos.parent)
        output = output and self.check_folder('output folder', output)
        if not output:
            return
        self.photos, self.paths = photos, ps.CapturePaths(output, ps.capture_name(photos, output))
        self.run = self.mesh_run = None
        for key in self.status:
            self.set_status(key, 'not started')
        self.fig.clear()
        self.canvas.draw()
        self.summary.config(text='')
        self.score_label.config(text='')
        self.where.config(text=f'photos {photos}  ->  {self.paths.align_run}')
        self.log(f'new capture "{self.paths.name}": photos {photos}, runs in {output}')
        self.refresh_buttons()

    def choose_run(self):
        folder = self.ask_folder('Choose a run folder with cameras.csv')
        if folder:
            self.open_run(folder)

    def open_run(self, folder):
        if self.worker:
            return messagebox.showinfo('Busy', 'Wait for the running step or cancel it first.')
        folder = self.check_folder('run folder', folder, need=['cameras.csv'])
        if not folder:
            return
        self.photos = self.paths = None
        self.run = folder
        self.mesh_run = folder if (folder / 'raw-mesh.obj').exists() else None
        self.set_status('align', 'done')
        self.set_status('mesh', 'done' if self.mesh_run else
                        'not started' if ps.is_alignment_checkpoint(folder) else 'not available')
        self.set_status('score', 'not started')
        self.score_label.config(text='')
        self.show_coverage()
        self.auto_score()

    def choose_reference(self):
        folder = self.ask_folder('Choose a reference run folder with cameras.csv')
        folder = folder and self.check_folder('reference folder', folder, need=['cameras.csv'])
        if folder:
            self.reference = folder
            self.set_status('score', 'not started')
            self.score_label.config(text='')
            if self.run:
                self.show_coverage()
            self.auto_score()

    def clear_reference(self):
        self.reference = None
        self.score_label.config(text='')
        self.set_status('score', 'not started')
        if self.run:
            self.show_coverage()
        self.refresh_buttons()

    def auto_score(self):
        self.refresh_buttons()
        if self.buttons['score'].cget('state') == 'normal':
            self.run_score()

    # ------------------------------------------------------------ steps

    def run_align(self):
        python, photos, paths = self.py_var.get().strip(), self.photos, self.paths

        def job():
            self.post(self.log, f'license check with {python or "(BD3D_METASHAPE_PYTHON not set)"}')
            ok, detail = ps.license_ok(python)
            self.post(self.log, f'license: {detail}')
            if not ok:
                raise RuntimeError(detail)
            self.post(self.log, f'hashing and verifying photos in {photos} …')
            manifest = ps.alignment_manifest(photos, paths.name)
            ps.write_manifest(paths.align_manifest, manifest)
            self.post(self.log, f'manifest {paths.align_manifest}: {manifest["image_count"]} images, '
                                f'{manifest["experiment_id"]}')
            self.run_watched('align', ps.watched_command(python, paths.align_manifest, photos, paths.align_run,
                                                         ps.ALIGN_LIMITS), paths.align_run, self.align_done)
        self.start('align', job)

    def run_watched(self, key, cmd, run_dir, on_done):
        """Worker thread: run a watched command, then hand the outcome to on_done on the main thread."""
        wp = ps.WatchedProcess(cmd, run_dir, lambda line: self.post(self.log, line))
        self.post(self.set_proc, wp)
        self.post(self.log, 'command: ' + ' '.join(cmd))
        code = wp.run()
        state, detail = ps.run_outcome(run_dir)
        if wp.cancelled:
            state, detail = 'cancelled', f'stopped by Cancel; watchdog exit {code}; outputs kept in {run_dir}'
        self.post(on_done, key, run_dir, state, detail)

    def set_proc(self, wp):
        self.proc = wp
        self.refresh_buttons()

    def align_done(self, key, run_dir, state, detail):
        self.finish(key, state, detail, show_error=state == 'failed')
        if (run_dir / 'cameras.csv').exists():   # failed alignments still write their camera table
            self.run = run_dir
            self.where.config(text=str(run_dir))
            self.show_coverage()
        self.refresh_buttons()

    def show_coverage(self):
        try:
            cams, ref, note = self.run / 'cameras.csv', self.reference, ''
            try:
                cov = compare(cams, ref / 'cameras.csv') if ref else analyze(cams)
            except ValueError as exc:
                if ref is None or 'share fewer than 3' not in str(exc):
                    raise
                note = f' [reference {ref.name} shares no cameras with this run; shown without it]'
                self.log(f'coverage: {exc}; showing this run in its own frame')
                cov, ref = analyze(cams), None
            cov.summary += note
            self.draw(cov, self.run, ref)
        except Exception as exc:
            self.set_status('coverage', 'failed')
            self.error('Coverage view failed', exc)
            return
        self.where.config(text=str(self.run) + (f'   vs reference {self.reference}' if self.reference else ''))
        self.summary.config(text=cov.summary)
        self.set_status('coverage', 'done')
        self.log(f'coverage: {cov.summary}')
        self.refresh_buttons()

    def run_mesh(self):
        python, align_run, photos = self.py_var.get().strip(), self.run, self.photos
        if photos is None:   # opened an existing alignment run: the manifest has relative image paths
            photos = self.ask_folder(f'Photo folder used for {align_run.name}')
            photos = photos and self.check_folder('photo folder', photos)
            if not photos:
                return
            self.photos = photos
        paths = self.paths or ps.CapturePaths(align_run.parent, align_run.name)
        mesh_run = paths.mesh_run
        n = 1
        while mesh_run.exists() or mesh_run.with_name(mesh_run.name + '-watchdog.log').exists():
            n += 1
            mesh_run = paths.output / f'{paths.name}-mesh-{n}'

        def job():
            ok, detail = ps.license_ok(python)
            self.post(self.log, f'license: {detail}')
            if not ok:
                raise RuntimeError(detail)
            align = json.loads((align_run / 'manifest.json').read_text())
            manifest_path = paths.manifests / f'{mesh_run.name}.json'
            ps.write_manifest(manifest_path, ps.mesh_manifest(align))
            self.post(self.log, f'mesh manifest {manifest_path}, seeded from {align_run}')
            self.run_watched('mesh', ps.watched_command(python, manifest_path, photos, mesh_run, ps.MESH_LIMITS,
                                                        seed=align_run), mesh_run, self.mesh_done)
        self.start('mesh', job)

    def mesh_done(self, key, run_dir, state, detail):
        self.finish(key, state, detail, show_error=state == 'failed')
        if state == 'done' and (run_dir / 'raw-mesh.obj').exists():
            self.mesh_run = run_dir
            self.log(f'mesh: {run_dir / "raw-mesh.obj"}')
        self.auto_score()

    def run_score(self):
        run, ref = self.mesh_run, self.reference

        def job():
            self.post(self.log, f'scoring {run} against {ref} (evaluator v3) …')
            row = ps.score_run(run, ref)
            self.post(self.score_done, row)
        self.start('score', job)

    def score_done(self, row):
        text = ps.score_text(row)
        self.score_label.config(text=f'Score vs {self.reference.name}: {text}')
        self.finish('score', 'done', text)

    # ------------------------------------------------------------ cancel / close

    def cancel(self):
        if self.proc is not None:
            self.log(f'cancel requested: stopping the watchdog process group (pid {self.proc.proc and self.proc.proc.pid});'
                     ' nothing is deleted')
            self.proc.cancel()

    def close(self):
        if self.proc is not None:
            if not messagebox.askyesno('Step running', 'A step is running. Cancel it and quit? Outputs are kept.'):
                return
            self.proc.cancel()
        self.root.destroy()

    # ------------------------------------------------------------ figure

    def draw(self, cov, folder, reference):
        self.fig.clear()
        ax, cm = self.fig.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1.6]})
        ref = cov.reference

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
    root.geometry('1320x950')
    CoverageApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
