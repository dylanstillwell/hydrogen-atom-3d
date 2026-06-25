#!/usr/bin/env python3
"""
3D Hydrogen Atom — GPU glow edition
====================================

Same exact quantum mechanics as hydrogen_atom_3d.py — hydrogen's analytic
Schrödinger eigenstates psi_nlm = R_nl(r) Y_lm(theta,phi) with exact
time evolution  Psi(x,t) = sum_i c_i psi_i(x) exp(-i E_i t / hbar)  —
but rendered with OpenGL additive blending: 28,000 glowing particles in
a two-layer bloom (hot core + soft halo), so the probability cloud looks
like a nebula instead of a scatter plot.

  hue        = quantum phase arg(Psi)   (watch it rotate at E/hbar)
  brightness = probability density |Psi|^2  (white-hot where dense)
  shimmer    = Metropolis re-sampling — Heisenberg uncertainty, live

Controls
  1-9    choose state / superposition      space  pause time
  + / -  faster / slower time              r      resample cloud
  j      toggle quantum jitter             v      toggle camera orbit
  q/esc  quit                              mouse  drag to orbit, wheel zoom
Run:  python3 hydrogen_atom_glow.py
"""

import sys
import math
import numpy as np
from scipy.special import genlaguerre

try:  # scipy >= 1.15
    from scipy.special import sph_harm_y

    def Ylm(l, m, theta, phi):
        return sph_harm_y(l, m, theta, phi)
except ImportError:
    from scipy.special import sph_harm

    def Ylm(l, m, theta, phi):
        return sph_harm(m, l, phi, theta)

from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as gl

TEST = "--test" in sys.argv

HARTREE_EV = 27.211386245988
T_AU_AS = 24.188843265857
SUB = {"0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄", "-": "₋"}
L_LETTER = "spdfgh"


def subscript(m):
    return "".join(SUB[ch] for ch in str(m))


# ---------------------------------------------------------------- physics
def radial_R(n, l, r):
    rho = 2.0 * r / n
    norm = math.sqrt((2.0 / n) ** 3
                     * math.factorial(n - l - 1)
                     / (2.0 * n * math.factorial(n + l)))
    return norm * np.exp(-rho / 2.0) * rho ** l * genlaguerre(n - l - 1, 2 * l + 1)(rho)


def sample_inverse_cdf(grid, pdf, count, rng):
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]
    return np.interp(rng.random(count), cdf, grid)


class HydrogenState:
    def __init__(self, parts, name):
        self.name = name
        c = np.array([p[0] for p in parts], dtype=complex)
        self.coeffs = c / np.linalg.norm(c)
        self.quantum_numbers = [(p[1], p[2], p[3]) for p in parts]
        for n, l, m in self.quantum_numbers:
            assert 0 <= l < n and abs(m) <= l
        self.energies = np.array([-0.5 / n ** 2 for n, _, _ in self.quantum_numbers])
        self.probs = np.abs(self.coeffs) ** 2

    def eval_components(self, x, y, z):
        r = np.sqrt(x * x + y * y + z * z)
        with np.errstate(invalid="ignore", divide="ignore"):
            theta = np.arccos(np.clip(np.divide(z, r, where=r > 0), -1.0, 1.0))
        theta = np.where(r > 0, theta, 0.0)
        phi = np.arctan2(y, x)
        out = np.empty((len(self.coeffs), x.size), dtype=complex)
        for i, (n, l, m) in enumerate(self.quantum_numbers):
            out[i] = radial_R(n, l, r) * Ylm(l, m, theta, phi)
        return out

    def psi(self, comps, t):
        return (self.coeffs * np.exp(-1j * self.energies * t)) @ comps

    def sample(self, count, rng):
        which = rng.choice(len(self.coeffs), size=count, p=self.probs)
        x, y, z = np.empty(count), np.empty(count), np.empty(count)
        for i, (n, l, m) in enumerate(self.quantum_numbers):
            idx = np.flatnonzero(which == i)
            if idx.size == 0:
                continue
            r_grid = np.linspace(0, 2 * n * n + 12 * n, 6000)
            r = sample_inverse_cdf(r_grid, (r_grid * radial_R(n, l, r_grid)) ** 2,
                                   idx.size, rng)
            t_grid = np.linspace(0, np.pi, 2000)
            t_pdf = np.abs(Ylm(l, m, t_grid, 0.0)) ** 2 * np.sin(t_grid)
            theta = sample_inverse_cdf(t_grid, t_pdf, idx.size, rng)
            phi = rng.uniform(0, 2 * np.pi, idx.size)
            st = np.sin(theta)
            x[idx] = r * st * np.cos(phi)
            y[idx] = r * st * np.sin(phi)
            z[idx] = r * np.cos(theta)
        return x, y, z

    def label(self):
        terms = []
        for c, (n, l, m) in zip(self.coeffs, self.quantum_numbers):
            amp = f"{abs(c):.2f}" if len(self.coeffs) > 1 else ""
            terms.append(f"{amp}|{n}{L_LETTER[l]}{subscript(m)}⟩")
        return " + ".join(terms)

    def energy_text(self):
        ev = self.energies * HARTREE_EV
        if len(ev) == 1:
            return f"E = {ev[0]:.2f} eV"
        txt = "E = " + ", ".join(f"{e:.2f}" for e in ev) + " eV"
        de = np.abs(self.energies[:, None] - self.energies[None, :])
        de = de[de > 1e-12]
        if de.size:
            txt += (f" &nbsp;|&nbsp; beat ΔE = {de.min() * HARTREE_EV:.2f} eV, "
                    f"T = {2 * np.pi / de.min() * T_AU_AS:.0f} as")
        return txt


ISQ2 = 1 / math.sqrt(2)
PRESETS = {
    "1": HydrogenState([(1, 1, 0, 0)], "1s ground state"),
    "2": HydrogenState([(1, 2, 1, 0)], "2p₀ orbital"),
    "3": HydrogenState([(1, 3, 2, 0)], "3d₀ orbital"),
    "4": HydrogenState([(1, 3, 2, 2)], "3d₂ — probability current swirls around z"),
    "5": HydrogenState([(1, 4, 3, 0)], "4f₀ orbital"),
    "6": HydrogenState([(ISQ2, 1, 0, 0), (ISQ2, 2, 1, 0)],
                       "1s + 2p₀ — oscillating dipole (Lyman-α beat)"),
    "7": HydrogenState([(ISQ2, 1, 0, 0), (ISQ2, 2, 0, 0)],
                       "1s + 2s — radial breathing beat"),
    "8": HydrogenState([(ISQ2, 2, 1, 1), (ISQ2, 3, 2, 2)],
                       "circular states 2p₁ + 3d₂ — orbiting packet"),
    "9": HydrogenState([(ISQ2, 3, 2, 2), (ISQ2, 3, 2, -2)],
                       "3d₂ + 3d₋₂ — standing wave (4-lobe interference)"),
}

HELP = ("[1-9] state &nbsp; [space] pause &nbsp; [+/-] speed &nbsp; [j] jitter "
        "&nbsp; [v] orbit &nbsp; [r] resample &nbsp; [q] quit &nbsp; mouse: drag / wheel")


def hsv_to_rgb(h, s, v):
    """Vectorized HSV -> RGB, all arrays in [0,1]. Returns (N,3)."""
    h6 = (h % 1.0) * 6.0
    i = np.floor(h6).astype(int) % 6
    f = h6 - np.floor(h6)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    conds = [i == k for k in range(6)]
    r = np.select(conds, [v, q, p, p, t, v])
    g = np.select(conds, [t, v, v, q, p, p])
    b = np.select(conds, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)


class GlowView(gl.GLViewWidget):
    """GL view that forwards key presses to the app."""

    def __init__(self, on_key):
        super().__init__()
        self._on_key = on_key

    def keyPressEvent(self, ev):
        if not self._on_key(ev):
            super().keyPressEvent(ev)


class HydrogenGlowApp:
    def __init__(self, n_points=28000, seed=1):
        self.rng = np.random.default_rng(seed)
        self.n_points = n_points
        self.t = 0.0
        self.dt = 0.08
        self.paused = False
        self.jitter = True
        self.orbiting = True

        self.qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        pg.setConfigOptions(antialias=True)

        self.view = GlowView(self.handle_key)
        self.view.setBackgroundColor("#000005")

        # additive-blended particle layers: soft halo + hot core = bloom
        self.halo = gl.GLScatterPlotItem(pxMode=True)
        self.halo.setGLOptions("additive")
        self.core = gl.GLScatterPlotItem(pxMode=True)
        self.core.setGLOptions("additive")
        self.view.addItem(self.halo)
        self.view.addItem(self.core)

        # the proton: a tight glowing ember at the origin
        n_nuc = 260
        self.nuc_pos = self.rng.normal(0, 1.0, (n_nuc, 3))
        self.nuc_size = self.rng.uniform(2.0, 9.0, n_nuc)
        mix = self.rng.random(n_nuc)[:, None]
        self.nuc_base = (np.array([1.0, 0.55, 0.25]) * (1 - mix)
                         + np.array([1.0, 0.95, 0.85]) * mix)
        self.nucleus = gl.GLScatterPlotItem(pxMode=True)
        self.nucleus.setGLOptions("additive")
        self.view.addItem(self.nucleus)

        # window with HUD overlaid on the GL view
        self.win = QtWidgets.QWidget()
        self.win.setWindowTitle("Hydrogen atom — live Schrödinger evolution (GPU glow)")
        layout = QtWidgets.QGridLayout(self.win)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.view, 0, 0)

        def overlay(align, style):
            lab = QtWidgets.QLabel()
            lab.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
            lab.setStyleSheet("background: transparent; " + style)
            lab.setAlignment(QtCore.Qt.AlignHCenter)
            layout.addWidget(lab, 0, 0, alignment=align)
            return lab

        top = QtCore.Qt.AlignTop | QtCore.Qt.AlignHCenter
        bottom = QtCore.Qt.AlignBottom | QtCore.Qt.AlignHCenter
        self.lbl_title = overlay(top, "color:#f0f6ff; font-size:15px; padding-top:10px;")
        self.lbl_time = overlay(bottom,
                                "color:#8fa8cf; font-size:11px; font-family:Menlo,monospace;")
        self.lbl_help = overlay(bottom, "color:#46546e; font-size:10px; padding-bottom:6px;")
        self.lbl_help.setText(HELP)
        layout.setAlignment(self.lbl_time, bottom)
        self.lbl_time.setContentsMargins(0, 0, 0, 22)

        self.win.resize(1100, 880)
        self.set_state("6")

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.frame)

    # ------------------------------------------------------------ state
    def set_state(self, key):
        self.state = PRESETS[key]
        self.t = 0.0
        self.resample()
        st = self.state
        self.lbl_title.setText(
            f"<b>Ψ = {st.label()}</b> &nbsp;—&nbsp; {st.name}<br>"
            f"<span style='color:#9fb4d8; font-size:12px;'>{st.energy_text()}</span>")

    def resample(self):
        st = self.state
        x, y, z = st.sample(self.n_points, self.rng)
        self.pts = np.column_stack([x, y, z]).astype(np.float32)
        self.comps = st.eval_components(x, y, z)
        self.env = st.probs @ (np.abs(self.comps) ** 2)
        self.extent = 1.1 * np.percentile(np.linalg.norm(self.pts, axis=1), 99)
        self.step = 0.05 * self.extent
        self.nucleus.setData(pos=(self.nuc_pos * 0.006 * self.extent).astype(np.float32),
                             size=self.nuc_size,
                             color=np.column_stack([self.nuc_base,
                                                    np.full(len(self.nuc_base), 0.9)]))
        self.view.setCameraPosition(distance=2.55 * self.extent, elevation=16, azimuth=-60)

    # ------------------------------------------------------------ dynamics
    def metropolis_step(self):
        # re-thermalize a random quarter of the walkers each frame
        idx = self.rng.choice(self.n_points, self.n_points // 4, replace=False)
        prop = self.pts[idx] + self.rng.normal(0, self.step, (idx.size, 3)).astype(np.float32)
        comps_p = self.state.eval_components(prop[:, 0], prop[:, 1], prop[:, 2])
        env_p = self.state.probs @ (np.abs(comps_p) ** 2)
        acc = self.rng.random(idx.size) < env_p / np.maximum(self.env[idx], 1e-300)
        moved = idx[acc]
        self.pts[moved] = prop[acc]
        self.comps[:, moved] = comps_p[:, acc]
        self.env[moved] = env_p[acc]

    def frame(self):
        if not self.paused:
            self.t += self.dt
            if self.jitter:
                self.metropolis_step()

        psi = self.state.psi(self.comps, self.t)            # exact TDSE evolution
        w = np.abs(psi) ** 2 / np.maximum(self.env, 1e-300)
        b = np.clip(w / (np.percentile(w, 92) + 1e-300), 0.0, 1.0) ** 0.7
        b = 0.03 + 0.97 * b

        hue = (np.angle(psi) + np.pi) / (2 * np.pi)
        sat = 1.0 - 0.25 * b ** 3                           # slight white-hot at peaks
        rgb = hsv_to_rgb(hue, sat, b)

        halo_col = np.empty((self.n_points, 4), dtype=np.float32)
        halo_col[:, :3] = rgb * 0.085
        halo_col[:, 3] = 0.8
        core_col = np.empty_like(halo_col)
        core_col[:, :3] = rgb * 0.75
        core_col[:, 3] = 0.85

        self.halo.setData(pos=self.pts, color=halo_col, size=7.0 + 11.0 * b)
        self.core.setData(pos=self.pts, color=core_col, size=1.4 + 2.6 * b)

        pulse = 0.78 + 0.22 * math.sin(self.t * 2.2)        # nucleus heartbeat
        nuc_col = np.column_stack([self.nuc_base * pulse,
                                   np.full(len(self.nuc_base), 0.9)])
        self.nucleus.setData(color=nuc_col)

        if self.orbiting:
            self.view.orbit(0.18, 0.10 * math.sin(self.t * 0.15))

        self.lbl_time.setText(
            f"t = {self.t * T_AU_AS:9.1f} as   dt = {self.dt:.3f} au   "
            f"box ±{self.extent:.0f} a₀   jitter {'on' if self.jitter else 'off'}"
            f"{'   PAUSED' if self.paused else ''}")

    # ------------------------------------------------------------ input
    def handle_key(self, ev):
        key, text = ev.key(), ev.text()
        if text in PRESETS:
            self.set_state(text)
        elif key == QtCore.Qt.Key_Space:
            self.paused = not self.paused
        elif text in ("+", "="):
            self.dt = min(self.dt * 1.5, 5.0)
        elif text in ("-", "_"):
            self.dt = max(self.dt / 1.5, 0.005)
        elif text == "j":
            self.jitter = not self.jitter
        elif text == "v":
            self.orbiting = not self.orbiting
        elif text == "r":
            self.resample()
        elif text == "q" or key == QtCore.Qt.Key_Escape:
            self.qapp.quit()
        else:
            return False
        return True

    def run(self):
        self.win.show()
        self.view.setFocus()
        self.timer.start(16)
        self.qapp.exec_()


def main():
    app = HydrogenGlowApp()
    if TEST:  # offscreen smoke test: step frames, save framebuffer snapshots
        app.win.setAttribute(QtCore.Qt.WA_DontShowOnScreen, True)
        app.win.show()
        for key, fname in (("6", "hydrogen_glow_beat.png"),
                           ("4", "hydrogen_glow_3d2.png"),
                           ("8", "hydrogen_glow_orbit.png")):
            app.set_state(key)
            for _ in range(25):
                app.frame()
                app.qapp.processEvents()
            app.view.grabFramebuffer().save(fname)
            print("wrote", fname)
        return
    app.run()


if __name__ == "__main__":
    main()
