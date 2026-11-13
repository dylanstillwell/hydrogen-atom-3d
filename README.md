# Hydrogen Atom

A real-time 3D visualization of the hydrogen atom's quantum states.

It's exact quantum mechanics, not a stylized approximation: hydrogen's
analytic Schrödinger eigenstates

```
ψ_nlm(r, θ, φ) = R_nl(r) · Y_lm(θ, φ)
```

are evolved in time exactly via

```
Ψ(x, t) = Σ_i c_i ψ_i(x) exp(-i E_i t / ℏ)
```

28,000 particles are sampled from `|Ψ|²` and continuously re-thermalized
with a Metropolis step, so the cloud visibly shimmers with quantum
uncertainty even when "paused" in time.

- **Hue** — the quantum phase `arg(Ψ)` (watch it rotate at `E/ℏ`)
- **Brightness** — the probability density `|Ψ|²` (white-hot where dense)
- **Shimmer** — live Metropolis re-sampling of the Heisenberg-uncertain cloud

## Requirements

- Python 3.9+
- A GPU / OpenGL-capable display (this uses `pyqtgraph.opengl`)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python3 hydrogen_atom_glow.py
```

## Controls

| Key       | Action                          |
|-----------|----------------------------------|
| `1`–`9`   | Choose a state / superposition  |
| `space`   | Pause / resume time evolution   |
| `+` / `-` | Speed up / slow down time       |
| `j`       | Toggle quantum jitter (Metropolis resampling) |
| `v`       | Toggle camera auto-orbit        |
| `r`       | Resample the particle cloud     |
| `q` / `esc` | Quit                           |
| mouse     | Drag to orbit, wheel to zoom    |

## Preset states

1. `1s` ground state
2. `2p₀` orbital
3. `3d₀` orbital
4. `3d₂` — probability current swirls around the z-axis
5. `4f₀` orbital
6. `1s + 2p₀` — oscillating dipole (Lyman-α beat)
7. `1s + 2s` — radial breathing beat
8. `2p₁ + 3d₂` — circular states, an orbiting wave packet
9. `3d₂ + 3d₋₂` — standing wave, 4-lobe interference pattern

## Smoke test

Renders three presets offscreen and saves PNG snapshots:

```bash
python3 hydrogen_atom_glow.py --test
```

## License

MIT — see [LICENSE](LICENSE).
