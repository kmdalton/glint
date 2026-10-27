# Box vs profile-fit integration on real SFX data

GLINT's box integrator (`predict.integrate_spots`) against the laue-dials profile fit, ported to torch
as `glint.profile_integrate` (`--integrate --integrator profile`). The comparison runs end to end on
public data, from raw pixels to a refined structure: peak find, index, integrate both ways, merge
each with abismal, refine each against the deposited model, and score on Rwork/Rfree and on the
anomalous difference peaks.

## Data: CXIDB 33, Dataset A

Hen egg-white lysozyme at SACLA BL3, 7 keV (1.77 Å), MPCCD octal detector (Nakane *et al.* 2015,
*Acta Cryst.* D71, 2519, native S/Cl SAD; PDB 4YOP). 39 Cheetah run files, ~47k hits, 243 GB, CC0.
At 7 keV the 10 sulfurs (8 Cys SG, 2 Met SD) and the chloride carry a measurable anomalous signal,
so the anomalous map at those sites is the sharpest test of weak-signal integration.

**Why not CXIDB 81 (thermolysin, Zn).** Its pixels exist only as raw XTC on NERSC tape, and on
2026-10-26 the tape portal returned `can't log in as wwwhpss` for every file, as it did for every other
tape-backed CXIDB entry tried (22, 54, 62, 66, 73). Its web-hosted `data.tar.gz` holds DIALS tables
and max-projection images, not per-shot frames. CXIDB 33's images are web-hosted.

## Running it

```bash
cd experiments/profile_bench
snakemake -s Snakefile --configfile config.yaml -j 10 --resources gpu=2
```

Edit `data` and the tool paths in `config.yaml` first. The workflow needs:

* GLINT's python (torch), for peak finding, indexing and both integrators;
* abismal, for merging, plus reciprocalspaceship and gemmi in the same env, for the report;
* PHENIX, for refinement;
* the deposit's `cxidb_33_metadata.tar.gz`, unpacked into `data/metadata`;
* `reference/4YOP.pdb`;
* `reference/r-free-flags.mtz` from `rs.rfree -c 79.2 79.2 37.9 90 90 90 -sg "P 43 21 2" -d 1.0 -r 0.1`.

| step | what |
|---|---|
| `fetch` | stream-extract the 243 GB tar over HTTP, resumable, no second copy on disk (`fetch_tar.py`) |
| `md5` | check each run file against the deposit's md5 list |
| `stack` | zero-copy virtual `(event, ss, fs)` view of each Cheetah per-tag file (`stack_sacla.py`) |
| `peaks` | GLINT v4 peak finding, host CPU, one job per run (`find_peaks.py`) |
| `index_integrate` | GLINT known-cell indexing + strict gate on the GPU, then **the same orientations** integrated by each integrator (`index_integrate.py`) |
| `merge` | abismal, anomalous, identical settings per arm |
| `refine` | phenix.refine against 4YOP, B reset to 20, rigid body + ADP, shared free set (`refine.eff.in`) |
| `anom_peaks` | anomalous map height at every S, Cl and Na, and the best peak that is not a site (`anom_peaks.py`) |
| `report` | `summary.json`: throughput, per-observation CC to \|Fcalc\|², the arms against each other, R factors, anomalous peaks (`report.py`) |

Settings that apply to both arms, and why:

* **`adu_per_photon: 10`.** Both integrators put a Poisson variance on the counts, and in ADU every
  σ is √10 too small. The profile fit's strong-spot test (I ≥ 3σ) also moves with that.
* **`tol: 0.002` Å⁻¹.** This is the prediction gate, not GLINT's default 0.006. Measured on this data,
  found spots sit within 0.00084 Å⁻¹ of the sphere at the 95th percentile and 0.0014 at the 99th.
  0.006 predicts ~1770 reflections per frame, mostly empty. That also shrinks the profile fit's
  automatic window, which laue-dials sets from prediction spacing.
* **Peak finder `sig_floor: 10, abs_thr: 100`** (detector units). The frames are clipped at zero, so
  without a σ floor an empty patch has σ = 0 and the finder returns 500–3000 "peaks" per frame.

## The port

`glint/profile_integrate.py`. One padded `(frames, reflections, window pixels)` tensor per batch;
each frame keeps its own radius, k, strong set, convergence and failure.
`experiments/test_profile_integrate.py` checks it against laue-dials itself, on CPU and CUDA, with
the upstream module fetched at the pinned commit in CI. `bench_speed.py` repeats that check on real
frames and times it. On 56 CXIDB 33 frames, ~605 predictions each, on an RTX A6000 shared with
other jobs:

| integrator | ms / frame | agreement with laue-dials |
|---|---|---|
| laue-dials (numpy, upstream) | 572 | — |
| port, CPU float64 | 378 | max \|ΔI\|/σ 2e-11, same reflections kept |
| port, GPU float64 (default) | 74 | 2e-11, same reflections kept |
| port, GPU float32 | 31 | up to 0.13 σ |
| GLINT box (CPU) | 6 | — |

## Results

See `RESULTS.md` once the workflow has run.
