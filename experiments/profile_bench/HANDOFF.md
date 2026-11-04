# Handoff: laue-dials profile integration in GLINT, and the box-vs-profile benchmark

Status as of 2026-11-04. Branch `profile-integrate`.

## Goal

1. Port laue-dials' profile-fitting integrator into GLINT, on the GPU, in GLINT's idiom.
2. Process a real SFX dataset with GLINT end to end, integrating it both ways: GLINT's box and the
   ported profile fit.
3. Compare the two on a refined structure: Rwork/Rfree against a reference model, and anomalous
   peak heights. This runs as a workflow (Snakemake, `experiments/profile_bench/`).

Parts 1 and 2 are done. Part 3 ran end to end on the full dataset on Perlmutter on 2026-10-27, but
its numbers do not yet answer the question: see the open problems. `RESULTS.md` is not written.

## Results of the 2026-10-27 run

39 runs, 82,039 frames with peaks, 69,361 indexed (85%). Outputs are in
`/pscratch/sd/k/kmdalton/cxidb_33/work/report/` (`summary.json`, `cchalf*.json`, `cchalf.png`).

| | box | profile |
|---|---|---|
| reflections | 41.0 M | 42.3 M |
| CC1/2 overall (merge-free, `cchalf.py`) | 0.982 | 0.997 |
| CC1/2, 2.07-2.03 A | 0.57 | 0.83 |
| CC1/2, 2.03-1.99 A | 0.32 | 0.69 |
| final Rwork / Rfree (abismal, torchref) | 0.300 / 0.346 | 0.305 / 0.352 |
| mean S peak / best non-site peak (sigma) | 2.48 / 4.37 | 2.51 / 4.38 |

## Open problems

**Refinement statistics are far too high.** 4YOP's deposit reports 0.157 / 0.191 at 2.1 A.
- These merges are abismal's. The merge is to be switched to partialator, to find out how much of
  this is the merge.
- R is 0.32 even below 4 A, which suggests solvent or scaling. Not tested.
- ADP refinement raises Rfree in both arms (box 0.341 -> 0.346, profile 0.346 -> 0.352).
- Box's 0.5-point lead depends on scaling. With a separate scale per shell (F-obs vs F-model,
  work + free), profile has the lower R in every shell but 2.0-1.9 A, by 0.3-0.8 points.

**The anomalous peak heights do not measure the signal.** No S site beats the best non-site peak,
but the signal is in the merges.
- CC(F(+)-F(-), the difference predicted from 4YOP) is 0.33-0.44 per shell to 2.3 A with the right
  sign, and ~0.01 at 2.3-1.9 A. All 1,048 lattices of the one run checked (run179641) are right-handed.
- A map from F(+)-F(-) with phases phi_model - 90 reproduces `anom_peaks.py`'s heights, so
  torchref's ANOM map is not at fault.
- Cutting that map at 2.3 A lifts the S sites from 2.5 to ~7.5 sigma, but the best non-site peak
  reaches 8.2-8.5 sigma. It has not been located. These checks were throwaway scripts, not in git.
- `cchalf_anom.json` keeps Friedel mates apart but is dominated by the mean intensity. It is not
  CCanom.

**CC1/2 gains more than agreement with the model does.** At 2.1-2.0 A, CC(F-obs, F-model) only goes
from 0.46 to 0.49. `cchalf.py` is an unweighted merge with no partiality model and drops 12-14% of
frames, a slightly different set per arm.

**Sigma calibration is unchecked.** Profile's sigmas are 35% smaller than box's on the same
reflections, so its better I/sigma is not evidence yet.

**Speed.** The workflow's 68 vs 70 ms per frame (box vs profile) disagrees with the bench's 6 vs 42,
probably because image reading dominates. Re-measure on an idle A100 before quoting either.

## Next

1. Merge with partialator; compare refinement to the abismal numbers above.
2. Find out why R is high: solvent, scaling, ADP, and the 1.9 A limit (the deposit stops at 2.1 A).
3. A real anomalous test: CCanom from half-datasets, CC to the predicted difference, and peak heights
   at a cutoff chosen independently of the arms. Locate the non-site peak.
4. Check sigma calibration in both arms, then write `RESULTS.md`.
5. Run `cchalf.py` from the Snakefile; it was run by hand, and `plot_cchalf.py` hard-codes the frame count.

## What is done

### The port: `glint/profile_integrate.py`

- A torch translation of `laue_dials.algorithms.integration.Integrator`, plus the per-image driver
  rules in `laue_dials.command_line.integrate`: mask-dilation drop, MIN_WINDOW_PIXELS drop, and
  skip-the-image on RuntimeError. Pinned upstream commit `d57f96ee`. MIT, noticed in
  `THIRD_PARTY_NOTICES.md`.
- Batched over frames: one padded (frames, reflections, window pixels) tensor. Each frame keeps its
  own radius, k, strong set, convergence and failure, so a batch equals each frame run alone.
- Pixel convention: GLINT puts pixel centres at integers, laue-dials uses pixel i = [i, i+1); the port
  maps centroid = fs + 0.5. Nearest neighbours and the auto radius use lab-frame pixel coordinates
  from the .geom.
- Wired into GLINT: `glint --integrate --integrator profile` with `--profile-radius/-knn/-maxiter/-batch`,
  on both `integrate_cxi` (--images) and `integrate_frames` (--peaks). Frames are queued and fitted
  in device batches (`predict._ProfileQueue`). The box path is unchanged.
- Also added: `--adu-per-photon` / `adu_per_photon=`, which divides frames by the gain before either
  integrator. Both assume Poisson counts; in ADU every sigma is sqrt(gain) too small.

**Verification**
- `experiments/test_profile_integrate.py` runs the port against the laue-dials module itself. It passes
  on CPU and CUDA, float64 agreeing to ~1e-13 relative with the same kept sets. Cases: masks,
  panel-edge windows, auto radius, the early stop (maxiter=6), float counts, and a failed frame.
- Added to the torch-CPU CI job, which fetches the upstream module at the pinned commit.
- Existing integration tests pass. `test_negative_intensities.py` counted keep-rules in `predict.py`;
  its expected count went 2 -> 3, the new one being the batched queue's.

### Measured performance (RTX A6000, *shared with other jobs at 100% utilisation*, so pessimistic)

Real CXIDB 33 frames, ~605 predictions per frame (`bench_speed.py`):

| | ms/frame | vs laue-dials |
|---|---|---|
| laue-dials numpy | 572 | — |
| port CPU f64 | 378 | max \|dI\|/sigma 2e-11, same kept set |
| port GPU f64, batch 32 | 74 | 2e-11, same kept set |
| port GPU f32, batch 32 | 31 | up to 0.13 sigma |
| GLINT box (CPU) | 6–18 | — |

Batch sweep, 126 frames (GPU, ms/frame and peak memory):

| batch | f64 ms | f64 memory | f32 ms | f32 memory |
|---|---|---|---|---|
| 1 | 71.8 | 0.3 GB | 29.1 | 0.2 GB |
| 8 | **42.0** | 2.2 GB | 24.7 | 1.2 GB |
| 32 | 46.7 | 8.7 GB | 30.2 | 4.7 GB |
| 128 | 48.7 | 34.3 GB | 26.8 | 18.6 GB |

- The batch knee is at about 8, so the default is now 8 everywhere.
- Memory is about 0.27 GB per frame in f64.
- Of the f32 time, ~12 ms per frame is copying the whole 16.8 MB frame to the device. Gathering
  windows on the host, or shipping int16, would roughly halve it.
- Re-measure on an idle A100/H100 before quoting any of these.

### Dataset: CXIDB 33 Dataset A (not 81)

- HEWL, SACLA BL3, 7 keV / 1.77 Å, MPCCD; Nakane et al. 2015 native S/Cl SAD; PDB 4YOP; CC0.
- 39 Cheetah run files, 243 GB; 82,039 frames with peaks.
- Web-hosted at `https://www.cxidb.org/data/33/cxidb_33_DatasetA.h5.tar`, a single plain tar;
  `fetch_tar.py` stream-extracts it.
- Pixels are int16 at 10 ADU per photon, dark-subtracted and clipped at 0.

**CXIDB 81 (thermolysin, the preferred benchmark) was not reachable.** Its frames exist only as raw
XTC, ~130–650 GB per run, on NERSC tape behind portal.nersc.gov. On 2026-10-26 the portal failed
for every file ("can't log in as wwwhpss"), as it did for CXIDB 22/54/62/66/73. The web-hosted
`data.tar.gz` has DIALS tables and max projections only. **On an HPC this may be solvable:**

- at NERSC, the files are in HPSS under `/home/projects/cxidb/www/81/cxi78513/` (xtc + `calib.tar.gz`)
  and might be reachable with `hsi` from a login node;
- at S3DF, experiment cxi78513 may still be in the LCLS archive.

Neither was checked. GLINT's `experiments/xtc_bridge` reads LCLS-I xtc through psana1 in the GLINT
env (S3DF `ana-4.0.58-py3-minipytorch`); psana1 also installs from conda channel `lcls-i`. The
2tli model and refine settings for 81 are in your abismal-benchmarks `data/cxidb_81/reference_data`.

### The workflow: `experiments/profile_bench/`

`Snakefile` + `config.yaml`, with `perlmutter/` for slurm. Every step has run on all 39 runs;
`README.md` explains each step and the settings both arms share.

On the first 100-frame probe, box wrote 157k reflections and profile 162k. Box drops windows that
straddle an edge; profile keeps clipped ones.

Decisions both arms share, measured on this data:

- **Prediction gate `tol: 0.002` 1/A** instead of GLINT's 0.006. Found spots sit within 0.00084 of
  the Ewald sphere at the 95th percentile and 0.0014 at the 99th; 0.006 predicts ~1770 mostly empty
  reflections per frame. That would also shrink the profile fit's auto radius, which is set from
  prediction spacing.
- **v4 peak finder `sig_floor 10, abs_thr 100`** (ADU). Clipped-at-zero frames give an empty patch
  sigma = 0; without the floor the finder returns 500–3300 "peaks" per frame.
- **Index once, deep-copy the results into each arm** (`index_integrate.py`), so only the integrator
  differs.
- **Merge (2026-10-27 run):** abismal PyTorch port, `--anomalous --dmin 1.9`, P43212, the branch's
  defaults (10 epochs x 1000 steps).
- **Refine:** torchref against 4YOP, B reset to 20, rigid body, 5 cycles of scale + isotropic ADP,
  shared free set from `make_rfree.py`. Individual sites stay fixed.
- **Anomalous metric:** map height (sigma) within 1.5 Å of each Cys SG, Met SD, Cl and Na, and the
  best peak more than 3 Å from all of them as the null. Plus, merge-free, per-shell CC of each arm's
  partial intensities with \|Fcalc\|² of 4YOP.

## Stacking

`stack_sacla.py` once built zero-copy HDF5 virtual datasets. Through h5py's default chunk cache,
HDF5 keeps every source frame it has touched open with its decompressed chunk, ~17 GB a run. The
workflow now writes real int16 lzf stacks instead, ~6.8 GB a run. GLINT's image reads still have
that behaviour on any VDS input.

## Files

| path | what |
|---|---|
| `glint/profile_integrate.py` | the port |
| `glint/predict.py`, `glint/glint_cli.py` | integrator switch, batched queue, adu_per_photon, CLI flags |
| `experiments/test_profile_integrate.py` | agreement test vs laue-dials (CI: torch-CPU job) |
| `experiments/profile_bench/` | workflow, scripts, README, this handoff |
| `THIRD_PARTY_NOTICES.md`, `.github/workflows/ci.yml` | licence notice, CI step |

State on Perlmutter, not in git:

- data at `/pscratch/sd/k/kmdalton/cxidb_33`: raw Dataset A (39 files, md5 checked), reference/,
  and work/ with stacks, peaks, streams, merges, refinements and the report. Scratch can be purged;
- micromamba envs `glint` and `abismal` in `.mamba/` at the repository root.
