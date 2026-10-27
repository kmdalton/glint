# Handoff: laue-dials profile integration in GLINT, and the box-vs-profile benchmark

Status as of 2026-10-27. Branch `profile-integrate`.

## Goal

1. Port laue-dials' profile-fitting integrator into GLINT, on the GPU, in GLINT's idiom.
2. Process a real SFX dataset with GLINT end to end, integrating it both ways: GLINT's box and the
   ported profile fit.
3. Compare the two on a refined structure: Rwork/Rfree against a reference model, and anomalous
   peak heights. This runs as a workflow (Snakemake, `experiments/profile_bench/`).

Parts 1 and 2 are done; the workflow up to integration is proven. Part 3 has not produced numbers:
the full-dataset run was stopped by a memory problem in the data-staging step (below). The merge,
refine and report steps have never run.

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
- 39 Cheetah run files, ~1200 hits each (~47k), 243 GB.
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

`Snakefile` + `config.yaml`. Steps: fetch -> md5 -> stack -> peaks (CPU) -> index_integrate (GPU) ->
merge (abismal) -> refine (phenix) -> anom_peaks -> report. `README.md` explains each step and the
settings both arms share.

| step | status |
|---|---|
| fetch, md5, geom | ran |
| stack | ran; **has the memory bug below** |
| peaks, index_integrate | proven on a 100-frame probe of run179641: 99 frames with peaks, median 108 peaks per frame; 89 indexed after the strict gate (known cell); both arms wrote streams |
| merge, refine, anom_peaks, report | **written, never run.** Expect first-run fixes |

On the probe, box wrote 157k reflections and profile 162k. Box drops windows that straddle an edge;
profile keeps clipped ones.

Decisions both arms share, measured on this data:

- **Prediction gate `tol: 0.002` 1/A** instead of GLINT's 0.006. Found spots sit within 0.00084 of
  the Ewald sphere at the 95th percentile and 0.0014 at the 99th; 0.006 predicts ~1770 mostly empty
  reflections per frame. That would also shrink the profile fit's auto radius, which is set from
  prediction spacing.
- **v4 peak finder `sig_floor 10, abs_thr 100`** (ADU). Clipped-at-zero frames give an empty patch
  sigma = 0; without the floor the finder returns 500–3300 "peaks" per frame.
- **Index once, deep-copy the results into each arm** (`index_integrate.py`), so only the integrator
  differs.
- **Merge:** abismal `--anomalous --dmin 1.9`, P43212, 30 epochs x 1000 steps (guessed; tune).
- **Refine:** phenix.refine against 4YOP, B reset to 20, rigid body + ADP, 5 macro cycles, shared free
  set from `rs.rfree -d 1.0 -r 0.1`. This mirrors the cxidb_81 abismal-benchmarks protocol; individual
  sites are off there too.
- **Anomalous metric:** map height (sigma) within 1.5 Å of each Cys SG, Met SD, Cl and Na, and the
  best peak more than 3 Å from all of them as the null. Plus, merge-free, per-shell CC of each arm's
  partial intensities with \|Fcalc\|² of 4YOP.

## The open problem: stacking leaks memory

- `stack_sacla.py` builds a zero-copy HDF5 *virtual* dataset over each run's 1217 per-tag frames.
- Read through h5py's default chunk cache, HDF5 keeps every source frame it has touched open with its
  decompressed chunk: +14 MB per frame, ~17 GB per run.
- Eight parallel peak-finding jobs reached 82 GB on a 123 GB box. I stopped the workflow before
  anything was OOM-killed.
- `h5py.File(..., rdcc_nbytes=0)` fixes it (+0.06 GB over 150 frames against +2.1 GB), but GLINT opens
  the images itself. Pick one:
  1. Write real stacked copies: int16 with lzf, ~6.8 GB per run, ~265 GB in total. No GLINT change.
     Simplest on an HPC with scratch space.
  2. Give GLINT's image reads (`frames_from_cxi`, `integrate_cxi`, `_load_image`) a small chunk cache.
     This protects every VDS input, and Cheetah/psana .cxi often are VDS. But it re-decompresses a
     chunk per event where chunks hold several events, so measure before changing defaults.

Whichever you pick, also cap per-job memory in the Snakefile (`resources: mem_mb`) for the cluster.

## On the HPC: to do, in order

1. Move or re-fetch the data:
   - `snakemake ... fetch` re-downloads Dataset A (~2 h at 35 MB/s), resumable;
   - `metadata/` from `https://www.cxidb.org/data/33/cxidb_33_metadata.tar.gz`;
   - `reference/4YOP.pdb` from RCSB;
   - make `reference/r-free-flags.mtz` as in `README.md`.
2. Fix the stacking leak (above).
3. Edit `config.yaml` paths: data, python, abismal_bin, phenix_env. Add a Snakemake executor profile
   (slurm) with `gpu` and `mem_mb` resources.
4. **Peak finding is the bottleneck:** ~2–3 s per frame on one CPU core, ~35 CPU-hours for Dataset A.
   Fine as an array job. Or move it to GLINT's GPU v4 finder (needs cupy; `frames_from_cxi` hands it
   numpy now) for minutes instead of hours.
5. Run through `index_integrate` and look at `streams/*.json` (indexing rate, reflections, timings).
   Then run `merge`. abismal has never been fed a GLINT stream; check it parses (`abismal/io/crystfel.py`).
6. `refine`, `anom_peaks`, `report`. All three are untested; the `refine.eff.in` labels assume abismal's
   `F(+),SIGF(+),F(-),SIGF(-)`.
7. Write `RESULTS.md`.
8. Optional: the same workflow on CXIDB 81 if its xtc is reachable; the xtc bridge replaces stack/peaks.

## Files

| path | what |
|---|---|
| `glint/profile_integrate.py` | the port |
| `glint/predict.py`, `glint/glint_cli.py` | integrator switch, batched queue, adu_per_photon, CLI flags |
| `experiments/test_profile_integrate.py` | agreement test vs laue-dials (CI: torch-CPU job) |
| `experiments/profile_bench/` | workflow, scripts, README, this handoff |
| `THIRD_PARTY_NOTICES.md`, `.github/workflows/ci.yml` | licence notice, CI step |

Local state on the workstation, not in git:

- data at `/media/kmdalton/3bd97678-a8fe-47cb-bdd2-b2fdcc5569b1/data/xtal/cxidb_33`: raw Dataset A
  complete (39 files, md5 checked for 7), reference/, flags, and work/ with probe and speed outputs;
- `.../xtal/cxidb_81_raw/data.tar.gz`: CXIDB 81 processed data, no pixels, deletable;
- micromamba env `snakemake`.
