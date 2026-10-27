"""Collect the box-vs-profile benchmark into one summary.json.

  * throughput and counts per arm: frames, reflections, peak-find / index / integrate seconds;
  * merge-free accuracy: per resolution shell, the Pearson CC between each arm's single-shot
    (partial) intensities and |Fcalc|^2 from the reference model. Partiality spreads both arms
    the same way, so the difference between arms is the integrator's;
  * the arms against each other on the observations both kept: CC of I, median sigma ratio;
  * I/sigma per shell;
  * phenix.refine final Rwork/Rfree and the anomalous peak heights (anom_peaks.py).

    python report.py --work WORK --integrators box,profile --runs r1 r2 ... --pdb 4YOP.pdb -o summary.json
"""
import argparse
import glob
import json
import os
import re

import gemmi
import numpy as np
import reciprocalspaceship as rs


def read_stream(path, max_chunks=None):
    """GLINT .stream -> (chunk, h, k, l, I, sigma) arrays, chunk counted from 0 in file order."""
    rows = []
    chunk = -1
    inref = False
    with open(path) as f:
        for ln in f:
            if ln.startswith("----- Begin chunk"):
                chunk += 1
                if max_chunks and chunk >= max_chunks:
                    break
            elif ln.startswith("Reflections measured after indexing"):
                inref = True
            elif ln.startswith("End of reflections"):
                inref = False
            elif inref:
                p = ln.split()
                if len(p) >= 5 and p[0].lstrip("-").isdigit():
                    rows.append((chunk, int(p[0]), int(p[1]), int(p[2]), float(p[3]), float(p[4])))
    if not rows:
        return np.zeros((0, 6))
    return np.array(rows)


def fcalc_lookup(pdb, cell, sg, dmin):
    """|Fcalc|^2 on the reciprocal ASU to dmin, as a dict keyed by ASU hkl."""
    st = gemmi.read_structure(pdb)
    st.setup_entities()
    calc = gemmi.StructureFactorCalculatorX(st.cell)
    hkl = rs.utils.generate_reciprocal_asu(cell, sg, dmin, anomalous=False)
    return {tuple(h): abs(calc.calculate_sf_from_model(st[0], [int(x) for x in h])) ** 2 for h in hkl}


def shells(d, n=8):
    edges = np.quantile(1 / d ** 2, np.linspace(0, 1, n + 1))
    return np.clip(np.searchsorted(edges, 1 / d ** 2, side="right") - 1, 0, n - 1), 1 / np.sqrt(edges)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--integrators", default="box,profile")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--cell", nargs=6, type=float, default=[79.2, 79.2, 37.9, 90, 90, 90])
    ap.add_argument("--sg", default="P 43 21 2")
    ap.add_argument("--dmin", type=float, default=1.9)
    ap.add_argument("--max-chunks", type=int, default=0, help="per run, for the per-observation stats")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    integs = a.integrators.split(",")
    cell, sg = gemmi.UnitCell(*a.cell), gemmi.SpaceGroup(a.sg)
    out = {"runs": a.runs, "integrators": integs}

    # --- throughput -------------------------------------------------------------------------
    tot = {"t_peakfind_s": 0.0, "t_index_s": 0.0, "n_hits_with_peaks": 0, "n_indexed": 0}
    for i in integs:
        tot[i] = {"t_integrate_s": 0.0, "n_frames": 0, "n_refl": 0}
    for r in a.runs:
        j = json.load(open(f"{a.work}/streams/{r}.json"))
        for k in ("t_peakfind_s", "t_index_s", "n_hits_with_peaks", "n_indexed"):
            tot[k] += j[k]
        for i in integs:
            for k in tot[i]:
                tot[i][k] += j[i][k]
    for i in integs:
        tot[i]["ms_per_frame"] = 1e3 * tot[i]["t_integrate_s"] / max(tot[i]["n_frames"], 1)
        tot[i]["refl_per_frame"] = tot[i]["n_refl"] / max(tot[i]["n_frames"], 1)
    out["throughput"] = tot

    # --- per-observation accuracy against the model --------------------------------------------
    F2 = fcalc_lookup(a.pdb, cell, sg, a.dmin)
    obs = {}
    for i in integs:
        parts = []
        for ri, r in enumerate(a.runs):
            x = read_stream(f"{a.work}/streams/{i}/{r}.stream", a.max_chunks or None)
            if len(x):
                x = np.column_stack([np.full(len(x), ri), x])
                parts.append(x)
        x = np.concatenate(parts)
        H = rs.utils.hkl_to_asu(x[:, 2:5].astype(int), sg)[0]
        d = cell.calculate_d_array(H)
        f2 = np.array([F2.get(tuple(h), np.nan) for h in H])
        obs[i] = dict(run=x[:, 0], chunk=x[:, 1], H=H, hkl=x[:, 2:5].astype(int), I=x[:, 5],
                      sig=x[:, 6], d=d, f2=f2)
    ref_d = np.concatenate([obs[i]["d"] for i in integs])
    _, edges = shells(ref_d[ref_d >= a.dmin])
    def shell_of(d):
        e = 1 / edges ** 2
        return np.clip(np.searchsorted(e, 1 / d ** 2, side="right") - 1, 0, len(edges) - 2)
    acc = {"d_edges": [round(float(e), 3) for e in edges]}
    for i in integs:
        o = obs[i]
        ok = np.isfinite(o["f2"]) & (o["d"] >= a.dmin) & (o["sig"] > 0)
        sh = shell_of(o["d"][ok])
        I, s, f2 = o["I"][ok], o["sig"][ok], o["f2"][ok]
        acc[i] = {
            "cc_vs_fcalc2": [float(np.corrcoef(I[sh == b], f2[sh == b])[0, 1]) for b in range(len(edges) - 1)],
            "cc_vs_fcalc2_overall": float(np.corrcoef(I, f2)[0, 1]),
            "median_isigi": [float(np.median(I[sh == b] / s[sh == b])) for b in range(len(edges) - 1)],
            "frac_isigi_gt3": [float(np.mean(I[sh == b] / s[sh == b] > 3)) for b in range(len(edges) - 1)],
            "n_obs": int(ok.sum()),
        }
    out["accuracy"] = acc

    # --- the arms against each other, on the observations both kept ----------------------------
    if len(integs) == 2:
        A, B = (obs[i] for i in integs)
        key = lambda o: o["run"] * 1e12 + o["chunk"] * 1e6 + (o["hkl"] + 50) @ np.array([1e4, 1e2, 1])
        ka, kb = key(A), key(B)
        common, ia, ib = np.intersect1d(ka, kb, return_indices=True)
        out["paired"] = {
            "n_common": int(len(common)), f"n_only_{integs[0]}": int(len(ka) - len(common)),
            f"n_only_{integs[1]}": int(len(kb) - len(common)),
            "cc_I": float(np.corrcoef(A["I"][ia], B["I"][ib])[0, 1]),
            f"median_sigma_ratio_{integs[1]}_over_{integs[0]}": float(np.median(B["sig"][ib] / A["sig"][ia])),
        }

    # --- refinement and anomalous peaks -------------------------------------------------------
    ref = {}
    for i in integs:
        logp = f"{a.work}/refine/{i}/refine_1.log"
        txt = open(logp).read() if os.path.exists(logp) else ""
        m = re.findall(r"Final R-work = ([0-9.]+), R-free = ([0-9.]+)", txt)
        st = re.findall(r"Start R-work = ([0-9.]+), R-free = ([0-9.]+)", txt)
        ref[i] = {"rwork": float(m[-1][0]) if m else None, "rfree": float(m[-1][1]) if m else None,
                  "start_rwork": float(st[-1][0]) if st else None, "start_rfree": float(st[-1][1]) if st else None}
        pk = f"{a.work}/refine/{i}/anom_peaks.json"
        if os.path.exists(pk):
            ref[i]["anom"] = json.load(open(pk))
    out["refinement"] = ref
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
