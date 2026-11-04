"""CrystFEL merged reflection list (partialator -y 422) -> the MTZ refine_torchref.py reads.

French-Wilson per Bijvoet mate, so negative merged intensities become positive amplitudes. The
mates stay apart: F(+),SIGF(+),F(-),SIGF(-), plus the I(+),SIGI(+),I(-),SIGI(-) they came from.
Systematic absences are dropped.

    python hkl_to_mtz.py merged.hkl -o merged.mtz --cell 79.2 79.2 37.9 90 90 90 --sg "P 43 21 2"
"""
import argparse

import gemmi
import numpy as np
import reciprocalspaceship as rs


def read_hkl(path):
    """h, k, l, I, sigma(I), nmeas of a CrystFEL reflection list (the phase column is '-')."""
    rows = []
    with open(path) as f:
        for ln in f:
            w = ln.split()
            if ln.startswith("End of reflections"):
                break
            if len(w) == 7 and w[0].lstrip("-").isdigit():
                rows.append((int(w[0]), int(w[1]), int(w[2]), float(w[3]), float(w[5]), int(w[6])))
    return np.array(rows)


def to_mtz(H, I, sig, cell, sg, out):
    """Unmerged-in-Friedel (H, I, sigI) on any symmetry equivalent -> French-Wilson MTZ."""
    ds = rs.DataSet({"H": H[:, 0], "K": H[:, 1], "L": H[:, 2], "I": I, "SIGI": sig},
                    cell=cell, spacegroup=sg).infer_mtz_dtypes().set_index(["H", "K", "L"])
    ds = ds[ds["SIGI"] > 0].hkl_to_asu(anomalous=True).remove_absences()
    ds = ds.drop(columns=["M/ISYM"], errors="ignore")
    ds.merged = True
    ds = rs.algorithms.scale_merged_intensities(ds, "I", "SIGI", mean_intensity_method="isotropic",
                                                output_columns=["FW-I", "FW-SIGI", "F", "SIGF"])
    ds = ds[["I", "SIGI", "F", "SIGF"]].unstack_anomalous()
    ds.write_mtz(out)
    print(f"{out}: {len(ds)} ASU reflections, {ds[['F(+)', 'F(-)']].notna().all(axis=1).sum()} with both mates")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hkl")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--cell", nargs=6, type=float, required=True)
    ap.add_argument("--sg", required=True)
    a = ap.parse_args()
    x = read_hkl(a.hkl)
    to_mtz(x[:, :3].astype(int), x[:, 3], x[:, 4], gemmi.UnitCell(*a.cell), gemmi.SpaceGroup(a.sg), a.out)


if __name__ == "__main__":
    main()
