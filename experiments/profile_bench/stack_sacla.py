"""SACLA Cheetah multi-event HDF5 (one /tag-N group per hit) -> a stacked (event, ss, fs) view that
GLINT's --images route reads, without copying a pixel.

The view is an HDF5 virtual dataset: event i of /entry_1/data_1/data maps onto /tag-N/data of the
source file, so the 4.4 GB gzip-compressed run stays the only copy. A float32 photon-unit copy
would be ~2.3x larger, ~560 GB for CXIDB 33 Dataset A. The pixels stay in ADU; CXIDB 33 is 10 ADU
per photon, and ``glint --integrate --adu-per-photon 10`` converts them before integration.

    python stack_sacla.py RAW.h5 OUT.h5

Writes /entry_1/data_1/data (n, ss, fs) virtual int16, /LCLS/photon_energy_eV (n,), /LCLS/tag (n,).
The source is referenced by absolute path, so the view survives being read from anywhere.

KNOWN PROBLEM -- do not run the workflow on these views as they are. Opened with h5py's default
chunk cache, HDF5 keeps every source frame it has read open with its decompressed chunk:
memory grows ~14 MB per frame read, ~17 GB for a 1217-frame run. It took 8 peak-finding jobs to
82 GB on 2026-10-27. Opening with rdcc_nbytes=0 stops it (+0.06 GB over 150 frames, against
+2.1 GB), but GLINT opens the images itself. Either write real lzf-compressed int16 stacks
(~6.8 GB a run) or give GLINT's image reads a small chunk cache. See HANDOFF.md.
"""
import argparse
import os

import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("out")
    a = ap.parse_args()
    src = os.path.abspath(a.raw)
    with h5py.File(src, "r") as f:
        tags = sorted((k for k in f if k.startswith("tag-")), key=lambda k: int(k[4:]))
        d0 = f[tags[0]]["data"]
        shape, dtype = d0.shape, d0.dtype
        eV = np.array([f[t]["photon_energy_ev"][()] for t in tags])
    lay = h5py.VirtualLayout(shape=(len(tags),) + shape, dtype=dtype)
    for i, t in enumerate(tags):
        lay[i] = h5py.VirtualSource(src, f"/{t}/data", shape=shape)
    with h5py.File(a.out + ".part", "w") as g:
        g.create_virtual_dataset("/entry_1/data_1/data", lay, fillvalue=0)
        g["/LCLS/photon_energy_eV"] = eV
        g["/LCLS/tag"] = np.array([int(t[4:]) for t in tags])
        g.attrs["source"] = src
    os.replace(a.out + ".part", a.out)
    print(f"{a.out}: {len(tags)} events -> {src}, {eV.mean():.1f} eV mean")


if __name__ == "__main__":
    main()
