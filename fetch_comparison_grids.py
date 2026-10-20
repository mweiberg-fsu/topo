#!/usr/bin/env python3
"""Fetch other topography/bathymetry datasets for the Great Lakes box and resample each onto
the exact ETOPO1 1-arc-min grid (grid-registered nodes, lon -94..-74, lat 40..50, 1201x601).

Needs internet (run locally, not on mars); copy the output npz next to
make_great_lakes_topo.py, which embeds it in index.html.

    python3 fetch_comparison_grids.py [--raw-dir DIR]

Resampling: sources finer than 1' are block-averaged over the 1'x1' cell centred on each
ETOPO1 node (source samples binned to the nearest node); coarser or offset-registered 1'/2'
sources are bilinearly interpolated to the nodes. Output values are int16 metres relative to
mean sea level, FILL where a source has no data.
"""
import argparse
import gzip
import json
import os
import shutil
import urllib.request
import zipfile

import numpy as np

LON0, LAT0, NX, NY = -94.0, 40.0, 1201, 601
D = 1 / 60.0
TGT_LON = LON0 + np.arange(NX) * D
TGT_LAT = LAT0 + np.arange(NY) * D
FILL = -32768
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(OUT_DIR, "great_lakes_compare_grids.npz")

# low water datum of each NOAA Great Lakes grid (m, IGLD85), from each file's history attr
GL_LAKES = {"superior": 183.2, "michigan": 176.0, "huron": 176.0, "erie": 173.5, "ontario": 74.2}

DATASETS = {
    "etopo2022": dict(
        name="ETOPO 2022 (60″ surface)",
        info="NOAA NCEI ETOPO 2022 v1, 60 arc-sec, ice surface. Cell-centred (½′ offset "
             "from ETOPO1 nodes), bilinearly interpolated."),
    "gebco2020": dict(
        name="GEBCO 2020 (15″)",
        info="GEBCO_2020 grid via NOAA CoastWatch ERDDAP, 15 arc-sec, cell-centred; "
             "4×4 block mean per 1′ cell."),
    "srtm15": dict(
        name="SRTM15+ V2.7 (15″)",
        info="Tozer et al. SRTM15+ V2.7 (topex.ucsd.edu), 15 arc-sec, cell-centred; "
             "4×4 block mean per 1′ cell."),
    "glbathy": dict(
        name="NOAA Great Lakes bathymetry (3″)",
        info="NOAA NCEI Great Lakes bathymetry, 3 arc-sec, grid-registered; depths re "
             "low water datum converted to IGLD85 (Superior 183.2, Mich/Huron 176.0, "
             "Erie 173.5, Ontario 74.2 m). Lake-specific boxes incl. nearby land; "
             "blank elsewhere. Block mean per 1′ cell; overlaps averaged."),
    "gmted2010": dict(
        name="GMTED2010 (15″ mean)",
        info="USGS GMTED2010 mean statistic, 15 arc-sec. Land elevation only: lakes are "
             "flat water surface, no bathymetry. Block mean per 1′ cell."),
    "etopo2": dict(
        name="ETOPO2v2c (2′)",
        info="NOAA NGDC ETOPO2v2c (2006), 2 arc-min, cell-centred; bilinearly interpolated."),
}


def fetch(url, path):
    if not os.path.exists(path):
        print("downloading", url)
        tmp = path + ".part"
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, 2**20)
        os.replace(tmp, path)
    return path


def index_range(coords, lo, hi):
    """Slice of (ascending or descending) 1-D coords covering [lo, hi]."""
    idx = np.where((coords >= lo) & (coords <= hi))[0]
    return slice(idx[0], idx[-1] + 1)


class BlockMean:
    """Accumulates source samples into the 1' cell around the nearest ETOPO1 node."""

    def __init__(self):
        self.s = np.zeros(NX * NY)
        self.n = np.zeros(NX * NY)

    def add(self, lon, lat, z):
        z = np.asarray(z, float)
        ix = np.floor((np.asarray(lon) - LON0) / D + 0.5 + 1e-9).astype(int)
        iy = np.floor((np.asarray(lat) - LAT0) / D + 0.5 + 1e-9).astype(int)
        okx, oky = (ix >= 0) & (ix < NX), (iy >= 0) & (iy < NY)
        z = z[np.ix_(oky, okx)]
        cell = (iy[oky][:, None] * NX + ix[okx][None, :]).ravel()
        v = z.ravel()
        good = np.isfinite(v)
        self.s += np.bincount(cell[good], v[good], NX * NY)
        self.n += np.bincount(cell[good], minlength=NX * NY)

    def result(self):
        with np.errstate(invalid="ignore"):
            return (self.s / self.n).reshape(NY, NX)


def bilinear(lon, lat, z):
    from scipy.interpolate import RegularGridInterpolator
    if lat[0] > lat[-1]:
        lat, z = lat[::-1], z[::-1]
    f = RegularGridInterpolator((lat, lon), z, bounds_error=False, fill_value=np.nan)
    la, lo = np.meshgrid(TGT_LAT, TGT_LON, indexing="ij")
    return f(np.column_stack([la.ravel(), lo.ravel()])).reshape(NY, NX)


# margins: one source cell beyond the outermost 1' target cell
W, E, S, N = LON0 - D, LON0 + (NX - 1) * D + D, LAT0 - D, LAT0 + (NY - 1) * D + D


def get_etopo2022(raw):
    import netCDF4
    url = ("https://www.ngdc.noaa.gov/thredds/dodsC/global/ETOPO2022/60s/"
           "60s_surface_elev_netcdf/ETOPO_2022_v1_60s_N90W180_surface.nc")
    with netCDF4.Dataset(url) as d:
        lat, lon = d["lat"][:], d["lon"][:]
        sy, sx = index_range(lat, S - D, N + D), index_range(lon, W - D, E + D)
        z = np.ma.filled(d["z"][sy, sx].astype(float), np.nan)
    return bilinear(np.asarray(lon[sx]), np.asarray(lat[sy]), z)


def get_erddap_15s(raw, dataset_id, var):
    import netCDF4
    path = os.path.join(raw, f"{dataset_id}_greatlakes.nc")
    fetch(f"https://coastwatch.pfeg.noaa.gov/erddap/griddap/{dataset_id}.nc?"
          f"{var}[({S}):1:({N})][({W}):1:({E})]", path)
    with netCDF4.Dataset(path) as d:
        lat, lon = d["latitude"][:], d["longitude"][:]
        z = np.ma.filled(d[var][:].astype(float), np.nan)
    bm = BlockMean()
    bm.add(lon, lat, z)
    return bm.result()


def get_srtm15(raw):
    path = os.path.join(raw, "SRTM15_V2.7_greatlakes.npz")
    if not os.path.exists(path):
        import fsspec
        import h5py
        print("reading SRTM15_V2.7.nc subset over HTTP (6.5 GB file, range requests)")
        with fsspec.open("https://topex.ucsd.edu/pub/srtm15_plus/SRTM15_V2.7.nc",
                         block_size=2**22).open() as f, h5py.File(f, "r") as h:
            lat, lon = h["lat"][:], h["lon"][:]
            sy, sx = index_range(lat, S, N), index_range(lon, W, E)
            np.savez_compressed(path, lon=lon[sx], lat=lat[sy], z=h["z"][sy, sx])
    d = np.load(path)
    bm = BlockMean()
    bm.add(d["lon"], d["lat"], d["z"])
    return bm.result()


def get_glbathy(raw):
    import netCDF4
    bm = BlockMean()
    for lake, datum in GL_LAKES.items():
        gz = fetch(f"https://www.ngdc.noaa.gov/mgg/greatlakes/{lake}/data/netcdf/{lake}_lld.grd.gz",
                   os.path.join(raw, f"{lake}_lld.grd.gz"))
        grd = gz[:-3]
        if not os.path.exists(grd):
            with gzip.open(gz) as fi, open(grd, "wb") as fo:
                shutil.copyfileobj(fi, fo, 2**20)
        with netCDF4.Dataset(grd) as d:
            lon, lat = d["x"][:], d["y"][:]
            z = np.ma.filled(d["z"][:].astype(float), np.nan) + datum
        print(f"  {lake}: {z.shape}, datum {datum} m")
        bm.add(lon, lat, z)
    return bm.result()


def get_gmted(raw):
    import tifffile
    bm = BlockMean()
    base = ("https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/topo/downloads/GMTED/"
            "Global_tiles_GMTED/150darcsec/mea/")
    for tile, folder in (("30N120W", "W120"), ("30N090W", "W090")):
        name = f"{tile}_20101117_gmted_mea150.tif"
        with tifffile.TiffFile(fetch(base + f"{folder}/{name}", os.path.join(raw, name))) as t:
            p = t.pages[0]
            dx, dy = p.tags["ModelPixelScaleTag"].value[:2]
            x0, y0 = p.tags["ModelTiepointTag"].value[3:5]  # upper-left corner (PixelIsArea)
            nodata = float(p.tags["GDAL_NODATA"].value) if "GDAL_NODATA" in p.tags else -32768
            z = p.asarray().astype(float)
            z[z == nodata] = np.nan
        lon = x0 + (np.arange(z.shape[1]) + 0.5) * dx
        lat = y0 - (np.arange(z.shape[0]) + 0.5) * dy
        sy, sx = index_range(lat, S, N), index_range(lon, W, E)
        bm.add(lon[sx], lat[sy], z[sy, sx])
    return bm.result()


def get_etopo2(raw):
    import netCDF4
    nc = os.path.join(raw, "ETOPO2v2c_f4.nc")
    if not os.path.exists(nc):
        zp = fetch("https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2/ETOPO2v2-2006/"
                   "ETOPO2v2c/netCDF/ETOPO2v2c_f4_netCDF.zip",
                   os.path.join(raw, "ETOPO2v2c_f4_netCDF.zip"))
        with zipfile.ZipFile(zp) as z:
            z.extract("ETOPO2v2c_f4.nc", raw)
    with netCDF4.Dataset(nc) as d:
        # node_offset=1: x/y actual_range are cell edges; stored x/y are centres
        lon, lat = d["x"][:].astype(float), d["y"][:].astype(float)
        sy, sx = index_range(lat, S - 2 * D, N + 2 * D), index_range(lon, W - 2 * D, E + 2 * D)
        z = np.ma.filled(d["z"][sy, sx].astype(float), np.nan)
    return bilinear(lon[sx], lat[sy], z)


GETTERS = {
    "etopo2022": get_etopo2022,
    "gebco2020": lambda raw: get_erddap_15s(raw, "GEBCO_2020", "elevation"),
    "srtm15": get_srtm15,
    "glbathy": get_glbathy,
    "gmted2010": get_gmted,
    "etopo2": get_etopo2,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=os.path.join(OUT_DIR, "compare_raw"),
                    help="download cache (~1 GB)")
    ap.add_argument("--only", nargs="*", help="refresh only these keys, keep the rest")
    a = ap.parse_args()
    os.makedirs(a.raw_dir, exist_ok=True)
    grids = {}
    if os.path.exists(OUT):
        with np.load(OUT) as old:
            grids = {k: old[k] for k in old.files if k in DATASETS}
    for key, get in GETTERS.items():
        if a.only and key not in a.only and key in grids:
            continue
        print(f"== {key}: {DATASETS[key]['name']}")
        z = get(a.raw_dir)
        ok = np.isfinite(z)
        print(f"   valid {ok.mean() * 100:.1f}%  range {np.nanmin(z):.0f} .. {np.nanmax(z):.0f} m")
        grids[key] = np.where(ok, np.round(z), FILL).astype(np.int16)
    meta = {k: DATASETS[k] for k in GETTERS if k in grids}
    np.savez_compressed(OUT, meta=json.dumps(meta), lon=TGT_LON, lat=TGT_LAT, **grids)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
