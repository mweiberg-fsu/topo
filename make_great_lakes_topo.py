#!/usr/bin/env python3
"""Great Lakes ETOPO1 topography: static PNG figure + interactive Leaflet HTML.

Run on mars:
    HDF5_USE_FILE_LOCKING=FALSE python3 make_great_lakes_topo.py
(file locking must be off or the NFS-mounted netCDF4/HDF5 file fails with "NetCDF: HDF error")

Comparison datasets in index.html come from great_lakes_compare_grids.npz, made by
fetch_comparison_grids.py on a host with internet; copy it (and the shoreline JSON) here first.
"""
import base64
import gzip
import json
import os
import urllib.request

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import xarray as xr

ETOPO = "/Net/samosproc/data/auxiliary/landmask/ETOPO1_Ice_grid.nc"
LON_MIN, LON_MAX = -94.0, -74.0
LAT_MIN, LAT_MAX = 40.0, 50.0
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# GSHHG 2.3.7 full-resolution shorelines (L1 ocean coast, L2 lakes, L3 islands in lakes,
# L4 ponds on those islands). Cartopy's built-in GSHHS URL is dead, so fetch from the NCEI
# archive. The clipped result is cached in SHORE_JSON, so hosts without internet (mars)
# only need that small file copied next to the script.
GSHHG_URL = ("https://www.ncei.noaa.gov/data/oceans/archive/arc0234/0304143/1.1/data/"
             "0-data/GSHHS_shp/f/")
GSHHG_DIR = os.path.join(OUT_DIR, "gshhg_f")
SHORE_JSON = os.path.join(OUT_DIR, "great_lakes_shorelines_gshhg_f.json")
MIN_LAKE_KM2 = 1.0  # drop lakes/islands smaller than this

# other datasets resampled to this ETOPO1 grid by fetch_comparison_grids.py (needs internet)
COMPARE_NPZ = os.path.join(OUT_DIR, "great_lakes_compare_grids.npz")

# lake surface elevations (m above sea level, approx. long-term mean / IGLD85)
LAKES = {
    "Superior": (47.7, -87.5, 183),
    "Michigan": (44.0, -87.0, 176),
    "Huron": (44.8, -82.4, 176),
    "Erie": (42.2, -81.2, 174),
    "Ontario": (43.7, -77.9, 75),
}

# shared colormap stops: (elevation m, hex) -- same in PNG and HTML
CMAP_STOPS = [
    (-250, "#08306b"), (-100, "#2171b5"), (-0.5, "#9ecae1"),
    (0, "#1a9850"), (100, "#66bd63"), (200, "#d9ef8b"),
    (300, "#fee08b"), (450, "#fdae61"), (600, "#d73027"),
    (900, "#7f3b08"),
]


def load_subset():
    ds = xr.open_dataset(ETOPO)
    lon = ds.longitude.values
    lat = ds.latitude.values
    # grid-registered 1-arcmin; use index ranges to avoid float32 coord mismatch
    ix = np.where((lon >= LON_MIN - 1e-3) & (lon <= LON_MAX + 1e-3))[0]
    iy = np.where((lat >= LAT_MIN - 1e-3) & (lat <= LAT_MAX + 1e-3))[0]
    z = ds.topography[iy[0]:iy[-1] + 1, ix[0]:ix[-1] + 1].values.astype(np.int16)
    # recompute exact coords (stored float32 values are rounded)
    lon_s = np.round(lon[ix[0]] * 60) / 60 + np.arange(z.shape[1]) / 60.0
    lat_s = np.round(lat[iy[0]] * 60) / 60 + np.arange(z.shape[0]) / 60.0
    ds.close()
    return lon_s, lat_s, z


def load_shorelines():
    """GSHHG full-res shorelines clipped to the box: {"coast": [lines], "lake": [lines]},
    each line a list of [lon, lat]. Lake shores include L3 islands and L4 ponds."""
    if os.path.exists(SHORE_JSON):
        with open(SHORE_JSON) as f:
            return json.load(f)
    import shapefile  # pyshp, a cartopy dependency
    from shapely.geometry import box, shape

    clip = box(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)
    os.makedirs(GSHHG_DIR, exist_ok=True)
    out = {"source": "GSHHG 2.3.7 full resolution (Wessel & Smith), clipped "
                     f"{LON_MIN}..{LON_MAX} E, {LAT_MIN}..{LAT_MAX} N",
           "coast": [], "lake": []}
    for level in (1, 2, 3, 4):
        base = os.path.join(GSHHG_DIR, f"GSHHS_f_L{level}")
        for ext in ("shp", "shx", "dbf"):
            if not os.path.exists(f"{base}.{ext}"):
                print("downloading", f"GSHHS_f_L{level}.{ext}")
                urllib.request.urlretrieve(f"{GSHHG_URL}GSHHS_f_L{level}.{ext}", f"{base}.{ext}")
        with shapefile.Reader(base) as r:
            area_i = [fld[0] for fld in r.fields[1:]].index("area")
            for sr in r.iterShapeRecords(bbox=clip.bounds):
                if level > 1 and abs(sr.record[area_i]) < MIN_LAKE_KM2:
                    continue
                g = shape(sr.shape.__geo_interface__)
                if not g.intersects(clip):
                    continue
                # clip the ring, not the polygon, so box edges don't become fake shoreline
                cut = g.boundary.intersection(clip)
                for ln in getattr(cut, "geoms", [cut]):
                    if ln.geom_type == "LineString" and len(ln.coords) > 1:
                        out["coast" if level == 1 else "lake"].append(
                            [[round(x, 5), round(y, 5)] for x, y in ln.coords])
    with open(SHORE_JSON, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print("wrote", SHORE_JSON, {k: len(out[k]) for k in ("coast", "lake")})
    return out


def make_cmap():
    lo, hi = CMAP_STOPS[0][0], CMAP_STOPS[-1][0]
    pos = [(v - lo) / (hi - lo) for v, _ in CMAP_STOPS]
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "etopo_gl", list(zip(pos, [c for _, c in CMAP_STOPS])))
    return cmap, mcolors.Normalize(lo, hi)


def lake_stats(lon, lat, z):
    """Per-lake ETOPO1 stats over grid cells inside Natural Earth 10m lake polygons."""
    import cartopy.io.shapereader as shpreader
    from matplotlib.path import Path

    try:
        reader = shpreader.Reader(shpreader.natural_earth("10m", "physical", "lakes"))
    except Exception as e:
        print("lake polygons unavailable:", e)
        return {}
    lon2, lat2 = np.meshgrid(lon, lat)
    pts = np.column_stack([lon2.ravel(), lat2.ravel()])
    stats = {}
    for rec in reader.records():
        nm = rec.attributes.get("name") or ""
        key = nm.replace("Lake ", "")
        if key not in LAKES:
            continue
        geoms = getattr(rec.geometry, "geoms", [rec.geometry])
        inside = np.zeros(len(pts), bool)
        for g in geoms:
            inside |= Path(np.asarray(g.exterior.coords)[:, :2]).contains_points(pts)
            for hole in g.interiors:  # islands
                inside &= ~Path(np.asarray(hole.coords)[:, :2]).contains_points(pts)
        v = z.ravel()[inside]
        if v.size:
            stats[key] = dict(n=int(v.size), min=int(v.min()), mean=float(v.mean()),
                              max=int(v.max()), frac_le0=float((v <= 0).mean()))
    print(f"{'lake':10s}{'surf':>6s}{'cells':>8s}{'min':>7s}{'mean':>8s}{'max':>6s}{'%<=0':>7s}")
    for k, (_, _, surf) in LAKES.items():
        if k in stats:
            d = stats[k]
            print(f"{k:10s}{surf:6d}{d['n']:8d}{d['min']:7d}{d['mean']:8.1f}{d['max']:6d}"
                  f"{100 * d['frac_le0']:7.1f}")
    return stats


def make_figure(lon, lat, z, stats, shores):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from shapely.geometry import MultiLineString

    cmap, norm = make_cmap()
    proj = ccrs.LambertConformal(central_longitude=-84, central_latitude=45,
                                 standard_parallels=(42, 48))
    fig = plt.figure(figsize=(13, 8.5), dpi=150)
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent([-93, -75, 40.5, 49.6], crs=ccrs.PlateCarree())

    pm = ax.pcolormesh(lon, lat, z, cmap=cmap, norm=norm, shading="nearest",
                       transform=ccrs.PlateCarree(), rasterized=True)
    # 0 m contour: where the ETOPO1 value changes sign
    cs = ax.contour(lon, lat, z, levels=[0], colors="k", linewidths=0.6,
                    transform=ccrs.PlateCarree())

    if shores:
        for kind, color, lw in (("lake", "#0b2a5b", 0.6), ("coast", "k", 0.8)):
            if shores[kind]:
                ax.add_geometries([MultiLineString(shores[kind])], ccrs.PlateCarree(),
                                  facecolor="none", edgecolor=color, linewidth=lw, zorder=3)
    try:
        if not shores:
            ax.add_feature(cfeature.LAKES.with_scale("10m"), facecolor="none",
                           edgecolor="0.15", linewidth=0.7)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.6,
                       linestyle="--", edgecolor="0.3")
        ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.3,
                       edgecolor="0.5")
    except Exception as e:  # no Natural Earth cache / no internet on host
        print("Natural Earth features skipped:", e)

    gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False,
                      linewidth=0.3, color="0.4", alpha=0.6)
    gl.top_labels = gl.right_labels = False

    for name, (la, lo, surf) in LAKES.items():
        st = stats.get(name)
        txt = f"{name}\nsurface ≈{surf} m"
        if st:
            txt += f"\nmin {st['min']} m"
        ax.text(lo, la, txt,
                transform=ccrs.PlateCarree(), ha="center", va="center",
                fontsize=8, color="k",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75))

    cb = fig.colorbar(pm, ax=ax, orientation="vertical", shrink=0.8, pad=0.02,
                      extend="both")
    cb.set_label("ETOPO1 (Ice) topography [m relative to sea level]")
    ax.set_title("ETOPO1_Ice_grid.nc — Great Lakes region (1 arc-min)\n"
                 "black line = 0 m contour; navy = GSHHG full-res lake/coast shorelines; "
                 "lake beds stored as elevation (surface − depth)",
                 fontsize=11)
    out = os.path.join(OUT_DIR, "great_lakes_etopo1.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Great Lakes Topography</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html, body { margin: 0; height: 100%; font: 13px/1.35 system-ui, sans-serif; background: #fff; }
  #map { position: absolute; inset: 0; }
  .panel { background: rgba(255,255,255,.93); padding: 8px 10px; border-radius: 6px;
           box-shadow: 0 1px 5px rgba(0,0,0,.35); max-width: 290px; }
  .panel h3 { margin: 0 0 6px; font-size: 14px; }
  .panel label { display: block; margin: 5px 0 2px; }
  .panel select { max-width: 100%; }
  .panel input[type=number] { width: 70px; }
  .panel .row { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
  #readout { font-family: ui-monospace, monospace; font-size: 12px; white-space: pre; }
  #legend canvas { width: 100%; height: 14px; display: block; border: 1px solid #999; }
  #legend .ticks { display: flex; justify-content: space-between; font-size: 11px; }
  .topo-img { image-rendering: pixelated; image-rendering: crisp-edges; }
  .note { font-size: 11px; color: #444; margin-top: 6px; }
  .tip { position: absolute; z-index: 1000; pointer-events: none; display: none;
         background: rgba(255,255,255,.96); border-radius: 5px; padding: 6px 8px;
         box-shadow: 0 1px 6px rgba(0,0,0,.4); font-size: 12px; white-space: nowrap; }
  .tip .hd { font-family: ui-monospace, monospace; font-size: 11px; color: #333; margin-bottom: 3px; }
  table.cmp { border-collapse: collapse; }
  table.cmp td { padding: 1px 5px; }
  table.cmp td.n { text-align: right; font-family: ui-monospace, monospace; }
  table.cmp tr.cur { background: #fff3b0; font-weight: 600; }
  table.cmp th { font-weight: 600; text-align: right; padding: 1px 5px; border-bottom: 1px solid #ccc; }
  table.cmp th:first-child { text-align: left; }
</style>
</head>
<body>
<div id="map"></div>
<script>
const META = __META__;
const GRIDS = __GRIDS__;  // [{key, name, info, b64}], GRIDS[0] = ETOPO1 (reference)
const STOPS = __STOPS__;
const SHORES = __SHORES__;
const NX = META.nx, NY = META.ny, D = META.dlon, FILL = META.fill;

// int16 little-endian, row 0 = southernmost latitude, gzipped after differencing along rows
async function decodeGrid(b64) {
  const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
  const z = new Int16Array(await new Response(stream).arrayBuffer());
  for (let j = 0; j < NY; j++)  // undo row deltas; Int16Array stores wrap mod 2^16 like the encoder
    for (let i = j * NX + 1, e = (j + 1) * NX; i < e; i++) z[i] += z[i - 1];
  return z;
}

function cell(lat, lon) {
  const i = Math.round((lon - META.lon0) / D), j = Math.round((lat - META.lat0) / D);
  if (i < 0 || j < 0 || i >= NX || j >= NY) return null;
  return { i, j, k: j * NX + i, lon: META.lon0 + i * D, lat: META.lat0 + j * D };
}

function hex2rgb(h) { return [1, 3, 5].map(k => parseInt(h.substr(k, 2), 16)); }
const STOP_RGB = STOPS.map(([v, c]) => [v, hex2rgb(c)]);
function rampStops(stops, v) {
  if (v <= stops[0][0]) return stops[0][1];
  for (let k = 1; k < stops.length; k++) {
    if (v <= stops[k][0]) {
      const [v0, c0] = stops[k - 1], [v1, c1] = stops[k];
      const f = (v - v0) / (v1 - v0 || 1);
      return c0.map((c, n) => c + f * (c1[n] - c));
    }
  }
  return stops[stops.length - 1][1];
}
// map value -> rgb honoring user min/max but keeping 0 at the blue/green break
function colorFor(v, vmin, vmax) {
  const lo = STOP_RGB[0][0], hi = STOP_RGB[STOP_RGB.length - 1][0];
  let s;
  if (v < 0) s = vmin < 0 ? lo * Math.min(v / vmin, 1) : -0.5;
  else s = vmax > 0 ? hi * Math.min(v / vmax, 1) : 0;
  return rampStops(STOP_RGB, s);
}
const DIV_RGB = [[-1, "#2166ac"], [-0.5, "#67a9cf"], [0, "#f7f7f7"], [0.5, "#ef8a62"], [1, "#b2182b"]]
  .map(([v, c]) => [v, hex2rgb(c)]);
const diffColor = (d, r) => rampStops(DIV_RGB, Math.max(-1, Math.min(1, d / r)));

const state = { ds: 0, mode: "elev", vmin: META.zmin, vmax: META.zmax, thr: 0, drange: 25,
                opacity: 0.85 };

// render into Web-Mercator-spaced rows so the overlay is geometrically exact
const latN = META.lat0 + (NY - 0.5) * D, latS = META.lat0 - 0.5 * D;
const lonW = META.lon0 - 0.5 * D, lonE = META.lon0 + (NX - 0.5) * D;
const merc = la => Math.log(Math.tan(Math.PI / 4 + la * Math.PI / 360));
const imerc = y => (2 * Math.atan(Math.exp(y)) - Math.PI / 2) * 180 / Math.PI;
const H = Math.round(NY * 1.25);
const canvas = document.createElement("canvas");
canvas.width = NX; canvas.height = H;
const ctx = canvas.getContext("2d");
const rowJ = new Int32Array(H);
for (let r = 0; r < H; r++) {
  const y = merc(latN) - (r + 0.5) / H * (merc(latN) - merc(latS));
  rowJ[r] = Math.min(NY - 1, Math.max(0, Math.round((imerc(y) - META.lat0) / D)));
}

function render() {
  const Z = GRIDS[state.ds].z, Z0 = GRIDS[0].z;
  const img = ctx.createImageData(NX, H), px = img.data;
  for (let r = 0; r < H; r++) {
    const base = rowJ[r] * NX;
    for (let i = 0; i < NX; i++) {
      const v = Z[base + i], p = (r * NX + i) * 4;
      if (v === FILL) continue;  // no data: leave transparent
      let c;
      if (state.mode === "elev") c = colorFor(v, state.vmin, state.vmax);
      else if (state.mode === "mask") c = v <= state.thr ? [33, 113, 181] : [237, 201, 120];
      else {
        const v0 = Z0[base + i];
        if (v0 === FILL) continue;
        c = diffColor(v - v0, state.drange);
      }
      px[p] = c[0]; px[p + 1] = c[1]; px[p + 2] = c[2]; px[p + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  overlay.setUrl(canvas.toDataURL());
  drawLegend();
}

const map = L.map("map", { zoomSnap: 0.25, preferCanvas: true }).fitBounds([[41, -92.5], [49.3, -75.5]]);
const bases = {
  "Esri World Imagery": L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    { attribution: "Tiles &copy; Esri", maxZoom: 18 }),
  "OpenStreetMap": L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    { attribution: "&copy; OpenStreetMap contributors", maxZoom: 18 }),
  "Carto Light": L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
    { attribution: "&copy; OpenStreetMap &copy; CARTO", maxZoom: 18 }),
};
bases["Carto Light"].addTo(map);
const overlay = L.imageOverlay("data:,", [[latS, lonW], [latN, lonE]],
  { opacity: state.opacity, className: "topo-img" }).addTo(map);
const labels = L.tileLayer("https://{s}.basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}{r}.png",
  { pane: "shadowPane", maxZoom: 18 });
map.createPane("shores").style.zIndex = 450;  // above overlay, below popups
const shoreLayer = (lines, style) => L.polyline(
  lines.map(l => l.map(([x, y]) => [y, x])),
  { pane: "shores", interactive: false, ...style });
const lakes = shoreLayer(SHORES.lake, { color: "#0b2a5b", weight: 1 }).addTo(map);
const coast = shoreLayer(SHORES.coast, { color: "#000", weight: 1.3 }).addTo(map);
L.control.layers(bases, { "Topography overlay": overlay, "Lake shorelines (GSHHG f)": lakes,
  "Ocean coastline (GSHHG f)": coast, "Place labels": labels }).addTo(map);
L.control.scale().addTo(map);

// control panel
const Panel = L.Control.extend({
  onAdd() {
    const d = L.DomUtil.create("div", "panel");
    d.innerHTML = `
      <h3>Great Lakes topography — dataset comparison</h3>
      <div id="readout">loading grids…</div>
      <label>Dataset</label>
      <select id="ds">${GRIDS.map((g, n) => `<option value="${n}">${g.name}</option>`).join("")}</select>
      <div class="note" id="dsinfo"></div>
      <label>Display</label>
      <select id="mode">
        <option value="elev">Elevation (colour ramp)</option>
        <option value="mask">Threshold mask (≤ thr = blue)</option>
        <option value="diff">Difference vs ${GRIDS[0].name}</option>
      </select>
      <div id="elevCtl">
        <div class="row">min <input type="number" id="vmin" value="${state.vmin}">
                         max <input type="number" id="vmax" value="${state.vmax}"></div>
      </div>
      <div id="diffCtl" style="display:none">
        <div class="row">± range <input type="number" id="drange" value="${state.drange}" min="1"> m</div>
      </div>
      <div id="legend"><canvas width="256" height="1"></canvas>
        <div class="ticks"><span id="lgmin"></span><span id="lgmid"></span><span id="lgmax"></span></div></div>
      <div id="maskCtl" style="display:none">
        <div class="row">threshold <input type="number" id="thr" value="0"> m</div>
        <div class="row">preset:
          <button data-t="0">0 (sea level)</button><button data-t="75">75 (Ontario)</button>
          <button data-t="176">176 (Mich/Huron)</button><button data-t="183">183 (Superior)</button></div>
      </div>
      <label>Overlay opacity <span id="opv">${state.opacity}</span></label>
      <input type="range" id="op" min="0" max="1" step="0.05" value="${state.opacity}" style="width:100%">
      <div class="note">All datasets resampled to the ETOPO1 grid: ${NX}×${NY}, 1 arc-min,
        grid-registered. Hover a cell to compare every dataset; click to pin.<br>
        Lake beds are stored as elevation (lake surface − depth), so most lake floor is &gt; 0 m.<br>
        Shorelines: GSHHG 2.3.7 full resolution (lakes ≥ ${META.min_lake_km2} km²).</div>`;
    L.DomEvent.disableClickPropagation(d);
    L.DomEvent.disableScrollPropagation(d);
    L.DomEvent.on(d, "mousemove", L.DomEvent.stopPropagation);
    L.DomEvent.on(d, "mouseenter", () => { tip.style.display = "none"; });
    return d;
  },
});
new Panel({ position: "topright" }).addTo(map);

function drawLegend() {
  const lg = document.getElementById("legend");
  lg.style.display = state.mode === "mask" ? "none" : "";
  const lc = lg.querySelector("canvas"), lx = lc.getContext("2d");
  const im = lx.createImageData(256, 1);
  const diff = state.mode === "diff";
  const lo = diff ? -state.drange : state.vmin, hi = diff ? state.drange : state.vmax;
  for (let k = 0; k < 256; k++) {
    const v = lo + (k / 255) * (hi - lo);
    const c = diff ? diffColor(v, state.drange) : colorFor(v, state.vmin, state.vmax);
    im.data.set([c[0], c[1], c[2], 255], k * 4);
  }
  lx.putImageData(im, 0, 0);
  $("lgmin").textContent = lo;
  $("lgmid").textContent = diff ? "0" : "";
  $("lgmax").textContent = diff ? `+${hi} m` : hi;
}

const $ = id => document.getElementById(id);
const showCtl = () => {
  $("elevCtl").style.display = state.mode === "elev" ? "" : "none";
  $("maskCtl").style.display = state.mode === "mask" ? "" : "none";
  $("diffCtl").style.display = state.mode === "diff" ? "" : "none";
  $("dsinfo").textContent = GRIDS[state.ds].info;
};
$("ds").onchange = e => { state.ds = +e.target.value; showCtl(); render(); };
$("mode").onchange = e => { state.mode = e.target.value; showCtl(); render(); };
$("vmin").onchange = e => { state.vmin = +e.target.value; render(); };
$("vmax").onchange = e => { state.vmax = +e.target.value; render(); };
$("drange").onchange = e => { state.drange = Math.max(1, +e.target.value); render(); };
$("thr").onchange = e => { state.thr = +e.target.value; render(); };
document.querySelectorAll("#maskCtl button").forEach(b => b.onclick = () => {
  state.thr = +b.dataset.t; $("thr").value = state.thr; render();
});
$("op").oninput = e => { state.opacity = +e.target.value; $("opv").textContent = state.opacity; overlay.setOpacity(state.opacity); };

const ready = () => GRIDS.every(g => g.z);
const val = (g, k) => g.z[k] === FILL ? null : g.z[k];
function fmt(p) {
  const v = val(GRIDS[state.ds], p.k);
  return `lat ${p.lat.toFixed(4)}  lon ${p.lon.toFixed(4)}\nindex j=${p.j + META.j0} i=${p.i + META.i0}\n` +
         `${GRIDS[state.ds].name}: ${v === null ? "no data" : v + " m"}`;
}
function compareTable(p) {
  const v0 = val(GRIDS[0], p.k);
  const rows = GRIDS.map((g, n) => {
    const v = val(g, p.k);
    const d = n === 0 || v === null || v0 === null ? "" : (v - v0 > 0 ? "+" : "") + (v - v0);
    return `<tr class="${n === state.ds ? "cur" : ""}"><td>${g.name}</td>` +
           `<td class="n">${v === null ? "—" : v}</td><td class="n">${d}</td></tr>`;
  }).join("");
  return `<div class="hd">${p.lat.toFixed(4)}°N ${(-p.lon).toFixed(4)}°W</div>` +
         `<table class="cmp"><tr><th>dataset</th><th>m</th><th>Δ vs ETOPO1</th></tr>${rows}</table>`;
}

// hover tooltip comparing all datasets at the cell under the cursor
const tip = L.DomUtil.create("div", "tip", map.getContainer());
map.on("mousemove", e => {
  if (!ready()) return;
  const p = cell(e.latlng.lat, e.latlng.lng);
  $("readout").textContent = p ? fmt(p) : "outside grid";
  if (!p) { tip.style.display = "none"; return; }
  tip.innerHTML = compareTable(p);
  tip.style.display = "block";
  const sz = map.getSize(), w = tip.offsetWidth, h = tip.offsetHeight, o = 16;
  const x = e.containerPoint.x + o + w > sz.x ? e.containerPoint.x - o - w : e.containerPoint.x + o;
  const y = e.containerPoint.y + o + h > sz.y ? e.containerPoint.y - o - h : e.containerPoint.y + o;
  tip.style.left = Math.max(0, x) + "px"; tip.style.top = Math.max(0, y) + "px";
});
map.on("mouseout", () => { tip.style.display = "none"; });
map.on("click", e => {
  const p = ready() && cell(e.latlng.lat, e.latlng.lng);
  if (p) L.popup({ maxWidth: 420 }).setLatLng([p.lat, p.lon]).setContent(compareTable(p)).openOn(map);
});

Promise.all(GRIDS.map(async g => { g.z = await decodeGrid(g.b64); delete g.b64; }))
  .then(() => { $("readout").textContent = "hover over map…"; showCtl(); render(); })
  .catch(err => { $("readout").textContent = "failed to decode grids: " + err; });
</script>
</body>
</html>
"""

FILL = -32768


def encode_grid(z):
    """int16 grid -> base64(gzip(row-differenced int16 LE)); deltas wrap mod 2^16."""
    z = np.ascontiguousarray(z, dtype="<i2")
    d = z.copy()
    d[:, 1:] = z[:, 1:] - z[:, :-1]
    return base64.b64encode(gzip.compress(d.tobytes(), 9, mtime=0)).decode()


def load_compare(lon, lat):
    """Comparison grids from fetch_comparison_grids.py, if present and on this grid."""
    if not os.path.exists(COMPARE_NPZ):
        print("no", COMPARE_NPZ, "- run fetch_comparison_grids.py (needs internet)")
        return []
    with np.load(COMPARE_NPZ) as d:
        if d["lon"].shape != lon.shape or d["lat"].shape != lat.shape or \
                not (np.allclose(d["lon"], lon) and np.allclose(d["lat"], lat)):
            print("comparison grids are on a different grid; skipped")
            return []
        meta = json.loads(str(d["meta"]))
        return [dict(key=k, z=d[k], **meta[k]) for k in meta]


def make_html(lon, lat, z, shores, compare=()):
    meta = dict(nx=int(z.shape[1]), ny=int(z.shape[0]), dlon=1 / 60.0,
                lon0=float(lon[0]), lat0=float(lat[0]),
                i0=int(round((lon[0] + 180) * 60)), j0=int(round((lat[0] + 90) * 60)),
                zmin=int(z.min()), zmax=int(z.max()), min_lake_km2=MIN_LAKE_KM2, fill=FILL)
    grids = [dict(key="etopo1", name="ETOPO1 Ice (1′)", z=z,
                  info="ETOPO1_Ice_grid.nc (NOAA NGDC 2009), 1 arc-min, grid-registered. "
                       "Reference grid for differences.")] + list(compare)
    grids = [dict(key=g["key"], name=g["name"], info=g["info"], b64=encode_grid(g["z"]))
             for g in grids]
    html = (HTML.replace("__META__", json.dumps(meta))
                .replace("__STOPS__", json.dumps(CMAP_STOPS))
                .replace("__SHORES__", json.dumps(
                    {k: (shores or {}).get(k, []) for k in ("coast", "lake")},
                    separators=(",", ":")))
                .replace("__GRIDS__", json.dumps(grids, ensure_ascii=False)))
    out = os.path.join(OUT_DIR, "index.html")
    with open(out, "w") as f:
        f.write(html)
    print("wrote", out, f"({os.path.getsize(out) / 1e6:.1f} MB, {len(grids)} datasets)")

if __name__ == "__main__":
    lon, lat, z = load_subset()
    print("subset", z.shape, "lon", lon[0], lon[-1], "lat", lat[0], lat[-1],
          "z", z.min(), z.max())
    np.savez_compressed(os.path.join(OUT_DIR, "great_lakes_etopo1.npz"), lon=lon, lat=lat, z=z)
    try:
        shores = load_shorelines()
    except Exception as e:  # no cached JSON and no internet
        print("GSHHG shorelines unavailable:", e)
        shores = None
    make_html(lon, lat, z, shores, load_compare(lon, lat))
    make_figure(lon, lat, z, lake_stats(lon, lat, z), shores)
