"""Step 0. Download everything from the BC Data Catalogue into one bundle file.

Run this on your own computer: python src/fetch_data.py
It takes a few minutes and writes data/raw/bc_wildfire_bundle.json (~9 MB).

Sources, all from openmaps.gov.bc.ca under the Open Government Licence - BC:
  PROT_HISTORICAL_INCIDENTS_SP    every fire the BC Wildfire Service responded to
  PROT_HISTORICAL_FIRE_POLYS_SP   burned-area perimeters by year
  PROT_WEATHER_STATIONS_SP        BCWS fire weather stations (candidate base sites)
  BC_MAJOR_CITIES_POINTS_500M     towns and their populations
  BEC_BIOGEOCLIMATIC_POLY         biogeoclimatic zones (the ecosystem type)
  DRA_DGTL_ROAD_ATLAS_MPAR_SP     roads

Points come from the WFS service as CSV. Polygon and line layers are huge, so
instead of downloading their shapes we ask the WMS service to draw them as a
1 km-per-pixel image with a colour per category, then count pixels per 10 km
cell. Antialiasing is switched off so every pixel is exactly one category.

Note: the prototype's bundle was fetched with the same requests from a browser.
This Python version follows the same steps but was written without access to
the live server, so check its printed totals against the ones in the README.
"""
import io
import json
from datetime import datetime, timezone

import numpy as np
import requests
from PIL import Image

from config import RAW, GRID, NX, NY, FIRST_YEAR

WFS = "https://openmaps.gov.bc.ca/geo/pub/wfs"
WMS = "https://openmaps.gov.bc.ca/geo/pub/wms"
ZONES = ["BAFA", "BG", "BWBS", "CDF", "CMA", "CWH", "ESSF", "ICH", "IDF",
         "IMA", "MH", "MS", "PP", "SBPS", "SBS", "SWB"]
PX_PER_CELL = GRID["cell_m"] // 1000              # 10 pixels of 1 km per cell
W_PX, H_PX = NX * PX_PER_CELL, NY * PX_PER_CELL   # 1640 x 1400
S = requests.Session()


def wfs_csv(layer, props=None, cql=None, page=10000):
    base = dict(service="WFS", version="2.0.0", request="GetFeature",
                typeName=f"pub:{layer}", outputFormat="csv")
    if cql:
        base["CQL_FILTER"] = cql
    hits = S.get(WFS, params={**base, "resultType": "hits"}, timeout=120).text
    n = int(hits.split('numberMatched="')[1].split('"')[0])
    lines = []
    for start in range(0, n, page):
        p = {**base, "count": page, "startIndex": start, "sortBy": "OBJECTID"}
        if props:
            p["propertyName"] = props
        rows = S.get(WFS, params=p, timeout=300).text.strip().splitlines()
        lines += rows if start == 0 else rows[1:]
    print(f"  {layer}: {n:,} features")
    return "\n".join(lines)


def sld(layer, rules):
    return ('<?xml version="1.0" encoding="UTF-8"?><StyledLayerDescriptor version="1.0.0" '
            'xmlns="http://www.opengis.net/sld" xmlns:ogc="http://www.opengis.net/ogc">'
            f'<NamedLayer><Name>{layer}</Name><UserStyle><FeatureTypeStyle>{rules}'
            '</FeatureTypeStyle></UserStyle></NamedLayer></StyledLayerDescriptor>')


def eq_rule(prop, value, fill):
    return (f'<Rule><ogc:Filter><ogc:PropertyIsEqualTo><ogc:PropertyName>{prop}</ogc:PropertyName>'
            f'<ogc:Literal>{value}</ogc:Literal></ogc:PropertyIsEqualTo></ogc:Filter>'
            f'<PolygonSymbolizer><Fill><CssParameter name="fill">{fill}</CssParameter></Fill>'
            '</PolygonSymbolizer></Rule>')


def render(layer, style):
    """Draw a layer over the whole grid at 1 km per pixel. Returns an RGBA array."""
    params = dict(service="WMS", version="1.1.1", request="GetMap", layers=f"pub:{layer}",
                  srs="EPSG:3005", bbox=f"{GRID['x0']},{GRID['y0']},{GRID['x1']},{GRID['y1']}",
                  width=W_PX, height=H_PX, format="image/png", transparent="true",
                  format_options="antialias:none", SLD_BODY=style)
    r = S.post(WMS, data=params, timeout=600)
    r.raise_for_status()
    return np.asarray(Image.open(io.BytesIO(r.content)).convert("RGBA"))


def per_cell(mask):
    """Share of each 10 km cell covered by a boolean 1 km pixel mask."""
    m = mask.reshape(NY, PX_PER_CELL, NX, PX_PER_CELL).mean(axis=(1, 3))
    return m.ravel()


def sparse(a):
    return [[int(i), round(float(v), 3)] for i, v in enumerate(a) if v > 0]


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    print("points (WFS):")
    fires = wfs_csv("WHSE_LAND_AND_NATURAL_RESOURCE.PROT_HISTORICAL_INCIDENTS_SP",
                    props="FIRE_NUMBER,FIRE_YEAR,RESPONSE_TYPE_DESC,IGNITION_DATE,FIRE_OUT_DATE,"
                          "FIRE_CAUSE,FIRE_CENTRE,ZONE,FIRE_TYPE,LATITUDE,LONGITUDE,CURRENT_SIZE,OBJECTID",
                    cql=f"FIRE_YEAR>={FIRST_YEAR}")
    # drop the FID column the server adds, to match the bundle format
    fires = "\n".join(",".join(l.split(",")[1:]) for l in fires.splitlines())
    stations = wfs_csv("WHSE_LAND_AND_NATURAL_RESOURCE.PROT_WEATHER_STATIONS_SP")
    cities = wfs_csv("WHSE_BASEMAPPING.BC_MAJOR_CITIES_POINTS_500M")

    print("zones (WMS):")
    bec_layer = "WHSE_FOREST_VEGETATION.BEC_BIOGEOCLIMATIC_POLY"
    # zone i is painted with red = 16 + 10 * i so it can be read back exactly
    rules = "".join(eq_rule("ZONE", z, f"#{16 + 10 * i:02x}0000") for i, z in enumerate(ZONES))
    img = render(bec_layer, sld(f"pub:{bec_layer}", rules))
    solid = img[..., 3] >= 128
    zi = np.rint((img[..., 0].astype(int) - 16) / 10).astype(int)
    bec = []
    fr = np.stack([per_cell(solid & (zi == i)) for i in range(len(ZONES))], axis=1)
    for c in np.flatnonzero(fr.sum(axis=1) > 0):
        bec.append([int(c), [[int(z), round(float(v), 3)] for z, v in enumerate(fr[c]) if v > 0]])

    print("burned area by year (WMS):")
    fire_layer = "WHSE_LAND_AND_NATURAL_RESOURCE.PROT_HISTORICAL_FIRE_POLYS_SP"
    burned = {}
    for y in range(FIRST_YEAR, datetime.now().year):
        img = render(fire_layer, sld(f"pub:{fire_layer}", eq_rule("FIRE_YEAR", y, "#000000")))
        burned[str(y)] = sparse(per_cell(img[..., 3] >= 128))
        print(f"  {y}: {int((img[..., 3] >= 128).sum()):,} km2")

    print("roads (WMS, slow):")
    road_layer = "WHSE_BASEMAPPING.DRA_DGTL_ROAD_ATLAS_MPAR_SP"
    line = ('<Rule><LineSymbolizer><Stroke><CssParameter name="stroke">#000000</CssParameter>'
            '<CssParameter name="stroke-width">1</CssParameter></Stroke></LineSymbolizer></Rule>')
    img = render(road_layer, sld(f"pub:{road_layer}", line))
    roads = sparse(per_cell(img[..., 3] >= 128))

    bundle = dict(
        meta=dict(source="BC Data Catalogue via openmaps.gov.bc.ca WFS/WMS, Open Government Licence - British Columbia",
                  fetched=datetime.now(timezone.utc).isoformat(),
                  grid=dict(**GRID, crs="EPSG:3005", nx=NX, ny=NY), zones=ZONES),
        bec=bec, burned=burned, roads_dra=roads,
        fires_csv=fires, stations_csv=stations, cities_csv=cities)
    print("province outline (Natural Earth):")
    import geopandas as gpd
    ne = gpd.read_file("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
                       "ne_10m_admin_1_states_provinces.geojson")
    bc = ne[(ne["iso_a2"] == "CA") & (ne["name"] == "British Columbia")][["name", "geometry"]].to_crs(3005)
    bc["geometry"] = bc.geometry.simplify(200)
    bc.to_crs(4326).to_file(RAW / "bc_outline.geojson", driver="GeoJSON")

    out = RAW / "bc_wildfire_bundle.json"
    out.write_text(json.dumps(bundle))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
