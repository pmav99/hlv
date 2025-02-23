# pyright: reportUnknownMemberType=false
from __future__ import annotations

import functools
import importlib
import logging
import math
import subprocess
import typing as T
from collections import abc

import cartopy.crs as ccrs
import logfmter
import numpy as np
import pyproj
import shapely

__all__ = [
    "calc_area_and_perimeter",
    "GDF",
    "hist",
    "measure_distance",
    "PLOT",
    "setup",
    "show",
    "WGS84",
    "WGS84_GEOD",
]


WGS84 = pyproj.CRS(projparams=("epsg", 4326))
WGS84_GEOD = pyproj.Geod(ellps="WGS84")

if T.TYPE_CHECKING:  # pragma: no cover
    import bokeh.events
    import geopandas as gpd
    import geoviews as gv  # pyright: ignore[reportUnusedImport]
    import holoviews as hv
    import numpy.typing as npt
    import panel as pn  # pyright: ignore[reportUnusedImport]
    from holoviews.plotting.bokeh.plot import BokehPlot

    NPArray: T.TypeAlias = npt.NDArray[np.float64]

    assert isinstance(ccrs.GOOGLE_MERCATOR, ccrs.Mercator)  # type hints BS


def setup(include_logging: bool = True):
    if include_logging:
        logging.getLogger("asyncio").setLevel(logging.INFO)
        logging.getLogger("bokeh").setLevel(logging.WARNING)
        logging.getLogger("tornado").setLevel(logging.WARNING)
        formatter = logfmter.Logfmter(
            keys=["lvl", "at", "filename", "lineno", ],
            mapping={"lvl": "levelname", "at": "asctime"},
            datefmt="%Y-%m-%d"
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logging.basicConfig(level=logging.DEBUG, handlers=[handler])
    hv = importlib.import_module("holoviews")
    pn = importlib.import_module("panel")
    _ = hv.extension("bokeh")
    _ = pn.extension(throttled=True, inline=True, ready_notification="Ready")
    hv.opts.defaults(
        hv.opts.Polygons(
            tools=["hover", "crosshair", "undo"],
            active_tools=["wheel_zoom", "pan"],
            responsive=True,
            show_grid=False,
            show_legend=False,
        ),
        hv.opts.Path(
            tools=["hover", "crosshair", "undo"],
            active_tools=["wheel_zoom", "pan"],
            responsive=True,
            show_grid=False,
            show_legend=False,
        ),
        hv.opts.Points(
            tools=["hover", "crosshair", "undo"],
            active_tools=["wheel_zoom", "pan"],
            responsive=True,
            show_grid=False,
            show_legend=False,
        ),
        hv.opts.Histogram(
            tools=["hover", "crosshair", "undo"],
            active_tools=["wheel_zoom", "pan"],
            responsive=True,
            show_grid=True,
            show_legend=False,
        ),
    )


def is_geometry_collection(val: abc.Collection[object]) -> T.TypeGuard[abc.Collection[shapely.Geometry]]:
    """Determines whether all objects in the Collection are shapely Geometries"""
    return all(isinstance(x, shapely.Geometry) for x in val)


@T.overload
def GDF(geo: abc.Sequence[shapely.Geometry], crs: T.Any) -> gpd.GeoDataFrame: ...
@T.overload
def GDF(geo: abc.Sequence[shapely.Geometry], crs: None = None) -> gpd.GeoDataFrame: ...
@T.overload
def GDF(geo: shapely.Geometry, crs: T.Any) -> gpd.GeoDataFrame: ...
@T.overload
def GDF(geo: shapely.Geometry, crs: None = None) -> gpd.GeoDataFrame: ...
@T.overload
def GDF(geo: NPArray, crs: T.Any) -> gpd.GeoDataFrame: ...
@T.overload
def GDF(geo: NPArray, crs: None = None) -> gpd.GeoDataFrame: ...
@T.overload
def GDF(geo: gpd.GeoDataFrame, crs: None = None) -> gpd.GeoDataFrame: ...
@T.overload
def GDF(geo: gpd.GeoSeries, crs: None = None) -> gpd.GeoDataFrame: ...
def GDF(  # noqa: N802
    geo: NPArray | shapely.Geometry | abc.Sequence[shapely.Geometry] | gpd.GeoDataFrame | gpd.GeoSeries,
    crs: T.Any | None = None,
) -> gpd.GeoDataFrame:
    gpd = importlib.import_module("geopandas")
    match geo:
        # Geoseries and GeoDataFrames are collections themselves, therefore they must come before
        # the check for `abc.Collection`
        #
        # GeoSeries
        case gpd.GeoSeries() if geo.crs is None:
            raise ValueError("CRS must be specified. Please set it using `.set_crs()`.")
        case gpd.GeoSeries():
            gdf = T.cast(gpd.GeoDataFrame, geo.to_frame(name="geometry"))
        # GeoDataFrame
        case gpd.GeoDataFrame() if geo.crs is None:
            raise ValueError("CRS must be specified. Please set it using `.set_crs()`.")
        case gpd.GeoDataFrame():
            gdf = geo
        # Collection[shapely.Geometry]
        case abc.Collection() if is_geometry_collection(geo) and crs is None:
            raise ValueError("CRS must be specified when converting a sequence of shapely geometries to a GeoDataFrame.")
        case abc.Collection() if is_geometry_collection(geo):
            gdf = gpd.GeoDataFrame(geometry=geo, crs=crs)
        # shapely.Geometry
        case shapely.Geometry() if crs is None:
            raise ValueError("CRS must be specified when converting a shapely geometry to a GeoDataFrame.")
        case shapely.Geometry():
            gdf = gpd.GeoDataFrame(geometry=[geo], crs=crs)
        # Numpy
        case np.ndarray() if crs is None:
            raise ValueError("CRS must be specified when converting a numpy array to a GeoDataFrame.")
        case np.ndarray():
            gdf = gpd.GeoDataFrame(geometry=[shapely.Polygon(geo)], crs=crs)
        # Other
        case _:
            raise ValueError("Unsupported type provided for conversion.")
    return gdf


def calc_area_and_perimeter(
    gdf: gpd.GeoDataFrame,
    geod: pyproj.Geod = WGS84_GEOD,
    *,
    sort: bool = False,
) -> gpd.GeoDataFrame:
    area: NPArray
    perimeter: NPArray
    assert gdf.crs is not None, "CRS must be defined!"
    geometry_area_perimeter = np.vectorize(geod.geometry_area_perimeter)  # pyright: ignore[reportUnknownArgumentType]
    area, perimeter = geometry_area_perimeter(gdf.to_crs(WGS84).geometry.exterior.normalize())
    area *= -1
    gdf = gdf.assign(  # pyright:ignore[reportAssignmentType]
        wgs84_area=area,
        wgs84_perimeter=perimeter,
        no_coords=gdf.count_coordinates() - 1,
    )
    if sort:
        gdf = gdf.sort_values("no_coords", ascending=False, ignore_index=True)  # pyright:ignore[reportAssignmentType]
    return gdf


def show(*objs: T.Any, threaded: bool = True, **kwargs: T.Any) -> None:
    pn = importlib.import_module("panel")
    _ = pn.serve(panels=pn.Column(*objs), threaded=threaded, **kwargs)


def _plot_gdf(
    gdf: gpd.GeoDataFrame,
    *,
    crs: pyproj.CRS = ccrs.GOOGLE_MERCATOR,
    alpha: float = 0.5,
    size: float = 8,
    per_row: bool = False,
    **kwargs: T.Any,
) -> list[hv.Element]:
    # lazy imports
    hv = importlib.import_module("holoviews")  # pyright: ignore[reportUnusedVariable]
    gv = importlib.import_module("geoviews")

    plots: list[hv.Element] = []
    polygons = gdf[gdf.geom_type.isin(["Polygon", "MultiPolygon"])]
    points = gdf[gdf.geom_type.isin(["Point", "MultiPoint"])]
    lines = gdf[gdf.geom_type.isin(["LineString", "LinearRing"])]
    if per_row:
        polygons = polygons.reset_index()
        points = points.reset_index()
        lines = lines.reset_index()
    if not polygons.empty:
        plot = T.cast(hv.Element, gv.Polygons(polygons, crs=crs).opts(alpha=alpha, **kwargs))
        plots.append(plot)
    if not points.empty:
        plot = T.cast(hv.Element, gv.Points(points, crs=crs).opts(size=size, **kwargs))
        plots.append(plot)
    if not lines.empty:
        plot = T.cast(hv.Element, gv.Path(lines, crs=crs).opts(**kwargs))
        plots.append(plot)
    if per_row:
        plots = [T.cast(hv.Element, p.opts(color="index")) for p in plots]
    return plots


def PLOT(  # noqa: N802
    *geos: gpd.GeoDataFrame | gpd.GeoSeries | hv.Element,
    crs: pyproj.CRS = ccrs.GOOGLE_MERCATOR,
    alpha: float = 0.5,
    size: float = 8,
    tiles: str = "OSM",
    per_row: bool = False,
    **kwargs: T.Any,
) -> None:
    gpd = importlib.import_module("geopandas")
    gvts = importlib.import_module("geoviews.tile_sources")
    hv = importlib.import_module("holoviews")
    plots: list[hv.Element | hv.Layout]
    if crs is ccrs.GOOGLE_MERCATOR:
        plots = [gvts.tile_sources[tiles]]
    else:
        plots = []
    for geo in geos:
        match geo:
            case gpd.GeoDataFrame() | gpd.GeoSeries():
                gdf = GDF(geo).to_crs(crs)
                plots.extend(
                    _plot_gdf(gdf, crs=crs, alpha=alpha, size=size, per_row=per_row, **kwargs),
                )
            case hv.Element():
                plots.append(
                    T.cast(hv.Element, geo.opts(**kwargs)),
                )
            case _:
                raise ValueError(f"Unsupported type provided for plotting: {type(geo)}.")
    overlay = T.cast(hv.Overlay, hv.Overlay(plots).opts(hooks=[measure_distance]))
    show(overlay)


@functools.cache
def _get_transformer() -> pyproj.Transformer:
    transformer = pyproj.Transformer.from_crs(
        pyproj.CRS(3857),
        pyproj.CRS(4326),
        always_xy=True,
        allow_ballpark=True,
        only_best=True,
    )
    return transformer


def measure_distance(plot: BokehPlot, _: hv.Element) -> None:
    import panel as pn
    import bokeh.events

    def dist(event: bokeh.events.Tap):
        # type hints BS
        assert event.model is not None
        assert event.x is not None
        assert event.y is not None
        cache = T.cast(dict[T.Any, tuple[float, float]], pn.state.cache)
        # Get the unique ID of the plot. We will use it as the cache key
        model_id = event.model.id
        # Retrieve coords of previously clicked point, the current point and calculate
        # euclidean distance (depending on the projection this value may be way off from reality)
        px, py = cache.get(model_id, (0.0, 0.0))
        cx, cy = event.x, event.y
        distance_cart = math.sqrt((cx - px)**2 + (cy - py)**2)
        # Store current point to cache
        cache[model_id] = (cx, cy)
        # Transform coords to EPSG 4326 and calculdate distance on GEOID
        #transformer = _get_transformer(plot.projection)
        transformer = _get_transformer()
        plon, plat = transformer.transform(px, py)
        clon, clat = transformer.transform(cx, cy)
        _, _, distance_ellps = WGS84_GEOD.inv(plon, plat, clon, clat)
        # Show notification
        msg = f"x: {cx}\ny: {cy}\n\nlon: {clon}\nlat: {clat}\n\nCartesian: {distance_cart:.2f}\nEllipsoid: {distance_ellps:.2f}"
        print(msg)  # noqa: T201
        #_ = pn.state.notifications.info(msg, duration=15000)  #-pyright: ignore[reportOptionalMemberAccess]
        _ = subprocess.run(["ntfy", "send", msg], check=True, shell=False)  # noqa: S603,S607

    # Register callback
    plot.state.on_event(bokeh.events.Tap, dist)


def hist(array: NPArray, bins: int = 20) -> None:
    hv = importlib.import_module("holoviews")
    show(hv.Histogram(np.histogram(array, bins)[::-1]))
