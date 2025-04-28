# pyright: reportUnknownMemberType=false
from __future__ import annotations

import functools
import importlib.metadata
import logging
import math
import os
import subprocess
import typing as T
from collections import abc

import bokeh.events
import cartopy.crs as ccrs
import geopandas as gpd
import geoviews as gv
import geoviews.tile_sources as gvts
import holoviews as hv
import logfmter
import numpy as np
import panel as pn
import pyogrio  # pyright: ignore[reportMissingTypeStubs]
import pyproj
import shapely

__all__ = [
    "calc_area_and_perimeter",
    "GDF",
    "hist",
    "measure_distance",
    "PLOT",
    "read_file",
    "setup",
    "show",
    "to_file",
    "to_points_df",
    "WGS84",
    "WGS84_GEOD",
    "__version__",
]

__version__ = importlib.metadata.version(__name__)

WGS84 = pyproj.CRS(projparams=("epsg", 4326))
WGS84_GEOD = pyproj.Geod(ellps="WGS84")

if T.TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt
    from holoviews.plotting.bokeh.plot import BokehPlot

    NPArray: T.TypeAlias = npt.NDArray[np.float64]

    assert isinstance(ccrs.GOOGLE_MERCATOR, ccrs.Mercator)  # type hints BS


def to_file(gdf: gpd.GeoDataFrame, filename: str | os.PathLike[str], **kwargs: T.Any):
    if "layer_metadata" in kwargs or "metadata" in kwargs:
        raise ValueError("You can't pass layer metadata. They are inferred from the dataframe's attrs")
    attrs = {k: str(v) for k, v in gdf.attrs.items()}
    kwargs["layer_metadata"] = attrs
    gdf.to_file(filename, **kwargs)


def read_file(filename: str | os.PathLike[str], **kwargs: T.Any):
    info = T.cast(dict[str, T.Any], pyogrio.read_info(filename, **kwargs))
    gdf = gpd.read_file(filename, **kwargs)
    gdf.attrs.update(info["layer_metadata"] or {})
    return gdf


def setup(include_logging: bool = True):
    if include_logging:
        logging.getLogger("asyncio").setLevel(logging.INFO)
        logging.getLogger("bokeh").setLevel(logging.WARNING)
        logging.getLogger("tornado").setLevel(logging.WARNING)
        formatter = logfmter.Logfmter(
            keys=["lvl", "at", "filename", "lineno", ],
            mapping={"lvl": "levelname", "at": "asctime"},
            #datefmt="%Y-%m-%d"
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logging.basicConfig(level=logging.DEBUG, handlers=[handler])
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


def _kill_existing_server(server_id: str):
    if server_id in pn.state._threads:
        pn.state._threads[server_id].stop()
    else:
        try:
            pn.state._servers[server_id][0].stop()
        except AssertionError:  # can't stop a server twice
            pass
    pn.state._servers.pop(server_id, None)
    pn.state._threads.pop(server_id, None)


def show(*objs: T.Any, threaded: bool = True, port: int = 0, **kwargs: T.Any) -> None:
    if port != 0:
        for server_id, server in pn.state._servers.items():
            if server[0].port == port:
                _kill_existing_server(server_id)
                break
    _ = pn.serve(panels=pn.Column(*objs), threaded=threaded, port=port, **kwargs)


def _plot_gdf(
    gdf: gpd.GeoDataFrame,
    *,
    crs: pyproj.CRS = ccrs.GOOGLE_MERCATOR,
    alpha: float = 0.5,
    size: float = 8,
    per_row: bool = False,
    **kwargs: T.Any,
) -> list[hv.Element]:
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
    alpha: float = 0.25,
    size: float = 8,
    tiles: str = "OSM",
    per_row: bool = False,
    port: int = 0,
    **kwargs: T.Any,
) -> None:
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
    overlay = T.cast(hv.Overlay, hv.Overlay(plots).opts(hooks=[measure_distance], projection=crs))
    show(overlay, port=port)

@T.overload
def to_points_df(
    geo: gpd.GeoDataFrame,
    crs: int,
    *,
    src: str,
    include_indices: bool,
    geod: pyproj.Geod
) -> gpd.GeoDataFrame: ...
@T.overload
def to_points_df(
    geo: shapely.Polygon,
    crs: int,
    *,
    src: str,
    include_indices: bool,
    geod: pyproj.Geod
) -> gpd.GeoDataFrame: ...
@T.overload
def to_points_df(
    geo: NPArray,
    crs: int,
    *,
    src: str,
    include_indices: bool,
    geod: pyproj.Geod,
) -> gpd.GeoDataFrame: ...
def to_points_df(
    geo: shapely.Polygon | NPArray | gpd.GeoDataFrame,
    crs: int,
    *,
    src: str = "",
    include_angles: bool = True,
    include_distances: bool = True,
    include_indices: bool = False,
    geod: pyproj.Geod = WGS84_GEOD,
):
    if isinstance(geo, shapely.Polygon):
        coords = shapely.get_coordinates(geo.normalize())
    elif isinstance(geo, gpd.GeoDataFrame):
        if len(geo) > 1:
            raise ValueError("Too many rows. I can only handle 1")
        coords = shapely.get_coordinates(geo.iloc[0].geometry)
    elif not np.array_equal(geo[0], geo[-1]):
        coords = np.r_[geo, geo[:1]]
    else:
        coords = geo
    no_coords = len(coords)

    repeated = coords
    non_repeated = coords[:-1]

    # Convert to lon/lat
    transformer = pyproj.Transformer.from_crs(
        crs,
        pyproj.CRS(4326),
        always_xy=True,
        only_best=True,
    )
    lons, lats = transformer.transform(repeated[:, 0], repeated[:, 1])

    data: dict[str, str | NPArray] = {
        "lons": lons,
        "lats": lats,
    }

    if include_distances:
        # cartesian distances
        diff = np.diff(repeated, axis=0)
        dist = np.hypot(diff[:, 0], diff[:, 1])

        # geod distances
        ellps = geod.line_lengths(lons, lats)

        # repeat first point
        cartesians = np.concat((dist, dist[[0]]), axis=0)
        ellipsoids = np.concat((ellps, ellps[[0]]), axis=0)

        data.update({
            "cartesian": cartesians,
            "ellipsoid": ellipsoids,
        })

    # Angles
    if include_angles:
        doublepi: float = 2 * np.pi
        v1 = non_repeated - np.concat((non_repeated[-1:], non_repeated[:-1]), axis=0)
        v2 = non_repeated - np.concat((non_repeated[1:], non_repeated[:1]), axis=0)
        angles = (np.arctan2(v1[:, 1], v1[:, 0]) - np.arctan2(v2[:, 1], v2[:, 0])) % doublepi

        # repeat first point
        angles = np.concat((angles, angles[[0]]), axis=0)

        data.update({
            "angle": angles,
            "angle_deg": np.rad2deg(angles),
        })

    if src:
        data["src"] = src

    # indexes
    if include_indices:
        index_prev = np.concat(([no_coords - 2], range(no_coords - 2), [no_coords - 2]), axis=0)
        index_main = np.concat((np.arange(no_coords - 1), [0]), axis=0)
        index_next = np.concat((range(1, len(coords) - 1), [0, 1]), axis=0)
        index_next2 = np.concat((range(2, len(coords) - 1), [0, 1, 2]), axis=0)
        data.update({
            "index_prev": index_prev,
            "index_main": index_main,
            "index_next": index_next,
            "index_next2": index_next2,
        })

    gdf = gpd.GeoDataFrame(
        data=data,
        geometry=shapely.get_parts(shapely.multipoints(coords)),
        crs=crs,
    ).reset_index()

    return gdf


@functools.cache
def _get_transformer(from_crs) -> pyproj.Transformer:
    transformer = pyproj.Transformer.from_crs(
        from_crs,
        pyproj.CRS(4326),
        always_xy=True,
        allow_ballpark=True,
        only_best=True,
    )
    return transformer


def measure_distance(plot: BokehPlot, _: hv.Element) -> None:

    def dist(event: bokeh.events.Tap):
        # type hints BS
        assert event.model is not None
        assert event.x is not None
        assert event.y is not None
        assert isinstance(plot.projection, ccrs.CRS)
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
        transformer = _get_transformer(plot.projection)
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
    show(hv.Histogram(np.histogram(array, bins)[::-1]))
