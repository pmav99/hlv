# pyright: basic
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
import shapely

from hlv import GDF

BOX = shapely.box(0, 0, 1, 1)
BOX_GDF = gpd.GeoDataFrame(geometry=[BOX], crs=4326)

def test_gdf_numpy_coords():
    expected = BOX_GDF
    coords = shapely.get_coordinates(BOX)
    gdf = GDF(coords, crs=4326)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert gdf.equals(expected)


def test_gdf_numpy_coords_without_crs_raises():
    coords = shapely.get_coordinates(BOX)
    with pytest.raises(ValueError) as exc:
        GDF(coords)
    msg = "CRS must be specified when converting a numpy array to a GeoDataFrame."
    assert msg in str(exc)


def test_gdf_shapely_geometry():
    expected = BOX_GDF
    gdf = GDF(BOX, crs=4326)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert gdf.equals(expected)


def test_gdf_shapely_geometry_without_crs_raises():
    with pytest.raises(ValueError) as exc:
        GDF(BOX)
    msg = "CRS must be specified when converting a shapely geometry to a GeoDataFrame."
    assert msg in str(exc)


def test_gdf_sequence_of_shapely_geometriies():
    expected = gpd.GeoDataFrame(geometry=[BOX, BOX], crs=4326)
    gdf = GDF([BOX, BOX], crs=4326)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert gdf.equals(expected)


def test_gdf_numpy_arrays_of_shapely_geometriies():
    expected = gpd.GeoDataFrame(geometry=[BOX, BOX], crs=4326)
    gdf = GDF(np.array([BOX, BOX]), crs=4326)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert gdf.equals(expected)


def test_gdf_sequence_of_shapely_geometries_without_crs_raises():
    with pytest.raises(ValueError) as exc:
        GDF([BOX, BOX])
    msg = "CRS must be specified when converting a sequence of shapely geometries to a GeoDataFrame."
    assert msg in str(exc)


def test_gdf_geodataframe_passthrough():
    expected = BOX_GDF
    gdf = GDF(BOX_GDF)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert gdf.equals(expected)


def test_gdf_geodataframe_without_crs_raises():
    gdf = gpd.GeoDataFrame(geometry=[BOX])
    assert gdf.crs is None
    with pytest.raises(ValueError) as exc:
        GDF(gdf)
    msg = "CRS must be specified. Please set it using `.set_crs()`."
    assert msg in str(exc)


def test_gdf_geoseries_converts_to_geodataframe():
    expected = BOX_GDF
    orig = gpd.GeoSeries(data=BOX, crs=4326)
    gdf = GDF(orig)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert gdf.equals(expected)


def test_gdf_geoseries_without_crs_raises():
    geo = gpd.GeoSeries(data=BOX)
    with pytest.raises(ValueError) as exc:
        GDF(geo)
    msg = "CRS must be specified. Please set it using `.set_crs()`."
    assert msg in str(exc)


def test_gdf_other_types_raises():
    with pytest.raises(ValueError) as exc:
        GDF([1, 2, 3])  # pyright: ignore[reportCallIssue,reportArgumentType]
    msg = "Unsupported type provided for conversion."
    assert msg in str(exc)
