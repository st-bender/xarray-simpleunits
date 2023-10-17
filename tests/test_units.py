# -*- coding: utf-8 -*-
# vim:fileencoding=utf-8
import numpy as np
import xarray as xr

import pytest

from astropy import units as au

from xarray_simpleunits import init_units

init_units()


def _prep_ds():
    ds = xr.Dataset(
        data_vars={
            "s": ("x", [1.0, 2.0, 3.0], {"units": "m"}),
            "t": ("x", [3.0, 2.0, 1.0], {"units": "s"}),
        },
    )
    return ds


def test_add():
    ds = _prep_ds()
    v = ds["s"] + ds["s"]
    assert v.units == au.Unit("m")
    vp = ds["s"] + (2.0 * au.Unit("m"))
    assert vp.units == au.Unit("m")
    # conversion to units of first array
    vpp = ds["s"] + (6378 * au.Unit("km"))
    np.testing.assert_allclose(vpp.values, [6378001, 6378002, 6378003])
    assert vpp.units == au.Unit("m")
    with pytest.raises(ValueError):
        vp = ds["s"] + ds["t"]
    with pytest.raises(ValueError):
        vp = ds["s"] + (10.0 * au.Unit("s"))


def test_sub():
    ds = _prep_ds()
    v = ds["t"] - ds["t"]
    assert v.units == au.Unit("s")
    vp = ds["t"] - (5.0 * au.Unit("s"))
    assert vp.units == au.Unit("s")
    # conversion to units of first array
    vpp = ds["s"] - (10. * au.Unit("mm"))
    np.testing.assert_allclose(vpp.values, [0.99, 1.99, 2.99])
    assert vpp.units == au.Unit("m")
    with pytest.raises(ValueError):
        vp = ds["t"] - ds["s"]
    with pytest.raises(ValueError):
        vp = ds["t"] - (10.0 * au.Unit("m"))


def test_mul():
    ds = _prep_ds()
    v = ds["s"] * ds["t"]
    assert v.units == au.Unit("m * s")
    vp = ds["s"] * au.Unit("s")
    assert vp.units == au.Unit("m s")
    tp = 2 * au.Unit("s")
    vpp = ds["s"] * ds["t"] * tp
    assert vpp.units == au.Unit("m s2")
    vppp = ds["s"] * ds["t"] * au.Unit("kg / s2")
    assert vppp.units == au.Unit("kg m / s")


def test_div():
    ds = _prep_ds()
    v = ds["s"] / ds["t"]
    np.testing.assert_allclose(v.values, [1. / 3., 1., 3.])
    assert v.units == au.Unit("m / s")
    vpp = ds["s"] / au.Unit("s")
    assert vpp.units == au.Unit("m / s")
    tp = 2 * au.Unit("s")
    vp = ds["s"] / ds["t"] / tp
    np.testing.assert_allclose(vp.values, [1. / 6., 0.5, 1.5])
    assert vp.units == au.Unit("m / s2")
    vppp = ds["s"] / ds["t"] / au.Unit("s / kg")
    assert vppp.units == au.Unit("N")


def test_lin():
    ds = _prep_ds()
    v = ds["s"] + ds["t"] * (7.2 * au.Unit("km / h"))
    np.testing.assert_allclose(v.values, [7, 6, 5])
    assert v.units == au.Unit("m")


def test_lin_si():
    ds = _prep_ds()
    _si = getattr(xr.DataArray, "__keep_si__", False)
    setattr(xr.DataArray, "__keep_si__", True)
    v = ds["t"] * (7.2 * au.Unit("km / h"))
    np.testing.assert_allclose(v.values, [6, 4, 2])
    assert v.units == au.Unit("m")
    if not _si:
        # clean up
        delattr(xr.DataArray, "__keep_si__")


def test_to_u():
    ds = _prep_ds()
    v = ds["s"].to_unit("mm")
    np.testing.assert_allclose(v.values, [1000., 2000., 3000.])
    assert v.units == au.Unit("mm")
    vp = ds["t"].to_unit("ns")
    np.testing.assert_allclose(vp.values, [3e9, 2e9, 1e9])
    assert vp.units == au.Unit("ns")
    with pytest.raises(ValueError):
        vpp = ds["s"].to_unit("s")
