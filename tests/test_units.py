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
    assert v.units == au.Unit("m / s")
    vpp = ds["s"] / au.Unit("s")
    assert vpp.units == au.Unit("m / s")
    tp = 2 * au.Unit("s")
    vp = ds["s"] / ds["t"] / tp
    assert vp.units == au.Unit("m / s2")
    vppp = ds["s"] / ds["t"] / au.Unit("s / kg")
    assert vppp.units == au.Unit("N")


def test_lin():
    ds = _prep_ds()
    v = ds["s"] + ds["t"] * (7.2 * au.Unit("km / h"))
    np.testing.assert_allclose(v.values, [7, 6, 5])
    assert v.units == au.Unit("m")
