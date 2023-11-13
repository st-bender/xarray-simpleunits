# -*- coding: utf-8 -*-
# vim:fileencoding=utf-8
import numpy as np
import xarray as xr

import pytest

from astropy import units as au

from xarray_simpleunits import init_units


@pytest.fixture(autouse=True, scope="module")
def init_u():
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


def test_ds():
    ds = _prep_ds()
    d = ds * 2.0
    np.testing.assert_allclose(d.s.values, [2, 4, 6])
    np.testing.assert_allclose(d.t.values, [6, 4, 2])
    assert d.s.units == "m"
    assert d.t.units == "s"
    dp = ds / 2.0
    np.testing.assert_allclose(dp.s.values, [0.5, 1, 1.5])
    np.testing.assert_allclose(dp.t.values, [1.5, 1, 0.5])
    assert dp.s.units == "m"
    assert dp.t.units == "s"
    st = ds + ds.mean("x")
    assert st.s.units == "m"
    assert st.t.units == "s"
    stp = ds * ds.mean("x")
    assert stp.s.units == "m2"
    assert stp.t.units == "s2"
    stpp = ds / ds.mean("x")
    assert stpp.s.units == ""
    assert stpp.t.units == ""
    v1 = ds / (1. * au.Unit("h"))
    assert v1.s.units == "m / h"
    assert v1.t.units == "s / h"
    # conversion to units of first Dataset
    sp = ds[["s"]] + (6378 * au.Unit("km"))
    np.testing.assert_allclose(sp.s.values, [6378001, 6378002, 6378003])
    assert sp.s.units == au.Unit("m")


def test_reset():
    from xarray_simpleunits import reset_units
    reset_units()
    ds = _prep_ds()
    stpp = ds / ds.mean("x")
    # no unit handling, the originals are kept
    assert stpp.s.units == "m"
    assert stpp.t.units == "s"
    # astropy raises ValueError because the first part
    # does not carry (astropy) units.
    with pytest.raises((ValueError, au.UnitsError)):
        sp = ds[["s"]] + (6378 * au.Unit("km"))
    # Double reset does nothing
    reset_units()
    # Re-init
    init_units()
    # conversion to units of first Dataset
    sp = ds[["s"]] + (6378 * au.Unit("km"))
    np.testing.assert_allclose(sp.s.values, [6378001, 6378002, 6378003])
    assert sp.s.units == au.Unit("m")


def test_to_si():
    ds = _prep_ds()
    ds["v"] = ("x", [3.6, 7.2, 10.8], {"units": "km / h"})
    assert ds.v.units == "km / h"
    vp = ds.v.to_si_units()
    np.testing.assert_allclose(vp.values, [1, 2, 3])
    assert vp.units == "m / s"
