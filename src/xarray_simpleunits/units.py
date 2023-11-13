# -*- coding: utf-8 -*-
# vim:fileencoding=utf-8
#
# Copyright (c) 2023 Stefan Bender
#
# This module is part of xarray_simpleunits.
# xarray_simpleunits is free software: you can redistribute it or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation, version 2.
# See accompanying LICENSE file or http://www.gnu.org/licenses/gpl-2.0.html.
"""Unitful calculations with xarray

Provides unit conversions and calculations, basically "overloads"
`xarray.DataArray` multiplication and division to keep track of the units.
"""
from __future__ import absolute_import, division, print_function

# %%
import xarray as xr

# %%
from astropy import units as au

__all__ = ["init_units", "reset_units", "to_si_units", "to_unit"]


# %%
def _get_unit(x):
    """Infer unit(s) from attributes
    """
    if hasattr(x, "unit"):
        return x.unit
    elif hasattr(x, "units"):
        return au.Unit(x.units)
    elif hasattr(x, "attrs"):
        return au.Unit(x.attrs.get("units", "1"))
    elif isinstance(x, (au.CompositeUnit, au.IrreducibleUnit, au.Unit)):
        return x
    return au.dimensionless_unscaled


# %%
def _get_values(x):
    """Get values from quantity
    """
    if hasattr(x, "value"):
        return x.value
    return x


# %%
def _convert(a, u):
    try:
        ret = a.to(u)
    except AttributeError:
        try:
            ret = a.to_unit(u)
        except au.UnitConversionError:
            raise
    except au.UnitConversionError:
        raise
    return ret


# %%
def add_u(a, b):
    """Addition with units
    """
    a_u = _get_unit(a)
    try:
        b = _convert(b, a_u)
    except au.UnitConversionError:
        raise ValueError("Unit mismatch in additon.")
    ret = a.__add_orig__(_get_values(b))
    ret.attrs["units"] = str(_get_unit(a))
    return ret


# %%
def sub_u(a, b):
    """Subtraction with units
    """
    a_u = _get_unit(a)
    try:
        b = _convert(b, a_u)
    except au.UnitConversionError:
        raise ValueError("Unit mismatch in subtraction.")
    ret = a.__sub_orig__(_get_values(b))
    ret.attrs["units"] = str(_get_unit(a))
    return ret


# %%
def mul_u(a, b):
    """Multiplication with units
    """
    ret = a.__mul_orig__(b)
    ret_u = _get_unit(a) * _get_unit(b)
    ret.attrs["units"] = ret_u.si
    if getattr(a, "__keep_si__", False):
        _u = (1 * ret_u).si.unit
        ret = ret.to_unit(str(_u))
    return ret


# %%
def truediv_u(a, b):
    """Division with units
    """
    ret = a.__truediv_orig__(b)
    ret_u = _get_unit(a) / _get_unit(b)
    ret.attrs["units"] = ret_u.si
    if getattr(a, "__keep_si__", False):
        _u = (1 * ret_u).si.unit
        ret = ret.to_unit(str(_u))
    return ret


# %%
def add_method(cls):
    def decorator(func):
        # func needs to take `self` as the first argument
        setattr(cls, func.__name__, func)
        return func  # returning func means func can still be used normally
    return decorator


# %%
# @add_method(xr.DataArray)
def to_si_units(a):
    ret_u = _get_unit(a)
    _u = (1 * ret_u).si.unit
    return a.to_unit(str(_u))


# %%
# @add_method(xr.DataArray)
def to_unit(a, u):
    # multiply by conversion factor
    ret = _get_unit(a).to(u) * a
    # update "units" attribute
    ret.attrs["units"] = u
    return ret


# %%
FUNC_MAP = {
    # attribute name: new function
    "__add__": add_u,
    "__sub__": sub_u,
    "__mul__": mul_u,
    "__truediv__": truediv_u,
    "__div__": truediv_u,
}


# %%
def init_units(keep_si=False):
    # Only (re)set SI behaviour if units are already enabled.
    # Otherwise setting the functions will call themselves,
    # resulting in an infinite recursion.
    if getattr(xr.Variable, "__has_units__", False):
        setattr(xr.Variable, "__keep_si__", keep_si)
        return
    for _a, _f in FUNC_MAP.items():
        orig_attr = _a[:-2] + "_orig__"
        orig_func = getattr(xr.Variable, _a, None)
        # save "original" functions as attributes
        setattr(xr.Variable, orig_attr, orig_func)
        # point `Variable` operations to functions with units
        setattr(xr.Variable, _a, _f)
    # Add unit conversion function, needed for both, DataArray and Variable
    setattr(xr.DataArray, "to_unit", to_unit)
    setattr(xr.Variable, "to_unit", to_unit)
    setattr(xr.DataArray, "to_si_units", to_si_units)
    setattr(xr.Variable, "to_si_units", to_si_units)
    # Track conversion to SI units in multiplication and division
    setattr(xr.Variable, "__keep_si__", keep_si)
    # set `xarray` to keep track of attributes, that includes units
    xr.set_options(keep_attrs=True)
    # Mark as initialized.
    xr.DataArray.__has_units__ = True
    xr.Variable.__has_units__ = True


# %%
def reset_units():
    # No units, nothing to reset here.
    if not getattr(xr.Variable, "__has_units__", False):
        return
    for _a, _f in FUNC_MAP.items():
        orig_attr = _a[:-2] + "_orig__"
        orig_func = getattr(xr.Variable, orig_attr, None)
        setattr(xr.Variable, _a, orig_func)
        delattr(xr.Variable, orig_attr)
        if orig_func is None:
            delattr(xr.Variable, _a)
    # Delete unit conversion function from attributes
    delattr(xr.DataArray, "to_unit")
    delattr(xr.Variable, "to_unit")
    delattr(xr.DataArray, "to_si_units")
    delattr(xr.Variable, "to_si_units")
    # Delete SI conversion attribute
    delattr(xr.Variable, "__keep_si__")
    # Delete unit indicator attributes
    delattr(xr.DataArray, "__has_units__")
    delattr(xr.Variable, "__has_units__")
