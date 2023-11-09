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

__all__ = ["init_units", "to_unit"]


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
def to_unit(a, u):
    # multiply by conversion factor
    ret = _get_unit(a).to(u) * a
    # update "units" attribute
    ret.attrs["units"] = u
    return ret


# %%
def init_units(keep_si=False):
    # Only (re)set SI behaviour if units are already enabled.
    # Otherwise setting the functions will call themselves,
    # resulting in an infinite recursion.
    if getattr(xr.Variable, "__has_units__", False):
        setattr(xr.Variable, "__keep_si__", keep_si)
        return
    # save "original" functions
    xr.Variable.__add_orig__ = xr.Variable.__add__
    xr.Variable.__sub_orig__ = xr.Variable.__sub__
    xr.Variable.__mul_orig__ = xr.Variable.__mul__
    xr.Variable.__truediv_orig__ = xr.Variable.__truediv__
    if hasattr(xr.Variable, "__div__"):  # Py2
        xr.Variable.__div_orig__ = xr.Variable.__div__
    # point DataArray operations to our functions
    xr.Variable.__add__ = add_u
    xr.Variable.__sub__ = sub_u
    xr.Variable.__mul__ = mul_u
    xr.Variable.__truediv__ = truediv_u  # Py3
    if hasattr(xr.Variable, "__div__"):
        xr.Variable.__div__ = truediv_u  # Py2
    # Add unit conversion function, needed for both, DataArray and Variable
    setattr(xr.DataArray, "to_unit", to_unit)
    setattr(xr.Variable, "to_unit", to_unit)
    # Track conversion to SI units in multiplication and division
    setattr(xr.Variable, "__keep_si__", keep_si)
    # set `xarray` to keep track of attributes, that includes units
    xr.set_options(keep_attrs=True)
    # Mark as initialized.
    xr.DataArray.__has_units__ = True
    xr.Variable.__has_units__ = True
