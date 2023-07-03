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
    elif isinstance(x, (au.CompositeUnit, au.IrreducibleUnit, au.Unit)):
        return x
    return au.dimensionless_unscaled


# %%
def _get_values(x):
    """Infer unit(s) from attributes
    """
    if hasattr(x, "value"):
        return x.value
    return x


# %%
def add_u(a, b):
    """Addition with units
    """
    a_u = _get_unit(a)
    try:
        b = b.to(a_u)
    except AttributeError:
        b = b.to_unit(a_u)
    except au.UnitConversionError:
        raise ValueError("Unit mismatch in additon.")
    ret = a.__add_orig__(_get_values(b))
    ret.attrs["units"] = _get_unit(a)
    return ret


# %%
def sub_u(a, b):
    """Subtraction with units
    """
    a_u = _get_unit(a)
    try:
        b = b.to(a_u)
    except AttributeError:
        b = b.to_unit(a_u)
    except au.UnitConversionError:
        raise ValueError("Unit mismatch in subtraction.")
    ret = a.__sub_orig__(_get_values(b))
    ret.attrs["units"] = _get_unit(a)
    return ret


# %%
def mul_u(a, b):
    """Multiplication with units
    """
    ret = a.__mul_orig__(b)
    ret_u = _get_unit(a) * _get_unit(b)
    ret.attrs["units"] = ret_u.si
    return ret


# %%
def truediv_u(a, b):
    """Division with units
    """
    ret = a.__truediv_orig__(b)
    ret_u = _get_unit(a) / _get_unit(b)
    ret.attrs["units"] = ret_u.si
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
    ret = a * _get_unit(a).to(u)
    ret.attrs["units"] = u
    return ret


# %%
def init_units():
    # save "original" functions
    xr.DataArray.__add_orig__ = xr.DataArray.__add__
    xr.DataArray.__sub_orig__ = xr.DataArray.__sub__
    xr.DataArray.__mul_orig__ = xr.DataArray.__mul__
    if hasattr(xr.DataArray, "__truediv__"):  # Py3
        xr.DataArray.__truediv_orig__ = xr.DataArray.__truediv__
    else:
        xr.DataArray.__truediv_orig__ = xr.DataArray.__div__
    # point DataArray operations to our functions
    xr.DataArray.__add__ = add_u
    xr.DataArray.__sub__ = sub_u
    xr.DataArray.__mul__ = mul_u
    if hasattr(xr.DataArray, "__truediv__"):
        xr.DataArray.__truediv__ = truediv_u  # Py3
    else:
        xr.DataArray.__div__ = truediv_u  # Py2
    # Add a unit conversion function
    setattr(xr.DataArray, "to_unit", to_unit)
    # set `xarray` to keep track of attributes, that includes units
    xr.set_options(keep_attrs=True)

    au.add_enabled_aliases({"degrees_north": au.degree})
    au.add_enabled_aliases({"degrees_east": au.degree})
    au.add_enabled_aliases({"degree_north": au.degree})
    au.add_enabled_aliases({"degree_east": au.degree})
    au.add_enabled_aliases({"ppm": 1e-6 * au.mol / au.mol})
    au.add_enabled_aliases({"ppb": 1e-9 * au.mol / au.mol})
    au.add_enabled_aliases({"ppt": 1e-12 * au.mol / au.mol})
