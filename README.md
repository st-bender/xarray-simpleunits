# Xarray_SimpleUnits

[![builds](https://github.com/st-bender/xarray-simpleunits/actions/workflows/ci_build_and_test.yml/badge.svg?branch=main)](https://github.com/st-bender/xarray-simpleunits/actions/workflows/ci_build_and_test.yml)
[![package](https://img.shields.io/pypi/v/xarray-simpleunits.svg?style=flat)](https://pypi.org/project/xarray-simpleunits)
[![wheel](https://img.shields.io/pypi/wheel/xarray-simpleunits.svg?style=flat)](https://pypi.org/project/xarray-simpleunits)
[![pyversions](https://img.shields.io/pypi/pyversions/xarray-simpleunits.svg?style=flat)](https://pypi.org/project/xarray-simpleunits)
[![codecov](https://codecov.io/gh/st-bender/xarray-simpleunits/branch/main/graphs/badge.svg)](https://codecov.io/gh/st-bender/xarray-simpleunits)
[![coveralls](https://coveralls.io/repos/github/st-bender/xarray-simpleunits/badge.svg)](https://coveralls.io/github/st-bender/xarray-simpleunits)

**Unitful calculations with `xarray`**

Keeps track of units when working with `xarray.DataArray`s
and `xarray.Dataset`s using `astropy.units` for the conversions.

:warning: This package is in **alpha** stage, that is, it works mostly,
but the interface might still be subject to change.

## Install

### Requirements

- `astropy` - required
- `xarray` - required
- `pytest` - optional, for testing

### xarray_simpleunits

An installable `pip` package called `xarray-simpleunits` will be available soon
from the main package repository, it can then be installed with:
```sh
$ pip install xarray_simpleunits
```
The latest development version can be installed
with [`pip`](https://pip.pypa.io) directly from github
(see <https://pip.pypa.io/en/stable/reference/pip_install/#vcs-support>
and <https://pip.pypa.io/en/stable/reference/pip_install/#git>):

```sh
$ pip install [-e] git+https://github.com/st-bender/xarray-simpleunits.git
```

The other option is to use a local clone:

```sh
$ git clone https://github.com/st-bender/xarray-simpleunits.git
$ cd xarray-simpleunits
```
and then using `pip` (optionally using `-e`, see
<https://pip.pypa.io/en/stable/reference/pip_install/#install-editable>):

```sh
$ pip install [-e] .
```

or using `setup.py`:

```sh
$ python setup.py install
```

Optionally, test the correct function of the module with

```sh
$ py.test [-v]
```

or even including the [doctests](https://docs.python.org/library/doctest.html)
in this document:

```sh
$ py.test [-v] --doctest-glob='*.md'
```

## Usage

The python module itself is named `xarray_simpleunits` and is imported as usual.

All functions should be `numpy`-compatible and work with scalars
and appropriately shaped arrays.

```python
>>> import xarray_simpleunits as xru

```

The module basically works by adapting (“monkey-patching”) the `xarray.Variable` arithmetic
methods to honour and keep track of the "units" attribute.
To initialize unit handling with `xarray`, call `init_units()` first:

```python
>>> import xarray_simpleunits as xru
>>> xru.init_units()

```
(A similar method to restore the original behaviour is planned.)

So far, the supported operations are addition, subtraction, multiplication, and division.
Unit mismatch when adding or subtracting unitful arrays raises an exception.
Currently, unit handling with `xarray.DataArray` requires that the respective
array is on the left side of any calculation:

```python
>>> from astropy import units as au
>>> import numpy as np
>>> import xarray as xr
>>> import xarray_simpleunits as xru
>>> np.set_printoptions(precision=6)
>>> xru.init_units()
>>> ds = xr.Dataset(
...     data_vars={
...         "s": ("x", [1., 2., 3.], {"units": "m"}),
...         "t": ("x", [3., 2., 1.], {"units": "s"}),
...     },
... )
>>> v = ds["s"] / ds["t"]
>>> v  # doctest: +ELLIPSIS
<xarray.DataArray (x: 3)>...
array([0.333333, 1.      , 3.      ])
Dimensions without coordinates: x
Attributes:
    units:    m / s
>>> # using `astropy` units directly:
>>> v = ds["s"] / (2 * au.Unit("s"))
>>> v  # doctest: +ELLIPSIS
<xarray.DataArray 's' (x: 3)>...
array([0.5, 1. , 1.5])
Dimensions without coordinates: x
Attributes:
    units:    m / s

```

Basic class and method documentation is accessible via `pydoc`:

```sh
$ pydoc xarray_simpleunits
```

## License

This python package is free software: you can redistribute it or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 2 (GPLv2), see [local copy](./LICENSE)
or [online version](http://www.gnu.org/licenses/gpl-2.0.html).
