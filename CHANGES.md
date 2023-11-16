Changelog
=========

v0.0.4 (2023-11-16)
-------------------

### Fixes

- Correctly support numpy and pandas datetime and timedelta variables

### Changes

- Removed implicit conversion to SI units after multiplication
  and division.


v0.0.3 (2023-11-13)
-------------------

### New

- Adds explicit conversion to SI units: `to_si_units(x)` or `x.to_si_units()`
- `xarray.Dataset` support by monkey-patching `xarray.Variable`
  instead of `xarray.DataArray`
- Implement way to restore standard behaviour with `reset_units()`


v0.0.2 (2023-10-17)
-------------------

### New

- Enable option to keep strict SI units
- Deployment to PyPI via Github actions


v0.0.1 (2023-07-23)
-------------------

Initial alpha release.
