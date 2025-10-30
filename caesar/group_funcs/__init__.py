"""
Convenience exports for group property functions and utilities.

This module re-exports symbols from the Cython/Python implementation
`caesar.group_funcs.group_funcs` so that existing imports like
`from caesar.group_funcs import get_group_overall_properties` work
regardless of whether the package is installed as a namespace package
or with compiled extensions.
"""

# Always import from the concrete module; if the compiled extension is
# available it will be used, otherwise the pure-Python fallback loads.
from .group_funcs import (  # noqa: F401
    get_group_overall_properties,
    get_group_gas_properties,
    get_group_star_properties,
    get_group_bh_properties,
    get_group_dust_properties,
    get_half_mass_radius,
    get_full_mass_radius,
)

