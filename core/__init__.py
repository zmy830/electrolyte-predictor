# -*- coding: utf-8 -*-
"""
core - 电解液预测核心模块
"""

from .predictor import ElectrolytePredictor, create_predictor
from .formula_utils import (
    mass_to_mole_fraction,
    mole_to_mass_fraction,
    normalize_fractions,
    validate_formula,
    validate_conditions,
    format_formula_string,
    get_preset_formula,
    list_preset_formulas,
    PRESET_FORMULAS,
)
from .solvent_database import (
    SolventDatabase,
    BUILTIN_SOLVENTS,
    SALT_DATABASE,
    get_salt_list,
    get_salt_info,
)

__all__ = [
    "ElectrolytePredictor",
    "create_predictor",
    "mass_to_mole_fraction",
    "mole_to_mass_fraction",
    "normalize_fractions",
    "validate_formula",
    "validate_conditions",
    "format_formula_string",
    "get_preset_formula",
    "list_preset_formulas",
    "PRESET_FORMULAS",
    "SolventDatabase",
    "BUILTIN_SOLVENTS",
    "SALT_DATABASE",
    "get_salt_list",
    "get_salt_info",
]
