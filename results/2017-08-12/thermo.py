"""Unit conversions and other thermodynamics
"""
from metpy.units import units
import metpy.constants as mc

Q_ = units.Quantity
kelvin_to_mixing_ratio = mc.Cp_d/mc.Lv
rho0 = Q_(1.15, "kg/m^3")

