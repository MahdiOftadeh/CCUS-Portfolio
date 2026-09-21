# tests/test_hydrodynamics.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.hydrodynamics.ergun import ErgunEquation

def test_ergun_pressure_drop():
    # Typical Zeolite 13X bed params
    ergun = ErgunEquation(bed_porosity=0.37, particle_diameter=0.002)
    
    dp_dz = ergun.calculate_pressure_gradient(
        velocity=0.1,         # m/s
        fluid_density=1.2,    # kg/m^3
        fluid_viscosity=1.8e-5 # Pa.s
    )
    
    # dP/dz should be negative and around -100 to -1000 Pa/m for these conditions
    assert dp_dz < 0
    assert abs(dp_dz) > 10.0
    print("✅ Hydrodynamics (Ergun Equation) test passed!")

if __name__ == '__main__':
    test_ergun_pressure_drop()
