# src/cycles/vsa_cycle.py
from enum import Enum, auto

class CycleStepType(Enum):
    ADSORPTION = auto()
    BLOWDOWN = auto()
    EVACUATION = auto()
    REPRESSURIZATION = auto()

class CycleStep:
    """
    Defines a single step in a cyclic adsorption process.
    """
    def __init__(self, name: str, step_type: CycleStepType, duration: float, inlet_pressure: float, outlet_pressure: float):
        self.name = name
        self.step_type = step_type
        self.duration = duration  # seconds
        self.p_in = inlet_pressure  # Pa
        self.p_out = outlet_pressure # Pa

class VSACycleManager:
    """
    Manages sequence of steps in a VSA/TVSA process.
    """
    def __init__(self):
        self.steps = []

    def add_step(self, step: CycleStep):
        self.steps.append(step)

    def get_total_cycle_time(self) -> float:
        return sum(s.duration for s in self.steps)
