import os
from fiber import StepIndexFiber



def SIF2Modes():
    # get absolute path of current file
    fiber_path = os.path.dirname(os.path.abspath(__file__))
    fiber = StepIndexFiber(clad_index=1.46, delta=0.005, core_radius=6, clad_radius=60, data_path=fiber_path)
    return fiber


def SIF4Modes():
    # get absolute path of current file
    fiber_path = os.path.dirname(os.path.abspath(__file__))
    fiber = StepIndexFiber(clad_index=1.4545, delta=0.005, core_radius=6, clad_radius=60, data_path=fiber_path)
    return fiber