import os
from fiber import StepIndexFiber

clad_index = 1.46
delta = 0.005
core_radius=6
clad_radius=60



# get absolute path of current file
fiber_path = os.path.dirname(os.path.abspath(__file__))

SIF2Modes = StepIndexFiber(clad_index=1.46, delta=0.005, core_radius=6, clad_radius=60, data_path=fiber_path)
SIF4Modes = StepIndexFiber(clad_index=1.4545, delta=0.005, core_radius=6, clad_radius=60, data_path=fiber_path)