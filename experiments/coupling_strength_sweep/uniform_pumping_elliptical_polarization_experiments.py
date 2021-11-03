import sys
import os

# add path containing the code and data files
current_path = os.path.dirname(os.path.abspath(__file__))
experiments_path = os.path.dirname(current_path)
root_path = os.path.dirname(experiments_path)
experiments_path = sys.path.append(root_path)

import numpy as np
from scipy.constants import lambda2nu

from fiber import StepIndexFiber
from polarization import hyperstokes_to_jones
import raman_linear_coupling

from experiment import Experiment
from uniform_pumping_experiments import UniformPumpingExperiment

class ParallelEllipticalPolarizationsUniformPumpingExperiment(UniformPumpingExperiment):

    def run(self, thetas):
        """
        Parameters
        ----------
            thetas: np.ndarray
                Array of perturbation angles
        """

        signal_jones = np.ones((int(3 * self.num_modes_s / 2,))) / np.sqrt(3)
        signal_jones = hyperstokes_to_jones(signal_jones)

        pump_jones = np.ones((int(3 * self.num_modes_p / 2,))) / np.sqrt(3)
        pump_jones = hyperstokes_to_jones(pump_jones)

        return super().run(signal_jones, pump_jones, thetas)

    
class OrthogonalEllipticalPolarizationsUniformPumpingExperiment(UniformPumpingExperiment):
    def run(self, thetas):
        """
        Parameters
        ----------
            thetas: np.ndarray
                Array of perturbation angles
        """


        signal_jones = np.ones((int(3 * self.num_modes_s / 2),)) / np.sqrt(3)
        signal_jones = hyperstokes_to_jones(signal_jones)

        pump_jones = -np.ones((int(3 * self.num_modes_p / 2),)) / np.sqrt(3)
        pump_jones = hyperstokes_to_jones(pump_jones)

        return super().run(signal_jones, pump_jones, thetas)