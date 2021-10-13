from abc import ABC, abstractmethod
from multiprocessing import Pool

class Experiment(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def run(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)


class ParallelExperiment(ABC):
    def __init__(self) -> None:
        pass

    def run(self, **kwargs):
        iterables = self.inputs()
        inputs = zip(*iterables)

        with Pool() as pool:
            results = pool.starmap(self.experiment.run, inputs, **kwargs)


        self.results = results

        return results

    @abstractmethod
    def inputs(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)



