from abc import ABC, abstractmethod


class CDataLoader(ABC):
    """
    Generic abstract interface for data loaders
    """
    @abstractmethod
    def load_data(self):
        raise NotImplementedError("The method is abstract.")