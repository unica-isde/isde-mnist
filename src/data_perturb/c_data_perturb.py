from abc import ABC, abstractmethod


class CDataPerturb(ABC):

    @abstractmethod
    def data_perturbation(self, x):
        raise NotImplementedError()

    def perturb_dataset(self, X):
        Xp = X.copy()
        for i in range(X.shape[0]):
            Xp[i, :] = self.data_perturbation(Xp[i,:])
        return Xp
    