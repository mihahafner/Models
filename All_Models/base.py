from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def fit(self, train_data, **kwargs):
        ...

    @abstractmethod
    def predict(self, data, **kwargs):
        ...
