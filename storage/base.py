from abc import ABC, abstractmethod

class Storage(ABC):
    @abstractmethod
    def save_prediction(self, uid, original_image, predicted_image): pass

    @abstractmethod
    def save_detection(self, prediction_uid, label, score, box): pass

    @abstractmethod
    def get_prediction(self, uid): pass