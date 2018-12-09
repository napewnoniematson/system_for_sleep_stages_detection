from src.gui.controller.train.train_controller import TrainController
from src.gui.controller.result.result_controller import ResultController
from src.gui.controller.classify.classify_controller import ClassifyController

from src.model.result import Result


class MainController:
    def __init__(self):
        result = Result()
        self.train_controller = TrainController(result=result)
        self.classify_controller = ClassifyController(result=result)
        self.result_controller = ResultController(result=result)
