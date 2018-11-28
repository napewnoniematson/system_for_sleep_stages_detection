from src.gui.controller.train.train_controller import TrainController
from src.gui.controller.result.result_controller import ResultController
from src.gui.controller.classify.classify_controller import ClassifyController


class MainController:
    def __init__(self):
        self.train_controller = TrainController()
        self.classify_controller = ClassifyController()
        self.result_controller = ResultController()
