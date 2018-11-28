from src.gui.controller.train_controller import TrainController
from src.gui.controller.classify_controller import ClassifyController


class MainController:
    def __init__(self):
        self.train_controller = TrainController()
        self.classify_controller = ClassifyController()
