from tkinter import *
from tkinter.ttk import Separator

from src.utils.util import *
from src.gui.panel.train.train_panel import TrainPanel
from src.gui.panel.result.result_panel import ResultPanel
from src.gui.panel.classify.classify_panel import ClassifyPanel


class MainView:

    def __init__(self, root, controller):
        self.root = root
        root.title(TITLE)
        root.resizable(False, False)
        self.controller = controller
        if controller:
            self.train_panel = TrainPanel(root, self.controller.train_controller)
            self.train_panel.pack(side=LEFT, anchor=N)
            Separator(root, orient=VERTICAL).pack(side=LEFT, anchor=N, fill=Y)
            self.classify_panel = ClassifyPanel(root, self.controller.classify_controller)
            self.classify_panel.pack(side=LEFT, anchor=N)
            Separator(root, orient=VERTICAL).pack(side=LEFT, anchor=N, fill=Y)
            self.result_panel = ResultPanel(root, self.controller.result_controller)
            self.result_panel.pack(side=LEFT, anchor=N)
        else:
            raise Exception(CONTROLLER_NOT_FOUND_EXCEPTION)
