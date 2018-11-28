from tkinter import *
from tkinter.ttk import Separator

from src.utils.util import *
from src.gui.panel.train.train_panel import TrainPanel
from src.gui.panel.classify.classify_panel import ClassifyPanel


class MainView:

    def __init__(self, root, controller):
        examinations = ["subject{}".format(i) for i in range(20)]
        features = ["min", "max", "variety", "value", "mean", "rms"]
        window_sizes = [("{}s".format((i+1)*5), (i+1)*5) for i in range(5)]
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
        else:
            raise Exception(CONTROLLER_NOT_FOUND_EXCEPTION)
