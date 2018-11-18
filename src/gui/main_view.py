from tkinter import *

from src.utils.util import *
from src.gui.train_choice_panel import TrainChoicePanel
from src.gui.train_buttons_panel import TrainButtonsPanel


class MainView:

    def __init__(self, root):
        # examinations = ["subject{}".format(i) for i in range(20)]
        # features = ["min", "max", "variety", "value", "mean", "rms"]
        # window_sizes = [("{}s".format((i+1)*5), (i+1)*5) for i in range(5)]

        # ROOT #
        self.root = root
        root.title(TITLE)
        # TRAIN
        TrainChoicePanel(root).pack(side=TOP)
        TrainButtonsPanel(root).pack(side=TOP)
