from tkinter import *
from tkinter.ttk import Separator

from src.utils.util import *
from src.gui.train_panel import TrainPanel
from src.gui.classify_panel import ClassifyPanel


class MainView:

    def __init__(self, root):
        # examinations = ["subject{}".format(i) for i in range(20)]
        # features = ["min", "max", "variety", "value", "mean", "rms"]
        # window_sizes = [("{}s".format((i+1)*5), (i+1)*5) for i in range(5)]

        # ROOT #
        self.root = root
        root.title(TITLE)
        root.resizable(False, False)
        # TRAIN #
        TrainPanel(root).pack(side=LEFT, anchor=N)
        # SEPARATOR #
        Separator(root, orient=VERTICAL).pack(side=LEFT, anchor=N, fill=Y)
        # CLASSIFY #
        ClassifyPanel(root).pack(side=LEFT, anchor=N)
