from tkinter import *

from src.utils.util import *
from src.gui.train_choice_panel import TrainChoicePanel
from src.gui.train_buttons_panel import TrainButtonsPanel


class TrainPanel(Frame):

    def __init__(self, root):
        Frame.__init__(self, root)
        self.train_choice_panel = TrainChoicePanel(self)
        self.train_choice_panel.pack(side=TOP, anchor=N)
        self.train_buttons_panel = TrainButtonsPanel(self)
        self.train_buttons_panel.pack(side=TOP, anchor=N)
