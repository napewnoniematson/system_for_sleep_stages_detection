from tkinter import *

from src.utils.util import *
from src.gui.widgets.multiple_choice import MultipleChoice
from src.gui.widgets.one_choice import OneChoice


class TrainChoicePanel(Frame):

    def __init__(self, root, examinations, features, window_sizes):
        Frame.__init__(self, root)
        self.examination_view = MultipleChoice(self, EXAMINATIONS_TEXT, examinations)
        self.examination_view.pack(side=LEFT, anchor=N)
        self.features_view = MultipleChoice(self, FEATURES_TEXT, features)
        self.features_view.pack(side=LEFT, anchor=N)
        self.window_view = OneChoice(self, WINDOW_WIDTH_TEXT, window_sizes)
        self.window_view.pack(side=LEFT, anchor=N)

    def state(self):
        return {
            EXAMINATIONS_KEY: list(self.examination_view.state()),
            FEATURES_KEY: list(self.features_view.state()),
            WINDOW_WIDTH_KEY: self.window_view.state()
        }