from tkinter import *

from src.utils.util import *
from src.gui.multiple_choice_view import MultipleChoiceView
from src.gui.one_choice_view import OneChoiceView


class TrainChoicePanel(Frame):

    def __init__(self, root):
        Frame.__init__(self, root)
        examinations = ["subject{}".format(i) for i in range(20)]
        features = ["min", "max", "variety", "value", "mean", "rms"]
        window_sizes = [("{}s".format((i + 1) * 5), (i + 1) * 5) for i in range(5)]

        # EXAMINATIONS #
        self.examination_view = MultipleChoiceView(self, EXAMINATIONS_TEXT, examinations)
        self.examination_view.pack(side=LEFT, anchor=N)
        # FEATURES #
        self.features_view = MultipleChoiceView(self, FEATURES_TEXT, features)
        self.features_view.pack(side=LEFT, anchor=N)
        # WINDOW WIDTH #
        self.window_view = OneChoiceView(self, WINDOW_WIDTH_TEXT, window_sizes)
        self.window_view.pack(side=LEFT, anchor=N)

    def state(self):
        print("examination list: ", list(self.examination_view.state()))
        print("features list: ", list(self.features_view.state()))
        print("window width: ", self.window_view.state())