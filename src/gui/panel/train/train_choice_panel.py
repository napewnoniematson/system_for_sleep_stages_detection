from tkinter import *

from src.utils.util import *
from src.gui.widgets.multiple_choice import MultipleChoice
from src.gui.widgets.one_choice import OneChoice


class TrainChoicePanel(Frame):

    def __init__(self, root, examinations, features, window_sizes, stage_modes, epochs, training_part_percentage):
        Frame.__init__(self, root)
        self.examination_view = MultipleChoice(self, EXAMINATIONS_TEXT, examinations)
        self.examination_view.pack(side=LEFT, anchor=N)
        self.features_view = MultipleChoice(self, FEATURES_TEXT, features)
        self.features_view.pack(side=LEFT, anchor=N)
        self.window_view = OneChoice(self, WINDOW_WIDTH_TEXT, window_sizes)
        self.window_view.pack(side=LEFT, anchor=N)
        self.stage_mode_view = OneChoice(self, STAGE_MODE_TEXT, stage_modes)
        self.stage_mode_view.pack(side=LEFT, anchor=N)
        self.epochs_view = OneChoice(self, EPOCHS_TEXT, epochs)
        self.epochs_view.pack(side=LEFT, anchor=N)
        self.training_part_view = OneChoice(self, TRAINING_PART_TEXT, training_part_percentage)
        self.training_part_view.pack(side=LEFT, anchor=N)

    def state(self):
        return {
            EXAMINATIONS_KEY: list(self.examination_view.state()),
            FEATURES_KEY: list(self.features_view.state()),
            WINDOW_WIDTH_KEY: self.window_view.state(),
            STAGE_MODE_KEY: self.stage_mode_view.state(),
            EPOCHS_KEY: self.epochs_view.state(),
            TRAINING_PART_KEY: self.training_part_view.state()
        }
