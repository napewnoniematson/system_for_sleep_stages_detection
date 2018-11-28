from tkinter import *

from src.gui.panel.train.train_choice_panel import TrainChoicePanel
from src.gui.panel.train.train_buttons_panel import TrainButtonsPanel


class TrainPanel(Frame):

    def __init__(self, root, controller):
        Frame.__init__(self, root)
        self.controller = controller
        self.train_choice_panel = TrainChoicePanel(self, self.controller.examinations,
                                                   self.controller.features, self.controller.window_sizes)
        self.train_choice_panel.pack(side=TOP, anchor=N)
        self.train_buttons_panel = TrainButtonsPanel(self, train_callback=self.train_callback,
                                                     save_callback=self.controller.on_save_model_button_click,
                                                     load_callback=self.controller.on_load_model_button_click)
        self.train_buttons_panel.pack(side=TOP, anchor=N)

    def train_callback(self):
        data = self.train_choice_panel.state()
        self.controller.on_train_button_click(input_data=data)
