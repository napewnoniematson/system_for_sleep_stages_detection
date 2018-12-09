from tkinter import *
from tkinter import messagebox

from src.gui.panel.train.train_choice_panel import TrainChoicePanel
from src.gui.panel.train.train_buttons_panel import TrainButtonsPanel

from src.utils.util import *


class TrainPanel(Frame):

    def __init__(self, root, controller):
        Frame.__init__(self, root)
        self.controller = controller
        self.train_choice_panel = TrainChoicePanel(self, self.controller.examinations,
                                                   self.controller.features_titles, self.controller.window_sizes,
                                                   self.controller.stage_modes, self.controller.epochs,
                                                   self.controller.training_part_percentage)
        self.train_choice_panel.pack(side=TOP, anchor=N)
        self.train_buttons_panel = TrainButtonsPanel(self, train_callback=self.train_callback,
                                                     save_callback=self.save_callback,
                                                     load_callback=self.load_callback)
        self.train_buttons_panel.pack(side=TOP, anchor=N)

    def train_callback(self):
        data = self.train_choice_panel.state()
        self.controller.on_train_button_click(input_data=data)
        messagebox.showinfo(TRAINING_TEXT, TRAINING_FINISHED_MESSAGE)

    def load_callback(self):
        self.controller.on_load_model_button_click()
        messagebox.showinfo(LOAD_MODEL_TEXT, LOADING_MODEL_FINISHED_MESSAGE)

    def save_callback(self):
        self.controller.on_save_model_button_click()
        messagebox.showinfo(SAVE_MODEL_TEXT, SAVING_MODEL_FINISHED_MESSAGE)
