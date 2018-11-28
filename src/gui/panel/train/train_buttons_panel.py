from tkinter import *

from src.utils.util import *


class TrainButtonsPanel(Frame):

    def __init__(self, root, train_callback=None, save_callback=None, load_callback=None):
        Frame.__init__(self, root)
        self.training_button = Button(self, text=TRAINING_TEXT, command=train_callback)
        self.training_button.pack(side=LEFT, anchor=N)
        self.save_button = Button(self, text=SAVE_MODEL_TEXT, command=save_callback)
        self.save_button.pack(side=LEFT, anchor=N)
        self.load_button = Button(self, text=LOAD_MODEL_TEXT, command=load_callback)
        self.load_button.pack(side=LEFT, anchor=N)
