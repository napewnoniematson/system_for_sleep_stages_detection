from tkinter import *

from src.utils.util import *


class TrainButtonsPanel(Frame):

    def __init__(self, root, train_callback=None, load_callback=None):
        Frame.__init__(self, root)
        # START #
        self.training_button = Button(self, text=TRAINING_TEXT, command=train_callback)
        self.training_button.pack(side=LEFT, anchor=W)
        # LOAD MODEL #
        self.load_button = Button(self, text=LOAD_MODEL_TEXT, command=load_callback)
        self.load_button.pack(side=LEFT, anchor=W)
