from tkinter import *

from src.gui.widgets.one_choice import OneChoice
from src.utils.util import *


class ClassifyPanel(Frame):

    def __init__(self, root, controller):
        Frame.__init__(self, root)
        self.controller = controller
        self.examinations_view = OneChoice(self, EXAMINATIONS_FOR_CLASSIFY_TEXT, self.controller.examinations)
        self.examinations_view.pack(side=TOP, anchor=W)
        self.classify_button = Button(self, text=CLASSIFY_BUTTON_TEXT,
                                      command=self.controller.on_classify_button_click)
        self.classify_button.pack(side=TOP, anchor=W)

    def state(self):
        print(self.examinations_view.state())
