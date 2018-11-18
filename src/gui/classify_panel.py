from tkinter import *

from src.gui.one_choice_view import OneChoiceView
from src.utils.util import *


class ClassifyPanel(Frame):

    def __init__(self, root, classify_callback=None):
        Frame.__init__(self, root)
        examinations = [("subject{}".format(i), i) for i in range(20)]
        self.examinations_view = OneChoiceView(self, EXAMINATIONS_FOR_CLASSIFY_TEXT, examinations)
        self.examinations_view.pack(side=TOP, anchor=W)
        self.classify_button = Button(self, text=CLASSIFY_BUTTON_TEXT, command=classify_callback)
        self.classify_button.pack(side=TOP, anchor=W)

    def state(self):
        print(self.examinations_view.state())
