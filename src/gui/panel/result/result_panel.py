from tkinter import *

from src.utils.util import *


class ResultPanel(Frame):

    def __init__(self, root, controller):
        Frame.__init__(self, root)
        self.controller = controller
        self.label = Label(self, text=RESULT_TEXT)
        self.label.pack(side=TOP, anchor=N)
        self.result_text = Text(self, state='disabled')
        self.result_text.pack(side=TOP, anchor=N)
