from tkinter import *
from tkinter import messagebox

from src.utils.util import *


class ResultPanel(Frame):

    def __init__(self, root, controller):
        Frame.__init__(self, root)
        self.controller = controller
        self.label = Label(self, text=RESULT_TEXT)
        self.label.pack(side=TOP, anchor=N)
        self.result_text = Text(self, state=NORMAL)
        self.result_text.pack(side=TOP, anchor=N)
        self.update_button = Button(self, text=UPDATE_TEXT, command=self.update)
        self.update_button.pack(side=LEFT, anchor=N)

    def update(self):
        output = self.controller.update()
        BEGIN = 1.0
        self.result_text.delete(BEGIN, END)
        self.result_text.insert(END, output)
        messagebox.showinfo(UPDATE_TEXT, UPDATING_RESULT_FINISHED_MESSAGE)
