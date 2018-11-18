from tkinter import *

from src.gui.widgets.checkbox import Checkbox


class MultipleChoice(Frame):
    def __init__(self, root, title, picks):
        Frame.__init__(self, root)
        self.label = Label(self, text=title)
        self.label.pack(side=TOP)
        self.listbox = Listbox(self)
        self.listbox.pack(side=TOP)
        self.checkbox = Checkbox(self.listbox, picks)
        self.checkbox.pack(side=TOP)

    def state(self):
        return self.checkbox.state()
