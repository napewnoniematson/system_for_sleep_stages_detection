from tkinter import *

from src.gui.widgets.radiobox import Radiobox


class OneChoice(Frame):

    def __init__(self, root, title, picks):
        Frame.__init__(self, root)
        self.label = Label(self, text=title)
        self.label.pack(side=TOP)
        self.listbox = Listbox(self)
        self.listbox.pack(side=TOP)
        self.radiobox = Radiobox(self.listbox, picks)
        self.radiobox.pack(side=TOP)

    def state(self):
        return self.radiobox.state()


