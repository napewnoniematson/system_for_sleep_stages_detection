from tkinter import *


class Radiobox(Frame):

    def __init__(self, root=None, picks=[], side=TOP, anchor=W):
        Frame.__init__(self, root)
        self.var = IntVar()
        selected = picks[0][1] if picks else -1
        self.var.set(selected)
        for text, value in picks:
            radiobutton = Radiobutton(self, text=text, variable=self.var, value=value)
            radiobutton.pack(side=side, anchor=anchor, expand=YES)

    def state(self):
        return self.var.get()
