from tkinter import *


class Checkbox(Frame):

    def __init__(self, root=None, picks=[], side=TOP, anchor=W):
        Frame.__init__(self, root)
        self.vars = []
        for pick in picks:
            var = IntVar()
            checkbutton = Checkbutton(self, text=pick, variable=var)
            checkbutton.pack(side=side, anchor=anchor, expand=YES)
            self.vars.append(var)

    def state(self):
        return map((lambda var: var.get()), self.vars)
