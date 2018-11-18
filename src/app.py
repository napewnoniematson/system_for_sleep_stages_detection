from tkinter import Tk

from src.controller.controller import Controller
from src.logger.messages import *
import src.logger.logger as logger
from src.gui.main_view import MainView

if __name__ == '__main__':
    logger.info(START_APP)
    root = Tk()
    main_view = MainView(root)
    root.mainloop()
    # app = Controller()
    # app.run()
