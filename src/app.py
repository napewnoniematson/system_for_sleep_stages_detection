from tkinter import Tk

from src.gui.controller.main_controller import MainController as Controller
from src.logger.messages import *
import src.logger.logger as logger
from src.gui.main_view import MainView

if __name__ == '__main__':
    logger.info(START_APP)
    import src.test as t
    # t.start()
    root = Tk()
    ctr = Controller()
    main_view = MainView(root, ctr)
    root.mainloop()
