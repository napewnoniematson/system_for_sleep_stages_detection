from src.controller.controller import Controller
from src.logger.messages import *
import src.logger.logger as logger

if __name__ == '__main__':
    logger.info(START_APP)
    app = Controller()
    app.run()
