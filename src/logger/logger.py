from src.logger import type, colors


def info(message):
    """
    Displaying INFO message

    :param message: message to display
    """
    print(colors.BOLD + colors.BLUE + type.INFO + colors.DISABLE + message)


def data_info(message):
    """
    Displaying DATA message

    :param message: message to display
    """
    print(colors.BOLD + colors.CYAN + type.DATA_INFO + colors.DISABLE + message)


def warning(message):
    """
    Displaying WARNING message

    :param message: message to display
    """
    print(colors.BOLD + colors.YELLOW + type.WARNING + colors.DISABLE + message)


def error(message):
    """
    Displaying ERROR message

    :param message: message to display
    """
    print(colors.BOLD + colors.RED + type.ERROR + colors.DISABLE + message)
