from src.logger import type, colors


def info(message):
    print(colors.BOLD + colors.BLUE + type.INFO + colors.DISABLE + message)


def data_info(message):
    print(colors.BOLD + colors.CYAN + type.DATA_INFO + colors.DISABLE + message)


def warning(message):
    print(colors.BOLD + colors.YELLOW + type.WARNING + colors.DISABLE + message)


def error(message):
    print(colors.BOLD + colors.RED + type.ERROR + colors.DISABLE + message)
