from src.logger import type, colors


def info(message):
    print(colors.BOLD + colors.BLUE + type.INFO + colors.DISABLE + message)


def warning(message):
    print(colors.BOLD + colors.YELLOW + type.WARNING + colors.DISABLE + message)


def error(message):
    print(colors.BOLD + colors.RED + type.ERROR + colors.DISABLE + message)


def header(message):
    print(colors.BOLD + colors.MAGENTA + type.WARNING + colors.DISABLE + message)
