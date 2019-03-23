import datetime

LOG_DATE_TIME_FORMAT = '%Y-%m-%d %H:%M:%S'


def get_time():
    """
    Function return current date

    :return: current datetime (format - '%Y-%m-%d %H:%M:%S')
    """
    return datetime.datetime.now().strftime(LOG_DATE_TIME_FORMAT)
