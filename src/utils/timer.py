import datetime

LOG_DATE_TIME_FORMAT = '%Y-%m-%d %H:%M:%S'


def get_time():
    return datetime.datetime.now().strftime(LOG_DATE_TIME_FORMAT)
