from src.logger.messages import *
import src.logger.logger as logger


def __load_data_from_file(path):
    data = []
    try:
        logger.info(RK_LOAD_START_INFO)
        with open(path+"s", "r") as file:
            data = file.readlines()
            logger.info(RK_LOAD_END_INFO)
    except FileNotFoundError:
        logger.error(RK_LOAD_ERROR)

    return data


def __filter_data(data):
    # removed '\n' from hypno values e.g. "5\n" -> "5"
    filtered = []
    if data:
        filtered = list(map(lambda x: x.strip(), data))
    return filtered


def __split_data(data):
    title = ''
    hypnogram_data = []
    if data:
        title = data[0]
        hypnogram_data = data[1:]
    return title, hypnogram_data


def __prepare_data(hypnogram):
    logger.info(RK_PREPARE_DATA_START_INFO)
    filtered = __filter_data(hypnogram)
    title, hypnogram_data = __split_data(filtered)
    wake = []
    rem = []
    s1 = []
    s2 = []
    s3 = []
    s4 = []
    begin = 0
    if hypnogram_data:
        for i in range(len(hypnogram_data) - 1):
            if hypnogram_data[i] != hypnogram_data[i + 1] or i + 1 == len(hypnogram_data) - 1:
                end = i * 1000
                if hypnogram_data[i] == '5':
                    wake.append([begin, end])
                elif hypnogram_data[i] == '4':
                    rem.append([begin, end])
                elif hypnogram_data[i] == '3':
                    s1.append([begin, end])
                elif hypnogram_data[i] == '2':
                    s2.append([begin, end])
                elif hypnogram_data[i] == '1':
                    s3.append([begin, end])
                elif hypnogram_data[i] == '0':
                    s4.append([begin, end])
                begin = end + 1
        logger.info(RK_PREPARE_DATA_END_INFO)
    else:
        logger.warning(RK_PREPARE_DATA_END_WARNING)
    return {
        "title": title,
        "data": hypnogram_data,
        "wake": wake,
        "rem": rem,
        "s1": s1,
        "s2": s2,
        "s3": s3,
        "s4": s4
    }


def load(path):
    hypnogram = __load_data_from_file(path)
    return __prepare_data(hypnogram)
