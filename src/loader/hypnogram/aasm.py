def __load_data_from_file(path):
    with open(path, "r") as file:
        return file.readlines()


def __filter_data(data):
    # removed '\n' from hypno values e.g. "5\n" -> "5"
    filtered = list(map(lambda x: x.strip(), data))
    return filtered


def __split_data(data):
    title = data[0]
    hypnogram_data = data[1:]
    return title, hypnogram_data

def aasm_stages(path):
    hypnogram = __load_data_from_file(path)
    filtered = __filter_data(hypnogram)
    title, hypnogram_data = __split_data(filtered)
    wake = []
    rem = []
    n1 = []
    n2 = []
    n3 = []
    begin = 0
    for i in range(len(hypnogram_data) - 1):
        if hypnogram_data[i] != hypnogram_data[i + 1] or i + 1 == len(hypnogram_data) - 1:
            end = i * 1000
            if hypnogram_data[i] == '5':
                wake.append([begin, end])
            elif hypnogram_data[i] == '4':
                rem.append([begin, end])
            elif hypnogram_data[i] == '3':
                n1.append([begin, end])
            elif hypnogram_data[i] == '2':
                n2.append([begin, end])
            elif hypnogram_data[i] == '1':
                n3.append([begin, end])
            begin = end + 1
    return title, hypnogram_data, wake, rem, n1, n2, n3
