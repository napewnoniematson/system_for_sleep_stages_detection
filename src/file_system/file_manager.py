import csv


def save(path, features):
    with open(path + '.csv', 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', lineterminator='\r\n')
        for row in features:
            writer.writerow(row)


def load(path, return_int=False):
    with open(path + '.csv', 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',', lineterminator='\r\n')
        return [[int(i) for i in r] for r in reader] if return_int else [r for r in reader]


def save_json(path, json):
    with open(path + '.json', 'w') as f:
        f.write(json)
