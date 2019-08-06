import os


def get_physio_net(path):
    dl = os.listdir(path)
    dl.sort()
    hypnogram, signal = [], []
    [(hypnogram, signal)["PSG" in e].append(e) for e in dl]
    if len(hypnogram) == len(signal):
        for h, s in zip(hypnogram, signal):
            if h[:6] != s[:6]:
                raise Exception("Incorrect pair hypnogram <==> signal")
    else:
        raise Exception("Incorrect pair hypnogram <==> signal")
    return hypnogram, signal


def get_de_mons(path):
    dl = os.listdir(path)
    dl.sort()
    hypnogram, signal = [], []
    [(hypnogram, signal)["Hypnogram" not in e].append(e) for e in dl]
    if len(hypnogram) == len(signal):
        for h, s in zip(hypnogram, signal):
            if h[-5:-4] != s[-5:-4]:
                raise Exception("Incorrect pair hypnogram <==> signal")
    else:
        raise Exception("Incorrect pair hypnogram <==> signal")
    return hypnogram, signal
