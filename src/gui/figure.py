import matplotlib as plt


def plot(samples_indices, samples_values, classes, hypnogram):
    c = ['b' if c == h else 'r' for c, h in zip(classes, hypnogram)]
    plt.scatter(samples_indices, samples_values, c=c, s=0.05)
    plt.show()
