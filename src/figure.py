import matplotlib as plt


# todo make sure that classes and hypnogram have same stage mode
def plot(samples_indices, samples_values, classes, hypnogram):
    c = ['b' if c == h else 'r' for c, h in zip(classes, hypnogram)]
    plt.scatter(samples_indices, samples_values, c=c, s=0.05)
    plt.show()
