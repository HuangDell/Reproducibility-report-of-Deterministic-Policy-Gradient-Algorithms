import matplotlib.pyplot as plt


def show(x, y,title):
    plt.xlabel('Time-steps')
    plt.ylabel('Reward')
    plt.plot(x, y)
    plt.title(title)
    plt.savefig(f'{title}.jpg')
    plt.show()


