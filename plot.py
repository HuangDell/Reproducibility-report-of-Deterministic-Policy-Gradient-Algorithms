import matplotlib.pyplot as plt


def show(x, y,title):
    plt.xlabel('Time-steps(x200)')
    plt.ylabel('Reward')
    plt.plot(x, y,label='COPDAC-Q')
    # plt.plot(a, b,lable='AC')
    plt.title(title)
    plt.savefig(f'{title}.jpg')
    plt.show()
