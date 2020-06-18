import matplotlib.colors as clr
import matplotlib.pyplot as plt
import numpy as np
import scipy.special


def fig_rk():
    x_min, x_max = -4, 2
    y_min, y_max = -4, 4
    res_x = 0.01
    res_y = 0.01

    x = np.arange(x_min, x_max, res_x)
    y = np.arange(y_min, y_max, res_y)
    x, y = np.meshgrid(x, y)

    def r(k):
        return np.vectorize(lambda z: sum(np.power(z, range(k + 1)) / scipy.special.factorial(range(k + 1))))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    for axis in ('left', 'bottom'):
        ax.spines[axis].set_position('zero')
        ax.spines[axis].set_linewidth(2)
    for axis in ('right', 'top'):
        ax.spines[axis].set_color('none')
    ax.tick_params(width=2, labelsize=12)
    ax.yaxis.tick_left()
    ax.xaxis.tick_bottom()
    for (i, c) in zip((4, 3, 2, 1), ([.8, .0, .0, 1], [.0, .8, .0, 1], [.0, .0, .8, 1], [.5, .5, .5, 1])):
        rk_i = abs(r(i)(x + 1j * y)) < 1
        ax.pcolormesh(x, y, np.ma.masked_array(rk_i, mask=1 - rk_i), cmap=clr.ListedColormap([c]))
        ax.plot([None], [None], '--', c=c, lw=20, label='RK{0}'.format(i))
    ax.grid(True, lw=2)
    ax.legend(fontsize=16, loc='upper left', handletextpad=1, labelspacing=1)

    plt.show()


if __name__ == '__main__':
    fig_rk()
