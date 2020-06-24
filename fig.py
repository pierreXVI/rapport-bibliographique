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


def fig_bdf():
    def r(k):
        def sub_r(z):
            coeff = np.zeros(k + 1, dtype=complex)
            for i in range(1, k + 1):
                coeff[0] += 1 / i
                coeff[i] = np.power(-1, i) * scipy.special.binom(k, i) / i
            coeff[0] -= z
            return (np.abs(np.roots(coeff)) < 1).all()

        return np.vectorize(sub_r)

    fig = plt.figure()
    c = [.8, .0, .0, 1]
    for (i, x_min, x_max, y_min, y_max) in zip((1, 2, 3, 4, 5, 6),
                                               (-2, -2, -4, -4, -10, -20),
                                               (3, 5, 8, 14, 25, 40),
                                               (-2, -3, -5, -8, -15, -30),
                                               (2, 3, 5, 8, 15, 30),
                                               ):
        res_x = (x_max - x_min) / 500
        res_y = (y_max - y_min) / 500

        x = np.arange(x_min, x_max, res_x)
        y = np.arange(y_min, y_max, res_y)
        x, y = np.meshgrid(x, y)

        ax = fig.add_subplot(2, 3, i)
        ax.set_aspect('equal')
        for axis in ('left', 'bottom'):
            ax.spines[axis].set_position('zero')
            ax.spines[axis].set_linewidth(2)
        for axis in ('right', 'top'):
            ax.spines[axis].set_color('none')
        ax.tick_params(width=2, labelsize=12)
        ax.yaxis.tick_left()
        ax.xaxis.tick_bottom()
        rk_i = r(i)(x + 1j * y)
        ax.pcolormesh(x, y, np.ma.masked_array(rk_i, mask=1 - rk_i), cmap=clr.ListedColormap([c]))
        ax.plot([None], [None], '--', c=c, lw=20)
        ax.grid(True, lw=2)
        ax.set_title('BDF{0}'.format(i), fontsize=16, y=-.1)

    plt.show()


if __name__ == '__main__':
    fig_rk()
