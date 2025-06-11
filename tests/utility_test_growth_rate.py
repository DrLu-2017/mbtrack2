import h5py as hp
import matplotlib.pyplot as plt
import numpy as np


def r2(y, y_fit):
    # residual sum of squares
    ss_res = np.sum((y - y_fit)**2)

    # total sum of squares
    ss_tot = np.sum((y - np.mean(y))**2)

    # r-squared
    r2 = 1 - (ss_res/ss_tot)
    return r2


def get_growth(ring, start, end, bunch_num, g, plane="y", plot=False):
    plane_dict = {"x": 0, "y": 1, "long": 2}
    time = np.array(g["Beam"]["time"]) * ring.T0
    data = np.array(g["Beam"]["cs_invariant"][plane_dict[plane], bunch_num, :])
    idx = (time > ring.T0 * start) & (time < ring.T0 * end)
    time_fit = time[idx] - start * ring.T0
    data_fit = data[idx]

    ploy = np.polynomial.polynomial.Polynomial.fit(time_fit,
                                                   np.log(data_fit),
                                                   1,
                                                   w=np.sqrt(data_fit))

    if plot:
        plt.plot(time / ring.T0, np.log(data), '-')

    arr = ploy.convert().coef
    fit_vals = [np.exp(arr[0]) * np.exp(arr[1] * x) for x in time_fit]

    return (arr[0], arr[1] / 2, r2(data_fit, fit_vals))


def get_growth_beam(ring, path, start, end, plane="y", plot=False):
    g = hp.File(path, "r")
    gr_cs = np.zeros(ring.h)
    cste = np.zeros(ring.h)
    r2_cs = np.zeros(ring.h)

    for i in range(ring.h):
        cste[i], gr_cs[i], r2_cs[i] = get_growth(ring, start, end, i, g, plane,
                                                 plot)

    time = np.array(g["Beam"]["time"])
    idx = (time > start) & (time < end)
    time_fit = time[idx]
    fit = np.mean(cste) + np.mean(gr_cs) * 2 * (time_fit-start) * ring.T0

    print(f"growth rate = {np.mean(gr_cs)} +- {np.std(gr_cs)} s-1")
    print(f"r2 = {np.mean(r2_cs)} +- {np.std(r2_cs)}")
    if plot:
        plt.plot(time_fit, fit, 'k--')
        ymax = plt.gca().get_ylim()[1]
        ymin = plt.gca().get_ylim()[0]
        plt.vlines(start, ymin, ymax, "r")
        plt.vlines(end, ymin, ymax, "r")
        plt.grid()
        plt.xlabel("Turn number")
        plt.ylabel("log(CS_invariant)")
    g.close()
    return (np.mean(gr_cs), np.std(gr_cs))
