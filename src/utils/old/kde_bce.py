from scipy.stats import gaussian_kde

ts_bin_H0_BCE = np.load('../../data_bin/BCE_ref/ts_bin_H0_BCE.npy')
ts_bin_H1_BCE = np.load('../../data_bin/BCE_ref/ts_bin_H1_BCE.npy')

def KDE_H0(norm=True):
    n = min(ts_bin_H0_BCE.flatten()) if norm else 0
    dat = ts_bin_H0_BCE.flatten() - n
    return gaussian_kde(dat)

def KDE_H1(norm=True):
    n = min(ts_bin_H1_BCE.flatten()) if norm else 0
    dat = ts_bin_H1_BCE.flatten() - n
    return gaussian_kde(dat)