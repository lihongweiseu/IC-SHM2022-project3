import numpy as np
from numpy import linalg as LA
from scipy import signal
import os
import scipy.io as io
import matplotlib.pyplot as plt
from fembeam import beam_fem
plt.rcParams["font.family"] = "Times New Roman"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class rand_vib:
    def __init__(self, signal_mtx):
        # signal should in matrix form, of which dimension is 5*n_t
        self.signal_mtx = signal_mtx

    def psd_analysis(self, dim=0):
        signal_vec = self.signal_mtx[dim, :].reshape(-1, 1)
        f, pxx = signal.welch(signal_vec, fs=100, nperseg=2000, axis=0)
        return f, pxx

    def ms_ratio(self, ms):
        ms_r = []
        ms_r.append(ms[1] / ms[0])
        ms_r.append(ms[2] / ms[1])
        ms_r.append(ms[2] / ms[3])
        ms_r.append(ms[3] / ms[4])
        return ms_r

    def fdd(self, f_lb=8.5, f_ub=10.5, nperseg_num=40, type='peak'):
        # implementation of frequency domain decomposition
        # signal should in matrix form, whose dimension is 5*n_t
        # will return the mode shapes and the natural frequency
        # two ways to generate mode shapes, peak or average
        w_f = []
        w_acc = []
        for i in range(self.signal_mtx.shape[0]):
            for j in range(self.signal_mtx.shape[0]):
                w_f_temp, w_acc_temp = signal.csd(
                    self.signal_mtx[i, :], self.signal_mtx[j, :], fs=100, window='hann', nperseg=nperseg_num, axis=0, scaling='density', average='mean')
                w_f.append(w_f_temp)
                w_acc.append(w_acc_temp)
        idx = [i for i, v in enumerate(w_f[0]) if v <= f_ub and v >= f_lb]
        tru_w_acc = np.array(w_acc)[:, idx]
        nf_temp_idx = []
        ms = []
        for i in range(tru_w_acc.shape[1]):
            G_yy = tru_w_acc[:, i].reshape(5, 5)
            u, s, _ = LA.svd(G_yy, full_matrices=True)
            nf_temp_idx.append(s[0])
            ms.append(np.real(u[:, 0]))
        nf_temp_idx = np.argmax(np.array(nf_temp_idx))
        nf_idx = idx[0]+nf_temp_idx
        nf = w_f[0][nf_idx]
        if type == 'peak':
            ms_peak = np.array(ms)[nf_temp_idx, :]
            return ms_peak, nf
        elif type == 'average':
            ms_avg = np.average(np.array(ms), axis=0)
            return ms_avg, nf

    def neur_net_input(self, f_lb=8.5, f_ub=10.5, nperseg_num=30, type='peak'):
        # implementation of frequency domain decomposition
        ms, _ = self.fdd(f_lb=f_lb, f_ub=f_ub,
                         nperseg_num=nperseg_num, type=type)
        ms_r = self.ms_ratio(ms)
        beam = beam_fem()
        ms_r_undamaged = beam.md1st_ratio()
        return (ms_r - ms_r_undamaged).reshape(1, -1)
