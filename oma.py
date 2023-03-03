import numpy as np
from numpy import linalg as LA
from scipy import signal
import os
import scipy.io as io

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def oma_svd(signal_mtx,nperseg_num=2000):
    # signal should in matrix form, whose dimension is 5*n_t
    # will return the mode shape ratios
    w_f = []
    w_acc = []
    for i in range(signal_mtx.shape[0]):
        for j in range(signal_mtx.shape[0]):
            w_f_temp, w_acc_temp = signal.csd(signal_mtx[i,:], signal_mtx[i,:], fs = 100, window='hann', nperseg=nperseg_num, axis=0, scaling = 'density', average='mean')
            w_f.append(w_f_temp)
            w_acc.append(w_acc_temp)
    idx = [i for i, v in enumerate(w_f[0]) if v<=10 and v>=9]
    tru_w_acc = np.array(w_acc)[:,idx]
    idx_ms = np.sum(np.abs(tru_w_acc),axis = 0).argmax()
    G_yy = np.array(w_acc)[:,idx_ms+idx[0]].reshape(5,5)
    u, _, _ = LA.svd(G_yy,full_matrices=True)
    ms = np.real(u[:,0])
    ms_ratio = []
    ms_ratio.append(ms[1] / ms[0])
    ms_ratio.append(ms[2] / ms[1])
    ms_ratio.append(ms[2] / ms[3])
    ms_ratio.append(ms[3] / ms[4])
    return ms_ratio

mat = io.loadmat('./data/train_dataset/train_11.mat')
signal_mtx = mat['A']
ms_ratio = oma_svd(signal_mtx,nperseg_num=2000)
print(ms_ratio)