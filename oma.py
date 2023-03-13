import numpy as np
from numpy import linalg as LA
from scipy import signal
import os
import scipy.io as io
import matplotlib.pyplot as plt
from fembeam import beam_fem
plt.rcParams["font.family"] = "Times New Roman"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class rand_vib:
    def __init__(self,signal_mtx):
        # signal should in matrix form, of which dimension is 5*n_t
        self.signal_mtx = signal_mtx

    def plot_first_three_signal(self):
        T = np.linspace(0,2000-1/100,200000)
        plt.plot(T,self.signal_mtx[2,:],color = 'tab:green', label = 'Third sensor')
        plt.plot(T,self.signal_mtx[1,:],color = 'tab:orange', label = 'Second sensor')
        plt.plot(T,self.signal_mtx[0,:],color = 'tab:blue', label = 'First sensor')
        plt.xlabel('Time (s)',fontname='Times New Roman')
        plt.ylabel('Acceleration $\mathregular{(m/s^2)}$',fontname='Times New Roman')
        plt.xticks(fontname='Times New Roman')
        plt.yticks(fontname='Times New Roman')
        # plt.legend(['Third sensor','Second sensor','First sensor'],prop={"family":"Times New Roman"})
        # order = [2,1,0]
        handles, labels = plt.gca().get_legend_handles_labels()
        handles = [handles[2], handles[1], handles[0]]
        labels = [labels[2], labels[1], labels[0]]
        plt.legend(handles,labels)
        plt.savefig('fig_signals.pdf', dpi=1200,bbox_inches='tight')
        plt.show()        

    def psd_analysis(self,dim=0):
        signal_vec = self.signal_mtx[dim,:].reshape(-1,1)
        f, pxx = signal.welch(signal_vec, fs=100, nperseg=2000, axis=0)
        return f, pxx
    
    def plot_first_three_psd(self):
        f1, pxx1 = self.psd_analysis(dim=0)
        f2, pxx2 = self.psd_analysis(dim=1)
        f3, pxx3 = self.psd_analysis(dim=2)
        plt.plot(f1,np.log10(pxx1))
        plt.plot(f2,np.log10(pxx2))
        plt.plot(f3,np.log10(pxx3))
        plt.xlabel('Frequency (Hz)',fontname='Times New Roman')
        plt.ylabel('Power spectral density $\mathregular{((m/s^2)^2/Hz)}$',fontname='Times New Roman')
        plt.xticks(fontname='Times New Roman')
        plt.yticks(fontname='Times New Roman')
        plt.legend(['First sensor','Second sensor','Third sensor'], prop={"family":"Times New Roman"})
        plt.savefig('fig_psds.pdf', dpi=1200,bbox_inches='tight')
        plt.show()

    def ms_ratio(self,ms):
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
                w_f_temp, w_acc_temp = signal.csd(self.signal_mtx[i,:], self.signal_mtx[j,:], fs = 100, window='hann', nperseg=nperseg_num, axis=0, scaling = 'density', average='mean')
                w_f.append(w_f_temp)
                w_acc.append(w_acc_temp)
        idx = [i for i, v in enumerate(w_f[0]) if v<=f_ub and v>=f_lb]
        tru_w_acc = np.array(w_acc)[:,idx]
        nf_temp_idx = []
        ms = []
        for i in range(tru_w_acc.shape[1]):
            G_yy = tru_w_acc[:,i].reshape(5,5)
            u,s,_ = LA.svd(G_yy,full_matrices=True)
            nf_temp_idx.append(s[0])
            ms.append(np.real(u[:,0]))
        nf_temp_idx = np.argmax(np.array(nf_temp_idx))
        nf_idx = idx[0]+nf_temp_idx
        nf = w_f[0][nf_idx]
        if type == 'peak':
            ms_peak = np.array(ms)[nf_temp_idx,:]
            return ms_peak, nf
        elif type == 'average':
            ms = np.average(np.array(ms),axis=0)
        return ms,nf

mat = io.loadmat('./data/train_dataset/train_4.mat')
mtx = mat['A']
vib_analysis = rand_vib(signal_mtx=mtx)
ms = vib_analysis.fdd()
ms_r = vib_analysis.ms_ratio(ms)
print(ms_r)
beam = beam_fem()
beam_ms = beam.modeshape(1, alphas=[0.0,0.4,0.0])[1:-1]
beam_ms_r = vib_analysis.ms_ratio(beam_ms)
print(beam_ms_r)
print(LA.norm(ms_r-np.array(beam_ms_r),ord=2))

vib_analysis.plot_first_three_psd()
vib_analysis.plot_first_three_signal()
# ms_ratio = oma_svd(signal_mtx,nperseg_num=2000)
# print(ms_ratio)