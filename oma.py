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

    def psd_analysis(self,dim=0):
        signal_vec = self.signal_mtx[dim,:].reshape(-1,1)
        f, pxx = signal.welch(signal_vec, fs=100, nperseg=2000, axis=0)
        return f, pxx

    def plot_first_three_signal(self):
        num = 6000
        T = np.linspace(0,(num-1)/100,num)
        col = ['k', 'b', 'r']
        cm = 1 / 2.54
        fig = plt.figure(figsize=(16 * cm, 14 * cm))
        ax = fig.add_subplot(211)
        ax.plot(T,self.signal_mtx[0,0:num], color=col[0], label = 'First sensor')
        ax.plot(T,self.signal_mtx[1,0:num], color=col[1], label = 'Second sensor')
        ax.plot(T,self.signal_mtx[2,0:num], color=col[2], label = 'Third sensor', zorder=1)
        ax.set_xlabel(r'Time (s)', fontsize=9, labelpad=1)    
        ax.set_ylabel(r'Acceleration ($\mathregular{m/s^2}$)', fontsize=9, labelpad=1)      
        ax.set_ylim([-0.3, 0.3])
        ax.set_yticks(np.arange(-0.3, 0.31, 0.1))
        ax.set_xlim([0, 60])
        ax.set_xticks(np.arange(0, 60.1, 10))
        legend = ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), borderpad=0.3, borderaxespad=0, handlelength=2.8,
                        edgecolor='black', fontsize=8, ncol=3, columnspacing=0.5, handletextpad=0.3)  # labelspacing=0
        legend.get_frame().set_boxstyle('Square', pad=0.0)
        legend.get_frame().set_lw(0.75)
        legend.get_frame().set_alpha(None)
        for obj in legend.legendHandles:
            obj.set_lw(0.75)
        ax.text(2, 0.22, '(a)')
        ax.tick_params(axis='x', direction='in')
        ax.tick_params(axis='y', direction='in')

        ax = fig.add_subplot(212)    
        f1, pxx1 = self.psd_analysis(dim=0)
        f2, pxx2 = self.psd_analysis(dim=1)
        f3, pxx3 = self.psd_analysis(dim=2)
        ax.plot(f1,np.log10(pxx1),color=col[0], lw=1, label='First sensor')
        ax.plot(f2,np.log10(pxx2),color=col[1], dashes=[8, 4], lw=1, label='Second sensor')
        ax.plot(f3,np.log10(pxx3),color=col[2], dashes=[2, 2], lw=1, label='Third sensor')  
        ax.set_xlabel(r'Frequency (Hz)', fontsize=9, labelpad=1)
        ax.set_ylabel(r'PSD ($\mathregular{(m/s^2)^2}$/Hz)', fontsize=9, labelpad=1)      
        ax.set_xlim([5, 40])
        ax.set_xticks(np.arange(5, 40.1, 5))     
        ax.set_ylim([-7, -1])
        ax.set_yticks(np.arange(-7, -0.9, 1))  
        legend = ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), borderpad=0.3, borderaxespad=0, handlelength=2.8,
                   edgecolor='black', fontsize=8, ncol=3, columnspacing=0.5, handletextpad=0.3)  # labelspacing=0
        legend.get_frame().set_boxstyle('Square', pad=0.0)
        legend.get_frame().set_lw(0.75)
        legend.get_frame().set_alpha(None)
        for obj in legend.legendHandles:
            obj.set_lw(0.75)
        ax.text(6.2, -1.8, '(b)')
        ax.tick_params(axis='x', direction='in')
        ax.tick_params(axis='y', direction='in')

        plt.savefig('./figs/threesignal.pdf',  format = "pdf",
                    dpi=1200,bbox_inches='tight')

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
# fdd analysis
# ms,_ = vib_analysis.fdd()
# ms_r = vib_analysis.ms_ratio(ms)
# print(ms_r)
# beam = beam_fem()
# beam_ms = beam.modeshape(1, alphas=[0.0,0.4,0.0])[1:-1]
# beam_ms_r = vib_analysis.ms_ratio(beam_ms)
# print(beam_ms_r)
# print(LA.norm(ms_r-np.array(beam_ms_r),ord=2))

# vib_analysis.plot_first_three_psd()
vib_analysis.plot_first_three_signal()
plt.show()
# ms_ratio = oma_svd(signal_mtx,nperseg_num=2000)
# print(ms_ratio)