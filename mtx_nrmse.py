import numpy as np
import matplotlib.pyplot as plt
from fembeam import beam_fem
from numpy import linalg as LA

def nrmse(a,b):
    return LA.norm(a-b,2)/LA.norm(a,2)

A = np.array([1,0.8,0.96,1.1,0.48,0.6,0.4,0.36,0.53,0.62])
rho = np.array([1,5.6,4.2,2.1,3.0,1.2,3.4,0.3,1.5,0.4])*1e4
E = np.array([2,3.4,0.2,0.8,1.2,4.2,5.3,6.6,2.4,0.5])*1e10
I = np.array([8.33,6.67,8.00,9.17,4.00,5,3.33,3.00,4.33,5.17])*1e-2
phi  = []
for i in range(10):
    beam = beam_fem(rho=rho[i],E=E[i],A=A[i],I=I[i],L=0.5)
    ms_full = beam.modeshape(1,type='full')
    phi.append(ms_full)
phi = np.array(phi)
nr = np.zeros((10,10))
for i in range(10):
    for j in range(10):
        nr[i,j] = nrmse(phi[i,:],phi[j,:])
fig, ax = plt.subplots()
ax.matshow(nr,cmap='pink')
plt.xticks(np.arange(10),np.arange(10)+1,fontname='Times New Roman')
plt.yticks(np.arange(10),np.arange(10)+1,fontname='Times New Roman')
for i in range(10):
   for j in range(10):
      if i == j:
          pass
      else:
        c = '%.2f' %np.log10(nr[j, i])
        ax.text(i, j, str(c), va='center', ha='center',fontname='Times New Roman',fontsize =8)
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none') 
plt.savefig('./figs/nrmse_mtx.pdf', dpi=1200,bbox_inches='tight')
plt.show()

















