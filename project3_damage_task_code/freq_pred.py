import numpy as np
from fembeam import beam_fem

A = np.array([1, 0.8, 0.96, 1.1, 0.48, 0.6, 0.4, 0.36, 0.53, 0.62])
rho = np.array([1, 5.6, 4.2, 2.1, 3.0, 1.2, 3.4, 0.3, 1.5, 0.4])*1e4
E = np.array([2, 3.4, 0.2, 0.8, 1.2, 4.2, 5.3, 6.6, 2.4, 0.5])*1e10
I = np.array([8.33, 6.67, 8.00, 9.17, 4.00, 5, 3.33, 3.00, 4.33, 5.17])*1e-2
alphas_set = np.array([[0.0, 0.2, 0.0],
                       [0.0, 0.3, 0.0],
                       [0.0, 0.4, 0.0],
                       [0.1, 0.0, 0.0],
                       [0.3, 0.0, 0.0],
                       [0.5, 0.0, 0.0],
                       [0.2, 0.2, 0.0],
                       [0.2, 0.4, 0.0],
                       [0.4, 0.2, 0.0],
                       [0.4, 0.4, 0.0]
                       ])
beam = beam_fem(rho=rho[0], E=E[0], A=A[0], I=I[0], L=0.5)
ratio = beam.freqs_ratio()
pred_freq = []
for i in range(10):
    model_freq = beam.frequency(order=1, alphas=alphas_set[i, :])
    pred_freq.append(round(model_freq*ratio, 4))
print(pred_freq)
