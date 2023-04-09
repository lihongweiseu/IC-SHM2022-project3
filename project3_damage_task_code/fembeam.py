import numpy as np
from numpy import linalg as LA


class beam_fem:
    def __init__(self, rho=1e4, E=2e10, A=1, I=1/12, L=0.5):
        # system parameters
        self.rho = rho
        self.E = E
        self.I = I
        self.L = L
        self.A = A

    def ele_K(self):
        # element stiffness matrix
        EIL12 = 12*self.E*self.I/(self.L**3)
        EIL6 = 6*self.E*self.I/(self.L**2)
        EIL4 = 4*self.E*self.I/(self.L)
        EIL2 = 2*self.E*self.I/(self.L)
        K = np.diag(np.array([EIL12, EIL4, EIL12, EIL4]))
        K_down = np.zeros((4, 4))
        K_down[1, 0] = EIL6
        K_down[2, 0] = -EIL12
        K_down[2, 1] = -EIL6
        K_down[3, 0] = EIL6
        K_down[3, 1] = EIL2
        K_down[3, 2] = -EIL6
        K = K+K_down+K_down.transpose()
        return K

    def ele_M(self):
        # element mass matrix
        const = self.rho*self.A*self.L/(420)
        M = np.diag(np.array([156, 4*self.L**2, 156, 4*self.L**2]))
        M_down = np.zeros((4, 4))
        M_down[1, 0] = 22*self.L
        M_down[2, 0] = 54
        M_down[2, 1] = 13*self.L
        M_down[3, 0] = -13*self.L
        M_down[3, 1] = -3*self.L**2
        M_down[3, 2] = -22*self.L**2
        M = const*(M+M_down+M_down.transpose())
        return M

    def assemble_glb_mtx(self, type='stiff', alpha1=0.0, alpha2=0.0, alpha3=0.0):
        # global mass and stiffness matrices
        # there are 44 elements, 45 nodes, 85 dofs in total
        ele_num = np.linspace(0, 43, 44, dtype=int).reshape((-1, 1))
        dof_num = np.linspace(0, 89, 90, dtype=int).reshape((-1, 1))
        ele_node = np.hstack(
            (ele_num*2, ele_num*2+1, ele_num*2+2, ele_num*2+3))
        glb_mtx = np.zeros((45*2, 45*2))
        if type == 'stiff':
            K = self.ele_K()
            for i in range(44):
                if i == 6:
                    bv, cv = np.meshgrid(ele_node[i, :], ele_node[i, :])
                    glb_mtx[bv, cv] = glb_mtx[bv, cv]+K*(1-alpha1)
                elif i == 21:
                    bv, cv = np.meshgrid(ele_node[i, :], ele_node[i, :])
                    glb_mtx[bv, cv] = glb_mtx[bv, cv]+K*(1-alpha2)
                elif i == 37:
                    bv, cv = np.meshgrid(ele_node[i, :], ele_node[i, :])
                    glb_mtx[bv, cv] = glb_mtx[bv, cv]+K*(1-alpha3)
                else:
                    bv, cv = np.meshgrid(ele_node[i, :], ele_node[i, :])
                    glb_mtx[bv, cv] = glb_mtx[bv, cv]+K
        elif type == 'mass':
            M = self.ele_M()
            for i in range(44):
                bv, cv = np.meshgrid(ele_node[i, :], ele_node[i, :])
                glb_mtx[bv, cv] = glb_mtx[bv, cv]+M
        con_node_number = np.array([0, 24, 64, 88]).reshape((-1, 1))
        uncon_node = np.delete(dof_num, con_node_number)
        bv, cv = np.meshgrid(uncon_node, uncon_node)
        glb_mtx = glb_mtx[bv, cv]
        return glb_mtx

    def freqs_ratio(self):
        K = self.assemble_glb_mtx(type='stiff')
        M = self.assemble_glb_mtx(type='mass')
        undam_freq = 9.4367
        return undam_freq/(np.sqrt(np.min(LA.eig(LA.inv(M)@K)[0]))/(2*np.pi))

    def frequency(self, order, alphas=[0.0, 0.0, 0.0]):
        # return i-th natural frequency in Hertz
        M = self.assemble_glb_mtx(type='mass')
        K = self.assemble_glb_mtx(
            type='stiff', alpha1=alphas[0], alpha2=alphas[1], alpha3=alphas[2])
        freq = list(LA.eig(LA.inv(M)@K)[0])
        freq.sort()
        return np.sqrt(freq[order-1])/(2*np.pi)

    def modeshape(self, order, alphas=[0.0, 0.0, 0.0], dof_idx=[5, 30, 42, 54, 79], type='acc'):
        # return i-th mode shape value at the locations of two ends and five accelerometers
        # or return a 85*1 vector that represents the modeshape of al dof
        if hasattr(self, 'K'):
            K = self.K
        else:
            K = self.assemble_glb_mtx(
                type='stiff', alpha1=alphas[0], alpha2=alphas[1], alpha3=alphas[2])
        if hasattr(self, 'M'):
            M = self.M
        else:
            M = self.assemble_glb_mtx(type='mass')
        freq, Phi = LA.eig(LA.inv(M)@K)
        freq = list(freq)
        freq_copy = freq.copy()
        freq.sort()
        idx = freq_copy.index(freq[order-1])
        if type == 'acc':
            ms = Phi[dof_idx, idx]
            ms = np.insert(ms, 0, 0)
            ms = np.append(ms, 0)
            return ms
        else:
            ms = Phi[:, idx]
            return ms

    def md1st_ratio(self, alphas=[0.0, 0.0, 0.0]):
        ms = self.modeshape(1, alphas)
        ms_ratio = []
        ms_ratio.append(ms[2] / ms[1])
        ms_ratio.append(ms[3] / ms[2])
        ms_ratio.append(ms[3] / ms[4])
        ms_ratio.append(ms[4] / ms[5])
        return np.array(ms_ratio)

    def nn_input(self, alphas, ms_ratio_undam, subtract_norm=True):
        # if subtract_norm is True, the neural network input is ms in damaged case minus ms in health state
        # if subtract_norm is False, the neural network input is ms in damaged case
        # ms_ratio_undam is the ms ratio of undamaged case
        ms_ratio_dam = self.md1st_ratio(alphas)
        if subtract_norm:
            return ms_ratio_dam-ms_ratio_undam
        else:
            return ms_ratio_dam
