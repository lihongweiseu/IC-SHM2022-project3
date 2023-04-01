import numpy as np
import torch
from fembeam import beam_fem
from torch.utils.data import TensorDataset
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Gen_Data():
    def __init__(self):
        pass

    def gen_train_data(self, grid_num=91):
        # Generate training set
        alphas = np.linspace(0.1, 1, grid_num)
        alphas = np.array(np.meshgrid(alphas, alphas, alphas)
                          ).reshape(3, -1).transpose()
        alphas = np.ones_like(alphas)-alphas
        beam = beam_fem()
        md_1st_r = []
        ms_ratio_undam = beam.md1st_ratio()
        for i in range(grid_num**3):
            md_1st_r.append(beam.nn_input(alphas[i, :], ms_ratio_undam))
            if i % (grid_num**2) == 0:
                print("%.2f" % ((i/(grid_num**3))*100), '%')
        md_1st_r = np.array(md_1st_r)
        train_data = TensorDataset(
            torch.tensor(md_1st_r), torch.tensor(alphas))
        torch.save(
            train_data, './damage identification task/data/neural_nets/train_data.pt')

    def gen_test_data(self, num=300000):
        # Generate test set
        alphas_test = np.random.rand(num, 3)*0.9
        md_1st_r_test = []
        beam = beam_fem()
        ms_ratio_undam = beam.md1st_ratio()
        for i in range(num):
            md_1st_r_test.append(beam.nn_input(
                alphas_test[i, :], ms_ratio_undam))
            if i % (num/10) == 0:
                print("%.2f" % ((i/num)*100), '%')
        md_1st_r_test = np.array(md_1st_r_test)
        test_data = TensorDataset(torch.tensor(
            md_1st_r_test), torch.tensor(alphas_test))
        torch.save(
            test_data, './damage identification task/data/neural_nets/test_data.pt')


My_data = Gen_Data()
My_data.gen_train_data()
My_data.gen_test_data()
