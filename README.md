# IC-SHM2022-project3

The repository contains the code and documentation for our team's entry in the 3rd International Competition for Structural Health Monitoring(IC-SHM 2022) project 3. Our solution is designed to solve the following 2 tasks:

- Data-driven modeling
- Damage identification

## Getting Started

To get started with our solution, you can follow these steps:

- Clone this repository to your local machine
- Check if the installed version of the packages meets the requirements
- Set the folder IC-SHM2022-project3 as the workspace
- Run XXX in the folder of Task 1 data-driven modeling
- Run main.py in the folder of Task 2 damage identification

### Requirements

- Python >= 3.8.0
- Pytorch >= 1.12.1
- SciPy >= 1.9.3

## Task 1: Data-driven Modeling

## Task 2: Damage Identification

In this task, we aim to implement damage identification of three specified units (7, 22, 38) in a three-span continuous bridge (other units stay undamaged) through the use of random vibration data from five channels. There are six cases in the testing datasets, and the developed algorithm results in a $6\times3$ matrix. It is remarked that **the training
dataset provided by the IC-SHM 2022 Committee for this task is completely unused**.

### Methodology

The proposed method demonstrates that the mode shapes of a three-span continuous beam are independent of finite element model parameters and are only affected by the reduction rates of the elastic modulus (i.e., the damage factors). Moreover, we were able to construct a closed-form forward mapping that relates the damage factors to the mode shapes. By using this mapping, we generated a large number of label pairs whose feature and target domain are exchanged, which allowed us to train a deep neural network with mode shape ratios as input and damage factors as output. We used Bayesian optimization to automatically tune the hyper-parameters of the network for more accurate predictions. In this competition, we were provided with random vibration signals from five accelerometers. We used the frequency domain decomposition method to extract the mode shape of the beam, which was then fed into the neural network to compute the damage factors.

### Results

Given the test dataset, our deep neural network could accurately predict the elastic modulus reduction rates. The results that retain 6 decimal places are shown in the following table.

<div align="center">

| File name | Damage condition <br/> (Unit No.7) | Damage condition <br/>(Unit No.22) | Damage condition <br/>(Unit No.38) |
| :-------: | :--------------------------------: | :--------------------------------: | :--------------------------------: |
|  test_1   |              0.002931              |              0.097144              |              0.005024              |
|  test_2   |              0.000000              |              0.499517              |              0.004262              |
|  test_3   |              0.221431              |              0.000000              |              0.003660              |
|  test_4   |              0.413827              |              0.000000              |              0.018817              |
|  test_5   |              0.192929              |              0.200971              |              0.189067              |
|  test_6   |              0.397418              |              0.400919              |              0.403472              |

</div>

### Folder Structure

1. **project3_damage_task.txt**: The elastic modulus reduction rates of three units for 6 test data sets.
2. **project3_damage_task_code**: Python source codes, data, and figures to demonstrate and reproduce the results. The data and figures are organized in individual folders, and the source codes are presented directly in this folder.

   - **fembeam.py**: A class that constructs the finite element model for the three-span continuous bridge.
   - **neuralnets.py**: Several classes including training data set generation, neural network definition and training, as well as the Bayesian optimization for hyper-parameters tuning.
   - **oma.py**: A class for operational modal analysis using frequency domain decomposition.
   - **figs_and_tables.py**: Numerous methods to implement the figures and tables utilized in the report.
   - **main.py**: Routines to implement the reduction of elastic modulus reduction rates.
   - **data**: Training and test dataset provided in IC-SHM 2022 Committee, training histories for neural networks and Bayesian optimization.
   - **figs**: Figures for random vibration, normalized root mean square error matrix, Bayesian optimization and neural network convergence plots.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

Thank XXX
