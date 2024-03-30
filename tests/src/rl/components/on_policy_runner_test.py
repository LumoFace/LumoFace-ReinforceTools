import h5py
import matplotlib.pyplot as plt
import numpy as np


f = h5py.File("test_rl_components_on_policy_runner_buffer.h5", "r")

N_ENVIRONMENTS = 3

for k in f["buffer"]:
    if k != "data":
        data = f["buffer"][k][:]
        data_env_1 = data[np.arange(0, data.shape[0], N_ENVIRONMENTS), :]
  