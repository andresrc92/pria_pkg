import numpy as np
import sys
from matplotlib import pyplot as plt
import os
import time


with open(sys.argv[1], "rb") as f:
    npzfile = np.load(f)

    start = 1

    train = npzfile.files[0]
    val = npzfile.files[1]
    
    train_loss = npzfile[train]
    val_loss = npzfile[val]
    
    epochs = np.arange(0,len(val_loss),1)

    plt.plot(epochs[start:], train_loss[start:], label="train")
    plt.plot(epochs[start:], val_loss[start:], label="val")
    plt.legend()
    plt.show()

    if len(npzfile.files) > 2:
        train = npzfile.files[2]
        val = npzfile.files[3]
        
        train_error = npzfile[train]
        val_error = npzfile[val]
        
        epochs = np.arange(0,len(val_loss),1)

        # plt.plot(epochs[start:], train_error[start:], label="train")
        # plt.plot(epochs[start:], val_error[start:], label="val")
        # plt.legend()
        # plt.show()
        print(train_error.shape)
        print(val_error.shape)

