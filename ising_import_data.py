import pandas as pd
import numpy as np

def getData(L=40, periodic=False):
    # df = pd.DataFrame(pd.read_csv('isingData/2D' + str(L) + '_p_test_data.txt',header=None,sep='\s+'))
    if periodic == True:
        df = pd.DataFrame(pd.read_csv('isingData/2D' + str(L) + '_training_data_np.txt',header=None,sep='\s+'))
    else:
        df = pd.DataFrame(pd.read_csv('isingData/2D' + str(L) + '_training_data.txt',header=None,sep='\s+'))

    temp = df[[0]].values.reshape(-1)
    data = df[list(range(1,L**2+1))].values
    data = data.reshape(temp.shape[0],L,L)
    return temp, data
