import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def checkLayerTensors():
    for counter, U in enumerate([4,5,6,8,9,10,12,14,16,20]):
        datafiles = glob('10RunsFeb15_16/run*_U' + str(U) + '/layer_tensors/*/L4W*')
        for text in datafiles:
            df = pd.DataFrame(pd.read_csv(text, delimiter='\s+'))
            df.columns = range(5)
            matrix = np.ravel(df.values)
            # print(matrix)
            plt.subplot(2,5,counter)
            plt.plot(range(len(matrix)), matrix)
        plt.title('layer4 check, U = ' + str(U), size=8)
        plt.xlabel('flattened matrix index', size=8)
        plt.ylabel('matrix value', size=8)
    plt.show()

# checkLayerTensors()

def checkUniqueTempsLeft():
    for counter, U in enumerate([4,5,6,8,9,10,12,14,16,20]):
        datafiles = glob('10RunsFeb15_16/run*_U' + str(U) + '/reduced_data/*/AfterL4*')
        unique = []
        for text in datafiles:
            df = pd.DataFrame(pd.read_csv(text, delimiter='\s+'))
            df.columns = list(map(str,range(5))) + ['temp']
            unique.append(sorted(df['temp'].unique()))

        unique = sum(unique, [])
        unique = [round(x,2) for x in unique]
        plt.subplot(5,2,counter)
        (keys, values) = zip(*Counter(unique).items())
        plt.bar(keys, values, width=.005)
        plt.title('U = ' + str(U))
        plt.ylabel('temps')
        plt.xlabel('occurances')


    plt.show()

# checkUniqueTempsLeft()

def checkTempsLeft():
    for counter, U in enumerate([4,5,6,8,9,10,12,14,16,20]):
    # for counter, U in enumerate([8]):
        datafiles = glob('10RunsFeb15_16/run*_U' + str(U) + '/reduced_data/*/AfterL4*')
        unique = []
        for idx, text in enumerate(datafiles):
            # print(idx,text)
            df = pd.DataFrame(pd.read_csv(text, delimiter='\s+'))
            df.columns = list(map(str,range(5))) + ['temp']
            unique.append(sorted(df['temp']))

        unique = sum(unique, [])
        unique = [round(x,2) for x in unique]
        plt.subplot(5,2,counter)
        (keys, values) = zip(*Counter(unique).items())
        plt.bar(keys, values, width=.005)
        print(text)
        print(sorted(zip(keys,values)))
        plt.title('U = ' + str(U))
        plt.ylabel('temps')
        plt.xlabel('occurances')


    # plt.show()

# checkTempsLeft()

def checkNumDataPerTemp():
    datafiles = glob('Hubbard_Data/N*U8*/*')
    print (datafiles)
    for text in datafiles:
        df = pd.DataFrame(pd.read_csv(text, delimiter='\s+'))
        print len(df)

checkNumDataPerTemp()
