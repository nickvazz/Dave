import numpy as np
import glob
import pandas as pd

def getTempData(U=8):
    def getData(U=8):
        fileFolders = glob.glob('Hubbard*/*')
        for folder in fileFolders[:1]:
            str_ = folder.split('_')[:2] + folder.split('_')[-1:]
            str_.insert(2, 'U' + str(U))
            return '_'.join(str_)

    files = glob.glob(getData(U)+'/*')

    def fixRow(x):
        return x.split(' ')[0]


    num_data_points = 2
    data = np.asarray([])
    temps = np.asarray([])
    for f in files[15:25]:
        df = pd.DataFrame(pd.read_csv(f,header=None))[:num_data_points]
        # df[0] = [np.asarray(map(int,np.asarray(list(x.split(' ')[0])))).reshape(200,4,4,4) for x in df[0]]
        # df[0] = [np.asarray(map(int,np.asarray(list(x.split(' ')[0])))) for x in df[0]]
        for item in df[0]:
            item = list(item)[:12800]
            item = np.asarray(item, dtype=int)
            # item = item.reshape(64,200)
            # item = item.reshape(4,4,4,200).swapaxes(1,2).swapaxes(1,2)
            print item, f
            data = np.append(data, item)



        temperture = float(f.split('.')[0].split('T')[-1] + '.' + f.split('.')[-3])
        temps = np.append(temps, np.ones(num_data_points) * temperture)
    # print data, type(data), len(data)
    # print temps, type(temps), len(temps)
    print data.shape, 'shape'
    return data, temps
# getTempData()
