import glob, re
import numpy as np
import pandas as pd

def dataAndLabels(folder_name, ZoomTemps=False, tempMin=0, tempMax=10, changingVar='T'):
    class DataSet(object):
        def __init__(self, images, labels):
            self._images = np.asarray(images)
            self._labels = np.asarray(labels)
            self._num_examples = len(images)
            self._epochs_completed = 0
            self._index_in_epoch = 0
        @property
        def images(self):
            return self._images
        @property
        def labels(self):
            return self._labels
        @property
        def num_examples(self):
            return self._num_examples
        @property
        def epochs_completed(self):
            return self._epochs_completed

        def next_batch(self, batch_size):
            start = self._index_in_epoch
            self._index_in_epoch += batch_size
            # print self._index_in_epoch, self._num_examples
            if self._index_in_epoch > self._num_examples:
                # Finished epoch
                self._epochs_completed += 1
                # Shuffle the data
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                # print perm
                # print self._images[perm]
                self._images = self._images[perm]
                self._labels = self._labels[perm]
                # Start next epoch
                start = 0
                self._index_in_epoch = batch_size
                assert batch_size <= self._num_examples
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]

    allData = []
    allLabels = []
    dataTrain = []
    dataTest = []
    dataValidation = []
    labelsTest = []
    labelsTrain = []
    labelsValidation = []

    files = glob.glob(folder_name)


    labels = []
    files2keep = []
    counter = 0
    FullDataSet = pd.DataFrame()
    for thing in files:
    # for thing in files[:3]:
        if changingVar == 'T':
            newThing = re.split('[|_ |, |. |/]',thing)[-4:-2]
            label = float(newThing[0][1:] + '.' + newThing[1])

        elif changingVar == 'Mu':
            str1 = thing.split('.')[1][-1:]
            str2 = thing.split('.')[2][:1]
            label = float(str1 + '.' + str2)

        if ZoomTemps == True:
            if tempMin <= label <= tempMax:
                files2keep.append(counter)
                labels.append(label)
        else:
            labels.append(label)
            files2keep.append(counter)
        counter += 1

        currentFile = thing
        df = pd.DataFrame(pd.read_csv(currentFile))
        df.columns = ['data']
        df['label'] = np.full(len(df['data']), label)
        df['fixedData'] = df['data'].str[:-(12)]#.str.split(pat='')

        df['fixedData'] = list(map(list,df['fixedData']))
        newFixedData = []
        for i in range(len(df['data'])):
            newFixedData.append(list(map(float,df['fixedData'][i])))
        df['fixedData'] = newFixedData
        FullDataSet = FullDataSet.append(df)

        print (changingVar + ' =', label, ' loaded')

    FullDataSet = FullDataSet.sample(frac=1, random_state=1).reset_index(drop=True) # shuffles dataFrame
    FullDataSet = FullDataSet[FullDataSet['label'] <= tempMax]
    FullDataSet = FullDataSet[FullDataSet['label'] >= tempMin]

    dataLen = len(FullDataSet['label'])

    dataTrain = list(map(list,FullDataSet['fixedData'][:int(dataLen*.7)].values))
    labelsTrain = FullDataSet['label'][:int(dataLen*.7)].values

    dataValidation = FullDataSet['fixedData'][int(dataLen*.7):int(dataLen*.8)].values
    labelsValidation = FullDataSet['label'][int(dataLen*.7):int(dataLen*.8)].values

    dataTest = FullDataSet['fixedData'][int(dataLen*.8):].values
    labelsValidation= FullDataSet['label'][int(dataLen*.8):].values

    data_sets = DataSet([],[])
    data_sets.train = DataSet(dataTrain, labelsTrain)
    data_sets.validation = DataSet(dataValidation, labelsTrain)
    data_sets.test = DataSet(dataTest, labelsTest)

    return data_sets



    # data = []
    # files = np.take(files, files2keep)
    #
    # for i in range(len(files)):
    #     aFile = open(files[i], 'r')
    #     for line in aFile:
    #
    #         temp = list(line[:-(13+700)]) + [labels[i]]
    #         temp = list(map(float,temp))
    #         # temp = [*map(float,temp)]
    #         data.append(temp)
    #
    #     aFile.close()
    #     print (changingVar + ' =', labels[i], ' loaded')
    #
    # for i in range(len(data)):
    #     label = float(data[i][-1])
    #     dataval = data[i][:-1]
    #
    #     if i < len(data) * 0.7:
    #         labelsTrain.append(label)
    #         dataTrain.append(dataval)
    #
    #     elif 0.7 * len(data) < i < 0.8 * len(data):
    #         labelsValidation.append(label)
    #         dataValidation.append(dataval)
    #
    #     else:
    #         dataTest.append(dataval)
    #         labelsTest.append(label)
    #
    #     allLabels.append(label)
    #     allData.append(dataval)
    #
    # data_sets = DataSet([],[])
    # data_sets.train = DataSet(dataTrain, labelsTrain)
    # data_sets.validation = DataSet(dataValidation, labelsTrain)
    # data_sets.test = DataSet(dataTest, labelsTest)
    #
    # return data_sets
