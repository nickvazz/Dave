import glob
import numpy as np

def dataAndLabels(folder_name, tempMin=0, tempMax=.4, changingVar='T'):
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

    if changingVar == 'T':
        labels = []
        files2keep = []
        counter = 0
        for thing in files:
            # if it says "ValueError: could not convert string to float: _T0.03"
            # near this line, chance a/b in the thing[a:b]
            if folder_name[:3] == 'Hub':
                try:
                    label_temp = thing[76:90].split('.')[0:2]
                    label = float(label_temp[0] + '.' + label_temp[1])
                except:
                    label_temp = thing[78:90].split('.')[0:2]
                    label = float(label_temp[0] + '.' + label_temp[1])

            elif folder_name[:3] == 'N10':
                try:
                    label_temp = thing[39:45].split('.')[0:2]
                    label = float(label_temp[0] + '.' + label_temp[1])
                except:
                    print('2D failed')
            else:
                try:
                    label_temp = thing[53:57].split('.')[0:2]
                    label = float(label_temp[0] + '.' + label_temp[1])
                except:
                    label_temp = thing[55:58].split('.')[0:2]
                    label = float(label_temp[0] + '.' + label_temp[1])

            label = float(label_temp[0] + '.' + label_temp[1])
            if tempMin <= label <= tempMax:
                files2keep.append(counter)
                labels.append(label)
            counter += 1
        # print(labels, ':tempertures')
        square_len_fix = 700
        data = []
        files = np.take(files, files2keep)
        for i in range(len(files)):
            aFile = open(files[i], 'r')
            for line in aFile:
                if folder_name[:3] == 'Hub':
                    temp = list(line[:-(13+700)]) + [labels[i]]
                    temp = list(map(float,temp))
                    # temp = [*map(float,temp)]
                    data.append(temp)
                elif folder_name[:3] == 'N10':
                    temp = list(line[:-(13+119)]) + [labels[i]]
                    temp = list(map(float,temp))

                    data.append(temp)

            print (changingVar + ' =', labels[i], ' loaded')

        for i in range(len(data)):
            label = float(data[i][-1])
            dataval = data[i][:-1]

            if i < len(data) * 0.7:
                labelsTrain.append(label)
                dataTrain.append(dataval)

            elif 0.7 * len(data) < i < 0.8 * len(data):
                labelsValidation.append(label)
                dataValidation.append(dataval)

            else:
                dataTest.append(dataval)
                labelsTest.append(label)

            allLabels.append(label)
            allData.append(dataval)

        data_sets = DataSet([],[])
        data_sets.train = DataSet(dataTrain, labelsTrain)
        data_sets.validation = DataSet(dataValidation, labelsTrain)
        data_sets.test = DataSet(dataTest, labelsTest)

        return data_sets

    elif changingVar == 'Mu':
        labels = []
        files2keep = []
        counter = 0
        for thing in files:
            str1 = thing.split('.')[1][-1:]
            str2 = thing.split('.')[2][:1]
            label = float(str1 + '.' + str2)

            if tempMin <= label <= tempMax:
                files2keep.append(counter)
                labels.append(label)
            counter += 1

        square_len_fix = 700
        data = []
        files = np.take(files, files2keep)
        for i in range(len(files)):
            aFile = open(files[i], 'r')
            for line in aFile:
                temp = list(line[:-(13+700)]) + [labels[i]]
                temp = list(map(float,temp))
                data.append(temp)
            print (changingVar + ' =', labels[i], ' loaded')

        for i in range(len(data)):
            print(float(i)/len(data))
            label = float(data[i][-1])
            dataval = data[i][:-1]

            if i < len(data) * 0.7:
                labelsTrain.append(label)
                dataTrain.append(dataval)

            elif 0.7 * len(data) < i < 0.8 * len(data):
                labelsValidation.append(label)
                dataValidation.append(dataval)

            else:
                dataTest.append(dataval)
                labelsTest.append(label)

            allLabels.append(label)
            allData.append(dataval)

        data_sets = DataSet([],[])
        data_sets.train = DataSet(dataTrain, labelsTrain)
        data_sets.validation = DataSet(dataValidation, labelsTrain)
        data_sets.test = DataSet(dataTest, labelsTest)

        return data_sets
