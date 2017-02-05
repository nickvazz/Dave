import os
import random
import numpy as np
import hubbard_input_data

U = 20
tempMin = .18
tempMax = .22

data_file = 'Hubbard Data/N4x4x4_L200_U' + str(U) + '_Mu0/*'
mnist = hubbard_input_data.dataAndLabels(data_file, tempMin=tempMin,tempMax=tempMax)

layer_trials = [[200,100,10,5]]
batch_size = 10
learning_rate = 0.1
training_epochs = 1*1E2

all_images = []
all_labels = []
for i in range(len(mnist.test.images)):
    all_images.append(mnist.test.images[i])
    all_labels.append(mnist.test.labels[i])
for i in range(len(mnist.validation.images)):
    all_images.append(mnist.validation.images[i])
    all_labels.append(mnist.validation.labels[i])
for i in range(len(mnist.train.images)):
    all_images.append(mnist.train.images[i])
    all_labels.append(mnist.train.labels[i])

# print len(all_images)

final_l1 = []
final_l2 = []
final_l3 = []
final_l4 = []
cant_find = []
could_find = []

r = random.random()
random.shuffle(all_images, lambda:r)
random.shuffle(all_labels, lambda:r)

try:
    for layer in layer_trials:
        # print layer
        L1, L2, L3, L4 = layer
        n_encoder_hidden_1 = L1
        n_encoder_hidden_2 = L2
        n_encoder_hidden_3 = L3
        n_code = L4

        filename_p1 = str(n_encoder_hidden_1)+'_'+str(n_encoder_hidden_2)+'_'+str(n_encoder_hidden_3)+'_'
        filename_p2 = str(n_code)+'_'+str(batch_size)+'_'+str(learning_rate)+'_'+str(int(training_epochs))
        filename = filename_p1 + filename_p2

        L1W = np.loadtxt('U' + str(U) + '/' + 'layer_tensors/' + filename + '/L1W.txt')#, delimiter='')
        L2W = np.loadtxt('U' + str(U) + '/' + 'layer_tensors/' + filename + '/L2W.txt')#, delimiter='')
        L3W = np.loadtxt('U' + str(U) + '/' + 'layer_tensors/' + filename + '/L3W.txt')#, delimiter='')
        L4W = np.loadtxt('U' + str(U) + '/' + 'layer_tensors/' + filename + '/L4W.txt')#, delimiter='')
        L1b = np.loadtxt('U' + str(U) + '/' + 'layer_tensors/' + filename + '/L1b.txt')#, delimiter='')
        L2b = np.loadtxt('U' + str(U) + '/' + 'layer_tensors/' + filename + '/L2b.txt')#, delimiter='')
        L3b = np.loadtxt('U' + str(U) + '/' + 'layer_tensors/' + filename + '/L3b.txt')#, delimiter='')
        L4b = np.loadtxt('U' + str(U) + '/' + 'layer_tensors/' + filename + '/L4b.txt')#, delimiter='')
        
        for i in range(len(all_images)):
            after_l1 = np.matmul(all_images[i],L1W)
            after_l1 = np.add(after_l1, L1b)
            final_l1.append(np.append(after_l1, all_labels[i]))

            after_l2 = np.matmul(after_l1, L2W)
            after_l2 = np.add(after_l2, L2b)
            final_l2.append(np.append(after_l2, all_labels[i]))

            after_l3 = np.matmul(after_l2, L3W)
            after_l3 = np.add(after_l3, L3b)
            final_l3.append(np.append(after_l3, all_labels[i]))

            after_l4 = np.matmul(after_l3, L4W)
            after_l4 = np.add(after_l4, L4b)
            final_l4.append(np.append(after_l4, all_labels[i]))

        if not os.path.exists('reduced_data/'+ filename):
            os.makedirs('reduced_data/'+ filename)

        np.savetxt('reduced_data/' + filename + '/AfterL1_' + filename + '.txt', final_l1, delimiter=' ', fmt='%10.5f')
        np.savetxt('reduced_data/' + filename + '/AfterL2_' + filename + '.txt', final_l2, delimiter=' ', fmt='%10.5f')
        np.savetxt('reduced_data/' + filename + '/AfterL3_' + filename + '.txt', final_l3, delimiter=' ', fmt='%10.5f')
        np.savetxt('reduced_data/' + filename + '/AfterL4_' + filename + '.txt', final_l4, delimiter=' ', fmt='%10.5f')

        could_find.append((L1,L2,L3,L4))

        final_l4 = []; final_l3 = []; final_l2 = []; final_l1 = []

except:
    cant_find.append((L1,L2,L3,L4))

if len(cant_find) != 0:
    print 'cant find layers: ', cant_find

os.system("python lowerDimensionEmbed.py")
