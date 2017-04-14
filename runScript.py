import os
import numpy as np

dataDimension = '3D'
U = 8
Us = [4,5,6,8,9,10,12,14,16,20]
# Us = [8]
runs = 1
# learning_rate = 0.1
# training_epochs = 100
learning_rates = [0.001,0.01,0.1,1,10]
learning_rates = [0.1,0.01]
# epochs = [50,100,150,200,300]
epochs = [500,1000,2000]
epochs = [100]
batch_size = 10
layers = ['80,30,10,5', '30,25,10,5', '175,23,17,5']
layers = ['30,25,10,2']
layers = ['80,30,5,2','175,23,17,2']
ZoomTemps = False
Tmin = 0.28
Tmax = 0.35
plotOn = True
WhichPlots = '1,2,3'
# # Layers to try below
# # np.random.seed(1)
# # np.random.seed(100)
# # np.random.seed(200)
# potentialLayers = [10,30,50,100,200,150,75,90,80,17,23,54,25,69,125,140,175,183,136,14,36,39,22]
# layers = [list(map(str,sorted(np.random.choice(potentialLayers,size=3,replace=False),reverse=True))) for sets in range(20)]
# layers = [a+','+b+','+c+','+str(5) for a,b,c in layers]
# layers = list(set(layers))
# layers = ['80,22,10,5']
def run(U, runs, dataDimension, learning_rate, training_epochs, batch_size, layer_trials, ZoomTemps, Tmin, Tmax, plotOn, WhichPlots):
    flag_string = ' -U ' + str(U) \
                + ' -R ' + str(runs) \
                + ' -LR ' + str(learning_rate) \
                + ' -TE ' + str(training_epochs) \
                + ' -BS ' + str(batch_size) \
                + ' -LT ' + str(layer_trials) \
                + ' -ZT ' + str(ZoomTemps) \
                + ' -Tmin ' + str(Tmin) \
                + ' -Tmax ' + str(Tmax)

    plot_flag_string = flag_string \
                     + ' -P ' + str(plotOn) \
                     + ' -WP ' + str(WhichPlots)
    print(flag_string)
    os.system('python autoencoder.py' + flag_string + ' -DataDim ' + str(dataDimension))
    # os.system('python bottlenecks.py' + flag_string + ' -DataDim ' + str(dataDimension))
    # os.system('python lowerDimensionEmbed.py' + plot_flag_string)
    # os.system('python fanFitting.py' + plot_flag_string)
for runs in range(1,runs+1):
    for learning_rate in learning_rates:
        for layer_trials in layers:
            for U in Us:
                for training_epochs in epochs:
                    run(U, runs, dataDimension, learning_rate, training_epochs, batch_size, layer_trials, ZoomTemps, Tmin, Tmax, plotOn, WhichPlots)
