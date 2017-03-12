import os, platform

pV = platform.python_version()
pyVersion = float(pV[:3])
try:
    import tensorflow
    if pyVersion > 3:
        programString = 'python3 '
    else:
        programString = 'python2.7 '
except ImportError:
    if pyVersion > 3:
        programString = 'python2.7 '
    else:
        programString = 'python3 '

dataDimension = '2D'
U = 8
runs = 1
learning_rate = 0.1
training_epochs = 100
batch_size = 10
# layers = ['100,36,25,9','200,100,10,5','200,100,50,25']
layers = ['200,100,10,5']
ZoomTemps = True
Tmin = 0
Tmax = 0.13
plotOn = True
WhichPlots = '1,2'
# changingVar = 'T' # 'Mu' or 'T'
# data_file = 'N10x10_L200_U8_Mu0/'; changingVar = 'T'
# data_file = 'N4x4x4_L200_U9_T0.32/'; changingVar = 'Mu'
# data_file = '/home/kchng/Quantum Machine Learning/N4x4x4_L200_U' + str(U) + '_Mu0/'
data_file = 'Hubbard_Data/N4x4x4_L200_U' + str(U) + '_Mu0/'; changingVar = 'T'

def run(U, runs, dataDimension, learning_rate, training_epochs, batch_size, layer_trials, ZoomTemps, Tmin, Tmax, plotOn, WhichPlots, changingVar, data_file):
    flag_string = ' -U ' + str(U) \
                + ' -R ' + str(runs) \
                + ' -LR ' + str(learning_rate) \
                + ' -TE ' + str(training_epochs) \
                + ' -BS ' + str(batch_size) \
                + ' -LT ' + str(layer_trials) \
                + ' -ZT ' + str(ZoomTemps) \
                + ' -Tmin ' + str(Tmin) \
                + ' -Tmax ' + str(Tmax) \
                + ' -ChangeVar ' + changingVar \
                + ' -DataFile ' + data_file

    plot_flag_string = flag_string \
                     + ' -P ' + str(plotOn) \
                     + ' -WP ' + str(WhichPlots)
    print(flag_string)

    os.system(programString + 'autoencoder.py' + flag_string + ' -DataDim ' + str(dataDimension))
    # os.system(programString + 'bottlenecks.py' + flag_string + ' -DataDim ' + str(dataDimension))
    # os.system(programString + 'lowerDimensionEmbed.py' + plot_flag_string)
    # os.system(programString + 'fanFitting.py' + plot_flag_string)

# for runs in range(1,11):
layer_trials = layers[0]
for layer_trials in layers:
    for runs in range(1,runs+1):
        run(U, runs, dataDimension, learning_rate, training_epochs, batch_size, layer_trials, ZoomTemps, Tmin, Tmax, plotOn, WhichPlots, changingVar, data_file)
