import os

dataDimension = '2D'
U = 8
runs = 1
learning_rate = 0.1
training_epochs = 100
batch_size = 10
layers = ['200,100,50,25','400,200,100,50','100,36,25,9']
ZoomTemps = False
Tmin = 0.28
Tmax = 0.35
plotOn = True
WhichPlots = '1,2'

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

    os.system('python2 autoencoder.py' + flag_string + ' -DataDim ' + str(dataDimension))
    os.system('python3 bottlenecks.py' + flag_string + ' -DataDim ' + str(dataDimension))
    # os.system('python3 lowerDimensionEmbed.py' + plot_flag_string)
    # os.system('python3 fanFitting.py' + plot_flag_string)

# for runs in range(1,11):
for layer_trials in layers:
    for runs in range(1,runs+1):
        run(U, runs, dataDimension, learning_rate, training_epochs, batch_size, layer_trials, ZoomTemps, Tmin, Tmax, plotOn, WhichPlots)
