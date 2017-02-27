import os

dataDimension = '2D'
U = 8
runs = 1
learning_rate = 0.1
training_epochs = 100
batch_size = 10
layer_trials = '200,100,10,5'
ZoomTemps = False
Tmin = 0.2
Tmax = 0.22
plotOn = True
WhichPlots = '1,2,3'

def run(U, runs, dataDimension, learning_rate, training_epochs, batch_size, layer_trials, ZoomTemps, Tmin, Tmax, plotOn, WhichPlots):
    flag_string = ' -U ' + str(U) \
                + ' -R ' + str(runs) \
                + ' -DataDim ' + str(dataDimension) \
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

    os.system('python2 autoencoder.py' + flag_string)
    # os.system('python2 bottlenecks.py' + flag_string)
    # os.system('python2 lowerDimensionEmbed.py' + plot_flag_string)
    # os.system('python2 fanFitting.py' + plot_flag_string)

# for runs in range(1,11):
for runs in range(1,runs+1):
    run(U, runs, dataDimension, learning_rate, training_epochs, batch_size, layer_trials, ZoomTemps, Tmin, Tmax, plotOn, WhichPlots)
