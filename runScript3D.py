import os

dataDimension = '2D'
U = 12
runs = 1
learning_rate = 0.1
training_epochs = 100
batch_size = 10
layer_trials = '200,100,10,5'
ZoomTemps = False
Tmin = 0.2
Tmax = 0.22
plotOn = True
WhichPlots = '1,2'

Us = [4,5,6,8,9,10,12,14,16,20]
# Us = [8,9,10,12,14,16,20]

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

    # os.system('python3 autoencoder.py' + flag_string)
    # os.system('python3 bottlenecks.py' + flag_string)
    os.system('python3 lowerDimensionEmbed.py' + plot_flag_string)
    os.system('python3 fanFitting.py' + plot_flag_string)


    # Us = [4,5,6,8,9,10,12,14,16,20]
    # tempMins = [.18,.22,.29,.32,.38,.30,.29,.26,.22,.18]
    # tempMaxs = [.22,.27,.34,.37,.42,.35,.34,.31,.27,.22]
    # learning_rate = 0.1
    # training_epochs = 1*1E2
    # batch_size = 10
    # layer_trials = [[200,100,10,5]]

for U in Us:
    for runs in range(1,11):
    # for runs in range(1,2):
        run(U, runs, dataDimension, learning_rate, training_epochs, batch_size, layer_trials, ZoomTemps, Tmin, Tmax, plotOn, WhichPlots)
