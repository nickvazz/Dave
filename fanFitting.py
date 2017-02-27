import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
from scipy import stats
import os, argparse

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-U','--U', help='Us to train (separated by commas)', required=True)
parser.add_argument('-R', '--Runs', help='# of runs', required=True)
parser.add_argument('-LR', '--LearningRate', help='', required=True)
parser.add_argument('-TE', '--TrainingEpochs', help='Number of training steps', required=True)
parser.add_argument('-BS', '--BatchSize', help='Size of batch to lessen memory strain', required=True)
parser.add_argument('-LT', '--LayerTrials', help='L1,L2,L3,L4', required=True)
parser.add_argument('-P', '--Plotting', help='Turning plotting on/off', required=True)
parser.add_argument('-WP','--WhichPlots', help='Which specific plots', required=True)
parser.add_argument('-ZT', '--ZoomTemps', help='Zoom in on temps = True/False', required=True)
parser.add_argument('-Tmin', '--TempMin', help='Min temp to load from data', required=True)
parser.add_argument('-Tmax', '--TempMax', help='Max temp to load from data', required=True)

args = vars(parser.parse_args())

U = args['U']
run_num = int(args['Runs'])
learning_rate = float(args['LearningRate'])
training_epochs = int(args['TrainingEpochs'])
batch_size = int(args['BatchSize'])
layer_trials = [map(int, args['LayerTrials'].split(','))]
tempRange = eval(args['ZoomTemps'])
tempMin = float(args['TempMin'])
tempMax = float(args['TempMax'])
plotOn = eval(args['Plotting'])
whichPlots = args['WhichPlots']

run_str = 'run' + str(run_num) + '_U' + str(U) + '/'

if not os.path.exists(run_str + 'Random_Tree_Fits'):
    os.makedirs(run_str + 'Random_Tree_Fits')

embedding = 'Random_Trees'

def plotting(U, L, embedding):
    data = np.loadtxt(run_str + 'embedded_data/200_100_10_5_10_0.1_100/AfterL' + str(L) + '/randTrees.txt')

    df = pd.DataFrame(data=data).T
    df.columns = ['x','y','T']
    df.sort_values('T', inplace=True)
    temps = sorted(list(set(df['T'])))

    fit_xs = np.arange(0,1.01,0.01)
    plt.figure(figsize=(10,10))
    norm = mpl.colors.Normalize(vmin=.1,vmax=.4)
    slopes = []

    for temp in temps:
        xs = df.loc[df['T'] == temp]['x']
        ys = df.loc[df['T'] == temp]['y']
        slope, intercept, r_val, p_val, std_err = stats.linregress(xs,ys)
        fit_ys = []
        for xval in fit_xs:
            yval = xval*slope + intercept
            fit_ys.append(yval)
        slopes.append(slope)

        color = 'Vega20b'
        plt.scatter(xs,ys,color=cm.jet(norm(temp)),s=1.5, label='')
        plt.plot(fit_xs, fit_ys, label=temp, c=cm.jet(norm(temp)),linewidth=2) #, c=cm.cool(temp))

    plt.title('U' + str(U) + ' L' + str(L) + ' ' + embedding)
    plt.legend(loc='upper left')

    angles = list(map(np.arctan, slopes))
    angle_diff = []
    for i in range(len(angles)-1):
        dif = np.rad2deg(abs(angles[i+1] - angles[i]))
        angle_diff.append(dif)

    for i in range(len(angle_diff)):
        txt = r'$\angle$' + ' T = ' + str(temps[i+1]) + ' & ' + str(temps[i]) + ' : ' + str(round(angle_diff[i], 2)) + r'$\degree$'
        plt.annotate(txt, xy=(0.05, .2-(i*.025)), xycoords='axes fraction',size=8)
    # plt.savefig('U' + str(U) + '_' + str(L) + '_' + embedding + '_deep.png')
    # plt.savefig('Random_Tree_Fits/U' + str(U) + '_' + str(L) + '_' + embedding + '.png')
    if tempRange == True:
        plt.savefig(run_str + 'Random_Tree_Fits/U' + str(U) + '_' + str(L) + '_' + embedding + '_zoom.png')
    else:
        plt.savefig(run_str + 'Random_Tree_Fits/U' + str(U) + '_' + str(L) + '_' + embedding + '.png')
    # plt.show()

for L in [3,4]:
    plotting(U,L,embedding)
