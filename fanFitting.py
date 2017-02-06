import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
from scipy import stats

Us = [4,5,6,8,9,10,12,14,16,20]
tempMins = [.18,.22,.29,.32,.38,.30,.29,.26,.22,.18]
tempMaxs = [.22,.27,.34,.37,.42,.35,.34,.31,.27,.22]


for U in [8]
    for i in range(10):
        tempRange = False
        run_num = i + 1

        run_str = 'run' + str(run_num) + '_U' + str(U) + '/'

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
                plt.scatter(xs,ys,c=cm.plasma(norm(temp)),s=1, label='')
                plt.plot(fit_xs, fit_ys, label=temp, c=cm.plasma(norm(temp))) #, c=cm.cool(temp))
            plt.title('U' + str(U) + ' L' + str(L) + ' ' + embedding)
            plt.legend(loc='upper left')

            angles = map(np.arctan, slopes)
            angle_diff = []
            for i in range(len(angles)-1):
                dif = np.rad2deg(abs(angles[i+1] - angles[i]))
                angle_diff.append(dif)

            for i in range(len(angle_diff)):
                txt = r'$\angle$' + ' T = ' + str(temps[i+1]) + ' & ' + str(temps[i]) + ' : ' + str(round(angle_diff[i], 2)) + r'$\degree$'
                plt.annotate(txt, xy=(0.05, .2-(i*.025)), xycoords='axes fraction',size=8)
            # plt.savefig('U' + str(U) + '_' + str(L) + '_' + embedding + '_deep.png')
            # plt.savefig('Random_Tree_Fits/U' + str(U) + '_' + str(L) + '_' + embedding + '.png')
            plt.savefig(run_str + 'Random_Tree_Fits/U' + str(U) + '_' + str(L) + '_' + embedding + '_zoom.png')
            # plt.show()

        for L in [3,4]:
            plotting(U,L,embedding)
