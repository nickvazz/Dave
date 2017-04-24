import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,5,100)

f, ax = plt.subplots(2,10, sharex=True, figsize=(20,4))


for i in range(10):
    ax[0,i].plot(x,x**i)
    ax[0,i].set_title(i)

    ax[1,i].plot(x*2,x**(i+10))
    ax[1,i].set_title(i+10)

plt.show()
