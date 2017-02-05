import matplotlib.pyplot as plt
import numpy as np

U = [4, 5, 6, 8, 9, 10, 12, 14, 16, 20]
khatami = [.16,.22,.30,.34,.35,.33,.30,.26,.23,.19]
l3 = [.185, .225, .305, .345, .385, .315, .305, .285, .235, .185]
l4 = [.186, .235, .295, .325, .385, .305, .295, .265, .225, .185]

Ll4 = [.2,.245,.315,.345,.4,.325,.315,.285,.245,.2]


plt.ylim(0,.4)
plt.xlabel('U')
plt.ylabel(r'$T_N$')
plt.plot(U,khatami,c='g',label='Khatami')
plt.plot(U,Ll4,c='y', label='pre zoon L4')
# plt.plot(U,l3,c='r',label='L3')
# plt.plot(U,l4,c='b',label='L4')
plt.legend()
plt.show()
