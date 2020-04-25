import matplotlib.pyplot as plt
import numpy as np

L = 2
X = np.arange(0.01,L/2,1e-2)

fig = plt.figure(figsize=(10,10),dpi=300)
ax = fig.add_axes([0,0,10,10])
ax.set_ylim(0,10)
ax.set_xlim(0,10)
ax.plot(X,2/X)

fig.show()


