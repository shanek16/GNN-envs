import matplotlib.pyplot as plt
from matplotlib.pyplot import gca
import numpy as np

font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 14}
r_max=3
n_agents=10

x = np.zeros((n_agents, 2))
length = np.sqrt(np.random.uniform(0, r_max, size=(n_agents,)))
angle = np.pi * np.random.uniform(0, 2, size=(n_agents,))
x[:, 0] = length * np.cos(angle)
x[:, 1] = length * np.sin(angle)

fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot(x[:, 0], x[:, 1],
                        'b.')  # Returns a tuple of line objects, thus the comma
ax.plot([0], [0], 'kx')
ax.plot([1,1],[-1,1],'g')
plt.ylim(-1.0 * r_max, 1.0 * r_max)
plt.xlim(-1.0 * r_max, 1.0 * r_max)
a = gca()
a.set_xticklabels(a.get_xticks(),family='sans-serif',
        weight='bold',
        size=14)
a.set_yticklabels(a.get_yticks(),family='sans-serif',
        weight='bold',
        size=14)
plt.title('GNN Controller')
fig = fig
line1 = line1

line1.set_xdata(x[:, 0])
line1.set_ydata(x[:, 1])
fig.canvas.draw()
plt.show()