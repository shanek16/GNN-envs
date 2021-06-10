import numpy as np

v_max = 9
dt = 0.01

v_hist = np.zeros((10,2))
v_hist[0,:] = [v_max, 0]
print('v_hist:\n {}'.format(v_hist))

t = np.pi/4 * np.arange(1,10) * dt
print('t:\n {}'.format(t))
x = np.ones((2,9))*[-v_max * np.sin(t - np.pi/2), v_max * np.cos(t - np.pi/2)]
print('x.T:\n {}'.format(x.T))
v_hist[1:10, :] = x.T
print('v_hist:\n {}'.format(v_hist))
Rx_final = sum(v_hist[:,0]) * dt
Ry_final = sum(v_hist[:,1]) * dt
print('Rx:\n {}'.format(Rx_final))
print('Ry:\n {}'.format(Ry_final))