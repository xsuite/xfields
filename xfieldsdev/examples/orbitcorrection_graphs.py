import pickle
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats
import math

shifts = [0, 1, 2, 3, 4, 5, 6]

bbp = 0.00358
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
xshift_means_x_b1 = []
xshift_means_x_b2 = []
xshift_means_y_b1 = []
xshift_means_y_b2 = []
yshift_means_x_b1 = []
yshift_means_x_b2 = []
yshift_means_y_b1 = []
yshift_means_y_b2 = []
xyshift_means_x_b1 = []
xyshift_means_x_b2 = []
xyshift_means_y_b1 = []
xyshift_means_y_b2 = []

for shift in shifts:
    xfile_name = '/Users/chenying/xsuite_dev/freqanal/meanandrms_x_shift_%i.pickle'%shift
    yfile_name = '/Users/chenying/xsuite_dev/freqanal/meanandrms_y_shift_%i.pickle'%shift
    xyfile_name = '/Users/chenying/xsuite_dev/freqanal/meanandrms_xy_shift_%i.pickle'%shift
    
    with open(xfile_name, 'rb') as xfile:
        xdata_loaded = pickle.load(xfile)
    
    with open(yfile_name, 'rb') as yfile:
        ydata_loaded = pickle.load(yfile)
        
    with open(xyfile_name, 'rb') as xyfile:
        xydata_loaded = pickle.load(xyfile)
    
    xshift_means_x_b1.append(np.mean(xdata_loaded['xmean_b1']))
    xshift_means_x_b2.append(np.mean(xdata_loaded['xmean_b2']))
    xshift_means_y_b1.append(np.mean(xdata_loaded['ymean_b1']))
    xshift_means_y_b2.append(np.mean(xdata_loaded['ymean_b2']))
    
    yshift_means_x_b1.append(np.mean(ydata_loaded['xmean_b1']))
    yshift_means_x_b2.append(np.mean(ydata_loaded['xmean_b2']))
    yshift_means_y_b1.append(np.mean(ydata_loaded['ymean_b1']))
    yshift_means_y_b2.append(np.mean(ydata_loaded['ymean_b2']))
    
    xyshift_means_x_b1.append(np.mean(xydata_loaded['xmean_b1']))
    xyshift_means_x_b2.append(np.mean(xydata_loaded['xmean_b2']))
    xyshift_means_y_b1.append(np.mean(xydata_loaded['ymean_b1']))
    xyshift_means_y_b2.append(np.mean(xydata_loaded['ymean_b2']))

ax1.plot(shifts, -np.array(xshift_means_x_b1)-np.array(xshift_means_x_b2), label = "X_b1 - X_b2")
ax1.plot(shifts, -np.array(xshift_means_y_b1)+np.array(xshift_means_y_b2), label = "Y_b1 - Y_b2")
ax1.set_title("X shifts difference in centroids")
ax1.set_xlabel("Shift in x")
ax1.set_ylabel('Difference in centroids')
ax1.legend()

ax2.plot(shifts, -np.array(yshift_means_x_b1)-np.array(yshift_means_x_b2), label = "X_b1 - X_b2")
ax2.plot(shifts, -np.array(yshift_means_y_b1)+np.array(yshift_means_y_b2), label = "Y_b1 - Y_b2")
ax2.set_title("Y shifts difference in centroids")
ax2.set_xlabel("Shift in y")
ax2.set_ylabel('Difference in centroids')
ax2.legend()

ax3.plot(shifts, -np.array(xyshift_means_x_b1)+np.array(xyshift_means_x_b2), label = "X_b1 - X_b2")
ax3.plot(shifts, -np.array(xyshift_means_y_b1)+np.array(xyshift_means_y_b2), label = "Y_b1 - Y_b2")
ax3.set_title("X+Y shifts difference in centroids")
ax3.set_xlabel("Shift in x+y")
ax3.set_ylabel('Difference in centroids')
ax3.legend()
plt.show()