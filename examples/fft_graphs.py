import pickle
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats
import math

shifts = [0, 1, 2, 3, 4, 5, 6]
turns = np.arange(0,32768, 1)
bbp = 0.00358
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()
fig5, ax5 = plt.subplots()
fig6, ax6 = plt.subplots()
for shift in shifts:

# Specify the file name
    xshift_file_name = '/Users/chenying/xsuite_dev/freqanal/meanandrms_x_shift_%i.pickle'%shift
    yshift_file_name = '/Users/chenying/xsuite_dev/freqanal/meanandrms_y_shift_%i.pickle'%shift
    xyshift_file_name = '/Users/chenying/xsuite_dev/freqanal/meanandrms_xy_shift_%i.pickle'%shift

# Open the file for reading
    with open(xshift_file_name, 'rb') as xfile:
    # Load the data from the file
        x_data_loaded = pickle.load(xfile)
        
    with open(yshift_file_name, 'rb') as yfile:
    # Load the data from the file
        y_data_loaded = pickle.load(yfile)
    
    with open(xyshift_file_name, 'rb') as xyfile:
    # Load the data from the file
        xy_data_loaded = pickle.load(xyfile)
# Print the loaded data
    
    freqs = np.fft.fftshift(np.fft.fftfreq(len(turns)))
    mask = freqs > 0
    ax1.plot(freqs[mask], (np.abs(np.fft.fftshift(np.fft.fft(x_data_loaded['xmean_b1'])))[mask]), label = "Xmean_b1 with %i$\sigma$ xshift"%shift)
    ax1.plot(freqs[mask], (np.abs(np.fft.fftshift(np.fft.fft(x_data_loaded['xmean_b2'])))[mask]), label = "Xmean_b2 with %i$\sigma$ xshift"%shift)
    ax1.set_xlim(0.3,0.32)
    ax2.plot(freqs[mask], (np.abs(np.fft.fftshift(np.fft.fft(x_data_loaded['ymean_b1'])))[mask]), label = "Ymean_b1 with %i$\sigma$ xshift"%shift)
    ax2.plot(freqs[mask], (np.abs(np.fft.fftshift(np.fft.fft(x_data_loaded['ymean_b2'])))[mask]), label = "Ymean_b2 with %i$\sigma$ xshift"%shift)
    ax2.set_xlim(0.31, 0.33)
    ax3.plot(freqs[mask], (np.abs(np.fft.fftshift(np.fft.fft(y_data_loaded['xmean_b1'])))[mask]), label = "Xmean_b1 with %i$\sigma$ yshift"%shift)
    ax3.plot(freqs[mask], (np.abs(np.fft.fftshift(np.fft.fft(y_data_loaded['xmean_b2'])))[mask]), label = "Xmean_b2 with %i$\sigma$ yshift"%shift)
    ax3.set_xlim(0.3, 0.32)
    
    ax4.plot(freqs[mask], (np.abs(np.fft.fftshift(np.fft.fft(y_data_loaded['ymean_b1'])))[mask]), label = "Ymean_b1 with %i$\sigma$ yshift"%shift)
    ax4.plot(freqs[mask], (np.abs(np.fft.fftshift(np.fft.fft(y_data_loaded['ymean_b2'])))[mask]), label = "Ymean_b2 with %i$\sigma$ yshift"%shift)
    ax4.set_xlim(0.31, 0.33)
    
    ax5.plot(freqs[mask], (np.abs(np.fft.fftshift(np.fft.fft(y_data_loaded['xmean_b1'])))[mask]), label = "Xmean_b1 with %i$\sigma$ x+yshift"%shift)
    ax5.plot(freqs[mask], (np.abs(np.fft.fftshift(np.fft.fft(y_data_loaded['xmean_b2'])))[mask]), label = "Xmean_b2 with %i$\sigma$ x+yshift"%shift)
    ax5.set_xlim(0.3, 0.32)
    
    ax6.plot(freqs[mask], (np.abs(np.fft.fftshift(np.fft.fft(y_data_loaded['ymean_b1'])))[mask]), label = "Ymean_b1 with %i$\sigma$ x+yshift"%shift)
    ax6.plot(freqs[mask], (np.abs(np.fft.fftshift(np.fft.fft(y_data_loaded['ymean_b2'])))[mask]), label = "Ymean_b2 with %i$\sigma$ x+yshift"%shift)
    ax6.set_xlim(0.31, 0.33)
    
    ax1.set_title("X mean FFT %i$\sigma$ xshift"%shift)
    ax1.set_xlabel("Frequency")
    ax1.set_ylabel('Amplitude')
    ax1.legend()
    ax2.set_title("X mean FFT %i$\sigma$ yshift"%shift)
    ax2.set_xlabel("Frequency")
    ax2.set_ylabel('Amplitude')
    ax2.legend()

    ax3.set_title("Y mean FFT %i$\sigma$ xshift"%shift)
    ax3.set_xlabel("Frequency")
    ax3.set_ylabel('Amplitude')
    ax3.legend()
    ax4.set_title("Y mean FFT %i$\sigma$ yshift"%shift)
    ax4.set_xlabel("Frequency")
    ax4.set_ylabel('Amplitude')
    ax4.legend()    
    
    ax5.set_title("X mean FFT %i$\sigma$ x+yshift"%shift)
    ax5.set_xlabel("Frequency")
    ax5.set_ylabel('Amplitude')
    ax5.legend()
    ax6.set_title("Y mean FFT %i$\sigma$ x+yshift"%shift)
    ax6.set_xlabel("Frequency")
    ax6.set_ylabel('Amplitude')
    ax6.legend()
plt.show()

import numpy as np
import pickle
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import ffmpeg

shifts = [0, 1, 2, 3, 4, 5, 6]
turns = np.arange(0, 32768, 1)
freqs = np.fft.fftshift(np.fft.fftfreq(len(turns)))
mask = freqs > 0

fig1, ax1 = plt.subplots()

# Animation update function
def update(shift):
    ax1.clear()
    ax1.set_xlim(0.31, 0.33)
    ax1.set_title("Y mean FFT with shift in x+y")
    ax1.set_xlabel("Frequency")
    ax1.set_ylabel("Amplitude")
    
    # Load the data for the current shift
    xyshift_file_name = f'/Users/chenying/xsuite_dev/freqanal/meanandrms_xy_shift_{shift}.pickle'
    with open(xyshift_file_name, 'rb') as xyfile:
        xy_data_loaded = pickle.load(xyfile)
    
    # Plot the data for the current shift
    ax1.plot(freqs[mask], np.abs(np.fft.fftshift(np.fft.fft(xy_data_loaded['ymean_b1'])))[mask], label=f"Ymean_b1 with {shift}$\sigma$ yshift")
    ax1.plot(freqs[mask], np.abs(np.fft.fftshift(np.fft.fft(xy_data_loaded['ymean_b2'])))[mask], label=f"Ymean_b2 with {shift}$\sigma$ yshift")
    ax1.legend()

# Create animation
ani = FuncAnimation(fig1, update, frames=shifts, repeat=False, interval = 2000)

ani.save('/Users/chenying/Desktop/Project_Figures/meanFFT_xyshift_y_centroids_animation.mp4', writer='ffmpeg', fps=2)

