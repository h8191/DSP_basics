import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def my_wave(t, wave_ind = 0):
    if wave_ind == 0: return (np.sign(np.sin(t)) + 1) / 2  #square wave function
    if wave_ind == 1: return np.sin(t) + np.cos(2*t) + np.sin(3*t) #mix of sinusoids 
    return np.r_[np.linspace(0,1,len(t)//3), np.linspace(1,0,len(t)//3), np.ones(len(t) - 2*len(t)//3) * 0.5]
    return np.r_[np.linspace(0,1,len(t)//3), np.linspace(1,0,len(t)//3), np.zeros(len(t) - 2*len(t)//3)]

def compute_single_fs_coeff(t, w_n, wave, Type="a"):
    y_cos = np.cos(w_n * t)
    y_sin = np.sin(w_n * t)

    if Type == "a":
        if w_n == 0:    return np.mean(y)
        return np.mean(y_cos * wave) * 2
    
    if Type == "b":     
        if w_n == 0:    return 0 #ideally can be anything, but setting to 0 just for convenience
        return np.mean(y_sin * wave) * 2

def Fourier_series(t, y, n_coeffs):

    coeff_a = [compute_single_fs_coeff(t, i, y, "a") for i in range(n_coeffs)]
    coeff_b = [compute_single_fs_coeff(t, i, y, "b") for i in range(n_coeffs)]

    #reconstruct wavefrom from the coefficients
    y_est = np.zeros_like(y)
    for ind, val in enumerate(coeff_a):
        y_est += val * np.cos(ind * t)

    for ind, val in enumerate(coeff_b):
        y_est += val * np.sin(ind * t)

    return y_est



t = np.linspace(0, 2 * np.pi, 1000) #time from 0 to 2 * pi, sampling frequency = len(array) Ex: 1000 -> 1khz  
y = my_wave(t, 1)




fig, ax = plt.subplots()
# ax.set_ylim([-0.2, 1.2])
line, = ax.plot(t, y, color='red', label='Square wave')
plt.plot(t, y)

def update(n_coeffs):
    Fourier = Fourier_series(t, y, n_coeffs)
    line.set_ydata(Fourier)
    ax.set_title(f'Fourier series approximation with {n_coeffs} coefficients')
    return line,

ani = FuncAnimation(fig, update, frames=np.arange(1, 30), interval=500)

plt.show()
