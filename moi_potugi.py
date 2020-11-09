import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve as conv
from numpy.fft import ifft

t = np.arange(0, 10, 0.1)
t2 = np.arange(0, 9.95, 0.05)
f = 15e9
ta = 0.01
m1 = 0.092
m2 = -m1
T = 0.01
## I suppose the signal is just a sinus
sinus = np.sin(t2)




class Sa:
    #def __init__(self, mu1, mu2, f_average, delay, tau, time, freq):
    def __init__(self, mu1, f_average, tau, time, delay, signal, mu2):
        self.mu1 = mu1
        self.mu2 = mu2
        self.f_average = f_average
        self.delay = delay
        self.tau = tau
        self.time = time
        self.signal = signal
        #self.freq = freq

## Printing of the lasers diod signal

    def ld_sig(self):
        sig = 2 / (self.tau * 3.14 ** 0.5) * np.exp(-2 * self.time / self.tau) ** 2 * np.exp((0 + 1j) * self.f_average * self.time)
        plt.figure(1)
        graph1 = plt.plot(self.time, sig)
        print('Plot: ', len(graph1), graph1)
        plt.show()
        #plt.plot(self.time, sig)

## This function is comparing the result of the convolution of the lasers diod signal and impulse response of the
## dispersion element and theoretical result.

    def disp_el(self):
        sig = 2 / (self.tau * 3.14 ** 0.5) * np.exp(-2 * self.time / self.tau) ** 2 * np.exp((0 + 1j) * self.f_average * self.time)
        imp_disp1 = np.exp((0 + 1j)*(self.time - self.delay)**2 / self.mu1) * np.exp((0 + 1j) * self.f_average * self.time)
        disp1_out = conv(sig, imp_disp1)
        theory = 100 * np.exp((0+1j) * (self.f_average * self.time + self.time ** 2 / 2 * self.mu1))
        plt.figure()
        graph = plt.plot(disp1_out)
        graph1 = plt.plot(theory)
        print('Plot: ', len(graph), graph)
        print('Plot: ', len(graph1), graph1)
        plt.show()

## displays the EOM output signal
    def eom(self):
        sig = 2 / (self.tau * 3.14 ** 0.5) * np.exp(-2 * self.time / self.tau) ** 2 * np.exp((0 + 1j) * self.f_average * self.time)
        imp_disp1 = np.exp((0 + 1j) * (self.time - self.delay) ** 2 / self.mu1) * np.exp((0 + 1j) * self.f_average * self.time)
        disp1_out = conv(sig, imp_disp1)
        eom_out = self.signal * disp1_out
        plt.figure()
        graph = plt.plot(eom_out)
        print('Plot: ', len(graph), graph)
        plt.show()

    def disp_el2(self):
        sig = 2 / (self.tau * 3.14 ** 0.5) * np.exp(-2 * self.time / self.tau) ** 2 * np.exp((0 + 1j) * self.f_average * self.time)
        imp_disp1 = np.exp((0 + 1j) * (self.time - self.delay) ** 2 / self.mu1) * np.exp((0 + 1j) * self.f_average * self.time)
        disp1_out = conv(sig, imp_disp1)
        eom_out = self.signal * disp1_out
        imp_disp2 = np.exp((0 + 1j) * (self.time - self.delay) ** 2 / self.mu2) * np.exp((0 + 1j) * self.f_average * self.time)
        disp_out = conv(eom_out, imp_disp2)
        plt.figure(1) #output of disp el 2
        graph1 = plt.plot(disp_out)
        print('Plot: ', len(graph1), graph1)
        plt.show()








newsa = Sa(m1, f, ta, t, T, sinus, m2)
newsa.disp_el2()
