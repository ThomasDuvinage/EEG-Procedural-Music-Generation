# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 00:47:12 2019

@author: thoma
"""

import numpy as np
import matplotlib.pyplot as plt
import serial

number_data_max = 5
number_data = 0
buffer = np.zeros(shape=(5,1))
line = ''

with serial.Serial('COM8', 19200, timeout=1) as ser:

    while(number_data != number_data_max):
        bit = ser.read()
        bit = bit.decode("utf-8")
        if(bit == ','):
            line = line[0:-2]#permet d'enlever \r\n à la fin de la chaine de caracteres
            buffer[number_data] = float(line)
            print(buffer[number_data])
            number_data += 1
            line = ''
        else:
            line = line + bit
        
    
# definition du signal
dt = 0.1
T1 = 2
T2 = 5
t = np.arange(0, T1*T2, dt)
signal = 2*np.cos(2*np.pi/T1*t) + np.sin(2*np.pi/T2*t)

# affichage du signal
plt.subplot(211)
plt.plot(t,signal)

# calcul de la transformee de Fourier et des frequences
fourier = np.fft.fft(signal)
n = signal.size
freq = np.fft.fftfreq(n, d=dt)

# affichage de la transformee de Fourier
plt.subplot(212)
plt.plot(freq, fourier.real, label="real")
plt.plot(freq, fourier.imag, label="imag")
plt.legend()

plt.show()