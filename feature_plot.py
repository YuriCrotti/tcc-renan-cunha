from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import matplotlib as mpl

mpl.style.use('default')
rcParams.update({'figure.autolayout': True})

[fs, x_good] = audioBasicIO.readAudioFile("samples/good/5.wav");
F_good = audioFeatureExtraction.stFeatureExtraction(x_good, fs, 0.050*fs, 0.025*fs)

[fs, x_bad] = audioBasicIO.readAudioFile("samples/bad/5.wav");
F_bad = audioFeatureExtraction.stFeatureExtraction(x_bad, fs, 0.050*fs, 0.025*fs);



# ZCR
plt.figure()
plt.plot(F_good[0,:], 'C2', linestyle = '-.')
plt.plot(F_bad[0,:], 'C3', linestyle = ':')
plt.show()
plt.legend(['Sem defeito', 'Com defeito'], loc='upper right')
plt.title('ZCR do sinal por quadro')
plt.xlabel('Quadro')
plt.ylabel('ZCR')

# ENERGIA
plt.figure()
plt.plot(F_good[1,:], 'C2', linestyle = '-.')
plt.plot(F_bad[1,:], 'C3', linestyle = ':')
plt.show()
plt.legend(['Sem defeito', 'Com defeito'], loc='upper right')
plt.title('Energia do sinal por quadro')
plt.xlabel('Quadro')
plt.ylabel('Energia')

# ENTROPIA
plt.figure()
plt.plot(F_good[2,:], 'C2', linestyle = '-.')
plt.plot(F_bad[2,:], 'C3', linestyle = ':')
plt.show()
plt.legend(['Sem defeito', 'Com defeito'], loc='bottom left')
plt.title('Entropia do sinal por quadro')
plt.xlabel('Quadro')
plt.ylabel('Entropia')

# Spectral Centroid
plt.figure()
plt.plot(F_good[3,:], 'C2', linestyle = '-.')
plt.plot(F_bad[3,:], 'C3', linestyle = ':')
plt.show()
plt.legend(['Sem defeito', 'Com defeito'], loc='upper left')
plt.title('Centroide espectral do sinal por quadro')
plt.xlabel('Quadro')
plt.ylabel('Centroide espectral')

# Spectral Spread
plt.figure()
plt.plot(F_good[4,:], 'C2', linestyle = '-.')
plt.plot(F_bad[4,:], 'C3', linestyle = ':')
plt.show()
plt.legend(['Sem defeito', 'Com defeito'], loc='upper left')
plt.title('Spectral Spread do sinal por quadro')
plt.xlabel('Quadro')
plt.ylabel('Spectral Spread')

# Spectral Flux
plt.figure()
plt.plot(F_good[6,:], 'C2', linestyle = '-.')
plt.plot(F_bad[6,:], 'C3', linestyle = ':')
plt.show()
plt.legend(['Sem defeito', 'Com defeito'], loc='upper left')
plt.title('Spectral Flux do sinal por quadro')
plt.xlabel('Quadro')
plt.ylabel('Spectral Flux')