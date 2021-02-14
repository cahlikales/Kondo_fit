# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 22:32:15 2020

@author: Aleš Cahlík, FZU
"""
### lmfit package used for the fitting, see https://lmfit.github.io/lmfit-py/model.html
import numpy as np
import matplotlib.pyplot as plt
import nanonispy as nap
from lmfit import Model, Parameters, fit_report, CompositeModel
import math 

plt.close('all') 

### Fitting functions
### Derivative of Fermi-Dirac function FD(E-eV) with rescpect to the energy (Epsilon), 10.1103/PhysRevLett.88.077205
def fermi(eps, Ekk, T, Kb):
    answer = np.exp((Ekk-eps)/(Kb*T)) / ((Kb*T) * (np.exp((Ekk-eps)/(Kb*T)) + 1)**2)
    answer /= np.amax(answer)
    return answer

### Fano function according to 10.1088/0953-8984/21/5/053001
def kondo_fano_b(eps, gamma, amp, Ek, q):
    def epsilon(eps, Ek, gamma):
        return (eps - Ek) / gamma
    return amp*(q + epsilon(eps, Ek, gamma))**2/(1 + epsilon(eps, Ek, gamma)**2)

def kondo_fano(eps, gamma, amp, Ek, q):
    def epsilon(eps, Ek, gamma):
        return (eps - Ek) / gamma
    return amp*(q + epsilon(eps, Ek, gamma))**2/(1 + epsilon(eps, Ek, gamma)**2)

### Frota function according to SI in 10.1038/nphys1876
def kondo_frota(eps, amp, gamma):
    return amp*np.imag(-1j*np.sqrt(1j*gamma/(1j*gamma+eps)))

### Linear background with offset
def line(eps, slope, offset):
    return slope*eps + offset

### Binary operator fot a simple convolution of two arrays.
def convolve(arr, kernel):
    npts = min(arr.size, kernel.size)
    pad = np.ones(npts)
    tmp = np.concatenate((pad*arr[0], arr, pad*arr[-1]))
    out = np.convolve(tmp, kernel, mode='valid')
    noff = int((len(out) - npts) / 2)
    return out[noff:noff+npts]


### Definition of the fitting models
### Frota + linear background
model_frota = Model(kondo_frota)+Model(line) 
### Fano + linear background
model_fano = Model(kondo_fano)+Model(line) 
### Temperature broadedned Fano, according to https://doi.org/10.1103/PhysRevLett.88.077205
### Convolution of Fano and derivative of FD + linear background
model_convo = CompositeModel(Model(kondo_fano_b), Model(fermi), convolve) + Model(line) 


### Reading Nanponis dat file
file_name = 'dI_dV016.dat'
spec = nap.read.Spec(file_name)

#data = spec.signals['Input 3 (V)']
data = spec.signals['LI D1 Y 1 omega (A)'] 
x = spec.signals['Bias calc (V)']
x = x-x[np.argmax(data)] #OPTIONAL: centering of the experimental data


### OPTIONAL: symmetric cut of marginal data points from the experiment for fitting
data_cut = 1 #index of the data points  that will be cut out on left and right side
x_cut = x[data_cut:-data_cut]
data_cut = data[data_cut:-data_cut]


### Liear fit to the experimental data to estimate the linear background:
linfit_index = 5
linfit_x = np.append(x[:linfit_index], x[-linfit_index:])
linfit_data = np.append(data[:linfit_index], data[-linfit_index:])
linfit = np.polyfit(linfit_x, linfit_data, 1)       

### Definition of the fittign parameters
### Good initial guess of the parameters helps with successful fitting
params = Parameters()

### !!! PARAMETERS THAT SHOULD BE SET FOR EVERY FITTING!!!
params.add('T', value=6, vary=False) #Real temperature of the microscope during the data acquisition for the FD thermal broadening
params.add('gamma', value=5e-3) #needs to be guessed/estimated from the data plot

### Parameters that shall be changed only for specific situations
params.add('amp', value=-np.max(data)) #amplitude of Frota/Fano functions, estimated as max value of the data
params.add('slope', value=linfit[0]) #slope of the linear background, estimated from the linear fit
params.add('offset', value=0) #offset value
params.add('Ek', value=0) #Kondo peak position for the Fano fit
params.add('q', value=0.01)  #Fano parameter which interpolates between a Lorentzian peak and dip, more i
params.add('Ekk', value=0, vary=False) #Fermi energy for the Fermi-Dirac (FD) distributions
params.add('Kb', value=8.617333e-5, vary=False) #Boltzmann constant


### Frota fitting with lmfit Model.fit
params['q'].set(vary=False)
params['Ek'].set(vary=False)
result_frota = model_frota.fit(data_cut, params, eps=x_cut)

### Fano fitting with lmfit Model.fit
params['q'].set(vary=True)
params['Ek'].set(vary=True)
result_fano = model_fano.fit(data_cut, params, eps=x_cut)
comps_fano = result_fano.eval_components(eps=x_cut)

### FD broadened Fano fitting with lmfit Model.fit
result_convo = model_convo.fit(data_cut, params, eps=x_cut)
comps_convo = result_convo.eval_components(eps=x_cut)


### Results of the fitf
print(result_fano.fit_report())
print(result_frota.fit_report())
print(result_convo.fit_report())

### Kondo temperature from the broadened Fano fit:
Gamma_fano = result_fano.best_values['gamma'] #HWHM
Tk_fano = 1/params.valuesdict()['Kb'] * np.sqrt(((Gamma_fano)**2 - (math.pi * params.valuesdict()['Kb' * params.valuesdict()['T'])**2)/2)
print('Fano fit Tk = ' + '{:.2f}'.format(Tk_fano) + ' K')

### Kondo temperature from the broadened Fano fit:
Gamma_fano_b = result_convo.best_values['gamma'] #HWHM
Tk_fano_b = 1/params.valuesdict()['Kb'] * np.sqrt(((Gamma_fano_b)**2 - (math.pi * params.valuesdict()['Kb'] * params.valuesdict()['T'])**2)/2)
print('Broadened Fano fit Tk = ' + '{:.2f}'.format(Tk_fano_b) + 'K')

### Kondo temperature from the Frota fit:, according to SI in 10.1038/nphys1876
Gamma_frota = result_frota.best_values['gamma']
Tk_frota = np.abs(Gamma_frota / (1.455 * params.valuesdict()['Kb'])) 
print('Frota fit Tk = ' + '{:.2f}'.format(Tk_frota) + 'K')

### Kondo temperature from the Frota fit, according to SI in 10.1038/ncomms3110
Gamma_frota_c = np.sqrt(2.54 * result_frota.best_values['gamma']**2 - (1/2 * 3.5 * params.valuesdict()['Kb'] * params.valuesdict()['T'])**2)
Tk_frota_c = 1/params.valuesdict()['Kb'] * np.sqrt(((Gamma_frota_c)**2 - (math.pi * params.valuesdict()['Kb'] * params.valuesdict()['T'])**2)/2)
print('Frota fit Tk 2 = ' + '{:.2f}'.format(Tk_frota_c) + 'K')

### Figure
fig, ax = plt.subplots(2,2, figsize=(10,10))
fig_title= file_name + ', calculated for T = ' + str(params.valuesdict()['T']) + 'K, acq time:' + spec.header['Start time'] 
fig.suptitle(fig_title)
ax[0,0].plot(x*1e3, data, 'x', color = 'k')
ax[0,0].plot(x_cut*1e3, result_fano.best_fit, color='y', lw=2, label='Fano')
ax[0,0].plot(x_cut*1e3, result_convo.best_fit, color='c', lw=2, label='Fano + FD')
ax[0,0].plot(x_cut*1e3, result_frota.best_fit, color='r', lw=2, label='Frota')
ax[0,0].set_title('All fits')
ax[0,0].set_xlabel('Bias (mV)')
ax[0,0].set_ylabel('dIdV (a.u.)')
ax[0,0].legend()

ax[0,1].plot(x_cut*1e3, data_cut, 'x', color = 'k')
ax[0,1].plot(x_cut*1e3, result_fano.best_fit, color='y', lw=3, label='Tk = '+'{:.1f}'.format(Tk_fano)+'K and $\Gamma$ = '+'{:.2f}'.format(Gamma_fano*1e3)+'mV')
ax[0,1].set_title('Fano resonance')
ax[0,1].set_xlabel('Bias (mV)')
ax[0,1].set_ylabel('dIdV (a.u.)')
ax[0,1].set_yticks([])
ax[0,1].legend(handlelength=0)

ax[1,0].plot(x_cut*1e3, data_cut, 'x', color = 'k')
ax[1,0].plot(x_cut*1e3, result_frota.best_fit, color='r', lw=3, label='Tk = '+'{:.1f}'.format(Tk_frota_c)+'K and $\Gamma$ = '+'{:.2f}'.format(Gamma_frota_c*1e3)+'mV')
ax[1,0].set_title('Frota function')
ax[1,0].set_xlabel('Bias (mV)')
ax[1,0].set_ylabel('dIdV (a.u.)')
ax[1,0].set_yticks([])
ax[1,0].legend(handlelength=0)

ax[1,1].plot(x_cut*1e3, data_cut, 'x', color = 'k')
ax[1,1].plot(x_cut*1e3, result_convo.best_fit, color='c', lw=3, label='Tk = '+'{:.1f}'.format(Tk_fano_b)+'K and $\Gamma$ = '+'{:.2f}'.format(Gamma_fano_b*1e3)+'mV')

ax[1,1].set_title('Convolution Fano resonance + FD distribution')
ax[1,1].set_xlabel('Bias (mV)')
ax[1,1].set_ylabel('dIdV (a.u.)')
ax[1,1].set_yticks([])
ax[1,1].legend(handlelength=0)

fig.savefig('Kondo.png', dpi=400, bbox_inches='tight')

### OPTIONAL. plot of the results with residuals
# result_fano.plot()
# result_frota.plot()
# result_convo.plot()


