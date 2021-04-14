# -*- coding: utf-8 -*-
"""
MRPy: Multivariate Random Processes with Python

Author: Marcelo Maia Rocha

Please check https://github.com/mmaiarocha/MRPy for latest version

MIT License
"""

import sys
import gzip   as gz
import pickle as pk
import numpy  as np
import pandas as pd

from   warnings          import warn
from   scipy.interpolate import interp1d

import matplotlib.pyplot as plt

#=============================================================================
#=============================================================================
class MRPy(np.ndarray):
#=============================================================================
#=============================================================================
# 1. Class initialization
#=============================================================================
    
    def __new__(cls, np_array, fs=None, Td=None):

        X  =  np.asarray(np_array).view(cls)

        if (X.size == 0):
            sys.exit('Empty array not allowed for new objects!')
        
        sh =  X.shape
        
        if (len(sh) == 1): 
            X  = np.reshape(X,(1,sh[0]))
        elif (sh[0] > sh[1]):
            X = X.T
            
        sh   =  X.shape
        X.NX =  sh[0]
        X.N  =  sh[1]
        
        if (X.N < 2):
            sys.exit('Come on!!! Start with at least 2 elements!')

        err =  1.0
        if (np.mod(X.N, 2) != 0):         # enforce N to be even...
            X   =  X[:,:-1]               # ... odd element is discarded!!!
            err = (X.N - 1)/X.N           # correction over Td
            X.N =  X.N - 1

        if (fs != None):                  # if fs is prescribed...
            X.fs = np.float(fs)
            X.Td = X.N/X.fs               # ... Td is calculated

        elif (Td != None):                # but if Td is prescribed...
            X.Td = err*np.float(Td)
            X.fs = X.N/X.Td               # ... fs is calculated

        else: sys.exit('Either fs or Td must be provided!')

        X.M  = X.N//2 + 1
        return X

#-----------------------------------------------------------------------------

    def __array_finalize__(self, X):
        
        if X is None: return
        
        self.fs = getattr(X, 'fs', None)
        self.Td = getattr(X, 'Td', None)
        self.NX = getattr(X, 'NX', None)
        self.N  = getattr(X, 'N',  None)
        self.M  = getattr(X, 'M',  None)

#=============================================================================
# 2. Class constructors from other sources
#=============================================================================

    def from_file(filename, form='mrpy'):
        """
        Load time series from file. Please contact the author for 
        including other types of datafile.
 
        Parameters:  filename: file to be loaded, including path,
                               whithout file extension
                     form:     data formatting. Options are
                               'mrpy'      - default gzip pickle loading
                               'excel'     - excel generated with pandas
                               'columns'   - t, X1, [X2, X3 ...]
                               'invh     ' - csv file by iNVH app
                               'mpu6050'   - gzip excel 6 axis data
        """

        try:
            
#-------------        
            if (form.lower() == 'mrpy'):
                with gz.GzipFile(filename+'.csv.gz', 'rb') as target:                 
                    return MRPy(*pk.load(target))

#---------------     
            elif (form.lower() == 'excel'):
                with open(filename+'.xlsx', 'rb') as target:
                    
                    data =  pd.read_excel(target, 
                                          index_col=0, 
                                          sheet_name='MRPy')
                    
                    ti   =  np.array(data.index, dtype=float)
                    return MRPy.resampling(ti, data.values)

#--------------- 
            elif (form.lower() == 'columns'):
                with open(filename+'.txt', 'rb') as target:
    
                    data = np.genfromtxt(target, 
                                         delimiter='\t')
        
                    ti   = data[:,0]
                    return MRPy.resampling(ti, data[:,1:])

#---------------    
            elif (form.lower() == 'invh'):
                with open(filename+'.csv', 'rb') as target:
                
                    data =  np.genfromtxt(target, 
                                          delimiter=',',
                                          skip_header=1)
                    
                    ti   =  data[:,0]
                    return MRPy.resampling(ti, data[:,1:-1])
    
#---------------    
            elif (form.lower() == 'mpu6050'):
                with gz.open(filename+'.csv.gz', 'rb') as target:
                
                    data =  np.genfromtxt(target, 
                                          delimiter=',')
                    
                    ti   =  data[:,0] - data[0,0]
                    return MRPy.resampling(ti, data[:,1:]/16384)

#--------------- 
            else:
                sys.exit('Data formatting not available!')
                return None
            
        except:
            sys.exit('Could not read file "{0}"!'.format(filename))
            return None

#=============================================================================
# 3. Class constructors by modification
#=============================================================================

    def zero_mean(self):
        """
        Clean mean values.
        """

        X   = MRPy.copy(self)
        Xm  = X.mean(axis=1) 

        for k in range(self.NX):  
            X[k,:] -= Xm[k]

        return X

#-----------------------------------------------------------------------------

    def integrate(self, band=None):
        """
        Frequency domain integration with passing band.
 
        Parameters:  band: frequency band to keep, tuple: (f_low, f_high)
        """

        b0, b1 = MRPy.check_band(self.fs, band)

        X  = np.empty((self.NX, self.N))
        f  = self.f_axis(); f[0] = f[1]     # avoid division by zero

        for kX, row in enumerate(self):        
        
            Xw = np.fft.fft(row)[0:self.M]
            Xw = Xw / (2j*np.pi*f)          # division means integration
            
            Xw[0]  = 0.                     # disregard integration constant
            
            Xw[(f <= b0) | (f > b1)] = 0.
            
            X[kX,:] = np.real(np.fft.ifft(
                      np.hstack((Xw, np.conj(Xw[-2:0:-1])))))

        return MRPy(X, self.fs)

#-----------------------------------------------------------------------------
    
    def differentiate(self, band=None):
        """
        Frequency domain differentiation with passing band.
 
        Parameters:  band: frequency band to keep, tuple: (f_low, f_high)
        """

        b0, b1 = MRPy.check_band(self.fs, band)

        X  = np.empty((self.NX, self.N))
        f  = self.f_axis(); f[0] = f[1]     # avoid division by zero

        for kX, row in enumerate(self):        
        
            Xw = np.fft.fft(row)[0:self.M]
            Xw = Xw * (2j*np.pi*f)          # multiplication means derivation
            
            Xw[(f <= b0) | (f > b1)] = 0.
            
            X[kX,:] = np.real(np.fft.ifft(
                      np.hstack((Xw, np.conj(Xw[-2:0:-1])))))

        return MRPy(X, self.fs)

#-----------------------------------------------------------------------------
    
    def sdof_Fourier(self, fn, zeta):
        """
        Integrates the dynamic equilibrium differential equation by Fourier.
        The input MRPy is assumed to be an acceleration (force over mass),
        otherwise the result must be divided by system mass to have
        displacement unit.
        System properties (frequency and damping) may be provided as 
        scalars or lists. If they are scalars, same properties are used
        for all series in the MRP.
    
        Parameters:  fn:   sdof natural frequency (Hz)
                     zeta: sdof damping  (nondim)
        """
        
        if ~hasattr(fn, "__len__"):
            fn   = fn*np.ones(self.NX)
    
        if ~hasattr(zeta, "__len__"):
            zeta = zeta*np.ones(self.NX)

        X   =  MRPy(np.empty((self.NX, self.N)), self.fs)

        for kX, row in enumerate(self):

            zt  =  zeta[kX]
            wn  =  2*np.pi*fn[kX]        
            K   =  wn*wn
            
            b   =  2*np.pi*self.f_axis()/wn
            Hw  = (K*((1.0 - b**2) + 1j*(2*zt*b)))**(-1)
            Hw  =  np.hstack((Hw,np.conj(Hw[-2:0:-1])))

            X[kX,:]  =  np.real(np.fft.ifft(Hw*np.fft.fft(row)))
        
        return X

#=============================================================================
# 6. Utilities
#=============================================================================

    def printAttrib(self):
        
        s1 =  ' fs = {0:.1f}Hz\n Td = {1:.1f}s\n'
        s2 =  ' NX = {0}\n N  = {1}\n M  = {2}\n'   
        
        print(s1.format(self.fs, self.Td))
        print(s2.format(self.NX, self.N, self.M))

#-----------------------------------------------------------------------------

    def t_axis(self):        
        return np.linspace(0, self.Td, self.N)

#-----------------------------------------------------------------------------
    
    def f_axis(self):        
        return np.linspace(0, self.fs/2, self.M)
    
#-----------------------------------------------------------------------------
    
    def T_axis(self):        
        return np.linspace(0, (self.M - 1)/self.fs, self.M)
    
#-----------------------------------------------------------------------------
    
    def subplot_shape(self):        

        sp0 = self.NX
        sp1 = 1

        if   (sp0 > 12):
            sp0 = 4
            sp1 = 5
        elif (sp0 == 8):
            sp0 = 4
            sp1 = 2
        elif (sp0 > 6):
            sp0 = 4
            sp1 = 3
        elif (sp0 > 3): 
            sp0 = 3
            sp1 = 2
        
        return sp0, sp1

#-----------------------------------------------------------------------------
    
    def plot_time(self, fig=0, figsize=(12, 8), axis_t=None, unit=''):

        plt.figure(fig, figsize=figsize)
        plt.suptitle('Time Domain Amplitude', fontsize=14)
        
        t  = self.t_axis()
        
        if (axis_t == None): 
            axis_t = [0, self.Td, 1.2*self.min(), 1.2*self.max()]

        sp0, sp1 = self.subplot_shape()
        lines    = []

        for kX, row in enumerate(self):
            
            plt.subplot(sp0,sp1,kX+1)
            lines.append(plt.plot(t, row, lw=0.5))
        
            plt.axis(axis_t)
            plt.ylabel('Amplitude {0}'.format(kX) + unit)
            plt.grid(True)
            
        plt.xlabel('Time (s)')
        plt.tight_layout(rect=[0, 0, 1, 0.97]) 
            
        return lines

#-----------------------------------------------------------------------------
    
    def plot_corr(self, fig=0, figsize=(12, 8), axis_T=None):

        plt.figure(fig, figsize=figsize)
        plt.suptitle('Normalized Autocorrelation', fontsize=14)
        
        Rx, Tmax = self.autocorr()
        
        T  = self.T_axis()

        if (axis_T == None): 
            axis_T = [0, Tmax, -1.2, 1.2]

        sp0, sp1 = self.subplot_shape()
        lines    = []

        for kX, row in enumerate(Rx):
            
            plt.subplot(sp0,sp1,kX+1)
            lines.append(plt.plot(T, row, lw=0.5))
        
            plt.axis(axis_T)
            plt.ylabel('Autocorrelation {0}'.format(kX))
            plt.grid(True)

        plt.xlabel('Time gap (s)')

        return lines

#=============================================================================
# 7. Helpers
#=============================================================================
 
    def resampling(ti, Xi):
        """
        Resampling irregular time step to fixed time step. The last
        element of ti is taken as total series duration. Series length
        is kept unchanged. Returns a MRPy instance.
 
        Parameters:  ti:    irregular time where samples are avaible
                     Xi:    time series samples, taken at ti
        """
        
        sh =  Xi.shape
        if (len(sh) == 1): 
            Xi  = np.reshape(Xi,(1,sh[0]))
        elif (sh[0] > sh[1]):
            Xi = Xi.T
            
        sh =  Xi.shape
        NX =  sh[0]
        N  =  sh[1]

        if (N < 2):
            sys.exit('Come on!!! Start with at least 2 elements!')
        
        tsh =  ti.shape
        if (len(tsh) > 1): 
            sys.exit('Time markers must be a 1D vector!')

        t0 =  ti[0]
        t1 =  ti[-1]

        fs =  N/(t1 - t0)               # average sampling rate
        t  =  np.linspace(t0, t1, N)    # regularly spaced time markers
        X  =  np.empty((NX,N))          # interpolated series
        
        for k in range(NX):
            resX   =  interp1d(ti, Xi[k,:], kind='linear')
            X[k,:] =  resX(t)

        return MRPy(X, fs)

#-----------------------------------------------------------------------------

    def check_fs(N, fs, Td):
        """
        Verifies if either fs or Td are given, and returns both
        properties verifyed. Observe that N is not verified to be
        even, for this will be done later on one MRPy constructor 
        is called. This means that Td may be eventually modified.
        """

        if ((fs is not None) & (Td is None)):    # if fs is available...
            pass

        elif ((fs is None) & (Td is not None)):  # if Td is available
            fs = N/Td

        else: 
            sys.exit('Either fs or Td must be specified!')
        
        return fs

#-----------------------------------------------------------------------------

    def check_band(fs, band):
        """
        Verifies if provided frequency band is consistent.
        """
        
        if (band is None):
            b0 = 0.
            b1 = fs/2

        else:
            b0 = band[0]
            b1 = band[1]

        if (b0 < 0):
            warn('Lower band limit truncated to 0!')
            b0 = 0

        if (b1 > fs/2):
            warn('Upper band limit truncated to fs/2!')
            b1 = fs/2
        
        return b0, b1
    
#=============================================================================
#=============================================================================
