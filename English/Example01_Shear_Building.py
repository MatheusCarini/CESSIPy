# -*- coding: utf-8 -*-
"""
CESSIPy Example 01: simulation of a three-storey plane frame 
    
Author: Matheus Roman Carini 
Support email: matheuscarini@gmail.com
Site: https://github.com/MatheusCarini/CESSIPy
MIT License

Federal University of Rio Grande do Sul, Porto Alegre, Brazil

Version: 1.1
Date: 20211012
"""

#=============================================================================

import CESSIPy as     SSI
import numpy   as     np
from   MRPy    import MRPy

# =============================================================================
# Part 1: Compute the response of the structure
# ============================================================================= 
# Structural parameters
# -----------------------------------------------------------------------------
# Three-storey plane frame 
#           m
#        =======       ____
#     k |      | k      ↑
#       |   m  |         H
#        =======       _↓__ 
#     k |      | k
#       |   m  |
#        =======
#     k |      | k
#      _|_    _|_
#      '''    '''

k = 980               # horizontal stiffness of a single column [N/m]
m = 0.33              # storey mass     [kg]
H = 3                 # storey height   [m]
ζ = 0.015             # damping ratio   [-]

# -----------------------------------------------------------------------------

N  = 2**14            # number of time samples
fs = 64               # sampling frequency [Hz]

# -----------------------------------------------------------------------------
# Eigenfrequencies and Eigenvectors
# -----------------------------------------------------------------------------

K = np.array(([ 2*k,-2*k,   0],                  # stiffness matrix
              [-2*k, 4*k,-2*k],
              [   0,-2*k, 4*k]))
M = np.array(([m,0,0],                           # mass matrix
              [0,m,0],
              [0,0,m])) 

Λ, Q = np.linalg.eig(np.linalg.inv(K) @ M)       # solve eigenvalue problem

ωn = (1/Λ)**.5
fn = ωn/(2*np.pi)                                # eigenfrequencies [Hz]

Mk = np.diag(Q.T @ M @ Q)
Q  = Q/Mk**0.5                                   # mass-normalized eigenvectors

x  = np.linspace(3*H,H,3)
SSI.plot_1dshapes(fn,ζ*np.ones(3),Q,'True',x,fix=[0]) # simulated modal model

# -----------------------------------------------------------------------------
# Displacements and accelerations time series
# -----------------------------------------------------------------------------

F  = MRPy(np.random.randn(3,N),fs=fs)              # random nodal forces
Fk = MRPy(np.dot(Q.T,F),fs=fs)                     # modal forces
Xk = MRPy.sdof_Fourier(Fk,fn,ζ)                    # modal displacements
X  = MRPy(np.dot(Q,Xk),fs=fs)                      # displacements
A  = MRPy.differentiate(MRPy.differentiate(X))     # accelerations

# =============================================================================
# Part 2: System identification
# =============================================================================
# Initial parameters
# -----------------------------------------------------------------------------

i    = 40                            # number of time lags
refs = (0,1,2)                       # reference sensors (all three are ref.)
yk   = SSI.rearrange_data(A,refs)    # outputs time series
nps  = yk.N//16                      # length of each time segment to
                                     # estimate spectral densities

# -----------------------------------------------------------------------------
# Frequency-domain methods
# -----------------------------------------------------------------------------
                                     
# BFD  interactive: please remove quotation marks to use this mode
'''
PSD = SSI.SDM(yk, nperseg=nps, plot=True)   
PSD = SSI.ANPSD_from_SDM(PSD,plot=True)
FBFD, ZBFD, VBFD, PSD = SSI.BFD(yk, PSD, plot=True)
γ   = SSI.coherence(yk, PSD, nps, plot=True)
SSI.plot_1dshapes(FBFD,ZBFD[1],VBFD,'BFD',x,refs,fix=[0])
'''
# BFD  batch

PSD = SSI.SDM(yk, nperseg=nps, plot=True) 
PSD.pki  = np.array([87,245,354])          
PSD.MGi  = np.array([0,2,1])                
PSD.fint = np.array([4.5,  7.875, 13.9375, 17.3125, 20.625 , 24.25]) 
FBFD, ZBFD, VBFD, PSD = SSI.BFD(yk, PSD, plot=True, mode='batch')
SSI.plot_1dshapes(FBFD,ZBFD[1],VBFD,'BFD',x,refs,fix=[0])

# EFDD interactive: please remove quotation marks to use this mode
'''
P2SD = SSI.SDM(yk, nperseg=nps, plot=True,window='boxcar', nfft=2*nps)       
FFDD, ZTFDD, VFDD, P2SD = SSI.EFDD(yk,P2SD,plot=True)
SSI.plot_1dshapes(FFDD,ZTFDD,VFDD,'FDD',x,refs,fix=[0])
'''
#EFDD batch

P2SD = SSI.SDM(yk, nperseg=nps, plot=True,window='boxcar', nfft=2*nps)       
P2SD.pki  = np.array([ 174, 490, 706], dtype=int)
P2SD.svi  = np.array([0,0,0], dtype=int)
P2SD.fint = np.array([ 2.5   ,  8.875 , 12.5   , 17.5   , 19.8125, 25.8125])
P2SD.tint = np.array([0.03125, 2.5625 , 0.0625 , 1.28125, 0.     , 0.9375 ])
FFDD, ZTFDD, VFDD, P2SD = SSI.EFDD(yk,P2SD,plot=True, mode='batch')
SSI.plot_1dshapes(FFDD,ZTFDD,VFDD,'EFDD',x,refs,fix=[0])


# -----------------------------------------------------------------------------
# Time-domain methods
# -----------------------------------------------------------------------------
                                    # tolerance
tol = np.array(([0.01,0, 30],       # eigenfrequencies
                [0.05,0,.03],       # damping ratios
                [0.10,0,1  ]))      # MAC

# SSI COV

FNC, ZTC, VVC = SSI.SSI_COV_iterator(yk,i,2,20,2)
stbC = SSI.stabilization_diagram(FNC,ZTC,VVC,'SSI COV', tol=tol)
FNCR, ZTCR,VVCR = SSI.stable_modes(FNC, ZTC, VVC, stbC, tol=0.01, spo=6)
SSI.plot_1dshapes(FNCR,ZTCR,VVCR,'SSI COV',x,refs,fix=[0])

# SSI DATA

FND, ZTD, VVD = SSI.SSI_DATA_iterator(yk,i,2,20,2)
stbD = SSI.stabilization_diagram(FND,ZTD,VVD,'SSI DATA', tol=tol)
FNDR, ZTDR,VVDR = SSI.stable_modes(FND, ZTD, VVD, stbD, tol=0.01, spo=6)
SSI.plot_1dshapes(FNDR,ZTDR,VVDR,'SSI DATA',x,refs,fix=[0])

# Fast SSI COV

FNO, ZTO, VVO = SSI.Fast_SSI(yk,i,2,20,2, based='COV')
stbO = SSI.stabilization_diagram(FNO,ZTO,VVO,'Fast SSI COV', tol=tol)
FNFC, ZTFC,VVFC = SSI.stable_modes(FNO, ZTO, VVO, stbO, tol=0.01, spo=6)
SSI.plot_1dshapes(FNFC,ZTFC,VVFC,'Fast SSI COV',x,refs,fix=[0])

# Fast SSI DATA

FNA, ZTA, VVA = SSI.Fast_SSI(yk,i,2,20,2, based='DATA')
stbA = SSI.stabilization_diagram(FNA,ZTA,VVA,'Fast SSI DATA', tol=tol)
FNFD, ZTFD,VVFD = SSI.stable_modes(FNA, ZTA, VVA, stbA, tol=0.01, spo=6)
SSI.plot_1dshapes(FNFD,ZTFD,VVFD,'Fast SSI DATA',x,refs,fix=[0])

# IV

FNI, ZTI, VVI = SSI.IV_iterator(yk,i,4,15,1)
stbI = SSI.stabilization_diagram(FNI, ZTI, VVI,'IV', tol=tol)
FNIR, ZTIR,VVIR = SSI.stable_modes(FNI, ZTI, VVI, stbI, tol=0.01, spo=2)
SSI.plot_1dshapes(FNIR,ZTIR,VVIR,'IV',x,refs,fix=[0])
