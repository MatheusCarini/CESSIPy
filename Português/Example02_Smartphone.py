# -*- coding: utf-8 -*-
"""
CESSIPy Examplo 02
Estrutura: Laje em concreto armado
Excitação: pulos e caminhar de uma pessoa
Time series recorded by smartphone through iNVH Android app
    
Autor: Matheus Roman Carini 
Email: matheuscarini@gmail.com
Site: https://github.com/MatheusCarini/CESSIPy
Licença MIT

Universidade Federal do Rio Grande do Sul, Porto Alegre, Brasil

Version: 1.1
Date: 20211012
"""

#==============================================================================

import numpy                as     np
import CESSIPy              as     SSI
from   MRPy                 import MRPy

# -----------------------------------------------------------------------------
# Importa Séries Temporais de Acelerações
# -----------------------------------------------------------------------------

y  = MRPy.from_file('Example02_Smartphone_Data', form='invh')   
yk = MRPy(y[2,2000:2000+16384], fs=y.fs)  
# Seleciona o terceiro sinal (aceleração vertical) e descarta os valores
# iniciais e finais pois se referem ao manuseio do celular

h1 = yk.plot_time(unit=' (m/s²)')    # plota série temporal
# -----------------------------------------------------------------------------
# Identificação de Sistemas
# 
# Domínio da Frequência
# -----------------------------------------------------------------------------

nps = yk.N//8

# BFD interactive: remova as aspas para rodá-lo
'''
PSD = SSI.SDM(yk, nperseg=nps, plot=True) 
PSD = SSI.ANPSD_from_SDM(PSD,plot=True)
FBFD, ZBFD, VBFD, PSD = SSI.BFD(yk, PSD, plot=True)
'''
# BFD batch

PSD = SSI.SDM(yk, nperseg=nps, plot=True) 
PSD.pki  = np.array([86])
PSD.MGi  = np.array([0])
PSD.fint = np.array([15, 31])
FBFD, ZBFD, VBFD, PSD = SSI.BFD(yk, PSD, plot=True, mode='batch')

# EFDD interactive: remova as aspas para rodá-lo
'''
P2SD = SSI.SDM(yk, nperseg=nps, plot=True,window='boxcar', nfft=2*nps)  
FFDD, ZTFDD, VFDD, P2SD = SSI.EFDD(yk,P2SD,plot=True)
'''
# EFDD batch

P2SD = SSI.SDM(yk, nperseg=nps, plot=True,window='boxcar', nfft=2*nps) 
P2SD.pki  = np.array([172])
P2SD.svi  = np.array([0])
P2SD.fint = np.array([10, 26]) 
P2SD.tint = np.array([0, .6])
FFDD, ZTFDD, VFDD, P2SD = SSI.EFDD(yk,P2SD,plot=True, mode='batch')

# -----------------------------------------------------------------------------
# Domínio do Tempo
# -----------------------------------------------------------------------------

yk = SSI.rearrange_data(yk,[0])
i = 40

                                    # tolerâncias: porcentagem, mínimo e máximo
tol = np.array(([0.01,0, 80],       # frequência
                [0.10,0,.10],       # amortecimento
                [1   ,0,1  ]))         # MAC

# SSI COV

FNC, ZTC, VVC = SSI.SSI_COV_iterator(yk,i,2,40,1)
stbC = SSI.stabilization_diagram(FNC,ZTC,VVC,'SSI COV', tol=tol)
FNCR, ZTCR,VVCR = SSI.stable_modes(FNC, ZTC, VVC, stbC, tol=0.01, spo=6)

# SSI DATA

FND, ZTD, VVD = SSI.SSI_DATA_iterator(yk,i,2,40,1)
stbD = SSI.stabilization_diagram(FND,ZTD,VVD,'SSI DATA', tol=tol)
FNDR, ZTDR,VVDR = SSI.stable_modes(FND, ZTD, VVD, stbD, tol=0.01, spo=6)

# IV

FNI, ZTI, VVI = SSI.IV_iterator(yk,i,2,30,1)
stbI = SSI.stabilization_diagram(FNI, ZTI, VVI,'IV', tol=tol)
FNIR, ZTIR,VVIR = SSI.stable_modes(FNI, ZTI, VVI, stbI, tol=0.01, spo=6)

