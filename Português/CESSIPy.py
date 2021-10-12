# -*- coding: utf-8 -*-
"""
CESSIPy: um módulo em Python de código aberto para Identificação Modal 
Estocástica de Estruturas
    
Autor: Matheus Roman Carini 
Email para suporte: matheuscarini@gmail.com
Site: https://github.com/MatheusCarini/CESSIPy
Licença MIT

Universidade Federal do Rio Grande do Sul, Porto Alegre, Brasil

Version: 1.1
Date: 20211012
"""

#=============================================================================
import sys
import numpy             as np
import matplotlib.pyplot as plt
import matplotlib        as mpl

from MRPy                import MRPy  
from scipy               import signal
from scipy.optimize      import curve_fit
from matplotlib.gridspec import GridSpec

plt.rcParams["font.family"] = "Times New Roman"
mpl.rcParams['mathtext.fontset'] = 'cm'

#=============================================================================
# Classe auxiliar
#=============================================================================

class auxclass(np.ndarray):
    """
    Uma classe criada para permitir o uso de atributos
    Visa facilitar a programação e leitura do código
    """
   
    def __new__(cls, np_array):

        return np.asarray(np_array).view(cls)

#=============================================================================
# Domínio do Tempo
#=============================================================================
        
def rearrange_data(self,ref):
    """
    Rearranja as l saídas medidas posicionando as r saídas de referência 
    nas primeiras linhas.      
    
    Entradas
    -------
    self : MRPy_like
        Um MRPy que contém as séries temporais das saídas.            
    ref: tupple, list
        Lista dos sensores de referência.
    
    Saídas
    -------
    yk : MRPy_like
        Um MRPy com as saídas de referência nas primeiras linhas e com os
        atributos r e l.
    ..  l : atributo do MRPy
            Número de saídas
    ..  r : atributo do MRPy
            Número de saídas de referência.                
    """        
          
    r = len(ref)
    l = self.shape[0]
    
    yk   = MRPy(np.empty((l,self.N)),fs=self.fs)
    yk.r = r
    yk.l = l
    
    yk[:r,:] = self[ref,:]
    yk[r:,:] = np.delete(self, ref, 0)
    
    return yk

#-----------------------------------------------------------------------------    
    
def Toeplitz(self, i):
    """
    Cria a matriz bloco Toeplitz, a qual reune as covariâncias das saídas
    estimadas até 2*i-1 time lags.
    
    Entradas
    -------
    self : MRPy_like        
        Um MRPy que contém as séries temporais das saídas.
    i : int
        Número de atrasos no tempo associado com a duração das funções de
        covariâncias. Note que elas são estimadas até 2*i-1 time lags.
        
    Saídas
    -------
    T : auxclass_like
        Um auxclass que contém os valores da matriz bloco Toeplitz e os
        atributos r, l e i.
    """       
    
    N = self.N - 2*i + 1
    r = self.r
    l = self.l

    Ypref = np.zeros((r*i,N))
    Yf    = np.zeros((l*i,N))
        
    for k in range(i):
        Ypref[k*r:k*r+r,:] = self[:r,k:k+N]
        Yf   [k*l:k*l+l,:] = self[: ,k+i:k+i+N]
        
    Ypref = Ypref/N**0.5
    Yf    = Yf   /N**0.5
        
    T   = auxclass(Yf @ Ypref.T)
    
    T.fs, T.r, T.l, T.i = self.fs, r, l, i

    return T

#-----------------------------------------------------------------------------    
    
def SSI_COV(T, no):
    """
    Método de Identificação Estocástica de Subespaços baseado nas 
    Covariâncias
    
    Estima as frequências naturais, razões de amortecimentos e formas modais 
    a partir da matriz bloco Toeplitz informada.
    
    Entradas
    ------- 
    T : auxclass_like
        Um auxclass que contém os valores da matriz bloco Toeplitz e os
        atributos SVD, r, l e i.             
    no : int
        Número de ordem do modelo no espaço de estado.
        
    Saídas
    -------    
    fn : ndarray
        Array com as frequências naturais.
    zt : ndarray
        Array com as razões de amortecimento.
    V : ndarray
        Array com as formas modais nas colunas.
        
    Ver também
    ------- 
    Toeplitz, SSI_COV_iterator
    """
    
    l = T.l
    i = T.i       
    U, S, VT = T.SVD
               
    U1 = U[:,:no]
    S1 = np.eye(no)*S[:no]           
    Oi = U1 @ S1**0.5    
    C  = Oi[:l,:]
    
    A  = np.linalg.pinv(Oi[:l*(i-1),:]) @ Oi[l:l*i+1,:]
    Λd, Ψ = np.linalg.eig(A)
    
    λ  =  np.log(Λd)*T.fs
    fn =  np.abs(λ)/(2*np.pi)
    zt = -np.real(λ)/np.abs(λ)                
    V  =  C @ Ψ
    
    return fn, zt, V
 
#-----------------------------------------------------------------------------
    
def SSI_COV_iterator(yk,i,nmin,nmax,incr=2, plot=False):
    """
    Itera a função SSI_COV para modelos de ordem nmin a nmax com 
    incrementos incr.
    
    Estima as frequências naturais, razões de amortecimentos e formas modais
    pelo método SSI COV para os números de ordem informados.
    
    Entradas
    ------- 
    yk : MRPy_like
        Um MRPy com as séries temporais das saídas arranjadas pela função 
        rearrange_data.
    i : int
        Número atrasos no tempo associado com a duração das funções de
        covariâncias. Note que elas são estimadas até 2*i-1 time lags.
    nmin : int
        Número de ordem inicial do modelo no espaço de estado.
    nmax : int
        Número de ordem do modelo no espaço de estado.
    incr : int, optional
        Variação da ordem em relação à iteração anterior. Padrão = 2.
    plot : bool, optional
        Se verdadeiro, gera o gráfico dos valores singulares da matriz
        Toeplitz. O padrão é falso.
        
    Saídas
    ------- 
    FN : ndarray
        Array 2D com as frequências naturais. Cada linha origina-se do 
        mesmo modelo em espaço de estado. 
    ZT : ndarray
        Array 2D com as frequências naturais. Cada linha origina-se do 
        mesmo modelo em espaço de estado.         
    VV : ndarray
        Array 3D com as formas modais. A primeira dimensão refere-se à 
        ordem do modelo em espaço de estado.
    
    Notas
    ------- 
    As propriedades dinâmicas do modelo de ordem nmin estão em FN[0,:],
    ZT[0,:] e VV[0,:,:].            
    """
    
    T  = Toeplitz(yk, i)
    T.method = 'SSI COV'
    if plot: plot_singular_values(T)
    T.SVD = np.linalg.svd(T)
    
    n  = np.arange(nmin,nmax+incr,incr)        
    FN = np.zeros((n.shape[0],nmax))
    ZT = np.zeros((n.shape[0],nmax))
    VV = np.zeros((n.shape[0],T.l,nmax),dtype=np.complex_)
    
    for ii, no in np.ndenumerate(n):
        FN[ii,:no], ZT[ii,:no], VV[ii,:,:no] = SSI_COV(T,no) 
        
    return FN, ZT, VV

#-----------------------------------------------------------------------------
    
def projection(yk, i):
    """
    Realiza a fatoração QR da matriz Hankel e calcula as matrizes Piref, 
    Pi1ref e Yii.
    
    Entradas
    ------- 
    yk : MRPy_like
        Um MRPy com as séries temporais das saídas arranjadas pela função 
        rearrange_data.  
    i : int
        Número atrasos no tempo associado com a duração das funções de
        covariâncias. Note que elas são estimadas até 2*i-1 time lags.
        
    Saídas
    -------     
    Pi : auxclass_like
        Um auxclass com as projeções das saídas futuras no espaço das linhas
        das saídas passadas e os atributos r, l e i.
    Pi1 : array_like
        Array com as projeções considerando a separação entre as saídas
        passadas e futuras uma linha para baixo.
    Yii : array_like
        Array com parte da matriz bloco Hankel.
    """
    
    N = yk.N - 2*i + 1
    r = yk.r
    l = yk.l

    Ypref = np.zeros((r*i,N))
    Yf    = np.zeros((l*i,N))
        
    for k in range(i):
        Ypref[k*r:k*r+r,:] = yk[:r,k:k+N]
        Yf   [k*l:k*l+l,:] = yk[: ,k+i:k+i+N]
        
    Ypref = Ypref/N**0.5
    Yf    = Yf   /N**0.5        
    Href  = np.vstack([Ypref,Yf])
    
    R = np.linalg.qr(Href.T, mode='r').T
    
    Pi  = auxclass(R[r*i:,:r*i]        @ np.eye(r*i,N))
    Pi1 =         R[r*i+l:,:r*i+r]    @ np.eye(r*i+r,N)
    Yii =         R[r*i:r*i+l,:r*i+l] @ np.eye(r*i+l,N)
            
    Pi.fs, Pi.r, Pi.l, Pi.i = yk.fs, r, l, i
    
    return Pi, Pi1, Yii

#-----------------------------------------------------------------------------    
    
def SSI_DATA(Pi, Pi1, Yii, no):
    """
    Método de Identificação Estocástica de Subespaços baseado nas Séries 
    Temporais   
    
    Estima as frequências naturais, amortecimentos e formas modais a partir 
    das matrizes Piref, Pi1ref e Yii.
    
    Entradas
    ------- 
    Pi, Pi1, Yii 
        Ver projection.             
    no : int
        Número de ordem do modelo no espaço de estado.
        
    Saídas
    -------    
    fn : ndarray
        Array com as frequências naturais.
    zt : ndarray
        Array com as razões de amortecimento.
    V : ndarray
        Array com as formas modais nas colunas.
    """
       
    U, S, VT = Pi.SVD
                         
    U1 = U[:,:no]
    S1 = np.eye(no)*S[:no]
                
    Oi  = U1 @ S1**0.5
    Oi1 = Oi[:-Pi.l,:]
            
    Xi  = np.linalg.pinv(Oi) @ Pi
    Xi1 = np.linalg.pinv(Oi1) @ Pi1
            
    AC = np.vstack([Xi1,Yii]) @ np.linalg.pinv(Xi) 
    A  = AC[:no,:]
    C  = AC[no:,:]
            
    Λd, Ψ = np.linalg.eig(A)
    
    λ  =  np.log(Λd)*Pi.fs
    fn =  np.abs(λ)/(2*np.pi)
    zt = -np.real(λ)/np.abs(λ)                
    V  =  C @ Ψ
    
    return fn, zt, V    

#-----------------------------------------------------------------------------
        
def SSI_DATA_iterator(yk, i, nmin, nmax, incr=2,plot=False):
    """
    Itera a função SSI_DATA para modelos de ordem nmin a nmax com 
    incrementos incr.
    
    Estima as frequências naturais, razões de amortecimentos e formas modais
    pelo método SSI DATA para os números de ordem informados.

    Entradas
    ------- 
    yk : MRPy_like
        Um MRPy com as séries temporais das saídas arranjadas pela função 
        rearrange_data.
    i : int
        Número atrasos no tempo associado com a duração das funções de
        covariâncias. Note que elas são estimadas até 2*i-1 time lags.
    nmin : int
        Número de ordem inicial do modelo no espaço de estado.
    nmax : int
        Número de ordem do modelo no espaço de estado.
    incr : int, optional
        Variação da ordem em relação à iteração anterior. Padrão = 2.
    plot : bool, optional
        Se verdadeiro, gera o gráfico dos valores singulares da matriz
        das projeções. O padrão é falso.
        
    Saídas
    ------- 
    FN : ndarray
        Array 2D com as frequências naturais. Cada linha origina-se do 
        mesmo modelo em espaço de estado. 
    ZT : ndarray
        Array 2D com as frequências naturais. Cada linha origina-se do 
        mesmo modelo em espaço de estado.         
    VV : ndarray
        Array 3D com as formas modais. A primeira dimensão refere-se à 
        ordem do modelo em espaço de estado.
    
    Notas
    ------- 
    As propriedades dinâmicas do modelo de ordem nmin estão em FN[0,:],
    ZT[0,:] e VV[0,:,:].
    """
    
    Pi, Pi1, Yii = projection(yk, i)
    Pi.method = 'SSI DATA'
    if plot: plot_singular_values(Pi)        
    Pi.SVD = np.linalg.svd(Pi)
        
    n  = np.arange(nmin,nmax+incr,incr)        
    FN = np.zeros((n.shape[0],nmax))
    ZT = np.zeros((n.shape[0],nmax))
    VV = np.zeros((n.shape[0],Pi.l,nmax),dtype=np.complex_)
        
    for ii, no in np.ndenumerate(n):
        FN[ii,:no],ZT[ii,:no],VV[ii,:,:no] = SSI_DATA(Pi,Pi1,Yii,no) 
        
    return FN, ZT, VV

#-----------------------------------------------------------------------------
    
def Fast_SSI(yk, i, nmin, nmax, incr=2, plot=False, based='COV'):  
    """
    Estima as frequências naturais, razões de amortecimentos e formas modais
    pelo Algoritmo 2 de Identificação Estocástica de Subespaços proposto por 
    [1] para os modelos de números de ordem informados.
       
    Entradas
    ------- 
    yk : MRPy_like
        Um MRPy com as séries temporais das saídas arranjadas pela função 
        rearrange_data.
    i : int
        Número atrasos no tempo associado com a duração das funções de
        covariâncias. Note que elas são estimadas até 2*i-1 time lags.
    nmin : int
        Número de ordem inicial do modelo no espaço de estado.
    nmax : int
        Número de ordem do modelo no espaço de estado.
    incr : int, optional
        Variação da ordem em relação à iteração anterior. Padrão = 2.
    plot : bool, optional
        Se verdadeiro, gera o gráfico dos valores singulares da matriz
        das projeções. O padrão é falso.
    based : string, optinal
        Método SSI. Se 'COV', utiliza o método baseado nas covariâncias. Se
        'DATA, utiliza o método baseado nas séries temporais. O padrão é 'COV'.
        
    Saídas
    ------- 
    FN : ndarray
        Array 2D com as frequências naturais. Cada linha origina-se do 
        mesmo modelo em espaço de estado. 
    ZT : ndarray
        Array 2D com as frequências naturais. Cada linha origina-se do 
        mesmo modelo em espaço de estado.         
    VV : ndarray
        Array 3D com as formas modais. A primeira dimensão refere-se à 
        ordem do modelo em espaço de estado.
    
    Notas
    ------- 
    As propriedades dinâmicas do modelo de ordem nmin estão em FN[0,:],
    ZT[0,:] e VV[0,:,:].

    Referência
    ----------
    .. [1] Döhler M; Mevel L. Fast Multi-Order Computation of System 
           Matrices in Subspace-Based System Identification. Control 
           Engineering Practice, Elsevier, 2012, 20 (9), pp.882-894. 
           10.1016/j.conengprac.2012.05.005. hal-00724068     
    """
    
    if based.lower() == 'cov':
        
        T  = Toeplitz(yk, i)
        T.method = 'SSI COV'
        if plot: plot_singular_values(T)
        U, S, VT = np.linalg.svd(T)
                          
        U1 = U[:,:nmax]
        S1 = np.eye(nmax)*S[:nmax]           
        Oi = U1 @ S1**0.5     

    
    elif based.lower() == 'data':
    
        Pi, Pi1, Yii = projection(yk, i)
        Pi.method = 'SSI DATA'
        if plot: plot_singular_values(Pi)        
        U, S, VT = np.linalg.svd(Pi)
                                 
        U1 = U[:,:nmax]
        S1 = np.eye(nmax)*S[:nmax]           
        Oi = U1 @ S1**0.5 
            
    else:
        sys.exit('Método base deve ser COV ou DATA')       
    
    l = yk.l

    Oiu = Oi[:l*(i-1),:]
    Oid = Oi[l:l*i+1 ,:]    
    C  = Oi[:l,:] 
    
    Q, R = np.linalg.qr(Oiu)
    St = Q.T @ Oid
    
    n  = np.arange(nmin,nmax+incr,incr) 
    FN = np.zeros((n.shape[0],nmax))
    ZT = np.zeros((n.shape[0],nmax))
    VV = np.zeros((n.shape[0],l,nmax),dtype=np.complex_)
    
    for ii, no in np.ndenumerate(n):
        A = np.linalg.inv(R[:no,:no]) @ St[:no,:no]
        Cj = C[:,:no]
    
        Λd, Ψ = np.linalg.eig(A)
        
        λ  =  np.log(Λd)*yk.fs
        
        FN[ii,:no] =  np.abs(λ)/(2*np.pi)
        ZT[ii,:no] = -np.real(λ)/np.abs(λ)  
              
        VV[ii,:,:no]  =  Cj @ Ψ
        
    return FN, ZT, VV

#-----------------------------------------------------------------------------
        
def IV(T,no):
    """
    Método das Variáveis Instrumentais
    
    Estima as frequências naturais, razões de amortecimentos e formas modais 
    a partir da matriz bloco Toeplitz informada.     
    
    Entradas
    ------- 
    T : auxclass_like
        Um auxclass que contém os valores da matriz bloco Toeplitz e os
        atributos r, l e i.             
    no : int
        Número de ordem do modelo no espaço de estado.
        
    Saídas
    -------    
    fn : ndarray
        Array com as frequências naturais
    zt : ndarray
        Array com as razões de amortecimento
    V : ndarray
        Array com as formas modais nas colunas
        
    Ver também
    ------- 
    Toeplitz
    """
    
    r = T.r
    l = T.l
    
    αb = np.linalg.lstsq(T[:,-no*r:], 
                        -T[:,-(no+1)*r:-no*r], rcond=None)[0]
    
    Apcomp = np.zeros((no*r,no*r))
    Apcomp[:-r,r:] += np.eye((no-1)*r)
    for kk in range(no):
        Apcomp[-r:,r*kk:r*(kk+1)] -= αb.T[:,r*(no-kk)-r:r*(no-kk)]
    
    Λd, Ψ = np.linalg.eig(Apcomp)
    
    λ  =  np.log(Λd)*T.fs
    fn =  np.abs(λ)/(2*np.pi)
    zt = -np.real(λ)/np.abs(λ)                

    Gmref = (Ψ[:r,:]).T
    Γmref = np.zeros((no*r,no*r),dtype=np.complex_)
    
    for ii in range(no):
        Γmref[:,ii*r:(ii+1)*r] = np.diag(Λd**(no-ii-1)) @ Gmref
        
    V = T[:l,-no*r:] @ np.linalg.inv(Γmref)
    
    return fn, zt, V

#-----------------------------------------------------------------------------
        
def IV_iterator(yk, i,nmin,nmax,incr=2,plot=False):
    """
    Itera a função IV para modelos de ordem nmin a nmax com incrementos 
    incr.
    
    Estima as frequências naturais, razões de amortecimentos e formas modais
    pelo método IV para os números de ordem informados.        
    
    Entradas
    ------- 
    yk : MRPy_like
        Um MRPy com as séries temporais das saídas arranjadas pela função 
        rearrange_data.
    i : int
        Número de atrasos no tempo associado com a duração das funções de
        covariâncias. Note que elas são estimadas até 2*i-1 time lags.
    nmin : int
        Número de ordem inicial do modelo no espaço de estado.
    nmax : int
        Número de ordem do modelo no espaço de estado.
    incr : int, optional
        Variação da ordem em relação à iteração anterior. Padrão = 2.
    plot : bool, optional
        Se verdadeiro, gera o gráfico dos valores singulares da matriz
        Toeplitz. O padrão é falso.
        
    Saídas
    ------- 
    FN : ndarray
        Array 2D com as frequências naturais. Cada linha origina-se do 
        mesmo modelo em espaço de estado. 
    ZT : ndarray
        Array 2D com as frequências naturais. Cada linha origina-se do 
        mesmo modelo em espaço de estado.         
    VV : ndarray
        Array 3D com as formas modais. A primeira dimensão refere-se à 
        ordem do modelo em espaço de estado.
    
    Notas
    ------- 
    Observe que um modelo ARMA de ordem p tem p * r polos
    As propriedades dinâmicas do modelo de ordem nmin estão em FN[0,:],
    ZT[0,:] e VV[0,:,:].  
    """

    T  = Toeplitz(yk,i)
    T.method = 'IV'
    if plot: plot_singular_values(T)        
    
    n  = np.arange(nmin,nmax+incr,incr)        
    FN = np.zeros((n.shape[0],nmax*T.r))
    ZT = np.zeros((n.shape[0],nmax*T.r))
    VV = np.zeros((n.shape[0],T.l,nmax*T.r),dtype=np.complex_)
    
    for ii, no in np.ndenumerate(n):
        FN[ii,:no*T.r], ZT[ii,:no*T.r], VV[ii,:,:no*T.r] = IV(T,no) 
        
    return FN, ZT, VV

#-----------------------------------------------------------------------------    
    
def stabilization_diagram(FN, ZT, VV, title, 
                         tol = np.array(([0.01,0, 100],
                                         [0.05,0,0.05],
                                         [0.10,0,   1])), plot=True):
    """
    Plota o diagrama de estabilização a partir das matrizes com as 
    frequências naturais, razões de amortecimentos e formas modais.
    
    Entradas
    -------     
    FN, ZT, VV
        Propriedades dinâmicas fornecidas pelas funções SSI_COV_Iterator,
        SSI_DATA_Iterator ou IV_Iterator.
    title : str
        Título do gráfico.
    tol : ndarray, optional
        Tolerâncias, valores mínimos e máximos utilizados para indicar a 
        estabilidade do pivô. A primeira linha refere-se à frequência, a 
        segundo às razões de amortecimento e a terceira aos coeficientes de 
        MAC. O padrão é
        [0.01,0,100 ] Δf = 1%; fmin = 0 Hz; fmax = 100 Hz
        [0.05,0,0.05] Δζ = 5%; ζmin = 0%;   ζmax = 5%
        [0.10,0,1   ] MAC >= (1 - 0.10) = 0.90     
    plot : bool, optional
        Se verdadeiro, gera o gráfico do diagrama de estabilização. 
        O padrão é falso.        
    
    Saídas
    -------   
    stb : array_like
        Array booleano indicando os polos estáveis como verdadeiro. Cada 
        linha origina-se do mesmo modelo em espaço de estado.        
    
    Notas
    ------- 
    Os polos estáveis do modelo de ordem nmax estão em stb[-1,:].
    """
    
    nmin = np.count_nonzero(FN, axis=1)[0]
    nmax = np.count_nonzero(FN, axis=1)[-1]
    incr = (nmax-nmin)//(FN.shape[0]-1)
    n    = np.arange(nmin,nmax+incr,incr)
    stb  = np.full(FN.shape, False)
    stbf = np.full(FN.shape, False)
    stbz = np.full(FN.shape, False)
    
    for ii in range(1,n.shape[0]): 
        
        no = n[ii]; ia = ii - 1
        na = n[ia]         
        
        # Frequências
        
        b1 = (FN[ii,:no] >= tol[0,1]) & (FN[ii,:no] <= tol[0,2])                     
        dif = FN[ia,:na] - FN[ii,:no].reshape(-1,1)
        ind = np.abs(dif).argmin(axis=1)
        res = np.diagonal(dif[:,ind])
        b1 = (np.abs(res/FN[ii,:no]) < tol[0,0]) & b1            
        
        # Amortecimento
        
        b2 = (ZT[ii,:no] >= tol[1,1]) & (ZT[ii,:no] <= tol[1,2])
        dif = ZT[ia,:na] - ZT[ii,:no].reshape(-1,1)
        res = np.diagonal(dif[:,ind])
        b2 = (np.abs(res/ZT[ii,:no]) < tol[1,0]) & b2 & b1      
        
        # MAC
               
        MCv = MAC(VV[ia,:,:na],VV[ii,:,:no])           
        res = np.abs(np.diag(MCv[ind,:]))                       
        b3 = (res > 1 - tol[2,0]) & b2          
        
        stbf[ii,:no] = b1
        stbz[ii,:no] = b2
        stb [ii,:no] = b3

    if plot:
        
        a_for = {'fontname':'Times New Roman','size':16}
        l_for = {'fontname':'Times New Roman','size':14}
        t_for = {'fontname':'Times New Roman','size':12}
        g_for = {'family'  :'Times New Roman','size':12}
        
        plt.figure(figsize=(10,5))     
                    
        for ii in range(n.shape[0]): 
                            
            yi = n[ii]*np.ones(n[ii])  
            ko = plt.scatter(FN[ii,:n[ii]],yi,s=2,c='k')
            go = plt.scatter(FN[ii,:n[ii]][stbf[ii,:n[ii]]],
                             yi[stbf[ii,:n[ii]]],s=4,c='g')
            bo = plt.scatter(FN[ii,:n[ii]][stbz[ii,:n[ii]]],
                             yi[stbz[ii,:n[ii]]],s=4,c='b')
            ro = plt.scatter(FN[ii,:n[ii]][stb [ii,:n[ii]]],
                             yi[stb [ii,:n[ii]]],s=8,c='r')
            
        plt.xlim((0,tol[0,2]))
        plt.ylim((0,n[-1]))
        plt.xticks(**t_for)
        plt.yticks(n,**t_for)
        plt.xlabel('f (Hz)',**l_for)
        plt.ylabel('Ordem do Modelo',**l_for)
        plt.suptitle(title + ' Diagrama de Estabilização',**a_for)
        plt.legend([ko, go, bo, ro], 
                  ["Novo polo", 
                   "Frequência estável",
                   "Frequência e amortecimento estáveis",
                   "Frequência, amortecimento e forma modal estáveis"],
                                                               prop=g_for)
        plt.tight_layout(rect=[0, 0, 1, 0.97])

    return stb

#-----------------------------------------------------------------------------
        
def stable_modes(FN, ZT, V, stb, tol=0.01, spo=6):
    """
    Automatiza o processo de escolha dos modos estruturais.
    
    Analisa os pontos estáveis do diagrama de estabilização e retorna as 
    características dinâmicas daqueles que apresentarem pelo menos spo/2 
    polos para uma mesma frequência.
    
    Entradas
    -------   
    FN, ZT, V
        Propriedades dinâmicas fornecidas pelas funções SSI_COV_Iterator,
        SSI_DATA_Iterator ou IV_Iterator.
    stb : array_like
        Array booleano indicando os polos estáveis como verdadeiro,
        fornecida pela função stabilization_diagram.
    tol : float
        Tolerância para frequência ser considerada do mesmo modo.
    spo : int
        Número mínimo de modelos com polos estáveis para que o modo seja
        considerado estrutural.
    
    Saídas
    ------- 
    fn : ndarray
        Array com as frequências naturais.
    zt : ndarray
        Array com as razões de amortecimento.         
    vv : ndarray
        Array 2D com as formas modais como colunas

    Notas
    -------
    Como as propriedades dinâmicas estão em pares, spo seleciona spo/2 
    polos estáveis
    """

    FN = FN[stb]
    ZT = ZT[stb]      
    VV = V[0,:,stb[0]].T
    
    for j in range(stb.shape[0]):
        VV = np.hstack((VV,V[j,:,stb[j]].T))
    
    fsi = np.argsort(FN)
        
    FNs, ZTs, VVs = FN[fsi], ZT[fsi], VV[:,fsi]
    
    fn, zt, v = [], [], V[0,:,stb[0]].T 
    
    k = 0   
    
    for i in range(len(FN)):
        
        b0 = (FNs > (1-tol)*FNs[k]) & (FNs < (1+tol)*FNs[k])
        
        if b0.sum() >= spo:
            
            fn = np.append(fn,(FNs[b0]).mean())
            zt = np.append(zt,(ZTs[b0]).mean())
            
            mv = np.argmax(np.abs(VVs[:,b0]),axis=0)[0]
            nv = np.mean(VVs[:,b0]/VVs[mv,b0],axis=1).reshape(-1,1)
            v  = np.hstack((v,nv))
            
        k += b0.sum() 
            
        if k > len(FN)-1: break

    return fn, zt, v

#-----------------------------------------------------------------------------
    
def plot_singular_values(T, figsize=(14, 4), nmx=40):
    """
    Calcula e plota os valores singulares da matriz informada.
    
    Entradas
    -------   
    T : auxclass_like
        Um auxclass com a matriz e atributo method.
    figsize : tuple, optional
        Tamanho do gráfico. Padrão é (14,4).
    nmx : int, optional
        Número de valores singulares a serem plotados.
    """
    
    a_for = {'size':16}
    l_for = {'size':16}
    
    S   = np.linalg.svd(T, compute_uv=False)[:nmx]
    idx = np.argmin(S[1:]/S[:-1])
 
    fig, ax = plt.subplots(1, 3,figsize=figsize)
    fig.suptitle('%s Valores Singulares' %(T.method), **a_for) 
    
    label = ['\n(a) valores singulares',
             'Ordem do Modelo\n(b) normalizados pelo primeiro',
             '\n(c) normalizados pelo anterior']
       
    ax[0].plot(np.arange(1,nmx+1),S,'bo',ms=4)
    ax[0].set_ylabel('Valores Singulares', **l_for)
    ax[0].set_ylim(bottom=0)
    
    ax[1].semilogy(np.arange(1,nmx+1),S/S[0],'b',idx+1,(S/S[0])[idx],'ro')
    ax[1].annotate('%.0f' %(idx+1),(idx+1.5,(S/S[0])[idx]),**l_for)
    
    ax[2].semilogy(np.arange(1,nmx+1), np.hstack((1,S[1:]/S[:-1])),'b',
                 idx+1,(S[1:]/S[:-1])[idx-1],'ro')
    ax[2].annotate('%.0f' %(idx+1),(idx+1.5,(S[1:]/S[:-1])[idx-1]),**l_for)
                     
    for i in range(3): 
        ax[i].set_xticks(np.linspace(0,nmx,nmx//2+1))
        ax[i].tick_params(labelsize=12)
        ax[i].set_xlim((0,nmx))
        ax[i].set_xlabel(label[i], **l_for)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    
    return

#=============================================================================
# Domínio da Frequência
#=============================================================================  
    
def SDM(self, nperseg=None, plot=False, window='hann', nfft=None, 
        figsize=(10,10)):
    """      
    Calcula a matriz das densidades espectrais através da função signal.csd
    
    Os sinais podem ser divididos em segmentos com nperseg valores visando
    obter estimativas suavizadas.        

    Entradas
    -------          
    self : MRPy_like
        Um MRPy com as séries temporais.
    nperseg : int, optional
        Comprimento de cada segmento utilizado para das densidades
        espectrais suavizadas. O padrão é o comprimento do sinal.
    plot : bool, optional
        Se verdadeira, gera o gráfico das densidades espectrais separadas
        em valor absoluto e fase. O padrão é falso.
    window : string, optional
        Janela de dados. O padrão é 'hann'.
    nfft : int, optional
        Comprimento da FFT utilizada, se deseja-se adicionar zeros na FFT. Se
        None, o comprimento da FFT é nperseg. O padrão é None.
    figsize : tuple, optional
        Tamanho do gráfico. Padrão é (12,12)
        
    Saídas
    -------   
    PSD : auxclass
        Um auxclass com as densidades espectrais, as frequências no atributo 
        f e o comprimento do trecho no atributo nperseg.
    
    Ver também
    -------              
    scipy.signal.csd 
    """                
    
    if nperseg is None: nperseg = self.N
    if nfft is None: nfft = nperseg
    if nfft < nperseg:
        raise ValueError('nfft must be greater than or equal to nperseg.')
    
    G = np.empty((self.NX, self.NX, nfft//2+1), dtype=np.complex_)
    
    for i in range(self.NX):
        for j in range(self.NX):
            f, G[i,j] = signal.csd(self[i,:], self[j,:], self.fs, 
                window=window, nperseg=nperseg, nfft=nfft)               
                         
    if plot:

        l_for = {'fontname':'Times New Roman','size':13}        
        
        axis_f = [0, f[-1]]
        #axis_G = [10**7*np.min(np.abs(G[:,:,1:])),1.2*np.max(np.abs(G))]
        
        n = G.shape[0]   # number of double-rows
        m = G.shape[1]   # number of columns
        H = 3            # relação gráfica amplitude pela fase            
        t = 0.9          # 1-t == top space 
        b = 0.1          # bottom space      (both in figure coordinates)            
        w = 0.05         # side spacing
        μ = 0.1          # minor spacing
        Μ = 0.2          # major spacing
        
        spa  = (t-b)/(n*(1+μ+1/H)+(n-1)*Μ)
        offb = spa*(μ+1/H)
        offt = spa*(1+μ)
        hsp1 = Μ+μ+1/H
        hsp2 = (Μ+μ+1)*H
        
        gso = GridSpec(n,m, bottom=b+offb, top=t, hspace=hsp1, wspace=w)  
        gse = GridSpec(n,m, bottom=b, top=t-offt, hspace=hsp2, wspace=w)         
        
        fig = plt.figure(figsize=figsize)
        
        for i in range(n*m):        
            
            ax1 = fig.add_subplot(gso[i])
            ax2 = fig.add_subplot(gse[i])
            
            ax1.semilogy(f[1:],np.abs(G[i//m, i%m, 1:]),color='blue')
            ax1.set_xlim(axis_f)
            #ax1.set_ylim(axis_G) 
            ax1.set_xticklabels([])
            ax1.set_title(r'$\hat G_y[{:d},{:d}]$'.format(i//m+1,i%m+1),
                                                                  fontsize=15) 
            
            if i%m is not 0:
                ax1.set_yticklabels([])
            else:
                ax1.set_ylabel('Amplitude ((m/s²)²/Hz)',**l_for)
                         
            ax2.plot(f,np.angle(G[i//m, i%m]),color='blue')
            ax2.set_xlim(axis_f)
            ax2.set_ylim([-4,4])
            
            if i%m is not 0:
                ax2.set_yticklabels([])
            else:
                ax2.set_ylabel('Fase (rad)',**l_for)
            
            if i//m == n-1:
                ax2.set_xlabel('f (Hz)',**l_for)
            else:
                ax2.set_xticklabels([])
                
            ax1.tick_params(labelsize=12)
            ax2.tick_params(labelsize=12)
                        
        plt.show()       
     
    PSD                          = auxclass(G)
    PSD.f, PSD.nperseg, PSD.nfft = f, nperseg, nfft
        
    return PSD

#-----------------------------------------------------------------------------

def ANPSD_from_SDM(PSD, plot=False, mode='interactive'):
    """      
    Calcula e plota o Espectro Normalizado Médio a partir da matriz de
    densidades espectrais.

    Entradas
    -------       
    PSD : auxclass_like
        Um auxclass com as densidades espectrais, as frequências no atributo 
        f e o comprimento do trecho no atributo nperseg.
    plot : bool, optional
        Se verdadeira, gera o gráfico das densidades espectrais separadas
        em valor absoluto e fase. O padrão é falso.
    mode : string, optional
        Modo de detecção de picos. Se 'interactive', o usuário deve definir
        os picos com o mouse. Se 'batch', os picos são definidos pelo atributo
        pki. O padrão é 'interactive'.
    
    Saídas
    -------   
    PSD : auxclass_like
        Um auxclass com o Espectro Normalizado Médio no atributo ANPSD e 
        os índices dos picos pki. 
        
    Modo batch
    ------- 
    Neste modo, o auxclass deve possuir o atributo pki.
    pki : list
        Lista dos índices das frequências dos picos.        
    """ 
    
    try:
        G = PSD.diagonal()
        f = PSD.f
    
    except AttributeError:
        sys.exit('PSD deve ser obtido da função SDM')
    
    NPSD  = np.real((G / G.sum(axis=0)).T)
    ANPSD = np.real(NPSD.sum(axis=0)) / G.shape[1]
    
    if mode.lower() == 'interactive':

        class SnaptoCursor:
            """
            Provides data cursor
            The crosshair snaps to the closest point
            Adapted from matplotlib gallery
            https://matplotlib.org/3.2.2/gallery/misc/cursor_demo_sgskip.html
            """
    
            def __init__(self, ax, x, y):
                self.ax  = ax
                self.txt = ax.text(0.8, 0.95, '', transform=ax.transAxes)
                self.lx, self.ly = ax.axhline(color='k'), ax.axvline(color='k')  
                self.x,  self.y  = x, y
        
            def mouse_move(self, event):
                if not event.inaxes: return
                x, y = event.xdata, event.ydata
                indx = np.argmin((f-x)**2+1E4*(ANPSD-y)**2)
                x, y = self.x[indx], self.y[indx]
                self.lx.set_ydata(y), self.ly.set_xdata(x)
                self.txt.set_text('f = %1.2f Hz'%x)
                self.ax.figure.canvas.draw()

        fig, ax = plt.subplots(figsize=(12,5))
        plt.semilogy(f,np.abs(ANPSD))
        snap_cursor = SnaptoCursor(ax, f, np.abs(ANPSD))
        plt.gcf().canvas.mpl_connect('motion_notify_event', 
               snap_cursor.mouse_move)           
        plt.gcf().canvas.mpl_connect
        plt.xlim([0, f[-1]])
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('ANPSD ((m/s²)²/Hz)')
        #plt.ylim([1E6*ANPSD.min(),1.6*ANPSD.max()])        

        plt.title('Click the left mouse button to select the peaks\n'
                  'Press middle mouse button to finalize')
        
        plt.text(.7,0.01,
                 'Peaking: left button\n'
                 'Undo: right button\n'
                 'Finalize: middle button',transform=plt.gca().transAxes)

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        pnt = np.array(plt.ginput(n=-1,timeout=0))
        fn = np.zeros(len(pnt))
        for ii in range(len(pnt)):
            fn[ii]  =  f[np.argmin((f-pnt[ii,0])**2+1E4*(ANPSD-pnt[ii,1])**2)]
        pki = np.argmin(np.abs(f-fn.reshape(-1,1)),axis=1)    
        
        plt.close()
                    
    elif mode.lower() == 'batch':
        
        try:
            pki = PSD.pki
            
        except AttributeError:
            sys.exit('PSD deve ter atributo pki no modo batch')
                    
    else:
        sys.exit('mode deve ser interactive ou batch')
    
    if plot:
        plt.figure(figsize=(7,7)) 
        plt.subplot(211)
        plt.title('NPSD')
        for ii, row in enumerate(NPSD):
            plt.semilogy(f,row,label=ii+1)            
        plt.legend()
        plt.xlim([0, f[-1]])
        plt.xlabel('Frequência (Hz)')
        plt.ylabel('Amplitude ((m/s²)²/Hz)')
        #plt.ylim([1E5*NPSD.min(),1.6*NPSD.max()])
        
        plt.subplot(212)
        plt.title('ANPSD')
        plt.semilogy(f,np.abs(ANPSD))
        plt.xlim([0, f[-1]])
        plt.xlabel('Frequência (Hz)')
        plt.ylabel('Amplitude ((m/s²)²/Hz)')
        #plt.ylim([1E5*ANPSD.min(),1.6*ANPSD.max()])
        plt.tight_layout()
        plt.plot(f[pki], ANPSD[pki], "x")
    
        for i in pki:
            plt.annotate('{:.3f} Hz'.format(f[i]),
                         (f[i],ANPSD[i]*1.08), ha='center')            
        plt.show()
           
    PSD.ANPSD  = ANPSD
    PSD.pki    = pki
        
    return PSD
 
#-----------------------------------------------------------------------------
    
def coherence(self, PSD=None, nperseg=None, plot=False):
    """      
    Calcula a matriz das funções de coerência
    
    Entradas
    -------       
    self : MRPy_like
        Um MRPy com as séries temporais.
    PSD : auxclass_like, optional
        Um auxclass com o atributo pki.
    nperseg : int, optional
        Comprimento de cada segmento utilizado para as funções de coerência
        suavizadas. O padrão é igual ao comprimento do sinal.
    plot : bool, optional
        Se verdadeira, gera o gráfico das funções de coerência.
        O padrão é falso.

    Saídas
    -------   
    γ : auxclass_like
        Um auxclass das funções de coerência e as frequências no atributo f.  
    
    Ver também
    -------  
    scipy.signal.coherence
    """     
    if nperseg is None:
        try:
            nperseg = PSD.nperseg
        except AttributeError:
            sys.exit('nperseg deve ser informado ou atributo')      
    
    γ = np.empty((self.NX,self.NX,nperseg//2+1))
    
    for i in range(self.NX):
        for j in range(self.NX):
            f, γ[i,j] = signal.coherence(self[i,:], self[j,:],
                 self.fs, nperseg=nperseg)                     

    γ   = auxclass(np.real(γ))
    γ.f = f
            
    if plot:
        
        a_for = {'fontname':'Times New Roman','size':14} 
        l_for = {'fontname':'Times New Roman','size':12}        
        t_for = {'family':'Times New Roman','size':10}
        
        NX = self.NX
        
        plt.figure(figsize=(8,8))
        
        for i in range(NX):
            for j in range(NX):
                ax = plt.subplot(NX,NX,i*NX+j+1)
                ax.plot(f,γ[i,j])
                if PSD is not None: ax.plot(f[PSD.pki],γ[i,j][PSD.pki],'ro')
                ax.set_title(r'$\gamma^2_{{{:d},{:d}}}$'
                             .format(i+1,j+1),**a_for)
                ax.set_xlim([0,f[-1]])
                ax.set_ylim([0, 1.05])
                
                if j is 0:
                    #ax.set_yticklabels(np.linspace(0,1,6),**t_for)
                    plt.yticks(**t_for)
                    ax.set_ylabel('Coerência',**l_for)
                else:
                    ax.set_yticklabels([])
                                   
                if i == NX - 1:
                    plt.xticks(**t_for)
                    ax.set_xlabel('f (Hz)',**l_for)
                else:
                    ax.set_xticklabels([])
          
        plt.tight_layout()
    
    return γ

#-----------------------------------------------------------------------------

def BFD(self, PSD, plot=False, mode='interactive'):
    """      
    Método Básico no Domínio da Frequência
    
    Estima as frequências naturais, razões de amortecimentos e formas modais 
    a partir das densidades espectrais.
    
    Estima as razões de amortecimento pelos métodos de meia-potência e 
    ajuste a um espectro analítico

    Entradas
    -------       
    self : MRPy_like
        Um MRPy com as séries temporais.
    PSD : auxclass_like
        Um auxclass com as densidades espectrais e 
        as frequências no atributo f.
    mode : string, optional.
        Modo de operação. Se 'interactive', o usuário deve definir
        parâmetros com o mouse. Se 'batch', estes parâmetros devem ser 
        atributos de PSD. O padrão é 'interactive'.
    
    Saídas
    -------    
    fn : ndarray
        Array com as frequências naturais.
    zt : list
        Lista com as razões de amortecimento. zt[0] para coeficientes do
        método de meia-potência e zt[1] para ajuste ao espectro analítico.
    V : ndarray
        Array com as formas modais nas colunas.
    PSD : auxclass_like
        Um auxclass com os atributos ANPSD, pki, MGi e fint.
        
    Modo batch
    -------  
    Neste modo, o PSD deve ter os atributos fint, MGi e pki.
    fint : array
        Array com as frequências inicial e final a serem utilizadas para 
        ajuste do espectro.
    MGi : integer array
        Array com os índices dos autoespectros a serem utilizados para
        determinação das formas modais e amortecimentos.
    pki : integer array
        Array com os índices das frequências dos picos.    
    fint : array
        Array com as frequências inicial e final a serem utilizadas para 
        ajuste do espectro.
        
    Notas
    -------  
    Antes de usar essa função, empregar as funções SDM e ANPSD_from_SDM.
    """                
    try:
        f, pki = PSD.f, PSD.pki
        G      = np.abs(PSD.diagonal().T)  # autoespectros
    except AttributeError:
        sys.exit('PSD deve ter os atributos f, nperseg e pki')   
        
    #-------------------------------------------------------------------
           
    if mode.lower() == 'interactive':
        
        print('Selecione o autoespectro de referência')

        global MGi, NX
        MGi, NX = [], self.NX 

        def onclick_select(event):  # obtém número do subplot selecionado
            global MGi, NX
            for i in range(NX):
                if event.inaxes == ax[0,i]:
                    MGi = np.array(np.append(MGi,i),dtype='int')
        
        for i, j in enumerate(pki):            
            fig, ax = plt.subplots(1, self.NX,figsize=(10, 4),sharey=True,squeeze=False)
            plt.suptitle('Clique no autoespectro de referência\n'
                         'a ser utilizado para amortecimento e forma modal')
                            
            for k in range(self.NX): 
                ax[0,k].semilogy(f,G[k]) 
                ax[0,k].semilogy(f[j],G[k,j],'ro')
                ax[0,k].set_xlim((0,f[-1]))
                ax[0,k].annotate('{:.2E}'.format(G[k,j]),(f[j],G[k,j]*1.05))
            
            ax[0,0].set_ylabel('Amplitude ((m/s²)²/Hz)')
            ax[0,k//2].set_xlabel('Frequência (Hz)')
            plt.tight_layout(rect=[0, 0.03, 1, 0.92])    
            fig.canvas.mpl_connect('button_press_event', onclick_select)
            plt.ginput(n=1,timeout=30)
            plt.close()        
        
        fint = np.zeros(2*len(pki))
        
        for i,(j,k) in enumerate(zip(pki,MGi)):
            plt.figure(figsize=(10,6))
            plt.title('Clique nos extremos do intervalo ' 
                      'para ajuste ao espectro analítico')
            plt.semilogy(f,G[k])
            plt.semilogy(f[j],G[k][j],'ro')   
            plt.annotate('{:.3f} Hz'.format(f[j]),(f[j],G[k][j]*1.15), ha='center')     
            #plt.xlabel(r'$f_n$ = {:.3f} Hz'.format(f[j]))
            plt.xlabel('Frequência (Hz)')
            plt.ylabel('Amplitude ((m/s²)²/Hz)')
            plt.xlim([0,f[-1]])
            
            pnt = np.array(plt.ginput(n=2,timeout=0))[:,0]
            
            id1 = np.argmin(np.abs(f-pnt.reshape(-1,1)),axis=1)
            fint[2*i:2*i+2] = f[id1]
            
            plt.close()     
            
        PSD.fint = fint
        PSD.MGi  = MGi
       
    elif mode.lower() == 'batch':
        try:
            MGi  = PSD.MGi
            fint = PSD.fint
        except AttributeError:
            sys.exit('PSD deve ter atributo MGi e fint no modo batch')   
    
    else:
        sys.exit('mode deve ser interactive ou batch')   

    #-------------------------------------------------------------------
    def Sy(f,c1,c2,fn,ζ):

        return c1*np.abs(2*np.pi*f**2/(1-(f/fn)**2+2j*ζ*(f/fn)))**2 + c2       
    #-------------------------------------------------------------------       
                                                                                        
    ζmp = np.zeros((len(pki)))
    ζft = np.zeros((len(pki)))
    P   = np.zeros((len(pki),4))
    
    idx = np.argmin(np.abs(f-fint.reshape(-1,1)),axis=1)
    
    if plot: fig, ax = plt.subplots(1,len(MGi),figsize=(len(MGi)*4, 3),squeeze=False)

    for i, (j, k, ii, si) in enumerate(zip(MGi,pki,idx[::2],idx[1::2])):
        
        mG = G[j,k]
        fa = np.interp( mG/2, G[j,ii:k+1],f[ii:k+1])
        fb = np.interp(-mG/2,-G[j, k:si ],f[ k:si ])
        f0 = f[k]
        
        ζmp[i] = (fb**2-fa**2)/(4*f[k]**2)    # meia potência
        
        Pmin = (0     , 0    , fa, 0.000)     # lower bounds
        P0   = (0     , 0    , f0, 0.010)     # initial guesses 
        Pmax = (mG/1E2,mG/1E3, fb, 0.05 )     # upper bounds   
        
        P[i,:], _ = curve_fit(Sy,f[ii:si],G[j,ii:si],
                                         p0=P0,bounds=(Pmin, Pmax))
        ζft[i] = P[i,3]
    
        if plot:
            ax[0,i].semilogy(f[ii:si],G[j,ii:si])
            ax[0,i].semilogy(np.linspace(f[ii],f[si],100),
                         Sy(np.linspace(f[ii],f[si],100),*P[i,:]),'k:') 
            ax[0,i].plot(f0,mG,'rx')
            ax[0,i].plot([fa,fb],[mG/2,mG/2],'ro')
            ax[0,i].annotate('{:.3f} Hz'.format(f0),(f0,G[j,k]*1.05), 
                                                                  ha='center')
            ax[0,i].annotate('{:.3f} Hz'.format(fa),(fa,  mG/2*1.05), 
                                                                  ha='right')
            ax[0,i].annotate('{:.3f} Hz'.format(fb),(fb,  mG/2*1.05), 
                                                                  ha='left') 
            ax[0,i].text(.99, .99, r'$\xi_{{mp}}$ = {:.2f}%'.format(ζmp[i]*100) 
                +'\n'+ r'$\xi_{{ft}}$ = {:.2f}%'.format(ζft[i]*100), 
                horizontalalignment='right',verticalalignment='top', 
                transform=ax[0,i].transAxes,fontsize=11)
            
    if plot: 
        ax[0,0].set_ylabel('Amplitude ((m/s²)²/Hz)')
        ax[0,i//2].set_xlabel('Frequência (Hz)')   
        ax[0,i].legend(['Densidade espectral','Função ajustada',
          'Frequência natural','Frequência de meia-potência'])
        fig.tight_layout()
            
    fn = f[pki]
    V  = PSD[MGi,:,pki]/PSD[MGi,MGi,pki].reshape(-1,1)            
    V  = np.abs(V)*(1-2*((np.angle(V)>np.pi/2)+(np.angle(V)<-np.pi/2)))        
    
    return fn, [ζmp,ζft], V.T, PSD
   
#-----------------------------------------------------------------------------
    
def EFDD(self, PSD, plot=False, mode='interactive'):
    """      
    Método de decomposição no domínio da frequência
    
    Obtém as frequências naturais e formas modais a partir das densidades 
    espectrais.

    Entradas
    -------       
    self : MRPy_like
        Um MRPy com as séries temporais.
    PSD : auxclass_like
        Um auxclass com as densidades espectrais 
        e as frequências no atributo f.
    plot : bool, optional
        Se verdadeira, gera o gráfico das densidades espectrais dos valores
        singulares. O padrão é falso.
    mode : string, optional.
        Modo de operação. Se 'interactive', o usuário deve definir
        parâmetros com o mouse. Se 'batch', estes parâmetros devem ser 
        atributos de PSD. O padrão é 'interactive'.
    
    Saídas
    -------    
    fn : ndarray
        Array com as frequências naturais.
    zt : list
        Lista com as razões de amortecimento. zt[0] para coeficientes do
        método de meia-potência e zt[1] par ajuste ao espectro analítico.
    V : ndarray
        Array com as formas modais nas colunas.
    PSD : auxclass_like
        Um auxclass com os atributos pki, svi, fint e tint.
           
    Modo batch
    -------  
    Neste modo, o PSD deve ter os atributos pki e MGi.
    pki : list
        Índices dos picos do array das frequências.
    svi : list
        Índices dos valores singulares a serem utilizados para
        determinação das formas modais.   
    fint : array
        Array com as frequências iniciais e finais a serem utilizadas para 
        ajuste da obtenção das funções de autocorrelação.
    tint : array
        Array com os tempos iniciais e finais a serem utilizados para ajuste da 
        função de autocorrelação. 
    """ 
    G, f, nperseg, nfft = PSD, PSD.f, PSD.nperseg, PSD.nfft
    
    U, S, VH = np.zeros_like(G), np.zeros_like(G), np.zeros_like(G)
    
    USV = np.zeros((self.NX,len(f)))
    
    for i in range(len(f)):
        U[:,:,i],S[:,:,i],VH[:,:,i] = np.linalg.svd(G[:,:,i])    
        
    for i in range(self.NX):
        for j in range(len(f)):
            USV[i,j] = np.abs(U[i,:,j] * S[i,i,j] @ VH[:,i,j])  
        
    #----------------------------------------
    
    if mode.lower() == 'interactive':
               
        plt.figure(figsize=(10,5))
        
        for i in range(self.NX):
            plt.semilogy(f[1:],USV[i,1:])
        
        print('Selecione os picos no gráfico')
        
        plt.text(.7,0.01,
                 'Selecionar: botão esquerdo\n'
                 'Desfazer: botão direito\n'
                 'Concluir: botão do meio',transform=plt.gca().transAxes)

        plt.title('Selecione os picos')
        plt.legend(["1o","2o","3o","4o"])
        plt.xlabel('Frequência (Hz)')
        plt.ylabel('Densidades Espectrais dos Valores Singulares')
        plt.xlim([0, f[-1]])
        plt.tight_layout()

        pnt = plt.ginput(n=-1,timeout=0)          
        x   = np.array(pnt)[:,0]
        pki = np.abs(f-x.reshape(-1,1)).argmin(axis=1)  
        y   = np.array(pnt)[:,1]
        svi = np.abs(USV[:,pki]-y).argmin(axis=0)   
        
        plt.close()
                
        #----------------------------------------
        
        fint = np.zeros(2*len(pki))
    
        for i, (j, k) in enumerate(zip(svi,pki)):
            
            MACv = MAC(U[:,j,[k]],U[:,j,:])[0]
            
            fig = plt.figure(figsize=(10, 6)) 
            
            try:
                imin = np.abs(np.max(f[:k][MACv[:k] < 0.8])-f).argmin() + 1
            except ValueError:
                imin = 0
            try:
                imax = np.abs(np.min(f[k:][MACv[k:] < 0.8])-f).argmin()
            except ValueError:
                imax = -1            
            
            gs = GridSpec(2, 1, height_ratios = [4, 1]) 
            
            ax0 = plt.subplot(gs[0])
            ax0.semilogy(f[1:],USV[j,1:])  
            ax0.semilogy(f[imin:imax], USV[j][imin:imax])
            ax0.semilogy(f[k],USV[j,k],'ro') 
            ax0.legend(["Espectro do Valor Singular",
                        "Trecho com MAC > 0.8","Frequência Natural"])
            ax0.set_xlim([0,f[-1]])    
            ax0.set_ylabel('Densidades Espectrais dos Valores Singulares')
            ax0.set_xticklabels([])
            ax0.set_title('Selecione os extremos do intervalo')
            
            ax1 = plt.subplot(gs[1])
            ax1.plot(f,MACv)   
            ax1.set_xlim([0,f[-1]]) 
            ax1.set_xlabel('Frequência (Hz)')
            ax1.set_ylabel('MAC')
            
            gs.tight_layout(fig)
            
            pnt = np.array(plt.ginput(n=2,timeout=0))[:,0]
            
            id1 = np.argmin(np.abs(f-pnt.reshape(-1,1)),axis=1)
            fint[2*i:2*i+2] = f[id1]
            
            plt.close()    

        PSD.pki, PSD.svi, PSD.fint = pki, svi, fint
        
    #----------------------------------------

    elif mode.lower() == 'batch':
        try:
            pki  = PSD.pki
            svi  = PSD.svi
            fint = PSD.fint
        except AttributeError:
            sys.exit('PSD deve ter os atributos pki, svi e fint no modo batch')   
    
    #----------------------------------------
    
    else:
        sys.exit('mode deve ser interactive ou batch')           
            
    #----------------------------------------

    idx = np.argmin(np.abs(f-fint.reshape(-1,1)),axis=1)
    
    FSD  = np.zeros((len(pki),S.shape[2]))
    MACv = np.zeros((len(pki),S.shape[2]))

    for i, (j, k, ii, si) in enumerate(zip(svi,pki,idx[::2],idx[1::2])):
        FSD[i,ii:si] = USV[j,ii:si] 
        MACv[i] = MAC(U[:,j,[k]],U[:,j,:])[0]
    
    R   = np.fft.irfft(FSD)          # autocorrelação
    env = np.abs(np.fft.ifft(FSD))   # envelope da função de autocorrelação
    
    R   = R  /np.max(np.abs(R),axis=1).reshape(-1,1)   
    env = env/np.max(      env,axis=1).reshape(-1,1)   
    
    t   = np.linspace(0,self.Td*nfft/self.N,  R.shape[1])   
    te  = np.linspace(0,self.Td*nfft/self.N,env.shape[1])
  
    win = (nperseg-np.arange(0,nperseg))/nperseg       
    
    R   =   R[:,:nperseg]/win        
    env = env[:,:nperseg]/win
    
    fn, zt, PSD = fit_autc(PSD, t, te, R, env, mode, plot)     
    
    #----------------------------------------          
        
    if plot:
        
        fig = plt.figure(figsize=(8,6))
        gs = GridSpec(2, 1, height_ratios = [3, 1]) 
        
        ax0 = plt.subplot(gs[0])     
        
        leg = ['1° valor singular','2° valor singular','3° valor singular']
        
        for ii in range(G.shape[0]):
            ax0.semilogy(f[1:],USV[ii,1:],label=leg[ii])
            
        for i, (ii, si) in enumerate(zip(idx[::2],idx[1::2])):
            ax0.semilogy(f[ii:si],np.abs(FSD[i,ii:si]),'r',label=(i//1)*"_"+'Modo')
        
        ax0.legend()
        ax0.plot(f[pki], USV[svi,pki], "x")
        
        for jj,kk in zip(pki,svi):
            ax0.annotate('{:.3f} Hz'.format(f[jj]),
                         (f[jj],USV[kk,jj]*1.25), ha='center')
            
        ax0.set_xlim([0,f[-1]])  
        ax0.set_xticklabels([])
        ax0.set_ylabel('Amplitude ((m/s²)²/Hz)')
        ax0.set_title('Densidades Espectrais dos Valores Singulares')
        
        leg = ['1° modo','2° modo','3° modo','4° modo','5° modo','6° modo']
        
        ax1 = plt.subplot(gs[1])
        for i, (ii, si) in enumerate(zip(idx[::2],idx[1::2])):
            ax1.plot(f[ii:si],MACv[i,ii:si],label=leg[i])  
        
        ax1.legend()
        ax1.set_xlim([0,f[-1]])
        ax1.set_xlabel('Frequência (Hz)')
        ax1.set_ylabel('MAC')
        
        gs.tight_layout(fig)
        
    #------------------------------------------
    
    V = U[:,svi,pki]
     
    return fn, zt, V, PSD

#----------------------------------------------------------------------------- 
    
def fit_autc(PSD, t, te, R, env, mode='interactive', plot=False):
    """
    Ajusta a função de autocorrelação teórica aos pontos, estimando a 
    frequência natural e a razão de amortecimento.
    
    Entradas
    -------       
    PSD : auxclass_like
        Um auxclass com os atributos f e pki.
    t : ndarray
        Array unidimensional com a série temporal da função de autocorrelação.
    te : ndarray
        Array unidimensional com a série temporal do envelope.
    R : ndarray
        Array com as funções de autocorrelação nas linhas.
    env : ndarray
        Array com os envelopes da função de autocorrelação.
    mode : string, optional.
        Modo de escolha do intervalo de tempo. Se 'interactive', o usuário 
        deve definir os limites com o mouse. Se 'batch', os limites devem 
        estar no atributo PSD.tint.
    
    Saídas
    -------    
    fn : ndarray
        Array com as frequências naturais
    zt : ndarray
        Array com as razões de amortecimento
    PSD : auxclass_like
        Um auxclass com o atributo tint.

    Modo batch
    -------  
    Neste modo, o PSD deve ter os atributos f, pki e tint.
    f : Array
        Array com as frequências das densidades espectrais.
    pki : integer array
        Array com os índices das frequências dos picos. 
    tint : array
        Array com os tempos inicial e final a serem utilizadas para ajuste da 
        função de autocorrelação.  
    """    

    #--------------------------------------------------
    def envelope(t, Xp, η):
        
        return Xp*np.exp(-η*t)
    
    def decay(t, Xp, η, fn):

        ωn = 2*np.pi*fn
        ζ  = η/ωn
        ωd = ωn * (1-ζ**2)**.5
        
        return Xp*np.exp(-η*t)*np.cos(ωd*t)
    
    #--------------------------------------------------
    
    if mode.lower() == 'interactive':    

        idx = np.zeros(2*len(PSD.pki),dtype=int)
            
        for ii in range(len(PSD.pki)):  
            plt.figure(figsize=(6,4))
            plt.plot(te[:len(te)//4],env[ii][:len(te)//4],'bo')
            plt.xlim([0,te[len(te)//4]])
            plt.xlabel('Tempo (s)')
            plt.ylabel('Autocorrelação Normalizada')
            plt.title('Selecione os limites do trecho a ser ajustado')
            plt.tight_layout()       
            
            pnt = np.array(plt.ginput(n=2,timeout=0))[:,0]
            idx[2*ii:2*ii+2] = np.argmin(np.abs(te-pnt.reshape(-1,1)),axis=1)
            
            plt.close()
            
        PSD.tint = te[idx]
    
    #----------------------------------------
    
    elif mode.lower() == 'batch':
        try:
            tint = PSD.tint            
        except AttributeError:
            sys.exit('PSD deve ter o atributo tint no modo batch')   
         
        idx = np.argmin(np.abs(te-tint.reshape(-1,1)),axis=1)
        
    #----------------------------------------
    
    else:
        sys.exit('mode deve ser interactive ou batch')           
            
    #----------------------------------------
       
    P   =  np.zeros((len(PSD.pki), 2))
    Q   =  np.zeros((len(PSD.pki), 1))
       
    for i, (j, k) in enumerate(zip(idx[::2],idx[1::2])):

        X0 =  1.00                   # initial amplitude value  
        ζ0 =  0.01                   # initial damping value  
        fn =  PSD.f[PSD.pki[i]]      # initial natural frequency
        η0 =  2*np.pi*fn*ζ0              
        
        Pmin = (1.00*X0, 0*η0)       # lower bounds
        P0   = (     X0,   η0)       # initial guesses
        Pmax = (1.25*X0, 5*η0)       # upper bounds   
        
        P[i,:], cv = curve_fit(envelope, te[j:k], env[i,j:k],       # fit for
                                        p0=P0, bounds=(Pmin, Pmax)) # X and η        
                
        Qmin = (0.97*fn)             # lower bounds
        Q0   = (     fn)             # initial guesses
        Qmax = (1.03*fn)             # upper bounds   
        
        Q[i,:], cv = curve_fit(lambda x, fn: decay(x,*P[i,:], fn),  
             t[2*j:2*k], R[i,2*j:2*k], p0=Q0, bounds=(Qmin, Qmax)) # fit for fn

    #--------------------------------------------------

    fn = Q[:,0]
    zt = P[:,1]/(2*np.pi*fn)

    #--------------------------------------------------
    
    if plot:
        
        tf = np.linspace(0,t[-1],len(t)*100)

        fig, ax = plt.subplots(1, len(PSD.pki), figsize=(4*len(PSD.pki),4),
                               sharey=True,squeeze=False)   
        
        for i, (j, k) in enumerate(zip(idx[::2],idx[1::2])):
            ax[0,i].plot(t[2*j:2*k],R[i,2*j:2*k],'bo')
            #ax[0,i].plot(te[j:k],env[i,j:k],'ro')
            ax[0,i].plot(tf,decay(tf, *P[i,:], *Q[i,:])) #fitted curve
            ax[0,i].set_xlim(0,t[2*k])
            
            ax[0,i].text(.99, .99, r'$f_n$ = {:.3f} Hz'.format(fn[i]) 
                +'\n'+ r'$\xi$ = {:.2f}%'.format(zt[i]*100), 
                horizontalalignment='right',verticalalignment='top', 
                transform=ax[0,i].transAxes,fontsize=11)

           
        ax[0,i//2].set_xlabel("Tempo (s)")
        ax[0,0].set_ylabel("Autocorrelação Normalizada")
        fig.suptitle('Funções de Autocorrelação Normalizadas')     
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])    
    
    return fn, zt, PSD

#=============================================================================
# Funções auxiliares: MAC e gráficos das formas modais
#=============================================================================  
    
def MAC(Ψi,Ψj, plot=False):
    """
    Calcula o Critério de Concordância Modal [1] a partir das formas modais 
    localizadas nas colunas de Ψi e Ψj.
    
    Entradas
    -------     
    Ψi, Ψj : array_like
        2D arrays com as formas modais nas colunas.
    plot : bool, optional
        Se verdadeiro, gera um gráfico dos MACs. O padrão é falso.        

    Saídas
    -------     
    MAC : array_like
    
    Referências
    ----------
    .. [1] Allemang, R. J.; Brown, D. L. "A correlation coefficient for 
           modal vector analysis", In: 1st International Modal Analysis
           Conference, p. 110-116, 1982.
    """
    
    MOMij =         Ψi.T @ np.conj(Ψj)
    MOMii = np.diag(Ψi.T @ np.conj(Ψi))
    MOMjj = np.diag(Ψj.T @ np.conj(Ψj))
    
    MAC   = np.abs(MOMij)**2 / np.outer(MOMii,MOMjj)
    
    if plot:
        plt.figure()
        plt.pcolormesh(np.real(MAC), cmap='Blues', vmin=0, vmax=1, 
                       edgecolors='k', linewidth=.5)
        cb = plt.colorbar()
        cb.ax.set_title('MAC')
        plt.xticks(np.arange(.5,MAC.shape[1]  ,1),
                   np.arange( 1,MAC.shape[1]+1,1))
        plt.yticks(np.arange(.5,MAC.shape[0]  ,1),
                   np.arange( 1,MAC.shape[0]+1,1))
        plt.gca().invert_yaxis()
        plt.gca().xaxis.tick_top()
        plt.tight_layout()
    
    return np.real(MAC)

#-----------------------------------------------------------------------------
        
def plot_1dshapes(fn,zt,vv,title,X,ref=False, fix=False):
    """      
    Plota formas modais unidimensionais
    
    Entradas
    -------   
    fn : ndarray
        Array com as frequências naturais.
    zt : ndarray
        Array com as razões de amortecimento.         
    vv : ndarray
        Array 2D com as formas modais como colunas.
    title : string
        Título do gráfico.
    X : ndarray
        Coordenadas dos l sensores.
    ref : tuple, list, optional
        Especifica quais sensores são de referência.
    fix : list, optional
        Adiciona componente modal nula no gráfico para as coordenadas 
        especificadas. Por exemplo se [0,L], adiciona a componente nula na
        coordenada zero e na coordenada L. O padrão é False.
        
    Notas
    ------- 
    As primeiras componentes das formas modais são dos sensores de 
    referência. Por isso a necessidade de informar ref.
    """        
    a_for = {'fontname':'Times New Roman','size':14}
    l_for = {'fontname':'Times New Roman','size':12}
    t_for = {'fontname':'Times New Roman','size':10}
    
    if ref is not False:  
        X  = np.hstack((X[ref,],np.delete(X,ref)))      

    if fix is not False:
        X  = np.hstack((np.array(fix),X))
        vv = np.vstack((np.zeros((len(fix),vv.shape[1])),vv))
    
    vv  = np.sign(np.real(vv))*np.abs(vv)     
    idx = np.argsort(X)         
    it  = np.argsort(fn)
    
    plt.figure(figsize=(2*len(fn),5))
    
    for i, k in enumerate(it):
        plt.subplot(1,fn.shape[0],i+1)
        plt.plot(0.97*vv[idx,k]/np.max(np.abs(vv[:,k])),X[idx])
        plt.xlim((-1,1))                    
        plt.ylim((0,X.max()))
        plt.xticks(**t_for)
        plt.yticks([])
        plt.xlabel(r'$f_n$ = {:.3f} Hz''\n'r'$\zeta$ = {:.2f} %'
             .format(fn[k],zt[k]*100),**l_for)            
        if i == 0:
            plt.yticks(np.linspace(0,X.max(),10),**t_for)
      
    plt.suptitle(title + ' Modal Shapes',**a_for)
    plt.tight_layout(rect=[0, -0.02, 1, 0.97]) 
    
    return

#-----------------------------------------------------------------------------
        
def plot_3das1d(fn,zt,q,X,title,ref=False):
    """
    Plota as três formais modais de edíficios no formato unidimensional.
    
    Entradas
    -------          
    fn : ndarray
        Array com as frequências naturais.
    zt : ndarray
        Array com as razões de amortecimento.         
    q : ndarray
        Array 2D com as formas modais como colunas.
    title : string
        Título do gráfico.
    X : ndarray
        Array com as alturas. 
    title : string
        Título do gráfico.
    ref : tuple, list, optional
        Especifica quais sensores são de referência.        
    
    Notas
    ------- 
    As primeiras componentes das formas modais são dos sensores de 
    referência. Por isso a necessidade de informar ref.
    A componente modal nula na base é adicionada automaticamente.     
    """ 

    a_for = {'fontname':'Times New Roman','size':14} 
    g_for = {'family'  :'Times New Roman','size':12}
    l_for = {'fontname':'Times New Roman','size':12}        
    t_for = {'fontname':'Times New Roman','size':10}

    if ref is not False: 
        X  = np.hstack((X[ref],np.delete(X,ref)))
    
    q   = np.sign(np.real(q))*np.abs(q)  
    it  = np.argsort(fn)
    X   = np.hstack((0,X)) 
    q   = np.vstack((np.zeros((3,len(fn))),q))
    idx = np.argsort(X)
    
    plt.figure(figsize=(2*len(fn),5))

    for ii, kk in np.ndenumerate(it):
        
        q[:,kk] = q[:,kk]/np.max(np.abs(q[:,kk])) 
        
        plt.subplot(1,fn.shape[0],ii[0]+1)
        plt.plot(q[0::3,kk][idx],X[idx],'k',  linewidth=3)
        plt.plot(q[1::3,kk][idx],X[idx],'r--',linewidth=2)
        plt.plot(q[2::3,kk][idx],X[idx],'g:', linewidth=4)
        plt.xticks(**t_for)
        plt.yticks([])
        plt.xlim((-1,1))
        plt.ylim((0,X.max()))
        plt.xlabel(r'$f_n$ = {:.3f} Hz''\n'r'$\zeta$ = {:.2f} %'
             .format(fn[kk],zt[kk]*100),**l_for)
        if ii[0] == 0:
            plt.yticks(np.linspace(0,X.max(),10),**t_for)
            plt.legend(('x', 'y',r'$\theta_z$'), loc='lower left',
                       prop=g_for, handlelength=1.2, handletextpad=0.4)
        
    plt.suptitle(title +' Modal Shapes',**a_for)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])    
    plt.show()        
    
    return