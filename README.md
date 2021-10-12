# CESSIPy
## A Python open source module for Stochastic System Identification in Civil Engineering

CESSIPy is a Python module for estimating a modal model of a vibrating structure from output-only measurements. The identified model consists of natural frequencies, damping ratios and mode shapes and the inputs are assumed to be realizations of a white noise process. 

The following methods are avaliable: 

1. Peak-Picking method also called Basic Frequency Domain method
2. Enhanced Frequency Domain Decomposition method
3. Instrumental Variable method
4. Covariance-driven Stochastic Subspace Identification
5. Data-driven Stochastic Subspace Identification

### Further reading

* [**User Guide**](https://nbviewer.jupyter.org/github/MatheusCarini/CESSIPy/blob/main/English/User_Guide.ipynb)
* [Script example 1 (simulation of a shear building identification)](https://github.com/MatheusCarini/CESSIPy/blob/main/English/Example01_Shear_Building.py)
* [Script example 2 (time data from iNVH app)](https://github.com/MatheusCarini/CESSIPy/blob/main/English/Example02_Smartphone.py)

## Um módulo de código aberto em Python para Identificação Modal Estocástica de Estruturas

O principal objetivo deste módulo é a identificação das propriedades dinâmicas de estruturas, ou seja as frequências naturais, razões de amortecimento e formas modais de estruturas, quando apenas as saídas (acelerações) são medidas. Programaram-se:

1. Método de Detecção de Picos (*Peak Picking Method* - PP) também chamado de Método Básico no Domínio da Frequência
2. Método de Aperfeiçoado de Decomposição no Domínio da Frequência (*Enhanced Frequency Domain Decomposition* - FDD também chamado de *Complex Mode Indication Function* - CMIF)
3. Método das Variáveis Instrumentais (*Instrumental Variable* - IV)
4. Método de Identificação Estocástica de Subespaços baseados nas Covariâncias (*Covariance-driven Stochastic Subspace Identification* - SSI COV)
5. Método de Identificação Estocástica de Subespaços baseados nas Séries Temporais (*Data-driven Stochastic Subspace Identification* - SSI DATA)

### Leitura adicional

* [**Guia do usuário**](https://nbviewer.jupyter.org/github/MatheusCarini/CESSIPy/blob/main/Portugu%C3%AAs/Guia_do_usu%C3%A1rio.ipynb) 
* [Exemplo 1: simulação de um pórtico plano](https://github.com/MatheusCarini/CESSIPy/blob/main/Portugu%C3%AAs/Example01_Shear_Building.py) 
* [Exemplo 2: identificação de uma laje excitada pelo caminhar humano e gravada com o aplicativo iNVH](https://github.com/MatheusCarini/CESSIPy/blob/main/Portugu%C3%AAs/Example02_Smartphone.py)
