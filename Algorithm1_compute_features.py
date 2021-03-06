# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 11:47:26 2021

@author: Yann
"""

#   COMPUTATION OF FEATURES (WITHOUT USING INVERSION SYMMETRY)

###############################################################################
#   To use this algorithm, just set correct values for nmax, lmax and numax
#   and then run the algorithm. All features are printed, i.e. non-zeros  
#   features first followed by zero features. Non-zero features are given 
#   in valriable lsNZ
###############################################################################

import numpy as np
from sympy import *
from sympy.physics.quantum.cg import CG
from sympy.physics.quantum import Ket, Bra
import time

##########################################################

# Initialisation of the limiting parameters, i.e. max value of n, l and nu
nmax = 2
lmax = 2
numax = 5

##########################################################

init_printing()

# Creation of a class COEFF that is composed of a list of non-zero coefficients,
# a list of zero coefficients, and printing functions to print these lists
class COEFF(object):
    listcoeff=[]
    def printerNZ(self):
        print()
        print("\n".join( [ str(elem) for elem in self.listcoeff ] ))
    listcoeffZ=[]
    def printerZ(self):
        print()
        print("\n".join( [ str(elem) for elem in self.listcoeffZ ] ))
    def adds(self,rho):
        self.listcoeff.append(rho)
    def addsZero(self,rhoZ):
        self.listcoeffZ.append(rhoZ)
    def writerNZ(self):
        f=open('Feature.txt', 'w+')
        f.write("\n".join( [ str(elem) for elem in self.listcoeff ] ))
        f.close()
    def writerZ(self):
        f=open('Zeros.txt', 'w+')
        f.write("\n".join( [ str(elem) for elem in self.listcoeffZ ] ))
        f.close()

# Creation of a class RHO that is composed of the following elements:
# -value of nu
# -index feature (n,l,m) of rho coefficient
# -value of sigma
# -value of lambda
# -value of mu and value of the coefficient
class RHO(object):
    def __init__(self, nu, den,  sig, lam, muc):
        """RHO(nu,  feature_indices(column n,l,k), sig, lam, [mu,coeff])"""
        self.nu=nu
        self.den=den
        self.sig=sig
        self.lam=lam
        self.muc=muc
    def __str__(self):
        text='<'+":".join([ 'Rho = %s' %(self.nu) , str(self.den),
                            '%s %s' %(self.sig, self.lam)])+'>'
        for m,c in zip(self.muc[:,0],self.muc[:,1]):
            text=text+'\n \t \u03BC=%s \t = %s' %(m, c) 
        return text+'\n'
    
# Function calulating coefficients for nu=1
def Listnu1(n_max,l_max,ls, nber_nu):
    counter=0
    coeff=symbols('c')
    for n in range(1,n_max+1):
        for l in range(l_max+1):
            nber=int(2*l+1)
            muc=zeros(nber,2)
            mu=np.linspace(-l,l,nber)
            mu=mu.astype(int)
            muc[:,0]=mu
            for i in range(nber):
#               Compute the value of the coefficient of the monomial according
#               to the NICE formula
                muc[i,1]=Bra(n)*Ket('\u03C1', l, mu[i])*(1/sqrt(nber))*(-1)**(l-mu[i])
            x=RHO(1,list([n,l,l]),1,l,muc)
            ls.adds(x)
            counter+=1

#   Printing non-zero features 
    ls.printerNZ()
#   Actuallisation of the number of non-zero elements for nu=1
    nber_nu[0]=counter
#   For each components, there exists 2*l+1 more elements correspnding to the 
#   the value mu can take
    return

# Function calulating coefficients for nu>1
def Listnu(nu, n_max, l_max, ls, nber_nu):
    counter=0
    nberNZ=0
    nberZ=0
    knu=0
    muc=zeros(2*l_max+1,2)
    cg=symbols('CG')
    coeff=symbols('coeff')
    c=symbols('coeff_interm')
    A=np.sum(nber_nu[:(nu-2)])
    B=np.sum(nber_nu[:(nu-1)])+1
#   Loop over (nu-1)-body-order features, calculated previously
    for rho in ls.listcoeff[A:B]: #nu=nu-1
        knu=rho.lam
#       Loop over the nu-th n, from 1 to nmax
        for nnu in range(1,n_max+1):
#           Loop over the nu-th l, from the (nu-1)-th l to lmax, ordered l
            for lnu in range(rho.den[-2],l_max+1):
#               Order n, if (nu-1)-th l = nu-th l, then (nu-1)-th n <= nu-th n
                if (lnu==rho.den[-2] and nnu>=rho.den[-3]) or (lnu>rho.den[-2]):
#                   Loop over lambda, according to possible values from CG
                    for LAM in range(abs(knu-lnu),min(l_max, knu+lnu)+1):
                        nberNZ=0
                        nberZ=0
                        muc=muc*0
#                       Loop over mu, from -lambda to lambda
                        for MU in range(-LAM, LAM+1):
                            coeff=0
#                           Loop over q and corresponding coefficients of 
#                           previous (nu-1)-body-order features
                            for qnu,coeffrho in zip(rho.muc[:,0],rho.muc[:,1]):
#                               Compute only possible value m according to CG
                                mnu=MU-qnu
#                               Compute CG coefficient
                                cg=CG(lnu,mnu,knu,qnu,LAM,MU).doit()
                                c=0
                                if N(cg)!=0:
#                                   Compute the feature coefficient according
#                                   to NICE formula
                                    c=Bra(nnu)*Ket('\u03C1', lnu, mnu)*coeffrho 
                                    if ((LAM-knu-mnu)%2)==1: cg=-cg
                                    c=c*cg*sqrt(2*knu+1)
                                    coeff+=c
                            coeff=coeff/sqrt(2*LAM+1)
                            coeff=coeff.expand()
#                           Check if it is non-zero
                            if coeff!=0:
                                muc[nberNZ,0]=MU
                                muc[nberNZ,1]=coeff
                                nberNZ+=1
                            else :
                                nberZ+=1
                                muc[-nberZ,0]=MU
#                       Compute value of sigma
                        s=(-1)**(lnu+knu+LAM)*rho.sig
#                       Add feature to the corresponding list
                        if nberNZ!=0 : 
                            x=RHO(nu, list(rho.den + [nnu, lnu, knu]), s, LAM, muc[:nberNZ,:])
                            ls.adds(x)
                            counter+=1
                        if nberZ!=0 :
                            x=RHO(nu, list(rho.den + [nnu, lnu, knu]), s, LAM, muc[-nberZ:,:])
                            ls.addsZero(x)
                    
                
#   List.printer() non-zeros
    print('\n\n List NON-ZEROS \n')
    ls.printerNZ()
#   List.printer() zeros
    print('\n\n List ZEROS \n')
    ls.printerZ()
#   Actuallisation of the number of non-zero elements for nu=1
    nber_nu[nu-1]=counter
#   For each components, there exists 2*l+1 more elements correspnding to the 
#   the value mu can take
    return   

# List with non-zero coefficients 
lsNZ=COEFF()
# Array keeping track of the number of non-zero coefficients 
# there are for each nu
nberNU=np.zeros(numax).astype(int)

# Run the following section for each nu, in order to compute nu=3 features
# it is necessary to compute n=1 and nu=2 features beforehand

#%%             

# CALCULATIONS FOR NU=1

print('''
      -------------------------------------
      ''')
      
print('NU=1:')

start=time.time()
Listnu1(nmax,lmax, lsNZ, nberNU)
end=time.time()
print(end-start)


#%%

# CALCULATIONS FOR NU=2

print('''
      -------------------------------------
      ''')
      
print('NU=2:')   
start=time.time()                 

Listnu(2, nmax, lmax, lsNZ, nberNU)
end=time.time()
print(end-start)

#%%

# CALCULATIONS FOR NU=3

print('''
      -------------------------------------
      ''')
      
print('NU=3:')                    

start=time.time()
Listnu(3, nmax, lmax, lsNZ, nberNU)
end=time.time()
print(end-start)

#%%

# CALCULATIONS FOR NU=4

print('''
      -------------------------------------
      ''')
      
print('NU=4:')                    

start=time.time()
Listnu(4, nmax, lmax, lsNZ, nberNU)
end=time.time()
print(end-start)

#%%

# CALCULATIONS FOR NU=5

print('''
      -------------------------------------
      ''')
      
print('NU=5:')                    

start=time.time()
Listnu(5, nmax, lmax, lsNZ, nberNU)
end=time.time()
print(end-start)