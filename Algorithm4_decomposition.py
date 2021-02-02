# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 13:41:48 2021

@author: Yann
"""

#   COMPUTE IRREDUCIBLE FEATURES

#   FIRST COMPUTE FEATURES USING ALGORITHM 1 !!!

###############################################################################
#   To use this algorithm, just set correct values for nmax and lmax
#   and then run the algorithm. Irreducible basis is given in variable lsirred
###############################################################################

import numpy as np
from sympy import *
from sympy.physics.quantum.cg import CG
from sympy.physics.quantum import Ket, Bra
import time

########################################################

# Set nmax and lmax
nmax=2
lmax=2

########################################################

# Creation of a class IRRED that is composed of a list of irreducible features, 
# another of reducible features and a printing function
class IRRED(object):
    listirred=[]
    listred=[]
    def printer(self):
        print()
        print('LIST IRREDUCIBLE','\n')
        print("\n".join( [ str(elem) for elem in self.listirred ] ))
    def adds(self,rho):
        self.listirred.append(rho)
    def printerred(self):
        print()
        print('LIST REDUCIBLE','\n')
        print("\n".join( [ str(elem) for elem in self.listred ] ))
    def addsred(self,rho):
        self.listred.append(rho)
    def writer(self):
        f=open('Irreducible.txt', 'w+')
        f.write("\n".join( [ str(elem) for elem in self.listirred ] ))
        f.close()

# Creation of a class POLYNOMIALS that is composed of the following elements:
# -value of nu
# -value of the coefficient of the feature
# -index in the initial list from algorithm 1
# -vector composed of individual coefficients for each tuple of monomials
class POLYNOMIALS(object):
    def __init__(self, nu, coeff, idx, vect):
        """POLYNOMIALS(nu, coeff, componenet_index_in_List)"""
        self.nu=nu 
        self.coeff=coeff 
        self.idx=idx
        self.vect=vect
        
# Produces a single list of all coefficients (no longer lists within other lists)       
def List(ls):
    List=[]
    for x in ls.listcoeff:
        for m,c in zip(x.muc[:,0],x.muc[:,1]):
            List.append(RHO(x.nu, x.den, x.sig, x.lam, np.array([[m,c]])))
    return List

# Computes all monomials for given nmax and lmax
def Monomials(n_max, l_max):
    listmonomial=[]
    c=symbols('c')
    for n in range(1, n_max+1):
        for l in range(l_max+1):
            for m in range(-l, l+1):
                c=Bra(n)*Ket('r', l, m)
                listmonomial.append(c)
    return listmonomial

# Gives the list of all combinations for a given list of monomials
def List_Terms(listmono, numax):
    N=len(listmono)
    ls=list(listmono)
    v=np.ones(N).astype(int)
    for n in range(1,numax):
        z=ls[-np.sum(v):]
        for m in range(N):
            v[m]=np.sum(v[m:])
            for x in z[-v[m]:]:
                ls.append(x*listmono[m])
    return ls[-np.sum(v):]

# Gives a list lsnu of all candidate polyniomials from lower body-order
# irreducible features, for a given nu
def get_candidates(ls, nu, lsterms):
    lsnu=[]
    N=len(lsterms)
    v=np.zeros(N)
    M=int(np.ceil(nu/2))
    for i in range(M):
        j=nu-i-2
        for x in ls[i]:
            for y in ls[j]:
                lsindex=sorted(x.idx+y.idx)
                count=0
                v=v*0
                for z in lsnu:
                    if z.idx==lsindex: 
                        count+=1
                        break
                if count==0:
                    coeff=(x.coeff*y.coeff).expand()
                    nn=0
                    for c in lsterms:
                        v[nn]=float(coeff.coeff(c,1))
                        nn+=1
                    lsnu.append(POLYNOMIALS(numax, coeff, lsindex, v))
    return lsnu

# Computes the irreducible list (basis) of features
def Basis(ls, lsmono, basis):
    maxnu=ls[-1].nu
    print(maxnu)
    listcandidates=[]
    lsnu=[]
    index=0
    m=0
    for x in ls:
#       Start with 1-body-order features, all irreducible by deffinition
        if x.nu==1: 
            basis.adds(x)
            v=np.zeros(len(lsmono))
            m=0
            for c in lsmono:
                v[m]=float((x.muc[0,1]).coeff(c,1))
            lsnu.append(POLYNOMIALS(1, x.muc[0,1], [index], v))
        index+=1
#   Add list monomials to list candidates
    listcandidates.append(lsnu)
#   Determine irreducible features for higher body-order features
    for n in range(2,maxnu+1):
        index=0
        lsdep=[]
        lsindep=[]
        lsterms=List_Terms(lsmono,n)
#        Compute possible candidates from lower body-order irreducible features 
        lsnu=get_candidates(listcandidates,n,lsterms)
        listcandidates.append(lsnu)
        N=len(lsterms)
#       Test all nu-body-order features
        for x in ls:
            if x.nu==n :
                b=np.zeros([N])
                m=0
#               Get vector of coefficients from features
                for c in lsterms:
                    b[m]=float((x.muc[0,1]).coeff(c,1))
                    m+=1
                m=0
                bb=b.dot(b)
                for a in lsnu:
#                   Compute coefficent of collinearity
                    sol=a.vect.dot(b)/bb
#                   Check collinearity (up to a tolerance)
                    if np.all(abs(a.vect-sol*b)<(1e-18*np.ones(N))): 
                        m+=1
                        break
#               If never collinear then independent -> goes to indep list
                if m==0: 
                    lsindep.append(POLYNOMIALS(n, x.muc[0,1], [index], b))
#                -> otherwise goes to dep list
                else : 
                    lsdep.append(POLYNOMIALS(n, x.muc[0,1], [index], b))
            index+=1
        m=0
        A=np.zeros([N,0])
#       Create matrix A from dependent features (case polynomial dependence only)
        for x in lsdep:
            A=np.column_stack([A,x.vect.T])
        m=0
#       Check polynomial dependence of all features in indep list
        for x in lsindep:
#           Solve least square problem (sol[0]=vector, sol[1]=residuals)
            sol=np.linalg.lstsq(A,x.vect.T, rcond=None)
            mm=0
#           Vector coefficient under a tolerence are set to 0
            for c in sol[0]:
                if c<1e-8: 
                    sol[0][mm]=0
                mm+=1
#           Check norm of residuals
#           If residuals != 0 -> add to irreducible list and add column vector to matrix A
            if ((len(sol[1])>0) and (sol[1]>1e-8)) or (np.any(abs(A.dot(sol[0].T)-x.vect.T)>1e-8*np.ones(N))) :# residual not 0 => indep
                basis.adds(ls[x.idx[0]])
                listcandidates[-1].append(x)
                A=np.column_stack([A,np.transpose(x.vect)])
                lsdep.append(x)
#           If residuals == 0 -> Not independent and not irreducible and add to reducible list
            else :
                basis.addsred(ls[x.idx[0]])
            m+=1
        m=0 
    return 

lsirred=IRRED()
listmono=Monomials(nmax, lmax)
terms=List_Terms(listmono,nmax)
listtot=List(lsNZ)
Basis(listtot,listmono,lsirred)