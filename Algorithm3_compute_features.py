# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 00:39:59 2021

@author: Yann
"""


# COMPUTES OF FEATURES (USING LINEAR ALGEBRA WITH SPARSE MATRICES)

###############################################################################
#   To use this algorithm, just set correct values for nmax, lmax and tol
#   and then run the algorithm. Non-zero features are given 
#   in matrix valriable r1, r2, r3, r4, r5 ...
###############################################################################

import numpy as np
from sympy.physics.quantum.cg import CG
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse import hstack
import time

############################################################

# Set nmax and lmax
n_max=2
l_max=2
# Set tolerance for accuracy of coefficients  
tol=1e-8

############################################################

# Assignes a unique index to each monomial composing the tuple of monomials
def decompose(n,b,nu):
    "n=number, b=basis"
    t=np.zeros(nu).astype(int)
    if n==0:
        return [0]
    i=1
    while n>0:
        q,r=n//b,n%b
        t[-i]=r
        n=q
        i+=1
    return t

# Assignes a unique index to a tuple of monomials
def compose(ls,b):
    "ls=list number composition, b=basis"
    N=len(ls)
    n=0
    for i in range(N):
        n+=ls[i]*b**(N-i-1)
    return n

# Computes feature coefficients
def Features(nu,mono,prev,nmax,lmax,tol):
#   Compute all 1-body-order features
    if nu==1:
        L=nmax*(lmax+1)**2
        i=0
        rho=np.zeros([L+1,8+L])
#       Matrix is composed of n,l,m, sigma, lambda, mu indexes
#       followed by the corresponding coefficient vector
        for l in range(lmax+1):
            for n in range(1,nmax+1):
                rho[i,0]=1
                for m in range(-l,l+1):
#                   Feature indexes
                    rho[i,1:8]=[nu,n,l,l,1,l,m]
#                   Coefficient vector: 0 if index monomial != index coefficient
                    rho[i,8+i]=(-1)**(l-m)/np.sqrt(2*l+1)
                    i+=1
        rho[-1,8:]=np.linspace(0,L-1,L)
#   Compute all higher body-order features
    else :
#       Matrix LM composed of all possible (lambda,mu)-couples
        LM=np.zeros([(lmax+1)**2,2]).astype(int)
        i=0
        for l in range(lmax+1):
            for m in range(-l,l+1):
                LM[i,:]=[l,m]
                i+=1
        lsm=np.where(mono[:,0]==1)[0]
        prevr=prev[:,:3*(nu-1)+5].toarray()
        lsp=np.where(prevr[:,0]==1)[0]
        L=len(lsm)
#       Matrix index composed of all possible nu-body-order (n,l,k)-tuples
        index=np.zeros([0,nu*3+2]).astype(int)
        j=0   
        for x in prevr[lsp,1:-1]:
            lpr=int(x[-4])
            npr=int(x[-5])
            i=nmax*lpr+(npr-1)
            index=np.vstack([index,np.concatenate((np.tile(x[1:-2],(L-i,1)),mono[lsm[i:],2:4],np.tile([x[-1],x[-2],j],(L-i,1))),axis=1)])
            j+=1
#       Matrix rho composed of all possible ordered feature indexes
        rho=np.zeros([0,nu*3+5]).astype(int)
        for x in index:
            lnu=int(x[-4])
            knu=int(x[-3])
            bot=(abs(lnu-knu))**2
            top=(min(lmax,lnu+knu)+1)**2
            L=top-bot
            rho=np.vstack([rho,np.concatenate((np.tile([x[-1],nu],(L,1)),np.tile(x[:-1],(L,1)),LM[bot:top,:]),axis=1)])
        rho[:,-3]=(-1)**(np.sum(rho[:,3:-3:3],axis=1)+rho[:,-2]) #!!!
        rho=np.vstack([rho,np.zeros(nu*3+5)])
        B=len(mono)-1
        L=(B)**nu-1
        T=int(np.sum(2*rho[:,-4]+1))
#       Sparse matrix M composed of individual nu-body-order coefficient vectors
        M=lil_matrix((L,T+1))
#       Sparse matrix A composed of vectors for linear combination of individual nu-body-order coefficient vectors
        A=lil_matrix((T+1,len(rho)))
        lsp=np.hstack([lsp,len(prevr)-1])
        j=1
        jj=0
#       Computation of matrices M and A
        for x in rho[:-1]:
            [nnu,lnu,knu]=x[-6:-3]
            LAM=x[-2]
            MU=x[-1]
            i=int(x[0])
            for ii in range(lsp[i],lsp[i+1]):
                qnu=prevr[ii,-1]
                mnu=MU-qnu
                cg=float(CG(lnu,mnu,knu,qnu,LAM,MU).doit())
                if cg!=0:
                    for k in (prev[ii,3*(nu-1)+5:].nonzero()[1]+3*(nu-1)+5):
                        idx=int(compose(np.sort(np.hstack([decompose(prev[-1,k],B,nu-1),nmax*lnu**2+(nnu-1)*(2*lnu+1)+mnu+lnu])),B))
                        M[idx,j]=prev[ii,k]
                        M[idx,0]=idx
                    A[j,jj]=cg*np.sqrt((2*knu+1)/(2*LAM+1))*(-1)**(LAM-knu-mnu)
                    j+=1
            jj+=1
        A[0,-1]=1
        rho[:,0]=0
        M=M.tocsr()
        A=A.tocsr()
#       Calculation on nu-body-order coefficient vectors
        MA=M*A
        MA=MA[abs(MA[:,:-1]).max(1).toarray().reshape(-1)>tol]
        rho=csr_matrix(rho)
        rho=hstack([rho,MA.T],format='csr')
        rho=rho[abs(rho[:,3*nu+5:]).max(1).toarray().reshape(-1)>tol].tolil()
        rho[abs(rho)<tol]=0
        idx=np.unique(rho[:-1,:nu*3+4].toarray(),axis=0,return_index=True)[1]
        rho[idx,0]=1
    return rho

# Compute 1-body-order features
start=time.time()
r1=Features(1,0,0,n_max,l_max,tol)
end=time.time()
print(end-start)
print(len(np.where(r1[:,0]==1)[0]))

# Compute 2-body-order features
start=time.time()
r=Features(2,r1,lil_matrix(r1),n_max,l_max,tol)
end=time.time()
print(end-start)
r2=r.toarray()
print(len(np.where(r2[:,0]==1)[0]))

# Compute 3-body-order features
start=time.time()
r=Features(3,r1,r,n_max,l_max,tol)
end=time.time()
print(end-start)
r3=r.toarray()
print(len(np.where(r3[:,0]==1)[0]))

# Compute 4-body-order features
start=time.time()
r=Features(4,r1,r,n_max,l_max,tol)
end=time.time()
print(end-start)
r4=r.toarray()
print(len(np.where(r4[:,0]==1)[0]))

# Compute 5-body-order features
start=time.time()
r=Features(5,r1,r,n_max,l_max,tol)
end=time.time()
print(end-start)
r5=r.toarray()
print(len(np.where(r5[:,0]==1)[0]))