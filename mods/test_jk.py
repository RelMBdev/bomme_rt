import os
import sys
import psi4
import numpy as np

sys.path.insert(0, "./common")
modpaths = os.environ.get('MODS_PATH')

if modpaths is not None :
    for path in modpaths.split(";"):
        sys.path.append(path)
from scf_run import run
from init_run import initialize

bset,bsetH, molelecule_str, psi4mol, wfn, jkobj = initialize(False,'DIRECT','cc-pvdz','3-21G','H2O1.xyz',\
                   'hf','svwn',0,eri='nofit')

mints = psi4.core.MintsHelper(bset)

S = np.array(mints.ao_overlap())
numbas = bset.nbf()

I_Size = (numbas**4) * 8.e-9
nbfA = bsetH.nbf()

#make U matrix
U = np.eye(numbas)
S11=S[:nbfA,:nbfA]
S11_inv=np.linalg.inv(S11)
S12 =S[:nbfA,nbfA:]
P=np.matmul(S11_inv,S12)
U[:nbfA,nbfA:]=-1.0*P

#S block orthogonal
Stilde= np.matmul(U.T,np.matmul(S,U))
# get the ERI 4 index tensor
numpy_memory = 1

if I_Size > numpy_memory:
    psi4.core.clean()
    raise Exception("Estimated memory utilization (%4.2f GB) " +\
            "exceeds numpy_memory limit of %4.2f GB." % (memory_footprint, numpy_memory))
#Get Eri (2-electron repulsion integrals)
I = np.array(mints.ao_eri())

from Fock_helper import jkfactory

jknative =jkfactory(bset,psi4mol,jknative=True,eri=None)

jk_eri =jkfactory(bset,psi4mol,jknative=False,eri=I)

Cocc = np.array(wfn.Ca_subset('AO','OCC'))
Dmat = np.matmul(Cocc,Cocc.T)

nbfA = bsetH.nbf()
tmp = np.zeros_like(Cocc[nbfA:,:])
Cocc[nbfA:,:] = tmp
Jmat_nat = jknative.J(Cocc,out_idx=[[0,nbfA],[0,nbfA]])
print("J [native] dim : %i,%i" % Jmat_nat.shape)
#Kmat_nat = jknative.K(Cocc,out_idx=[[0,nbfA],[0,nbfA]])

Cocc = np.array(wfn.Ca_subset('AO','OCC'))
print("use eri tensor")
Jmat_eri =jk_eri.J(None,Dmat=Dmat,sum_idx=[[0,nbfA],[0,nbfA]],out_idx=[[0,nbfA],[0,nbfA]])



Jmat_referi=np.einsum('pqrs,rs->pq', I[:nbfA,:nbfA,:nbfA,:nbfA], Dmat[:nbfA,:nbfA])
print("Check J[class,native] against eri ref : %s" % np.allclose(Jmat_nat,Jmat_referi))
print()

print("J [class,eri] dim : %i,%i" % Jmat_eri.shape)
#Kmat_eri = jk_eri.K(Cocc,out_idx=[[0,nbfA],[0,nbfA]])

print("Check J[class,eri] against eri ref : %s" % np.allclose(Jmat_referi,Jmat_eri))
print()

print()
print()

print("test for K matrix")

tmp = np.zeros_like(Cocc[nbfA:,:])  # in order to recover the (nbfA,nbfA) block of the density matrix some element of complete Cocc matrix has to be set to zero
Cocc[nbfA:,:] = tmp

Kmat_nat = jknative.K(Cocc,out_idx=[[0,nbfA],[0,nbfA]])
print("K [native] dim : %i,%i" % Kmat_nat.shape)

Kmat_referi=np.einsum('prqs,rs->pq', I[:nbfA,:nbfA,:nbfA,:nbfA], Dmat[:nbfA,:nbfA])
print("Check K[class,native] against eri ref")
print(np.allclose(Kmat_referi,Kmat_nat))


Cocc = np.array(wfn.Ca_subset('AO','OCC'))

print("K class using eri tensor")
Kmat_eri =jk_eri.K(None,Dmat=Dmat,sum_idx=[[0,nbfA],[0,nbfA]],out_idx=[[0,nbfA],[0,nbfA]])
#Kmat_eri = jk_eri.K(Cocc,out_idx=[[0,nbfA],[0,nbfA]])

print("Check K[class,eri] against ref eri : %s" %np.allclose(Kmat_referi,Kmat_eri) )
print()
