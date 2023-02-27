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
from Fock_helper import fock_factory
print("test mixed basis / functionals")
func_high = 'b3lyp'
func_low = 'b3lyp'
bset,bsetH, molelecule_str, psi4mol, wfn, jkobj = initialize(False,'DIRECT','3-21G','3-21G','H2O1.xyz',\
                   func_high,func_low,0,eri='nofit')

mints = psi4.core.MintsHelper(bset)
#I = np.array(mints.ao_eri())

H = np.array(mints.ao_kinetic())+ np.array(mints.ao_potential())
S = np.array(mints.ao_overlap())
numbas = bset.nbf()

nbfA = bsetH.nbf()

#make U matrix for blend basis(cc-pvdz+3-21G)
U = np.eye(numbas)
S11=S[:nbfA,:nbfA]
S11_inv=np.linalg.inv(S11)
S12 =S[:nbfA,nbfA:]
P=np.matmul(S11_inv,S12)
U[:nbfA,nbfA:]=-1.0*P

#S block orthogonal
Stilde= np.matmul(U.T,np.matmul(S,U))
#refresh orbital and fockbase
Cocc = np.array(wfn.Ca_subset('AO','OCC'))

try:
    U_inv = np.linalg.inv(U)
except np.linalg.LinAlgError:
    print("Error in numpy.linalg.inv of inputted matrix")

Cocc = np.matmul(U_inv,Cocc)
Dinput = np.matmul(Cocc,Cocc.T)
fockbase = fock_factory(jkobj,H,Stilde,funcname=func_low,basisobj=bset)
F_bblock = fockbase.get_bblock_Fock(Dmat=Dinput,func_acc=func_high,basis_acc=bsetH,U=U)

print("F(BO) dim: %i,%i\n" % (F_bblock.shape[0],F_bblock.shape[1]))
Test_H = np.matmul(U.T,np.matmul(wfn.Fa(),U))

test = np.allclose(F_bblock,Test_H,atol=1.0e-12)
print("test GGA/hybrid Fock Block-Orth.: Passed .... %s\n" % test)
