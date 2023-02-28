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
print("***initializing jk with fitted eri***\n")
bset,bsetH, molelecule_str, psi4mol, wfn, jkobj = initialize(False,'MEM_DF','cc-pvdz','3-21G','H2O1.xyz',\
                   'hf','svwn',0,eri='fit')

mints = psi4.core.MintsHelper(bset)



Cocc = np.array(wfn.Ca_subset('AO','OCC'))
Dmat = np.matmul(Cocc,Cocc.T)

nbfA = bsetH.nbf()
tmp = np.zeros_like(Cocc[nbfA:,:])

Cocc = np.array(wfn.Ca_subset('AO','OCC'))
Jmat_eri =jkobj.J(None,Dmat=Dmat,sum_idx=[[0,nbfA],[0,nbfA]],out_idx=[[0,nbfA],[0,nbfA]])
Kmat_eri =jkobj.K(None,Dmat=Dmat,sum_idx=[[0,nbfA],[0,nbfA]],out_idx=[[0,nbfA],[0,nbfA]])


print()
print("***initializing jk using native PSI4 class***\n")
bset,bsetH, molelecule_str, psi4mol, wfn, jkobj_eri = initialize(True,'MEM_DF','cc-pvdz','3-21G','H2O1.xyz',\
                   'hf','svwn',0,eri=None)
Cocc[nbfA:,:] = tmp
Jmat_nat =jkobj.J(Cocc,sum_idx=[[0,nbfA],[0,nbfA]],out_idx=[[0,nbfA],[0,nbfA]])
Kmat_nat =jkobj.K(Cocc,sum_idx=[[0,nbfA],[0,nbfA]],out_idx=[[0,nbfA],[0,nbfA]])
print("Check fitted Jmat [class,native] vs [class,erifit]: %s\n" % np.allclose(Jmat_nat,Jmat_eri))
print("Check fitted Kmat [class,native] vs [class,erifit]: %s\n" % np.allclose(Kmat_nat,Kmat_eri))
exit()

"""
"""
