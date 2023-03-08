import os
import sys
import numpy as np
sys.path.insert(0, "../common")
modpaths = os.environ.get('COMMON_PATH')

if modpaths is not None :
    for path in modpaths.split(";"):
        sys.path.append(path)

import  cube_util
from localizer import Localizer 

def regional_localize_MO(wfn,wfnBO,nbf_A=None,cubelist="",dx=[0.2,0.2,0.2],margin=[4.5,4.5,4.5],dumpdir='./'):
       
       if not isinstance(wfnBO,dict):
           ovapm = wfn.S()  
           Dmtx = wfn.Da()
           molecule = wfn.molecule()
           U = None
           basis = wfn.basisset()
       else:
           ovapm = wfnBO['Ovap'] 
           Dmtx = wfnBO['Dmtx']
           molecule = wfnBO['molecule']
           U = wfnBO['Umat']
           basis = wfnBO['jkfactory'].basisset()
       
       if not isinstance(nbf_A,int):
           nbf_A = wfnBO['nbf_A']

       try :
         SAA_inv=np.linalg.inv(ovapm[:nbf_A,:nbf_A])
       except scipy.linalg.LinAlgError:
         print("Error in np.linalg.inv")

       localbas=Localizer(Dmtx,np.array(ovapm),nbf_A)
       localbas.localize()
       #unsorted orbitals
       unsorted_orbs=localbas.make_orbitals()
       #the projector P
       Phat=np.matmul(ovapm[:,:nbf_A],np.matmul(SAA_inv,ovapm[:nbf_A,:]))
       #The RLMO are ordered
       # by descending value of the locality parameter.
       sorted_orbs = localbas.sort_orbitals(Phat)
       #the occupation number and the locality measure
       locality,occnum=localbas.locality()
       #save the localization parameters and the occupation numbers  
       np.savetxt('locality_rlmo.out', np.c_[locality,occnum], fmt='%.12e')
       #check the occupation number of ndimA=nbf_A MOs (of A).
       occA = int(np.rint( np.sum(occnum[:nbf_A]) ))
       print("ndocc orbitals (A) after localization: %i\n" % occA)
       #set the sorted as new basis 
       C=sorted_orbs
       if cubelist != "":
          molist = cubelist.split("&")
          occlist = molist[0].split(";")
          occlist = [int(m) for m in occlist]
          if occlist[0] < 0 :
              occlist.pop(0)
          virtlist = molist[1].split(";")
          virtlist = [int(m) for m in virtlist]
          if U is not None:
              C = np.matmul(U,C)
              
          cube_util.orbtocube(molecule,margin,dx,C,occlist,basis,tag="PsiA_occ",path=dumpdir)
          cube_util.orbtocube(molecule,margin,dx,C,virtlist,basis,tag="PsiA_virt",path=dumpdir)
