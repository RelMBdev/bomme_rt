import os
import sys
import numpy as np
import scipy.linalg

sys.path.insert(0, "../common")
modpaths = os.environ.get('COMMON_PATH')

if modpaths is not None :
    for path in modpaths.split(";"):
        sys.path.append(path)
# localize the orbitals
import numpy as np
import localizer
from localizer import Localizer
import cube_util

def localize_orbs(locscheme, Dtile, OvapAO, nbfA, embmol = None, dump_cube=False, L=[4.5,4.5,4.5], Dx = [0.2,0.2,0.2] ):
   
    #make sure OvapAO and Dtilde are numpy object
    if not isinstace(Dtilde,np.ndarray):
       Dtilde = np.array(Dtilde)
    
    if not isinstace(Dtilde,np.ndarray):
       OvapAO = np.array(OvapAO)

    numbas = OvapAO.shape[0] 
    #make U matrix
    Umat = np.eye(numbas)
    S11=OvapAO[:nbfA,:nbfA]

    try:
       S11_inv=np.linalg.inv(S11)
    except np.linalg.LinAlgError:
       print("Error in linalg.inv")

    S12 =OvapAO[:nbfA,nbfA:]
    P=np.matmul(S11_inv,S12)
    Umat[:nbfA,nbfA:]=-1.0*P

    #S block orthogonal
    Stilde= np.matmul(U.T,np.matmul(OvapAO,U))

      
    if locscheme == 'rlmo':
        locorb=Localizer(Dtilde,Stilde,nbfA) #has two additional default arguments
        locorb.localize()
        DRo = locorb.DRo()
        C_RLMO=locorb.make_orbitals()
        locorb.dump_results()
    
    
        #Phat is the projector in the BO basis
        Phat=np.matmul(Stilde[:,:nbfA],np.matmul(S11_inv,Stilde[:nbfA,:]))
        
        #sort the regional localized mo according to decresing value of the localization parameter
        #rename from C_RLMO to Ctilde_loc (tilde stays for block-orthogonalized (tilde) basis)
        Ctilde_loc=locorb.sort_orbitals(Phat)
        #C <- C^AO, in our notation C are the orbitals in BO basis
        if dump_cube:
          tmp=np.matmul(U,Ctilde_loc)
          cube_util.orbtocube(embmol,L,Dx,tmp,occlist,bset,tag="PsiA_occ",path="./")
          cube_util.orbtocube(embmol,L,Dx,tmp,virtlist,bset,tag="PsiA_virt",path="./")
        locality,occnum = locorb.locality()
        #save the localization parameters and the occupation numbers  
        np.savetxt('locality_rlmo.out', np.c_[locality,occnum], fmt='%.12e')
     
        #sorting the occupied MO
     
        occnumA = occnum[:nbfA]
        mask = np.abs(locality[:nbfA]) > args.threshold
        occnumA = occnumA[mask]
        lenA = occnumA.shape[0]
     
        ndoccA = int(np.rint( np.sum(occnumA) ))
        ndoccB = ndocc - ndoccA
     
        print("Absolutely localized electrons in frag A: %.8f\n" % ndoccA)
     
        occnumB = occnum[lenA:]
        np.savetxt('occnum_fragA.out', occnumA, fmt='%.12e')
        np.savetxt('occnum_fragB.out', occnumB, fmt='%.12e')
        idoccA = occnumA.argsort()[::-1]
        idoccB = occnumB.argsort()[::-1]
     
        C_AA = (Ctilde_loc[:,:lenA])[:,idoccA]
        C_BB = (Ctilde_loc[:,lenA:])[:,idoccB]
     
        Cocc_AA = np.matmul(U,C_AA[:,:ndoccA])
        Cocc_BB = np.matmul(U,C_BB[:,:ndoccB])

        return Cocc_AA, Cocc_BB
    
    elif args.locscheme == 'spade':
        raise Exception("SPADE orbitals do not provide virtuals\n")
        from localizer import Spade
    
        loc=Spade(C[:,:ndocc],Stilde.np,S11_inv,nbfA)
        #for later use
        Phat=loc.get_projector()
        #the Cspade orbitals on the tilde base
        Ctilde_loc =loc.localize()
        sigmad = loc.get_sigmaval()


        test=np.matmul(np.conjugate(Ctilde_loc.T),np.matmul(Stilde,Ctilde_loc))
        print("Cspade orbitals satisfy the orthonormality condition: %s\n" % (np.allclose(test,np.eye(Stilde.shape[0]))))
       
        #measuring the "locality"
        locparam = loc.get_locality(Ctilde_loc)
        np.savetxt("SPADE_locality.txt", np.c_[locparam,sigmad], fmt='%.12e')

        #CAO matrix contains Cspade (occupied) orbitals on the standard AO basis
        CAO_can=np.matmul(U,Ctilde_loc)
        return CAO_can
