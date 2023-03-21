import os
import sys
import numpy as np
import scipy.linalg
import psi4
import time

sys.path.insert(0, "../common")
modpaths = os.environ.get('COMMON_PATH')

if modpaths is not None :
    for path in modpaths.split(";"):
        sys.path.append(path)

pyembpaths = os.environ.get('PYBERTHA_MOD_PATH')

if pyembpaths is not None :
    for path in pyembpaths.split(";"):
        sys.path.append(path)
#import bo_helper
import helper_HF
#from molecule import Molecule
import fde_utils
from pyembmod import GridDensityFactory
# Diagonalize routine
def build_orbitals(diag,O,nbasis,ndocc):
    Fp = psi4.core.triplet(O, diag, O, True, False, True)

    Cp = psi4.core.Matrix(nbasis,nbasis)
    eigvals = psi4.core.Vector(nbasis)
    Fp.diagonalize(Cp, eigvals, psi4.core.DiagonalizeOrder.Ascending)

    C = psi4.core.doublet(O, Cp, False, False)

    Cocc = psi4.core.Matrix(nbasis, ndocc)
    Cocc.np[:] = C.np[:, :ndocc]

    D = psi4.core.doublet(Cocc, Cocc, False, True)
    return C, Cocc, D

from dataclasses import dataclass


@dataclass
class bommedata:
    Cocc: np.ndarray
    Dtilde : np.ndarray
    ovapm : np.ndarray
    Htilde : np.ndarray
    func_h : str
    ndocc : int
    Umat : np.ndarray
    Bmat : np.ndarray
    Enuc : np.float64
    return_ene: bool
class scf_run():
    def __init__(self,data_scf,fockengine,bsetH,maxiter,E_conv,D_conv,embobj=None,opt=None):
       self.__fockfact = fockengine
       self.__data_scf = data_scf
       self.__bset_acc = bsetH
       self.__maxit = maxiter
       self.__maxfde = None
       self.__fde_thresh = None
       self.__e_conv = E_conv 
       self.__d_conv = D_conv
       self.__C = None # collect the final C mat
       self.__eigvals = None # collect eigs
       self.__pyemb = embobj
       if opt is not None:
         self.__maxfde = opt.maxit_fde
         self.__fde_thresh = opt.fde_thresh
    def __call__(self): 
       diis = helper_HF.DIIS_helper(max_vec=6)
         
       Eold = 0.
       func_h = self.__data_scf.func_h
       B = self.__data_scf.Bmat
       Cocc = self.__data_scf.Cocc
       Dtilde = self.__data_scf.Dtilde # can be derived from Cocc (BO basis)
       U = self.__data_scf.Umat
       fockengine = self.__fockfact
       bsetH = self.__bset_acc
       maxiter = self.__maxit 
       Stilde = self.__data_scf.ovapm 
       Htilde = self.__data_scf.Htilde
       Enuc = self.__data_scf.Enuc
       E_conv = self.__e_conv
       D_conv = self.__d_conv
       ndocc = self.__data_scf.ndocc

       for fde_iter in range(self.__maxfde+1): #maxfde specify the number of emb-potential updates
           Dmat_ext = np.matmul(Cocc,Cocc.T)
           for SCF_ITER in range(1, maxiter + 1):

               Eh, Exclow, ExcAAlow, ExcAAhigh, Ftilde=fockengine.get_bblock_Fock(Cocc=Cocc,func_acc=func_h,basis_acc=bsetH,U=U,return_ene=True)
               
               # DIIS error build and update
               diis_e = np.matmul(Ftilde, np.matmul(Dtilde, Stilde))
               diis_e -=np.matmul(Stilde, np.matmul(Dtilde, Ftilde))
               diis_e = np.matmul(B, np.matmul(diis_e, B))
           
               diis.add(psi4.core.Matrix.from_array(Ftilde), psi4.core.Matrix.from_array(diis_e) )
           
               # SCF energy and update
               
               #SCF_E = Eh + Exc + Enuc + 2.0*np.trace(np.matmul(Dtilde,Htilde))
               SCF_E = Eh + Exclow + ExcAAhigh -ExcAAlow + Enuc + 2.0*np.trace(np.matmul(Dtilde,Htilde))
               
               dRMS = np.sqrt(np.mean(diis_e**2))
           
               print('SCF Iteration %3d: Energy = %4.16f   dE = % 1.5E   dRMS = %1.5E'
                     % (SCF_ITER, SCF_E, (SCF_E - Eold), dRMS))
               if (abs(SCF_E - Eold) < E_conv) and (dRMS < D_conv):
                   break
           
               Eold = SCF_E
               Dold = Dtilde
           
               Ftilde = psi4.core.Matrix.from_array(diis.extrapolate())
           
               # Diagonalize Fock matrix
               #C, Cocc, Dtilde = build_orbitals(dFtilde)
               
               ####################################################################################################################
               # solve the generalized eigenvalue in the block-orthogonalized basis (Stilde)
               try: 
                       eigvals,C=scipy.linalg.eigh(Ftilde.np,Stilde,eigvals_only=False)
               except scipy.linalg.LinAlgError:
                       print("Error in scipy.linalg.eigh")
               Cocc=C[:,:ndocc]        
               Dtilde = psi4.core.doublet(psi4.core.Matrix.from_array(Cocc), psi4.core.Matrix.from_array(Cocc), False, True)
           
               if SCF_ITER == maxiter:
                   psi4.clean()
                   raise Exception("Maximum number of SCF cycles exceeded.\n")
               # end scf inner loop
           #update the embedding potential if maxfde >0
           if self.__maxfde > 0:
              vemb_in = fockengine.current_vemb()
              # set the Cocc [from SCF loop] 
              Cocc_tmp = np.matmul(U,Cocc)

              rho = self.__pyemb.set_density(Cocc_tmp)
              Vemb = self.__pyemb.make_embpot(rho)
              Dmat_out = np.asarray(Dtilde) 
              diff_emb = vemb_in - Vemb
              diff_D = Dmat_out - Dmat_ext
              diff_D = np.matmul(U,np.matmul(diff_D,U.T))
              norm_D=np.linalg.norm(diff_D,'fro')
              norm_v=np.linalg.norm(diff_emb,'fro')
              if (norm_D <  self.__fde_thresh) and (norm_v <self.__fde_thresh):
                 print("norm_D : %.12f\n" % norm_D)
                 print("norm_v : %.12f\n" % norm_v)
                 break
              else:
                 #set the updated embedding potential and do an additional iteration
                 fockengine.set_vemb(Vemb)
                 print("upgrading Vemb\n")
       self.__C = C
       self.__eigvals = eigvals
       return Cocc,Dtilde,Ftilde,SCF_E
    def C(self):
       return self.__C
    def epsilon(self):
       return self.__eigvals

def run(jkclass,embmol,bset,bsetH,guess,func_h,func_l,exmodel,wfn,pyembopt=None):
    
    numbas = bset.nbf()
    nbfA = bsetH.nbf()
    
    mints = psi4.core.MintsHelper(bset)
    S = np.array(mints.ao_overlap())

    #make U matrix
    U = np.eye(numbas)
    S11=S[:nbfA,:nbfA]
    S11_inv=np.linalg.inv(S11)
    S12 =S[:nbfA,nbfA:]
    P=np.matmul(S11_inv,S12)
    U[:nbfA,nbfA:]=-1.0*P

    #S block orthogonal
    Stilde= np.matmul(U.T,np.matmul(S,U))
    np.savetxt("ovap.txt",S)
    np.savetxt("ovapBO.txt", Stilde)

    #check S in the BO basis: the S^AA block should be the same as S (in AO basis)
    print("Overlap_AA in BO is Overlap_AA in AO: %s" %(np.allclose(Stilde[:nbfA,:nbfA],S[:nbfA,:nbfA])))
    #check the off diag block of Stilde
    mtest=np.zeros((nbfA,(numbas-nbfA)))
    print("Overlap_AB in BO is zero: %s" %(np.allclose(Stilde[:nbfA,nbfA:],mtest)))

    #CHECK
    print("using EX model: ......... %i\n" % exmodel)


    ndocc=wfn.nalpha()
    if wfn.nalpha() != wfn.nbeta():
            raise PsiException("Only valid for RHF wavefunctions!")

    print('\nNumber of occupied orbitals: %d\n' % ndocc)

    if guess== 'SAD':
       # Set SAD basis sets
       nbeta = wfn.nbeta()
       psi4.core.prepare_options_for_module("SCF")
       sad_basis_list = psi4.core.BasisSet.build(wfn.molecule(), "ORBITAL",
           psi4.core.get_global_option("BASIS"), puream=wfn.basisset().has_puream(),
                                            return_atomlist=True)

       sad_fitting_list = psi4.core.BasisSet.build(wfn.molecule(), "DF_BASIS_SAD",
           psi4.core.get_option("SCF", "DF_BASIS_SAD"), puream=wfn.basisset().has_puream(),
                                              return_atomlist=True)

       # Use Psi4 SADGuess object to build the SAD Guess
       SAD = psi4.core.SADGuess.build_SAD(wfn.basisset(), sad_basis_list)
       SAD.set_atomic_fit_bases(sad_fitting_list)
       SAD.compute_guess();

    V = np.asarray(mints.ao_potential())
    T = np.asarray(mints.ao_kinetic())
    # Build H_core: [Szabo:1996] Eqn. 3.153, pp. 141
    Hcore = T + V

    # Orthogonalizer A = S^(-1/2) using Psi4's matrix power.
    A = mints.ao_overlap()
    A.power(-0.5, 1.e-16)
    A = np.asarray(A)

    #initialise fock_factory
    from Fock_helper import fock_factory
    # for now exmodel=0 is assumed
    fock_help = fock_factory(jkclass,Hcore,Stilde,funcname=func_l,basisobj=bset,exmodel=exmodel)

    B = psi4.core.Matrix.from_array(Stilde)
    B.power(-0.5, 1.e-16)

    #adapted from 

    #"""
    #A restricted Hartree-Fock code using the Psi4 JK class for the 
    #4-index electron repulsion integrals.
    #
    #References:
    #- Algorithms from [Szabo:1996], [Sherrill:1998], and [Pulay:1980:393]
    #"""
    #
    #__authors__ = "Daniel G. A. Smith"
    #__credits__ = ["Daniel G. A. Smith"]
    #
    #__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
    #__license__ = "BSD-3-Clause"
    #__date__ = "2017-9-30"

    # no JK class used 

    # Build diis
    #diis = helper_HF.DIIS_helper(max_vec=6)



    #Htilde is expressed in BO basis
    Htilde=np.matmul(U.T,np.matmul(Hcore,U))
    
    try:
       u=np.linalg.inv(U)
    except np.linalg.LinAlgError:
       print("Error in linalg.inv")
    
    if guess == 'SAD': 
       #in AO basis
       Da=SAD.Da()
       Cocc = psi4.core.Matrix(numbas, ndocc)
       Cocc.np[:] = (SAD.Ca()).np[:,:ndocc]
       
       Dtilde=np.matmul(u,np.matmul(Da,u.T))
       Dtilde=psi4.core.Matrix.from_array(Dtilde)
       Cocc=np.matmul(u,Cocc)
    #elif args.guess == 'CORE':
       #core guess: inappropriate in most occasions 
       #C, Cocc, Dtilde = build_orbitals(psi4.core.Matrix.from_array(Htilde))
    elif guess == 'GS':
       print("using as guess the density from the low level theory hamiltonian")
       
       Dtilde=np.matmul(u,np.matmul(np.asarray(wfn.Da()),u.T))
       #Dtilde=psi4.core.Matrix.from_array(Dtilde)
       Cocc=np.matmul(u,np.asarray(wfn.Ca_subset('AO','OCC')))
    else:
       raise Exception("Invalid guess type.\n")


    # Set defaults
    maxiter = 80
    E_conv = 1.0E-8
    D_conv = 1.0E-8

    E = 0.0
    Enuc = embmol.nuclear_repulsion_energy()
    Eold = 0.0
    #Dold = np.zeros_like(Dtilde)

    #E_1el = np.einsum('pq,pq->', H + H, D) + Enuc
    #Ebo_1el = np.einsum('pq,pq->', Htilde + Htilde, Dtilde) + Enuc
    #print('One-electron energy calculated in AO and BO : %4.16f, %4.16f' % (E_1el,Ebo_1el))
    #print(np.allclose(E_1el,Ebo_1el))

    # Cocc is in BO basis
    if pyembopt is not None:
    # define a temporaty Cocc_AO
       Cocc_AO = np.matmul(U,Cocc)
    # initialize the embedding engine
       embed = fde_utils.emb_wrapper(embmol,pyembopt,bset)
       # here we need the active system density expressed on the grid
       rho = embed.set_density(Cocc_AO)
       nel_ACT =embed.rho_integral()
       print("N.el active system : %.8f\n" % nel_ACT)
       Vemb = embed.make_embpot(rho)

       # set the embedding potential
       fock_help.set_vemb(Vemb)
    else:
       embed = None

    # define the common quantities for bomme   
    data_scf = bommedata
    
    data_scf.Cocc   = Cocc
    data_scf.Dtilde = Dtilde
    data_scf.ovapm = Stilde
    data_scf.Htilde = Htilde
    data_scf.func_h = func_h
    data_scf.Umat = U
    data_scf.Bmat = B
    data_scf.return_ene = True
    data_scf.Enuc = Enuc
    data_scf.ndocc = ndocc
    
    Bo_scf = scf_run(data_scf,fock_help,bsetH,maxiter,E_conv,D_conv,embobj=embed,opt=pyembopt)
    print('\nStart SCF iterations:\n\n')
    t = time.time()
    Cocc,Dtilde,Ftilde,SCF_E = Bo_scf()
    eigvals = Bo_scf.epsilon()
    C = Bo_scf.C()
    """ 
    for SCF_ITER in range(1, maxiter + 1):

        Eh, Exclow, ExcAAlow, ExcAAhigh, Ftilde=fock_help.get_bblock_Fock(Cocc=Cocc,func_acc=func_h,basis_acc=bsetH,U=U,return_ene=True)
        
        # DIIS error build and update
        diis_e = np.matmul(Ftilde, np.matmul(Dtilde, Stilde))
        diis_e -=np.matmul(Stilde, np.matmul(Dtilde, Ftilde))
        diis_e = np.matmul(B, np.matmul(diis_e, B))

        diis.add(psi4.core.Matrix.from_array(Ftilde), psi4.core.Matrix.from_array(diis_e) )

        # SCF energy and update
        
        #SCF_E = Eh + Exc + Enuc + 2.0*np.trace(np.matmul(Dtilde,Htilde))
        SCF_E = Eh + Exclow + ExcAAhigh -ExcAAlow + Enuc + 2.0*np.trace(np.matmul(Dtilde,Htilde))
        
        dRMS = np.sqrt(np.mean(diis_e**2))

        print('SCF Iteration %3d: Energy = %4.16f   dE = % 1.5E   dRMS = %1.5E'
              % (SCF_ITER, SCF_E, (SCF_E - Eold), dRMS))
        if (abs(SCF_E - Eold) < E_conv) and (dRMS < D_conv):
            break

        Eold = SCF_E
        Dold = Dtilde

        Ftilde = psi4.core.Matrix.from_array(diis.extrapolate())

        # Diagonalize Fock matrix
        #C, Cocc, Dtilde = build_orbitals(dFtilde)
        
        ####################################################################################################################
        # solve the generalized eigenvalue in the block-orthogonalized basis (Stilde)
        try: 
                eigvals,C=scipy.linalg.eigh(Ftilde.np,Stilde,eigvals_only=False)
        except scipy.linalg.LinAlgError:
                print("Error in scipy.linalg.eigh")
        Cocc=C[:,:ndocc]        
        Dtilde = psi4.core.doublet(psi4.core.Matrix.from_array(Cocc), psi4.core.Matrix.from_array(Cocc), False, True)

        if SCF_ITER == maxiter:
            psi4.clean()
            raise Exception("Maximum number of SCF cycles exceeded.\n")
    """
    print('Total time for SCF iterations: %.3f seconds \n\n' % (time.time() - t))

    print('Final scf BO energy: %.8f hartree\n' % SCF_E)


    print('Orbital (BO) Energies [Eh]\n')
    print('Doubly Occupied:\n')
    for k in range(ndocc):
        print('%iA : %.6f' %(k+1,eigvals[k]))
    print('Virtual:\n')

    for k in range(ndocc,numbas):
        print('%iA : %.6f'% (k+1,eigvals[k]))

    dipole=mints.ao_dipole()

    #density corrected energy based on density transformed to AO
    dummy_U=np.eye(U.shape[0])
    Dscf=np.matmul(U,np.matmul(Dtilde,U.T))
    Fscf=np.matmul(U.T,np.matmul(Ftilde,U))
    Cocc_scf=np.matmul(U,Cocc)

    Eh, Exclow, ExcAAlow, ExcAAhigh, dummy=fock_help.get_bblock_Fock(Cocc=Cocc_scf,func_acc=func_h,basis_acc=bsetH,U=dummy_U,return_ene=True)
    
    SCF_E = Eh + Exclow + ExcAAhigh -ExcAAlow + Enuc + 2.0*np.trace(np.matmul(Dscf,Hcore))
    print('Final SCF (Density Corrected) energy: %.8f hartree\n' % SCF_E)

    mux=np.trace(np.matmul(dipole[0],2.0*Dscf))
    muy=np.trace(np.matmul(dipole[1],2.0*Dscf))
    muz=np.trace(np.matmul(dipole[2],2.0*Dscf))
    print()

    print('electric dipole moment in ea0\n')
    print('x : %.8f, y : %.8f, z : %.8f\n' % (mux,muy,muz))
    #psi4.compare_values(-132.9157890983754555, SCF_E, 8, 'SCF Energy')

    #print some orbitals

    #In our notation C are the orbitals in BO basis
    C_AO=np.matmul(U,C)

    print('Compare wfn.Da() and Dscf. SUCCESS: %s\n' % np.allclose(np.array(wfn.Da()),Dscf))
    diff= Dscf-np.array(wfn.Da())
    print('Max of the abs density difference : %.12e\n' % np.max(np.abs(diff)))
    #np.savetxt("dmat_BO.txt",Dtilde) #the density matrix in BO basis
    ###
    
    wfn_BObasis = {'Fock' : Ftilde, 'Hcore': Htilde, 'epsilon_a' : eigvals, 'energy': SCF_E, 'Dmtx' : Dtilde, 'Ccoeff' : C, 'Ovap' : Stilde, 'Umat' : U,\
                    'nbf_A': nbfA, 'nbf_tot' : numbas, 'ndocc' : ndocc,'jkfactory' : jkclass,\
                    'func_h': func_h, 'func_l' : func_l, 'exmodel':exmodel, 'molecule' : embmol }
    
    return wfn, wfn_BObasis
