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

import bo_helper
import helper_HF
from molecule import Molecule

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

def run(jkflag,scf_type,embmol,bset,bsetH,guess,func_h,func_l,exmodel,wfn,numpy_mem):
    
    numpy_memory=numpy_mem
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
    #get the 2eri tensor 
    # Run a quick check to make sure everything will fit into memory
    I_Size = (numbas**4) * 8.e-9
    print("\nSize of the ERI tensor would be %4.2f GB." % I_Size)

    # Estimate memory usage
    memory_footprint = I_Size * 1.5
    scf_type = psi4.core.get_global_option("scf_type")
    #CHECK
    print("using EX model: ......... %i\n" % exmodel)

    print("JK Class: %s" % jkflag)
    if jkflag:

      # Initialize the JK object
      if (scf_type=='DIRECT' or scf_type=='PK'): 
         print("using %s scf and JK class\n" % scf_type)
         jk = psi4.core.JK.build(bset)
      elif scf_type == 'MEM_DF':
         print("using %s\n" % scf_type)
         auxb = psi4.core.BasisSet.build(embmol,"DF_BASIS_SCF", "", fitrole="JKFIT",other=psi4.core.get_global_option("BASIS"))
         jk = psi4.core.JK.build_JK(bset,auxb)
      else:
           print(scf_type)
           raise Exception("Invalid scf_type.\n")
      jk.set_memory(int(4.0e9))  # 1GB
      jk.set_do_wK(False)
      jk.initialize()
      jk.print_header()
      
    else:
      if I_Size > numpy_memory:
          psi4.core.clean()
          raise Exception("Estimated memory utilization (%4.2f GB) " +\
                  "exceeds numpy_memory limit of %4.2f GB." % (memory_footprint, numpy_memory))
      #Get Eri (2-electron repulsion integrals)
      I = np.array(mints.ao_eri())
    # test: check the 2eri tensor for the high level part
    #Get Eri (2-electron repulsion integrals)
    #I = np.array(mints.ao_eri())
    #I11=I[:nbfA,:nbfA,:nbfA,:nbfA]
    #I22=I[nbfA:,nbfA:,nbfA:,nbfA:]


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

    # Calculate initial core guess on the original AB basis
    Hp = A.dot(Hcore).dot(A)           
    e, C2 = np.linalg.eigh(Hp)     
    C = A.dot(C2)                  
    Cocc = C[:, :ndocc]
    D=np.matmul(Cocc,Cocc.T)
    # Calculate initial core guess on the BO basis
    # Orthogonalizer B = Sbo^(-1/2) using Psi4's matrix power.
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
    diis = helper_HF.DIIS_helper(max_vec=6)



    #Htilde is expressed in BO basis
    Htilde=np.matmul(U.T,np.matmul(Hcore,U))
    if guess == 'SAD': 
       #in AO basis
       Da=SAD.Da()
       Cocc = psi4.core.Matrix(numbas, ndocc)
       Cocc.np[:] = (SAD.Ca()).np[:,:ndocc]
       try:
          u=np.linalg.inv(U)
       except np.linalg.LinAlgError:
          print("Error in linalg.inv")
       Dtilde=np.matmul(u,np.matmul(Da,u.T))
       Dtilde=psi4.core.Matrix.from_array(Dtilde)
       Cocc=np.matmul(u,Cocc)
    #elif args.guess == 'CORE':
       #core guess: inappropriate in most occasions 
       #C, Cocc, Dtilde = build_orbitals(psi4.core.Matrix.from_array(Htilde))
    elif guess == 'GS':
       print("using as guess the density from the low level theory hamiltonian")
       try:
          u=np.linalg.inv(U)
       except np.linalg.LinAlgError:
          print("Error in linalg.inv")
       Dtilde=np.matmul(u,np.matmul(np.asarray(wfn.Da()),u.T))
       Dtilde=psi4.core.Matrix.from_array(Dtilde)
       Cocc=np.matmul(u,np.asarray(wfn.Ca_subset('AO','OCC')))
    else:
       raise Exception("Invalid guess type.\n")


    Stilde=psi4.core.Matrix.from_array(Stilde)
    # Set defaults
    maxiter = 80
    E_conv = 1.0E-8
    D_conv = 1.0E-8

    E = 0.0
    Enuc = embmol.nuclear_repulsion_energy()
    Eold = 0.0
    Dold = np.zeros_like(D)

    #E_1el = np.einsum('pq,pq->', H + H, D) + Enuc
    #Ebo_1el = np.einsum('pq,pq->', Htilde + Htilde, Dtilde) + Enuc
    #print('One-electron energy calculated in AO and BO : %4.16f, %4.16f' % (E_1el,Ebo_1el))
    #print(np.allclose(E_1el,Ebo_1el))


    print('\nStart SCF iterations:\n\n')
    t = time.time()

    for SCF_ITER in range(1, maxiter + 1):

        #Eh,Exc,Ftilde=bo_helper.test_Fock(Dtilde, Htilde, I,U,'blyp', bset)
        if jkflag:
            Eh,Exclow,ExcAAhigh,ExcAAlow,Ftilde=bo_helper.get_BOFock_JK(Dtilde,Cocc,Htilde,jk,U,func_h,func_l,bset,bsetH,nbfA,exmodel)
        else:
            Eh,Exclow,ExcAAhigh,ExcAAlow,Ftilde=bo_helper.get_BOFock(Dtilde,Htilde,I,U,func_h,func_l,bset,bsetH,exmodel)
        #print(np.allclose(Ftilde,dFtilde))
        
        # DIIS error build and update
        diis_e = psi4.core.triplet(Ftilde, Dtilde, Stilde, False, False, False)
        diis_e.subtract(psi4.core.triplet(Stilde, Dtilde, Ftilde, False, False, False))
        diis_e = psi4.core.triplet(B, diis_e, B, False, False, False)

        diis.add(Ftilde, diis_e)

        # SCF energy and update
        
        #SCF_E = Eh + Exc + Enuc + 2.0*np.trace(np.matmul(Dtilde,Htilde))
        SCF_E = Eh + Exclow + ExcAAhigh -ExcAAlow + Enuc + 2.0*np.trace(np.matmul(Dtilde,Htilde))
        
        dRMS = diis_e.rms()

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
                eigvals,C=scipy.linalg.eigh(Ftilde.np,Stilde.np,eigvals_only=False)
        except scipy.linalg.LinAlgError:
                print("Error in scipy.linalg.eigh")
        Cocc=C[:,:ndocc]        
        Dtilde = psi4.core.doublet(psi4.core.Matrix.from_array(Cocc), psi4.core.Matrix.from_array(Cocc), False, True)

        if SCF_ITER == maxiter:
            psi4.clean()
            raise Exception("Maximum number of SCF cycles exceeded.\n")

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

    if jkflag:
        Eh,Exclow,ExcAAhigh,ExcAAlow,dummyF=bo_helper.get_BOFock_JK(psi4.core.Matrix.from_array(Dscf),Cocc_scf,Hcore,jk,dummy_U,func_h,func_l,bset,bsetH,nbfA,exmodel)
    else:
        Eh,Exclow,ExcAAhigh,ExcAAlow,dummyF=bo_helper.get_BOFock(psi4.core.Matrix.from_array(Dscf),Hcore,I,dummy_U,func_h,func_l,bset,bsetH,exmodel)
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
    #update the wfn object
    wfn.Da().copy( psi4.core.Matrix.from_array(Dscf) )
    wfn.Db().copy( psi4.core.Matrix.from_array(Dscf) )
    wfn.Ca().copy( psi4.core.Matrix.from_array(C_AO) )
    wfn.Cb().copy( psi4.core.Matrix.from_array(C_AO) )
    wfn.Fa().copy( psi4.core.Matrix.from_array(Fscf) )
    wfn.Fb().copy( psi4.core.Matrix.from_array(Fscf) )
    
    wfn_BObasis = {'Fock' : Ftilde, 'Dmtx' : Dtilde, 'Ccoeff' : C, 'Ovap' : Stilde, 'Umat' : U}
    
    return wfn, wfn_BObasis
