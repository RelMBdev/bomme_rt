#Real-time TDDFT using Block-Orthogonalized Manby-Miller Embedding Theory
#written by Matteo De Santis (see J. Chem. Theory Comput. 2017, 13, 4173-4178)
#bwmcc19
import os
import sys
import argparse
sys.path.append(os.environ['PSI4_LIBS'])
sys.path.append(os.environ['PSI4_BOMME_RLMO'])

import time
import scipy.linalg
import psi4
import util
import numpy as np
import helper_HF
import argparse
from util import Molecule
from localizer import Localizer

parser = argparse.ArgumentParser()

parser.add_argument("-gA","--geomA", help="Specify geometry file for the subsystem A", required=True, 
        type=str, default="XYZ")
parser.add_argument("-gB","--geomB", help="Specify geometry file for the subsystem B", required=True, 
        type=str, default="XYZ")
parser.add_argument("-d", "--debug", help="Debug on, prints debug info to err.txt", required=False,
        default=False, action="store_true")
parser.add_argument("--locbasis", help="Use a basis for the propagation obtained from the localization of GS orbitals.", required=False,
        default=False, action="store_true")

parser.add_argument("-a", "--axis", help="The axis of  electric field direction (x = 0, y = 1, z = 2, default 2)",
        default=2, type = int)
parser.add_argument("-o1","--obs1", help="Specify the orbital basis set for subsys A (default 6-31G*)", required=False, 
        type=str, default="6-31G*")
parser.add_argument("-o2","--obs2", help="Specify the orbital basis set for subsys B (default 6-31G*)", required=False, 
        type=str, default="6-31G*")
parser.add_argument("-f2","--func2", help="Specify the low level theory functional", required=False, 
        type=str, default="blyp")
parser.add_argument("-f1","--func1", help="Specify the high level theory functional", required=False, 
        type=str, default="blyp")
parser.add_argument("-exA", "--exciteA", help="Only the A subsystem is excited : set model (mod 1, mod 2, default 0)", required=False,
        default=0, type = int)

parser.add_argument("--guess", help="Specify the initial guess for MME SCF", required=False, 
        type=str, default="SAD")
parser.add_argument("-m", "--numpy_mem", help="Set the memeory for the PSI4 driver (default 2 Gib)", required=False,
        default=2, type = int)
parser.add_argument("--ftden", help="On the fly FT of the TD density matrix", required=False,
        default=False, action="store_true" )
parser.add_argument("--npad", help="Number of padding points for the density matrix FT (the total number of sampling points has to be a power of 2)",
        default=0, type = int)
parser.add_argument("--freq", help="Specify the preset frquencies (FTDen) in a.u (n=>2)",
        default="0;0", type=str)
parser.add_argument("--select", help="Specify the occ-virt MO weighted dipole moment. (-2; occ_list & virt_list).\
         Occ/virt_list is a list of numbers separated by semicolon. To include all the virtual mainfold set virt_list=-99 (default: 0; 0 & 0)",
        default="0; 0 & 0", type=str)

parser.add_argument("--selective_pert", help="Selective perturbation on", required=False,
        default=False, action="store_true")
parser.add_argument("-p", "--principal", help="Restrict weighted dipole moment to HOMO->LUMO", required=False,
        default=False, action="store_true")
parser.add_argument("--input-param-file", help="Add input parameters filename [default=\"input.inp\"]", 
            required=False, default="input.inp", type=str, dest='inputfname')
parser.add_argument("-z", "--charge", help="Charge of the core system",
        default=0, type = int)
args = parser.parse_args()

debug = args.debug

acc_bset = args.obs1
gen_bset = args.obs2

HL=args.principal
direction = args.axis

fgeomA = args.geomA
fgeomB = args.geomB

#internally defined
svwn5_func = {
    "name": "SVWN5",
    "x_functionals": {
        "LDA_X": {}
    },
    "c_functionals": {
        "LDA_C_VWN": {}
    }
}

if args.func2 == 'svwn5':
  func_l=svwn5_func
else :
  func_l=args.func2

if args.func1 == 'svwn5':
  func_h=svwn5_func
else :
  func_h=args.func1

print("High Level functional : %s\n" % func_h)
print("Low Level functional : %s\n" % func_l)
print("Low Level basis : %s\n" % gen_bset)
print("High Level basis : %s\n" % acc_bset)



molA = Molecule(fgeomA,label=True)
molA.set_charge(args.charge)
speclist = molA.labels()

molB = Molecule(fgeomB)

#molA has been augmented by molecule B geometry
molA.append(molB.geometry())

#append some options, we can also include total charge (q) and multiplicity (S) of the total system
molA.append("symmetry c1" + "\n" + "no_reorient" + "\n" + "no_com")
molA.display_xyz()

embmol=psi4.geometry(molA.geometry())
embmol.print_out()
psi4.core.IO.set_default_namespace("embmol")
def basisspec_psi4_yo__anonymous775(mol,role):
        mol.set_basis_all_atoms(gen_bset, role=role)
        for k in speclist:
          mol.set_basis_by_label(k, acc_bset,role=role)
        return {}



#the basis set object for the complex: a composite basis set for rt applications
psi4.qcdb.libmintsbasisset.basishorde['USERDEFINED'] = basisspec_psi4_yo__anonymous775

L=[6.0,6.0,6.0]
Dx=[0.15,0.15,0.15]

#memory
psi4.set_memory(int(2e9))

psi4.set_options({'basis': 'userdefined',
                  'puream': 'True',
                  'DF_SCF_GUESS': 'False',
                  'scf_type': 'direct',
                  'dft_radial_scheme' : 'becke',
                  #'dft_radial_points': 49,
                  'dft_spherical_points' : 434,
                  'cubeprop_tasks': ['orbitals'],
                  'cubeprop_orbitals': [1, 2, 3, 4,5,6,7,8,9,10],
                  'CUBIC_GRID_OVERAGE' : L,
                  'CUBEPROP_ISOCONTOUR_THRESHOLD' : 1.0,
                  'CUBIC_GRID_SPACING' : Dx,
                  'e_convergence': 1e-8,
                  'd_convergence': 1e-8})

rt_bomme_nthreads = int(os.getenv('OMP_NUM_THREADS', 1))
psi4.set_num_threads(rt_bomme_nthreads)
#eb, wfn = psi4.energy('scf', return_wfn=True)
mol_wfn = psi4.core.Wavefunction.build( \
                    embmol,psi4.core.get_global_option('basis'))
mol_wfn.basisset().print_detail_out()

ene,wfn=psi4.energy('scf',dft_functional=func_l ,return_wfn=True)

# mHigh is the portion of the system treated at the higher level

mHigh=psi4.geometry(molA.core_geom(molA.natom()) +"symmetry c1" +"\n" +"no_reorient" +"\n" +"no_com")

print()
print("centers of high level subsys: %i" % molA.natom())


bset=mol_wfn.basisset()
numshell=bset.nshell()
print("Number of shells of the total AB basis:  %i" % numshell)
numbas=bset.nbf()
print("Number of functions of the total AB basis:  %i" % numbas)
natoms=embmol.natom()
#the basis set object for the high level portion
bsetH=psi4.core.BasisSet.build(mHigh,'ORBITAL',acc_bset,puream=-1)
counter=bsetH.nbf() #counter is the number of basis funcs of subsys A
#bsetH.print_detail_out()


print("functions in subsys1: %i" % counter)
mints = psi4.core.MintsHelper(bset)
S = np.array(mints.ao_overlap())

#make U matrix
U = np.eye(numbas)
S11=S[:counter,:counter]
S11_inv=np.linalg.inv(S11)
S12 =S[:counter,counter:]
P=np.matmul(S11_inv,S12)
U[:counter,counter:]=-1.0*P

#S block orthogonal
Stilde= np.matmul(U.T,np.matmul(S,U))
np.savetxt("ovap.txt",S)
np.savetxt("ovapBO.txt", Stilde)

#check S in the BO basis: the S^AA block should be the same as S (in AO basis)
print("Overlap_AA in BO is Overlap_AA in AO: %s" %(np.allclose(Stilde[:counter,:counter],S[:counter,:counter])))
#check the off diag block of Stilde
mtest=np.zeros((counter,(numbas-counter)))
print("Overlap_AB in BO is zero: %s" %(np.allclose(Stilde[:counter,counter:],mtest)))

numpy_memory=args.numpy_mem

#get the 2eri tensor 
# Run a quick check to make sure everything will fit into memory
I_Size = (numbas**4) * 8.e-9
print("\nSize of the ERI tensor will be %4.2f GB." % I_Size)

# Estimate memory usage
memory_footprint = I_Size * 1.5
if I_Size > numpy_memory:
    psi4.core.clean()
    raise Exception("Estimated memory utilization (%4.2f GB) " +\
            "exceeds numpy_memory limit of %4.2f GB." % (memory_footprint, numpy_memory))
#Get Eri (2-electron repulsion integrals)
from datetime import datetime

 
now = datetime.now()

# dd/mm/YY H:M:S
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("Fetching eri tensor started at: %s" % dt_string)
I = np.array(mints.ao_eri())

now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("Fetching eri tensor finished at: %s" % dt_string)
# test: check the 2eri tensor for the high level part
#Get Eri (2-electron repulsion integrals)
#I = np.array(mints.ao_eri())
#I11=I[:counter,:counter,:counter,:counter]
#I22=I[counter:,counter:,counter:,counter:]

#the low level theory G[D] matrix (J+ Vxc) don't require the K matrix, so we use the JK object
# Initialize the JK object
#jk = psi4.core.JK.build(bset)
#jk.set_do_K(False)
#jk.set_memory(int(1.25e8))  # 1GB
#jk.initialize()
#jk.print_header()

ndocc=mol_wfn.nalpha()
if mol_wfn.nalpha() != mol_wfn.nbeta():
        raise PsiException("Only valid for RHF wavefunctions!")

print('\nNumber of occupied orbitals: %d\n' % ndocc)

V = np.asarray(mints.ao_potential())
T = np.asarray(mints.ao_kinetic())
# Build H_core: [Szabo:1996] Eqn. 3.153, pp. 141
H = T + V

# Orthogonalizer A = S^(-1/2) using Psi4's matrix power.
A = mints.ao_overlap()
A.power(-0.5, 1.e-16)
A = np.asarray(A)

# Calculate initial core guess on the original AB basis
Hp = A.dot(H).dot(A)           
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


# Diagonalize routine
def build_orbitals(diag):
    Fp = psi4.core.triplet(B, diag, B, True, False, True)

    Cp = psi4.core.Matrix(numbas,numbas)
    eigvals = psi4.core.Vector(numbas)
    Fp.diagonalize(Cp, eigvals, psi4.core.DiagonalizeOrder.Ascending)

    C = psi4.core.doublet(B, Cp, False, False)

    Cocc = psi4.core.Matrix(numbas, ndocc)
    Cocc.np[:] = C.np[:, :ndocc]

    D = psi4.core.doublet(Cocc, Cocc, False, True)
    return eigvals, C, Cocc, D

# Set SAD prerequisites
if args.guess == 'SAD':
   nbeta = mol_wfn.nbeta()
   psi4.core.prepare_options_for_module("SCF")
   sad_basis_list = psi4.core.BasisSet.build(mol_wfn.molecule(), "ORBITAL",
       psi4.core.get_global_option("BASIS"), puream=mol_wfn.basisset().has_puream(),
                                        return_atomlist=True)

   sad_fitting_list = psi4.core.BasisSet.build(mol_wfn.molecule(), "DF_BASIS_SAD",
       psi4.core.get_option("SCF", "DF_BASIS_SAD"), puream=mol_wfn.basisset().has_puream(),
                                        return_atomlist=True)

   # Use Psi4 SADGuess object to build the SAD Guess
   SAD = psi4.core.SADGuess.build_SAD(mol_wfn.basisset(), sad_basis_list)
   SAD.set_atomic_fit_bases(sad_fitting_list)
   SAD.compute_guess();

#Htilde is expressed in BO basis
Htilde=np.matmul(U.T,np.matmul(H,U))

if args.guess=='SAD': 
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

elif args.guess == 'CORE':
   # CORE GUESS
   eigvals, C, Cocc, Dtilde = build_orbitals(psi4.core.Matrix.from_array(Htilde))
   #print("Intial density (diagonalizing Htilde) calculated in BO basis (transformed back to original AO) is the same as D: %s" %  np.allclose(np.matmul(U,np.matmul(Dtilde,U.T)),D,atol=1.0e-14))
   #C, Cocc, Dtilde = build_orbitals(psi4.core.Matrix.from_array(Htilde))
elif args.guess =='GS':
   print("using as guess the density from the low level theory hamiltonian")
   try:
      u=np.linalg.inv(U)
   except np.linalg.LinAlgError:
      print("Error in linalg.inv")
   Dtilde=np.matmul(u,np.matmul(np.asarray(wfn.Da()),u.T))
   Dtilde=psi4.core.Matrix.from_array(Dtilde)
   Cocc=np.matmul(u,np.asarray(wfn.Ca_subset('AO','OCC')))
else:
   print("Invalid choice\n")
import bo_helper

Stilde=psi4.core.Matrix.from_array(Stilde)
# Set defaults
maxiter = 80
E_conv = 1.0E-8
D_conv = 1.0E-8

E = 0.0
Enuc = embmol.nuclear_repulsion_energy()
Eold = 0.0
Dold = np.zeros_like(D)

E_1el = np.einsum('pq,pq->', H + H, D) + Enuc
Ebo_1el = np.einsum('pq,pq->', Htilde + Htilde, Dtilde) + Enuc
print('One-electron energy calculated in AO and BO : %4.16f, %4.16f' % (E_1el,Ebo_1el))
print(np.allclose(E_1el,Ebo_1el))


print('\nStart SCF iterations:\n\n')
t = time.time()

for SCF_ITER in range(1, maxiter + 1):

    #Eh,Exc,Ftilde=bo_helper.test_Fock(Dtilde, Htilde, I,U,'blyp', bset)
    Eh,Exclow,ExcAAhigh,ExcAAlow,Ftilde=bo_helper.get_BOFock(Dtilde,Htilde,I,U,func_h,func_l,bset,bsetH)
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
    eigvals, C, Cocc, Dtilde = build_orbitals(Ftilde)
    
    ####################################################################################################################
    # solve the generalized eigenvalue in the block-orthogonalized basis (Stilde)
    #try: 
    #        eigvals,C=scipy.linalg.eigh(Ftilde.np,Stilde.np,eigvals_only=False)
    #except scipy.linalg.LinAlgError:
    #        print("Error in scipy.linalg.eigh")
    #Cocc=C[:,:ndocc]        
    #Dtilde = psi4.core.doublet(psi4.core.Matrix.from_array(Cocc), psi4.core.Matrix.from_array(Cocc), False, True)

    if SCF_ITER == maxiter:
        psi4.core.clean()
        raise Exception("Maximum number of SCF cycles exceeded.\n")

print('Total time for SCF iterations: %.3f seconds \n\n' % (time.time() - t))

print('Final SCF energy: %.8f hartree\n' % SCF_E)
print('Orbital Energies [Eh]\n')
print('Doubly Occupied:\n')
for k in range(ndocc):
    print('%iA : %.6f' %(k+1,np.asarray(eigvals)[k]))
print('Virtual:\n')

for k in range(ndocc,numbas):
    print('%iA : %.6f'% (k+1,np.asarray(eigvals)[k]))

dipole=mints.ao_dipole()

Dscf=np.matmul(U,np.matmul(Dtilde,U.T))
mux=np.trace(np.matmul(dipole[0],2.0*Dscf))
muy=np.trace(np.matmul(dipole[1],2.0*Dscf))
muz=np.trace(np.matmul(dipole[2],2.0*Dscf))
print()
print('electric dipole moment in ea0\n')
print('x : %.8f, y : %.8f, z : %.8f\n' % (mux,muy,muz))
#psi4.compare_values(-132.9157890983754555, SCF_E, 8, 'SCF Energy')

#print some orbitals
#orblist=[0,1,2,3,4,5,6,7,8,9]

import cube_util
#C -> C^AO, in our notation C are the orbitals in BO basis
CAO=np.matmul(U,C)
np.savetxt("vct.txt", CAO)
#cube_util.orbtocube(embmol,L,Dx,CAO,orblist,bset,tag="PsiA",path="./")

###################################################################
#a minimalistic RT propagation code
molist = args.select.split("&")
occlist = molist[0].split(";")
occlist = [int(m) for m in occlist]
do_weighted = occlist.pop(0)
virtlist = molist[1].split(";")
virtlist = [int(m) for m in virtlist]

#for TNO analysis
#tlist = args.tlist.split("&")
#mlist = tlist[0].split(";")
#mlist = [int(m) for m in mlist]
#plist = tlist[1].split(";")
#plist = [int(m) for m in plist]

if (do_weighted == -2):
  if debug:
    print("Selected transitions from %s to %s MOs"% (str(occlist), str(virtlist)))

# frequency bin container
freqlist = args.freq.split(";")
freqlist = [np.float_(m) for m in freqlist]
if ( (freqlist == np.zeros(len(freqlist))).all() and args.ftden ):
         raise Exception("Check list of frequencies for ftden")
imp_opts, calc_params = util.set_params(args.inputfname)

if imp_opts['imp_type'] == 'analytic' :
    analytic = True
else:
    analytic = False

#dt in a.u
dt =  calc_params['delta_t']
#time_int in atomic unit
time_int=calc_params['time_int']
niter=int(time_int/dt)

if args.ftden:
  # define the number of sampling points
  Ns = niter+1
  if args.npad:
    Ns = Ns + args.npad
  Tperiod = (Ns)*dt
  print("FT sampling point : %i\n" % Ns)
  
  dw = 2.0*np.pi/Tperiod
  print("T period and dw : %12.3f ; %.8f\n" % (Tperiod,dw))
  #from freqlist derive the binlist
  binlist = []
  for k in freqlist:
   tmp = np.rint(k/dw)
   binlist.append(tmp)
  print("FT binlist ... :\n")
  print(binlist) 

print('Compare wfn.Da() and Dscf. SUCCESS: %s\n' % np.allclose(np.array(wfn.Da()),Dscf,atol=1.0e-10))
diff= Dscf-np.array(wfn.Da())
print('Max of the abs density difference : %.12e\n' % np.max(np.abs(diff)))

##DEBUG
#Htilde=H
#Ftilde=np.matmul(U.T,np.matmul(Ftilde,U))
#C=np.matmul(U,C)
#U=np.eye(numbas)
#%%%%
#Ftilde=np.array(wfn.Fa())
#Dtilde=np.array(wfn.Da())
#C=np.array(wfn.Ca())
#%%%%
#Dtilde=Dscf

#Stilde=S
Ftilde = np.asarray(Ftilde)
Dtilde = np.asarray(Dtilde)
C = np.asarray(C)
#dip_mat is transformed in the BO basis
dip_mat=np.matmul(U.T,np.matmul(np.array(dipole[direction]),U))

dip_dir = [0,1,2]
dip_dict={'0' : 'x', '1' : 'y', '2' : 'z'}
dip_dir.pop(direction)
dmat_offdiag = []

for i in dip_dir:
     dmat_offdiag.append(np.matmul(U.T,np.matmul(np.array(dipole[i]),U)))

test=np.matmul(C.T,np.matmul(Stilde,C))
print("in BO basis C^T Stilde C = 1 : %s" % np.allclose(test,np.eye(C.shape[0])))

if args.locbasis:
        localbas=Localizer(Dtilde,np.array(Stilde),counter)
        localbas.localize()
        #unsorted orbitals
        unsorted_orbs=localbas.make_orbitals()
        #the projector P
        Phat=np.matmul(Stilde.np[:,:counter],np.matmul(S11_inv,Stilde.np[:counter,:]))
        #The RLMO are ordered
        # by descending value of the locality parameter.
        sorted_orbs = localbas.sort_orbitals(Phat)
        #the occupation number and the locality measure
        locality,occnum=localbas.locality()
        #save the localization parameters and the occupation numbers  
        np.savetxt('locality_rlmo.out', np.c_[locality,occnum], fmt='%.12e')
        #check the occupation number of ndimA=counter MOs (of A).
        occA = int(np.rint( np.sum(occnum[:counter]) ))
        print("ndocc orbitals (A) after localization: %i\n" % occA)
        #set the sorted as new basis 
        C=sorted_orbs
        #use the density corresponding to (sorted) localized orbitals
        Dp_0 = np.diagflat(occnum)
else:
  Dp_0=np.zeros((numbas,numbas))
  for num in range(int(ndocc)):
      Dp_0[num,num]=1.0
try :
  C_inv=np.linalg.inv(C)
except scipy.linalg.LinAlgError:
  print("Error in np.linalg.inv")

# in an orthonormal basis Dtilde should be diagonal with oo=1, vv=0
ODtilde=np.matmul(C_inv,np.matmul(Dtilde,C_inv.T))
print("Dtilde is diagonal in the orbital basis: %s" % np.allclose(Dp_0,ODtilde,atol=1.0e-14))


#nuclear dipole for non-homonuclear molecules
Ndip= embmol.nuclear_dipole()
Ndip_dir=Ndip[direction]
#for the time being the nuclear dipole contribution to the dipole and energy
# is not considered
Ndip_dir = 0.0

Enuc_list=[]
print(niter)

#set the functional type
#if (calc_params['func_type'] == 'svwn5'):
#   func = svwn5_func
#else:
#   func=calc_params['func_type']

fo  = open("err.txt", "w")
print("analytic : %i" % analytic)
if (analytic):
   print('Perturb density with analytic delta')
   # set the perturbed density -> exp(-ikP)D_0exp(+ikP)
   k = imp_opts['Fmax']
   if   args.exciteA == 1:
     print("Excite only the A subsystem (High level)\n")
     #excite only A to prevent leak of the hole/particle density across the AA-BB  boundary
     #select AA block in BO basis
     tmpA=np.zeros_like(dip_mat)    
     tmpA[:counter,:counter]=dip_mat[:counter,:counter]
     dip_mo=np.matmul(np.conjugate(C.T),np.matmul(tmpA,C))
   else: 
        #dip_mat is transformed to the reference MO basis
        print("Dipole matrix is transformed to the MO basis\n")
        print("Local basis: %s\n" % args.locbasis)
        dip_mo=np.matmul(np.conjugate(C.T),np.matmul(dip_mat,C))
   if args.locbasis and (virtlist[0] == -99) and args.selective_pert:
        #use occnum to define a virtlist
        virtlist=[]
        for m in range(numbas):
          if np.rint(np.abs(occnum))[m] < 1.0: 
            virtlist.append(m+1)
        dip_mo=util.dipole_selection(dip_mo,-1,ndocc,occlist,virtlist,fo,debug)
   elif args.selective_pert:
         dip_mo=util.dipole_selection(dip_mo,virtlist[0],ndocc,occlist,virtlist,fo,debug)
       
   u0=util.exp_opmat(dip_mo,np.float_(-k))
   Dp_init= np.matmul(u0,np.matmul(Dp_0,np.conjugate(u0.T)))
   func_t0=k
   #backtrasform Dp_init
   D_init=np.matmul(C,np.matmul(Dp_init,np.conjugate(C.T)))
   Dtilde = D_init
   Dp_0 = Dp_init 
   
   #J0p,Exc0p,F_t0=util.get_Fock(D_ti,H,I,func,basisset)
   #if (func == 'hf'):                                  
   #    testene = np.trace(np.matmul(D_init,(H+F_t0)))  
   #else:                                               
   #    testene = 2.00*np.trace(np.matmul(D_init,H))+J0p+Exc0p
   #print('trace D(0+): %.8f' % np.trace(Dp_init).real)       
   #print(testene+Nuc_rep)                                    

#containers
ene_list = []
dip_list = []
dip_offdiag0 = []
dip_offdiag1 = []
imp_list=[]
weighted_dip = []
time_list = []
td_container = []
dip_list_p = []
#for molecules with permanent nuclear dipole add Enuc_list[k] to ene
#note that : i.e in the case of linear symmetric molecule the nuclear dipole can be zero depending
#on the choice of reference coordinates system

dipmo_mat=np.matmul(np.conjugate(C.T),np.matmul(dip_mat,C))
#C_inv used to backtransform D(AO)

C_inv=np.linalg.inv(C)
now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("RT-propagation started at: %s" % dt_string)
print('Entering in the first step of propagation')
start =time.time()
cstart =time.process_time()
Eh0,Exclow0,ExcAAhigh0,ExcAAlow0,func_t0,F_t0,fock_mid_init=util.mo_fock_mid_forwd_eval(Dtilde,Ftilde,\
                        0,np.float_(dt),Htilde,I,U,dip_mat,C,C_inv,Stilde,numbas,imp_opts,func_h,func_l,fo,bset,bsetH,args.exciteA)

#DEBUG
#import bo_helper
#dum1,dum2,dum3,dum4,F_t0=bo_helper.get_BOFockRT(Dtilde, Htilde, I,U, func_h, func_l, bset,bsetH)
diff_midF=fock_mid_init-Ftilde

norm_diff=np.linalg.norm(diff_midF,'fro')
fo.write('||Fock_mid(%i+1/2)-Fock_mid(%i-1/2)||_fro : %.8f\n' % (0,0,norm_diff))

if args.ftden:
    #tdmat = np.matmul(U,np.matmul((Dtilde-Dtilde),np.conjugate(U.T)))
    tdmat = Dtilde-Dtilde
    for n in  binlist:
       td_container.append(tdmat * np.exp( -2.0j*np.pi*0*n/Ns))
#if args.dump:
#  os.mkdir("td.0000000")
#  xs,ys,zs,ws,N,O = cube_util.setgridcube(mol,L,D)
#  phi,lpos,nbas=cube_util.phi_builder(mol,xs,ys,zs,ws,basis_set)
#  cube_util.denstocube(phi,Da,S,ndocc,mol,"density",O,N,D)
#  os.rename("density.cube","td.0000000/density.cube")

#check the Fock
if debug :
     print('F_t0 is equal to GS Ftilde : %s' % np.allclose(Ftilde,F_t0,atol=1.0e-12))
     print('Max of  abs(F_t0 - GS Ftilde) : %.12e\n' % np.max(np.abs(F_t0-Ftilde)))
#check hermicity of fock_mid_init

Ah=np.conjugate(fock_mid_init.T)
fo.write('Fock_mid hermitian: %s\n' % np.allclose(fock_mid_init,Ah))

#propagate D_t0 -->D(t0+dt)
#
#fock_mid_init is transformed in the MO ref basis
fockp_mid_init=np.matmul(np.conjugate(C.T),np.matmul(fock_mid_init,C))

#u=scipy.linalg.expm(-1.j*fockp_mid_init*dt)
u=util.exp_opmat(fockp_mid_init,np.float_(dt))

temp=np.matmul(Dp_0,np.conjugate(u.T))

Dp_t1=np.matmul(u,temp)

#check u if unitary
test_u=np.matmul(u,np.conjugate(u.T))
fo.write('U is unitary :%s\n' % np.allclose(test_u,np.eye(u.shape[0])))

fock_mid_backwd=np.copy(fock_mid_init)

#backtrasform Dp_t1

D_t1=np.matmul(C,np.matmul(Dp_t1,np.conjugate(C.T)))

ene_list.append(Eh0 + Exclow0 + ExcAAhigh0 -ExcAAlow0 + Enuc + 2.0*np.trace(np.matmul(Dtilde,Htilde)) )

dip_list.append(np.trace(np.matmul(Dtilde,dip_mat)))
dip_offdiag0.append(np.trace(np.matmul(Dtilde,dmat_offdiag[0])))
dip_offdiag1.append(np.trace(np.matmul(Dtilde,dmat_offdiag[1])))
#weighted dipole
if (do_weighted == -2):
  if virtlist[0] ==-99 and ((not args.locbasis ) or (not args.selective_pert)) :
    virtlist=[]
    for m in range(ndocc,numbas):
        virtlist.append(m+1)
  res = util.dipoleanalysis(dipmo_mat,Dp_0,ndocc,occlist,virtlist,debug,HL)
  weighted_dip.append(res)

fock_mid_backwd=np.copy(fock_mid_init) #prepare the fock at the previous midpint
D_ti=D_t1
Dp_ti=Dp_t1
Enuc_list.append(-func_t0*Ndip_dir) #just in case of non-zero nuclear dipole
#
imp_list.append(func_t0)
if debug :  
  #trace of D_t1
  fo.write('%.8f\n' % np.trace(Dp_ti).real)
  fo.write('Trace of DS %.8f\n' % np.trace(np.matmul(Stilde,D_ti)).real)
  fo.write('Trace of SD.real %.14f\n' % np.trace(np.matmul(Stilde,D_ti.real)))
  fo.write('Trace of SD.imag %.14f\n' % np.trace(np.matmul(Stilde,D_ti.imag)))
  fo.write('Dipole %.8f %.15f\n' % (0.000, 2.00*dip_list[0].real))

for j in range(1,niter+1):


    Eh_i,Exclow_i,ExcAAhigh_i,ExcAAlow_i,func_ti,F_ti,fock_mid_tmp=util.mo_fock_mid_forwd_eval(np.copy(D_ti),fock_mid_backwd,\
                                            j,np.float_(dt),Htilde,I,U,dip_mat,C,C_inv,Stilde,numbas,imp_opts,func_h,func_l,fo,bset,bsetH,args.exciteA)

    diff_midF=fock_mid_tmp-fock_mid_backwd

    norm_diff=np.linalg.norm(diff_midF,'fro')
    fo.write('||Fock_mid(%i +1/2)- Fock_mid(%i-1/2)||_fro: %.8f\n' % (j,j,norm_diff))

    if args.ftden:
       #tdmat =np.matmul(U,np.matmul((D_ti-Dtilde),np.conjugate(U.T)))
       tdmat =D_ti-Dtilde

       count=0
       for n in  binlist:
          tmp=td_container[count]
          tmp+=tdmat * np.exp( -2.0j*np.pi*j*n/Ns)
          td_container[count]=tmp
          count+=1
#    if args.dump:
#        if ( ( j % args.oint ) == 0 ) :
#           path="td."+str(j).zfill(7)
#           os.mkdir(path)
#           cube_util.denstocube(phi,D_ti,S,ndocc,mol,"density",O,N,D)
#           os.rename("density.cube",path+"/density.cube")

    Ah=np.conjugate(fock_mid_tmp.T)
    fo.write('Fock_mid hermitian: %s\n' % np.allclose(fock_mid_tmp,Ah))
    #transform fock_mid_init in MO basis
    fockp_mid_tmp=np.matmul(np.conjugate(C.T),np.matmul(fock_mid_tmp,C))
    u=util.exp_opmat(np.copy(fockp_mid_tmp),np.float_(dt))
    #u=scipy.linalg.expm(-1.0j*fockp_mid_tmp*dt)
    #check u is unitary
    test_u=np.matmul(u,np.conjugate(u.T))
    if (not np.allclose(np.eye(u.shape[0]),test_u)):
        print('U is not unitary\n')
    
    #check the trace of density to evolve
    fo.write('tr of density to evolve: %.8f\n' % np.trace(Dp_ti).real)
    
    #evolve the density in orthonormal basis
    temp=np.matmul(Dp_ti,np.conjugate(u.T))
    Dp_ti_dt=np.matmul(u,temp)

    #backtransform Dp_ti_dt
    D_ti_dt=np.matmul(C,np.matmul(Dp_ti_dt,np.conjugate(C.T)))
    fo.write('%.8f\n' % np.trace(Dp_ti_dt).real)
    #dipole expectation for D_ti
    dip_list.append(np.trace(np.matmul(dip_mat,D_ti)))
    dip_offdiag0.append(np.trace(np.matmul(D_ti,dmat_offdiag[0])))
    dip_offdiag1.append(np.trace(np.matmul(D_ti,dmat_offdiag[1])))
    
    #for debug
    if debug:
      fo.write('Dipole  %.8f %.15f\n' % (j*dt, 2.00*dip_list[j].real))

    if (do_weighted == -2):
      #weighted dipole 
      res = util.dipoleanalysis(dipmo_mat,Dp_ti,ndocc,occlist,virtlist,debug,HL)
      weighted_dip.append(res)
    #Energy expectation value at t = t_i 
    ene_list.append(Eh_i + Exclow_i + ExcAAhigh_i -ExcAAlow_i + Enuc + 2.0*np.trace(np.matmul(D_ti,Htilde)) )
    Enuc_list.append(-func_ti*Ndip_dir) #just in case of non-zero nuclear dipole
    imp_list.append(func_ti)
   
    #update D_ti and Dp_ti for the next step
    
    if debug :
      fo.write('here I update the matrices Dp_ti and D_ti\n')
    D_ti=np.copy(D_ti_dt)
    Dp_ti=np.copy(Dp_ti_dt)
    #update fock_mid_backwd for the next step
    fock_mid_backwd=np.copy(fock_mid_tmp)

fo.close()
end=time.time()
cend = time.process_time()
ftime  = open("timing.txt", "w")
ftime.write("time for %i time iterations : (%.3f s, %.3f s)\n" %(niter+1,end-start,cend-cstart))
ftime.close()
now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("RT-propagation ended at: %s" % dt_string)
t_point=np.linspace(0.0,niter*dt,niter+1)
dip_t=2.00*np.array(dip_list).real
ene_t=np.array(ene_list).real
imp_t=np.array(imp_list)

if (do_weighted == -2):
  wd_dip=2.00*np.array(weighted_dip).real
  np.savetxt('weighteddip.txt', np.c_[t_point,wd_dip], fmt='%.12e')

np.savetxt('dipole-'+2*dip_dict[str(direction)]+'.txt', np.c_[t_point,dip_t], fmt='%.12e')
# dipole_ij, i denotes the ith component of dipole vector, j denote the direction of the field
np.savetxt('dipole-'+dip_dict[str(dip_dir[0])]+dip_dict[str(direction)]+'.txt', np.c_[t_point,2.00*np.array(dip_offdiag0).real], fmt='%.12e')
np.savetxt('dipole-'+dip_dict[str(dip_dir[1])]+dip_dict[str(direction)]+'.txt', np.c_[t_point,2.00*np.array(dip_offdiag1).real], fmt='%.12e')
np.savetxt('imp.txt', np.c_[t_point,imp_t], fmt='%.12e')
np.savetxt('ene.txt', np.c_[t_point,ene_t], fmt='%.12e')

#post-propagation stuff
print("printing the values of the frequency bins ..\n")
#effective values of the bin (in au)
freqlist = [val*dw for val in binlist]
print(freqlist)

for num in range(len(td_container)):
  tmp = td_container[num]
  tmp = np.matmul(U,np.matmul(tmp,np.conjugate(U.T)))
  td_container[num] = tmp

if args.ftden:	
  count = 0
  xs,ys,zs,ws,N,O = cube_util.setgridcube(embmol,L,Dx)   #here L and D are specified at the beginning and used for explicit dumping below
  phi,lpos,dum1=cube_util.phi_builder(embmol,xs,ys,zs,ws,bset)
  for w in freqlist:
      tmp = td_container[count].imag*Tperiod/Ns
      np.savetxt("FTD_AO."+str(w)+".imag"+".txt",tmp) #for post calc
      ###########################
      nx = N[0]+1
      ny = N[1]+1
      nz = N[2]+1
      
      ntot= nx*ny*nz
      rho = np.einsum('pm,mn,pn->p', phi, tmp, phi)   
      fo =open("FT_"+str(w)+".imag"+".cube","w")
      ### write cube header
      fo.write("\n")
      fo.write("\n")
      fo.write("     %i %.6f %.6f %.6f\n" % ( embmol.natom(),O[0],O[1],O[2]))
      fo.write("   %i  %.6f %.6f %.6f\n" % (nx, Dx[0],0.0,0.0))
      fo.write("   %i  %.6f %.6f %.6f\n" % (ny, 0.0,Dx[1],0.0))
      fo.write("   %i  %.6f %.6f %.6f\n" % (nz, 0.0,0.0,Dx[2]))
      for A in range(embmol.natom()):
         fo.write("  %i %.6f %.6f %.6f %.6f\n" %(embmol.charge(A), 0.0000, embmol.x(A),embmol.y(A),embmol.z(A)))
      for i in range(ntot):
         fo.write("%.5e\n" % (rho[i]))
      fo.close
      count+=1
