# sample demonstration of Localizer class (SPADE)
import numpy as np

sys.path.insert(0, "../")
modpaths = os.environ.get('COMMON_PATH')

if modpaths is not None :
    for path in modpaths.split(";"):
        sys.path.append(path)

from localizer import Localizer
from localizer import Spade
import argparse 
parser = argparse.ArgumentParser()
parser.add_argument("--cmat", help="Specify C matrix filename", required=False, 
        type=str, default="cmat")
parser.add_argument("--ovap", help="Specify geometry file overlap matrix (tilde basis)", required=True, 
        type=str, default="XYZ")

#parser.add_argument("--nbasis", help="The (default 2)",required=True,
#default=2, type = int)

args = parser.parse_args()

C_BO=np.loadtxt(args.cmat)
Stilde=np.loadtxt(args.ovap)

import psi4
mol=psi4.geometry("""
    O     1.568501    0.105892    0.000005
    H     0.606736   -0.033962   -0.000628
    H     1.940519   -0.780005    0.000222
    symmetry c1
    noreorient  
    no_com
""")
psi4.set_options({'basis': 'sto-3g',
                  'puream': 'True',
                  'DF_SCF_GUESS': 'False',
                  'scf_type': 'direct',
                  'dft_radial_scheme' : 'becke',
                  #'dft_radial_points': 49,
                  'dft_spherical_points' : 434,
                  'cubeprop_tasks': ['orbitals'],
                  'cubeprop_orbitals': [1, 2, 3, 4,5,6,7,8,9,10],
                  'CUBIC_GRID_OVERAGE' : [4.5,4.5,4.5],
                  'CUBEPROP_ISOCONTOUR_THRESHOLD' : 1.0,
                  'CUBIC_GRID_SPACING' :  [0.1,0.1,0.1],
                  'e_convergence': 1e-8,
                  'd_convergence': 1e-8})



mol_wfn = psi4.core.Wavefunction.build( \
                    mol,psi4.core.get_global_option('basis'))
#re-define
nbasA = mol_wfn.basisset().nbf()
ndoccA=mol_wfn.nalpha()
print("check ndocc of system A: ................ %i\n" % ndoccA)
print("Nbf  system A: %i\n" % nbasA)
psi4.core.clean()


totmol=psi4.geometry("""
    O     1.568501    0.105892    0.000005
    H     0.606736   -0.033962   -0.000628
    H     1.940519   -0.780005    0.000222
    N    -1.395591   -0.021564    0.000037
    H    -1.629811    0.961096   -0.106224
    H    -1.862767   -0.512544   -0.755974
    H    -1.833547   -0.330770    0.862307

    symmetry c1
    noreorient  
    no_com
""")

tot_wfn = psi4.core.Wavefunction.build( \
                    totmol,psi4.core.get_global_option('basis'))
ndocc=tot_wfn.nalpha()

print("check ndocc of total system: ................ %i\n" % ndocc)
Cocc=C_BO[:,:ndocc]
check=np.matmul(Cocc,Cocc.T)
print("n. el : %.8f" % (np.trace(np.matmul(check,Stilde))))

try:
  Saa_inv = np.linalg.inv(Stilde[:nbasA,:nbasA])
except np.linalg.LinAlgError:
  print("Error in numpy.linalg.svd")

#read the overlap matrix of standard AO basis  
ovapAO=np.loadtxt("ovap.txt")
nbas = tot_wfn.basisset().nbf()

#make U matrix
U = np.eye(nbas)
S12 =ovapAO[:nbasA,nbasA:]
P=np.matmul(Saa_inv,S12)
U[:nbasA,nbasA:]=-1.0*P


loc=Spade(C_BO,Stilde,Saa_inv,nbasA)


cspade =loc.localize()

#measure the "locality"
locparam = loc.get_locality()
np.savetxt("SPADE_locty.txt",locparam)

test=np.matmul(np.conjugate(cspade.T),np.matmul(Stilde,cspade))
print(np.allclose(test,np.eye(Stilde.shape[0])))

CspadeAO=np.matmul(U,cspade)
from cube_util import orbtocube
basis = psi4.core.BasisSet.build(totmol, 'ORBITAL',psi4.core.get_global_option('basis'))
orbtocube(totmol,[4.5,4.5,4.5],[0.1,0.1,0.1],CspadeAO,[1,2,3,4,5,6,7,8,9,10],basis,tag="PsiA_spade")
#TODO
#check if SPADE orbitals are canonical
