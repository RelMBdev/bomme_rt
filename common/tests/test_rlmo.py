# sample demonstration of Localizer class (RLMO)
import numpy as np
import sys

sys.path.insert(0, "../")
modpaths = os.environ.get('COMMON_PATH')

if modpaths is not None :
    for path in modpaths.split(";"):
        sys.path.append(path)

from localizer import Localizer
import argparse 
parser = argparse.ArgumentParser()
parser.add_argument("--density", help="Input density matrix required for RLMO calculation ", required=True, 
        type=str, default="dmat.out")
parser.add_argument("--ovap", help="Specify the overlap matrix", required=True, 
        type=str, default="ovap.out")
parser.add_argument("--ortho_basis", help="The basis is orthonormal", required=False, 
        type=bool, default=False, action="store_true")

parser.add_argument("--nbasis", help="The number of basis functions of the region to be localized (default 2)",required=True,
default=2, type = int)

args = parser.parse_args()

DBO=np.loadtxt(args.density)
Stilde=np.loadtxt(args.ovap)

print("n. el : %.8f" % (np.trace(np.matmul(DBO,Stilde))))
nbasA=args.nbasis

if (not args.ortho_basis) :
	#print("The metric is the block-orthogonalized overlap S_tilde\n")
	locorb=Localizer(DBO,Stilde,nbasA) #has no default arguments
	locorb.localize()
	locorb.print_results()
	print("Eigvals A block\n")
	print(locorb.eigenvalues('A'))

	print("Eigvals B block\n")
	print(locorb.eigenvalues('B'))
	print()
	print("trace of density (D^RLMO): %.8f\n" % np.trace(locorb.DRo()))

	#get some matrices
	Tmat=locorb.Tmat()
	UU=locorb.UU()

	S11=Stilde[:nbasA,:nbasA]
	S11_inv=np.linalg.inv(S11)
	projector=np.matmul(Stilde[:,:nbasA],np.matmul(S11_inv,Stilde[:nbasA,:]))
	#check the projector
	print("project^AA == S^AA : %s\n" % (np.allclose(projector[:nbasA,:nbasA],S11)))
	#check BB part of the projector
	lenB=DBO.shape[0]-nbasA
	print("project^BB == 0^AA : %s\n" % (np.allclose(projector[nbasA:,nbasA:],np.zeros( (lenB,lenB) ) )))
	print("project^BA == 0^BA : %s\n" % (np.allclose(projector[nbasA:,:nbasA],np.zeros( (lenB,nbasA) ) )))
	C_RLMO=locorb.make_orbitals()
	sorted=locorb.sort_orbitals(projector)

	locality,occnum = locorb.locality()
	print(np.sum(occnum[:nbasA]))
	print(locality)
	DRo1st = locorb.DRo1st()
	np.savetxt("DRo_1st_bxa.txt",DRo1st[nbasA:,:nbasA])
	np.savetxt("DRo_1st_axb.txt",DRo1st[:nbasA:,nbasA:])
	DRo1st_axb = DRo1st[:nbasA,nbasA:]
	maxel=np.max(DRo1st_axb)
	print("maxel : %.8f\n" % (maxel*2.0))
	indices = np.where(np.isclose(DRo1st_axb, maxel))
	idA,idB=locorb.get_singly()
	print(idA,idB)
	eigA,eigB=locorb.get_singlyval()
	print("Singly occupied in eigvalA list")
	print(eigA)
	print("Singly occupied in eigvalB list")
	print(eigB)
	print("indices of so\n")
	print(indices)

else:

	##test using a ortho normal basis set (S=1)
	print("The metric is the identity matrix (ortho-normal basis set)\n")
	from scipy.linalg import fractional_matrix_power
	O = fractional_matrix_power(np.array(Stilde), 0.5)
	D=np.matmul(O,np.matmul(DBO,O))
	locorb1=Localizer(D,np.eye(D.shape[0]),nbasA)
	locorb1.localize()
	locorb1.print_results()
	print("Eigvals A block\n")
	print(locorb1.eigenvalues('A'))

	print("Eigvals B block\n")
	print(locorb1.eigenvalues('B'))
	print()
	print("trace of density (D^RLMO): %.8f\n" % np.trace(locorb1.DRo()))
