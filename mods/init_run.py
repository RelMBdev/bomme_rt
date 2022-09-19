import os
import sys
import psi4

sys.path.insert(0, "../common")
modpaths = os.environ.get('COMMON_PATH')

if modpaths is not None :
    for path in modpaths.split(";"):
        sys.path.append(path)

from molecule import Molecule

def initialize(jkflag,scf_type,obs1,obs2,geomA,geomB,func1,func2,\
                charge):
    if (not jkflag) and (not scf_type == 'direct'):
        raise Exception("Bad keyword combination: scf_type and jkclass mode \n")

    acc_bset = obs1
    gen_bset = obs2


    #orblist = molist.split(";")
    #orblist = [int(m) for m in molist]

    fgeomA = geomA
    fgeomB = geomB
    func_l=func2
    func_h=func1
    print("High Level functional : %s\n" % func_h)
    print("Low Level functional : %s\n" % func_l)
    print("Low Level basis : %s\n" % gen_bset)
    print("High Level basis : %s\n" % acc_bset)


    moltot = Molecule(fgeomA,label=True)
    moltot.set_charge(charge)

    psi4.set_memory('2 GB')

    #iso_mol = psi4.geometry(molA.geometry())
    #repene_iso = iso_mol.nuclear_repulsion_energy()
    #psi4.core.clean()
    # nuclear repulsione energy of subsys A



    speclist = moltot.labels()

    molB = Molecule(fgeomB)

    #moltot has been augmented by molecule B geometry
    moltot.append(molB.geometry())

    #append some options, we can also include total charge (q) and multiplicity (S) of the total system
    moltot.append("symmetry c1" + "\n" + "no_reorient" + "\n" + "no_com")
    moltot.display_xyz()

    molA = Molecule(fgeomA)

    #molA.display_xyz()
    #molB.display_xyz()

    molobj=psi4.geometry(moltot.geometry())
    molobj.print_out()


    psi4.core.IO.set_default_namespace("molobj")
    def basisspec_psi4_yo__anonymous775(mol,role):
            mol.set_basis_all_atoms(gen_bset, role=role)
            for k in speclist:
              mol.set_basis_by_label(k, acc_bset,role=role)
            return {}


    #the basis set object for the complex: a composite basis set for rt applications
    psi4.qcdb.libmintsbasisset.basishorde['USERDEFINED'] = basisspec_psi4_yo__anonymous775

    L=[4.5,4.5,4.5]
    Dx=[0.15,0.15,0.15]

    psi4.set_options({'basis': 'userdefined',
                      'puream': 'True',
                      'DF_SCF_GUESS': 'False',
                      'scf_type': scf_type,
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

    job_nthreads = int(os.getenv('OMP_NUM_THREADS', 1))
    psi4.set_num_threads(job_nthreads)

    #mol_wfn = psi4.core.Wavefunction.build( \
    #                    embmol,psi4.core.get_global_option('basis'))
    #mol_wfn.basisset().print_detail_out()

    ene,wfn=psi4.energy(func_l ,return_wfn=True)

    # mHigh is the portion of the system treated at the higher level

    mHigh=psi4.geometry(molA.geometry() +"symmetry c1" +"\n" +"no_reorient" +"\n" +"no_com")
    #wfnA = psi4.core.Wavefunction.build( \
    #                    mHigh,acc_bset)
    #nelA=wfnA.nalpha()+wfnA.nbeta()
    #print("Frag. A el: %i" %nelA)
    #print()
    print("centers of high level subsys: %i" % molA.natom())


    bset=wfn.basisset()
    numshell=bset.nshell()
    print("Number of shells of the total AB basis:  %i" % numshell)
    numbas=bset.nbf()
    print("Number of functions of the total AB basis:  %i" % numbas)
    natoms=molobj.natom()
    #the basis set object for the high level portion
    bsetH=psi4.core.BasisSet.build(mHigh,'ORBITAL',acc_bset,puream=-1)

    #bsetH.print_detail_out()
    nbfA = bsetH.nbf()
    
    #nbfA = 0
    #for k in range(numbas):
    #    if bset.function_to_center(k) <= molA.natom()-1 :
    #      nbfA+=1
    
    print("functions in subsys1: %i" % nbfA)

    return bset,bsetH, moltot,molobj,wfn

####################################################################################

if __name__ == "__main__":
    import argparse

    ####################################
    # parse arguments from std input
    ####################################
    parser = argparse.ArgumentParser()
    parser.add_argument("-gA","--geomA", help="Specify geometry file for the subsystem A", required=True,    
            type=str, default="XYZ")
    parser.add_argument("-gB","--geomB", help="Specify geometry file for the subsystem B", required=True, 
            type=str, default="XYZ")
    parser.add_argument("-d", "--debug", help="Debug on, prints debug info to err.txt", required=False,
            default=False, action="store_true")

    parser.add_argument("--guess", help="Set the guess density ('SAD' or 'GS')", required=False,
            type=str, default='GS',)
    parser.add_argument("-o1","--obs1", help="Specify the orbital basis set for subsys A", required=False, 
            type=str, default="6-31G*")
    parser.add_argument("-o2","--obs2", help="Specify the general orbital basis set", required=False, 
            type=str, default="6-31G*")
    parser.add_argument("-f2","--func2", help="Specify the low level theory functional", required=False, 
            type=str, default="blyp")
    parser.add_argument("-f1","--func1", help="Specify the high level theory functional", required=False, 
            type=str, default="blyp")
    parser.add_argument("--scf_type", help="Specify the scf type: direct or df (for now)", required=False, 
            type=str, default="direct")
    parser.add_argument("-J", "--jkclass", help="Use JK class for J and K matrix computation", required=False,
            default=False, action="store_true")
    parser.add_argument("-m", "--numpy_mem", help="Set the memeory for the PSI4 driver (default 2 Gib)", required=False,
            default=2, type = int)

    parser.add_argument("-z", "--charge", help="Charge of the whole system",
            default=0, type = int)
    args = parser.parse_args()

    
    bset,bsetH,moltot,psi4mol,wfn = initialize(args.jkclass,args.scf_type,args.obs1,args.obs2,args.geomA,\
                   args.geomB,args.func1,args.func2,args.charge)
