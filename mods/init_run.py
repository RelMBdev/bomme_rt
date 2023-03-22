import os
import sys
import psi4
import numpy as np

sys.path.insert(0, "../common")
modpaths = os.environ.get('COMMON_PATH')

if modpaths is not None :
    for path in modpaths.split(";"):
        sys.path.append(path)

from molecule import Molecule, gparser

def initialize(jkflag,scf_type,obs1,obs2,fgeom,func1,func2,\
                charge,numpy_memory=8,eri=None,rt_HF_iexch=False, exch_model=0, debug=False):
    # rt_HF_iexch flag determines if the imaginary part of the exchange 
    # is accounted in the rt-evolution
    # scf_type controls the type of scf type in the initialization (the GS density
    # is optionally used as guess density) and the type of J (K) matrix ('direct',
    # 'mem_df', 'disk_df', 'pk' integrals)
    acc_bset = obs1
    gen_bset = obs2
    

    #orblist = molist.split(";")
    #orblist = [int(m) for m in molist]


    func_l=func2
    func_h=func1
    print("High Level functional : %s\n" % func_h)
    print("Low Level functional : %s\n" % func_l)
    print("Low Level basis : %s\n" % gen_bset)
    print("High Level basis : %s\n" % acc_bset)
    
    # corestr is a string containing only the 'high-level-theory' subsys
    speclist, geomstr, corestr, natom1 = gparser(fgeom)
    
    # dump tmp file
    with open("tmp.xyz","w") as fgeom_act:
        fgeom_act.write("%s\n" %  str(natom1) )
        fgeom_act.write('\n') # black line        
        for line in geomstr:
         fgeom_act.write(line)
    moltot = Molecule()
    moltot.set_charge(charge)
    moltot.geom_from_string(geomstr)

    psi4.set_memory('2 GB')

    #iso_mol = psi4.geometry(molA.geometry())
    #repene_iso = iso_mol.nuclear_repulsion_energy()
    #psi4.core.clean()
    # nuclear repulsione energy of subsys A


    #append some options, we can also include total charge (q) and multiplicity (S) of the total system
    moltot.append("symmetry c1" + "\n" + "no_reorient" + "\n" + "no_com")
    moltot.display_xyz()
    
    # used in fragment A basis definition, see below
    molA = Molecule()
    molA.geom_from_string(corestr,natom1)
    
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

    psi4.core.set_output_file('psi4.out', False)
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
    #warning if the low-level functional is hybrid/lrc/need HF exch
    V_pot = wfn.V_potential()
    
    if func_l == 'hf':     
         fun_low_hfexch = True
    if (V_pot is not None):
        if V_pot.functional().is_x_hybrid() :
         fun_low_hfexch = True
         print("The low-level-theory functional requires HF exch")
        else:
         fun_low_hfexch = False
    # mHigh is the portion of the system treated at the higher level

    mHigh=psi4.geometry(molA.geometry() +"symmetry c1" +"\n" +"no_reorient" +"\n" +"no_com")
    #wfnA = psi4.core.Wavefunction.build( \
    #                    mHigh,acc_bset)
    #nelA=wfnA.nalpha()+wfnA.nbeta()
    #print("Frag. A el: %i" %nelA)
    #print()
    print("centers of high level subsys: %i" % mHigh.natom())


    bset=wfn.basisset()
    numshell=bset.nshell()
    print("Number of shells of the total AB basis:  %i" % numshell)
    numbas=bset.nbf()
    print("Number of functions of the total AB basis:  %i" % numbas)
    #natoms=molobj.natom()
    #the basis set object for the high level portion
    bsetH=psi4.core.BasisSet.build(mHigh,'ORBITAL',acc_bset,puream=-1)

    #bsetH.print_detail_out()
    nbfA = bsetH.nbf()
    
    #nbfA = 0
    #for k in range(numbas):
    #    if bset.function_to_center(k) <= molA.natom()-1 :
    #      nbfA+=1
    
    print("functions in subsys1: %i" % nbfA)

    mints = psi4.core.MintsHelper(bset)
    if (eri != 'fit') and (rt_HF_iexch) and (not fun_low_hfexch) and (not jkflag):
        raise Exception("Fitted K imag-part required -> fit eri \n")
    if eri=='nofit' and (not jkflag):
       # Run a quick check to make sure everything will fit into memory
       I_Size = (numbas**4) * 8.e-9
       print("\nSize of the ERI tensor will be %4.2f GB." % I_Size)
       
       # Estimate memory usage
       memory_footprint = I_Size * 1.5
       if I_Size > numpy_memory:
           psi4.core.clean()
           raise Exception("Estimated memory utilization (%4.2f GB) exceeds numpy_memory limit of %4.2f GB." % (memory_footprint, numpy_memory))
       #Get Eri (2-electron repulsion integrals)
       eri_tensor=np.array(mints.ao_eri())
       print("eri n. axis: %i" % len(eri_tensor.shape))
    elif eri=='fit':
          #print("HERE set aux_basis") 
          aux_basis = psi4.core.BasisSet.build(molobj, "DF_BASIS_SCF", "", "RIFIT", psi4.core.get_global_option('basis'))
          n_aux = aux_basis.nbf()

          # Run a quick check to make sure everything will fit into memory
          #3-index ERIs, dimension (1, Naux, nbf, nbf)
          Q_Size = (n_aux*numbas*2) * 8.e-9
          print("\nSize of the 3-index ERI tensor will be %4.2f GB." % Q_Size)

          # Estimate memory usage
          memory_footprint = Q_Size * 1.5
          if Q_Size > numpy_memory:
              psi4.core.clean()
              raise Exception("Estimated memory utilization (%4.2f GB) " +\
                      "exceeds numpy_memory limit of %4.2f GB." % (memory_footprint, numpy_memory))

          #dummy zero basis
          zero_bas = psi4.core.BasisSet.zero_ao_basis_set() 
          #bset = wfn.basisset() already defined
          Ppq = mints.ao_eri(aux_basis, zero_bas, bset,bset)

          # Build and invert the metric
          metric = mints.ao_eri(aux_basis, zero_bas, aux_basis, zero_bas)
          metric.power(-0.5, 1.e-14)

          # Remove the excess dimensions of Ppq & metric
          Ppq = np.squeeze(Ppq)
          metric = np.squeeze(metric)

          # Contract Ppq & Metric to build Qpq
          eri_tensor = np.einsum('QP,Ppq->Qpq', metric, Ppq)
          print("eri shape [%i,%i,%i]" % (eri_tensor.shape[0],eri_tensor.shape[1],eri_tensor.shape[2]))
          print("eri n. axis: %i" % len(eri_tensor.shape))
    else:
          print("Use native Psi4 JK class: %s , real-time  imag. HF exchange required: %s\n" % (jkflag,rt_HF_iexch))
          eri_tensor = None

    # the following conditions, if met, allow to set, in a typical BOMME configuration (i.e high-level/low-level func mixture)
    # a jkfactory and Fock such that the native Psi4 JKclass is employed to get the real part of J and K matrices and
    # and 'hopefully' a shrinked 4-indiex eri tensor (corresponding to the AA subblock of the basis set) is used to get the
    # imaginary part of the HF exch if needed.
    if (not fun_low_hfexch) and eri=='nofit' and exch_model == 0 and jkflag :  # most common case the exchange_model = 0
       mints_AA = psi4.core.MintsHelper(bsetH) # the 4-index eri object is retrived on the AA sub-basis
       # Run a quick check to make sure everything will fit into memory
       I_AA_Size = (nbfA**4) * 8.e-9
       print("\nSize of the ERI tensor (AA block) will be %4.2f GB." % I_AA_Size)
       
       # Estimate memory usage
       memory_footprint = I_AA_Size * 1.5
       if I_AA_Size > numpy_memory:
           psi4.core.clean()
           raise Exception("Estimated memory utilization (%4.2f GB) exceeds numpy_memory limit of %4.2f GB." % (memory_footprint, numpy_memory))
       #Get Eri (2-electron repulsion integrals)
       eri_tensor=np.array(mints_AA.ao_eri())
    
    print("eri is instance: %s\n" % type(eri_tensor)) 
    from Fock_helper import jkfactory
    jkbase = jkfactory(bset,molobj,jkflag,scf_type,eri=eri_tensor,real_time=rt_HF_iexch,debug=debug)

    return bset,bsetH, moltot,molobj,wfn,jkbase

####################################################################################

if __name__ == "__main__":
    import argparse

    ####################################
    # parse arguments from std input
    ####################################
    parser = argparse.ArgumentParser()
    parser.add_argument("-gA","--geomA", help="Specify geometry file for the subsystem A", required=True,    
            type=str, default="XYZ")
    #parser.add_argument("-gB","--geomB", help="Specify geometry file for the subsystem B", required=True, 
    #        type=str, default="XYZ")
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

    
    bset,bsetH,moltot,psi4mol,wfn, jkbase = initialize(args.jkclass,args.scf_type,args.obs1,args.obs2,args.geomA,\
                   args.func1,args.func2,args.charge)
