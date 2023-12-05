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

def dump_geom(psimol,jobtype):
    # dump a txt file containing basis geometry 
    # (no charge/mult if jobtype == 'adf')  
    natom = psimol.nallatom()
    with open("tmp.xyz","w") as fgeom_act:
      if jobtype == 'adf':
         geom_mat = np.asarray(psimol.geometry())*0.52917720859
         symb = [psimol.symbol(id_atom) for id_atom in range(natom)]
         fgeom_act.write("%s\n" %  str(natom) )
         fgeom_act.write("\n")
         for k in range(natom):
             fgeom_act.write("%s %.6f  %.6f  %.6f\n" % (symb[k],geom_mat[k,0],\
                       geom_mat[k,1],geom_mat[k,2]))
      else:
          geomstr = psimol.save_string_xyz()
          fgeom_act.write("%s\n" %  str(natom) )
          fgeom_act.write("\n")
          fgeom_act.write(geomstr)

    fgeom_act.close()

class checkpoint_data:
    def __init__(self,wfn,func_h = 'blyp',func_l = 'blyp',restart=False, json_data = None):
        self.__restart = restart
        self.__hfunc = func_h
        self.__lfunc = func_l
        self.__wfn = wfn # can be either a dummy psi4 wfn object or a real wfn  
        self.__is_hybrid = None
        if restart:
          self.__is_hybrid = json_data["is_x_hybr"]
        else:
            if func_l.upper() == 'HF':
                self.__is_hybrid = True # is Hartree-Fock
            else:    
                pot_type = wfn.V_potential()
                self.__is_hybrid = pot_type.functional().is_x_hybrid()

           
    def get_specs(self):
        res = {'func_h' : self.__hfunc, 'func_l' : self.__lfunc, 'lfunc_hyb' : self.__is_hybrid} 
        return res


def initialize(jkflag,scf_type,obs1,frag_spec,fgeom,func1,func2,\
                numpy_memory=8,eri=None,rt_HF_iexch=False, exch_model=0, debug=False, fdejob='adf' ,restart=False, restart_json = None):
    # rt_HF_iexch flag determines if the imaginary part of the exchange 
    # is accounted in the rt-evolution
    # scf_type controls the type of scf type in the initialization (the GS density
    # is optionally used as guess density) and the type of J (K) matrix ('direct',
    # 'mem_df', 'disk_df', 'pk' integrals)
    

    #orblist = molist.split(";")
    #orblist = [int(m) for m in molist]


    func_l=func2
    func_h=func1
    print("High Level functional : %s\n" % func_h)
    print("Low Level functional : %s\n" % func_l)
    
    # corestr is a string containing only the 'high-level-theory' subsys
    parsed_geom, f_id = gparser(frag_spec,fgeom)    # natom_bomme is the number of atom of the bomme-fragment (in a bomme+FDE setup)
    #psi4 molecule object
    molobj = psi4.geometry(parsed_geom.geometry())

    print("number of fragmemts: %i\n" % molobj.nfragments())
    molobj.activate_all_fragments()
    
    # parse the basis string
    basis_sub_str = obs1.split(";")
    basis_str = "assign " + str(basis_sub_str[0])+"\n"

    if len(basis_sub_str) >1:
       for elm in basis_sub_str[1:]:
           tmp= elm.split(":")
           basis_str += "assign " + str(tmp[0]) + " " +str(tmp[1]) +"\n"
    # set psi4 option
    psi4.set_memory(str(numpy_memory) + 'GB')
    psi4.core.set_output_file('psi4.out', False)
    psi4.set_options({
                      'puream': 'True',
                      'DF_SCF_GUESS': 'False',
                      'scf_type': scf_type,
                     # 'reference' : reference,
                      'df_basis_scf': 'def2-universal-jkfit',
                      'df_ints_io' : 'save',       # ?
                      'dft_radial_scheme' : 'becke',
                       #'dft_radial_points': 80,
                      'dft_spherical_points' : 434,
                      'cubeprop_tasks': ['density'],
                      'e_convergence': 1.0e-8,
                      'd_convergence': 1.0e-8})

    psi4.basis_helper(basis_str,
                     name='mybas')


    job_nthreads = int(os.getenv('OMP_NUM_THREADS', 1))
    psi4.set_num_threads(job_nthreads)

    
    #get the basis set corresponding to total molecule
    
    bset = psi4.core.BasisSet.build(molobj, 'ORBITAL',\
            psi4.core.get_global_option('basis')) # or set the basis from input

    print("functions in general basis: %i" % bset.nbf())
    
    #save the geometry with cleaned up symbols for later use
    dump_geom(molobj,fdejob) 
    
    # get the reduced basis
    molobj.deactivate_all_fragments()
    molobj.set_active_fragments( [int(x) for x in range(1,len(f_id)+1)] )
    bsetH = psi4.core.BasisSet.build(molobj, 'ORBITAL',\
            psi4.core.get_global_option('basis')) # or set the basis from input

    # reactivate all fragments
    
    molobj.activate_all_fragments()
    molobj.update_geometry()

    #bsetH.print_detail_out()
    nbfA = bsetH.nbf()
    # the n of basis function of the blended basis
    numbas = bset.nbf()
    
    print("functions in subsys1: %i" % nbfA)
    
    #mol_wfn = psi4.core.Wavefunction.build( \
    #                    embmol,psi4.core.get_global_option('basis'))
    #mol_wfn.basisset().print_detail_out()
    if not restart:
       ene,wfn=psi4.energy(func_l ,return_wfn=True)
    else:
       #build dummy wfn just to refresh mints & co 
       wfn = psi4.core.Wavefunction.build(molobj, psi4.core.get_global_option('BASIS')) 
    #check if the low-level functional is hybrid/lrc/need HF exch
    checkdata = checkpoint_data(wfn, func_h, func_l, restart, json_data = restart_json)
    
    if func_l.upper() == 'HF':     
         fun_low_hfexch = True
    if (checkdata is not None):
        if checkdata.get_specs()['lfunc_hyb']:
         fun_low_hfexch = True
         print("The low-level-theory functional requires HF exch")
        else:
         fun_low_hfexch = False

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

    return bset,bsetH, parsed_geom, molobj, wfn, jkbase

####################################################################################

if __name__ == "__main__":
    import argparse

    ####################################
    # parse arguments from std input
    ####################################
    parser = argparse.ArgumentParser()
    parser.add_argument("-gA","--geomA", help="Specify geometry file for the subsystem A", required=True,    
            type=str, default="XYZ")
    parser.add_argument("--frag_spec", help="Specify the fragment id (separated by semicolon) to be included in the high level portion", required=False, 
            type=str, default="1")
    #parser.add_argument("-gB","--geomB", help="Specify geometry file for the subsystem B", required=True, 
    #        type=str, default="XYZ")
    parser.add_argument("-d", "--debug", help="Debug on, prints debug info to err.txt", required=False,
            default=False, action="store_true")

    parser.add_argument("--guess", help="Set the guess density ('SAD' or 'GS')", required=False,
            type=str, default='GS',)
    parser.add_argument("-o1","--obs1", help="Specify the orbital basis set for subsys A", required=False, 
            type=str, default="6-31G*")

    #parser.add_argument("-o2","--obs2", help="Specify the general orbital basis set", required=False, 
    #        type=str, default="6-31G*")
    
    parser.add_argument("-f2","--func2", help="Specify the low level theory functional", required=False, 
            type=str, default="blyp")
    parser.add_argument("-f1","--func1", help="Specify the high level theory functional", required=False, 
            type=str, default="blyp")
    parser.add_argument("--scf_type", help="Specify the scf type: direct or df (for now)", required=False, 
            type=str, default="DIRECT")
    parser.add_argument("-J", "--jkclass", help="Use JK class for J and K matrix computation", required=False,
            default=False, action="store_true")
    parser.add_argument("-m", "--numpy_mem", help="Set the memeory for the PSI4 driver (default 2 Gib)", required=False,
            default=2, type = int)
    parser.add_argument("--jobtype", help="set the adf/psi4 embedding job (default = adf)",
        required=False, type=str, default="adf")

    #parser.add_argument("-z", "--charge", help="Charge of the whole system",
    #        default=0, type = int)
    args = parser.parse_args()

    
    bset,bsetH,moltot,psi4mol,wfn, jkbase = initialize(args.jkclass,args.scf_type,args.obs1,args.frag_spec,args.geomA,\
                   args.func1,args.func2,fdejob = args.jobtype)
