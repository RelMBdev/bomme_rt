import os
import sys
import psi4
import numpy as np

sys.path.insert(0, "./common")
modpaths = os.environ.get('MODS_PATH')

if modpaths is not None :
    for path in modpaths.split(";"):
        sys.path.append(path)
from scf_run import run
from init_run import initialize
from fde_utils import embedoption
import argparse
####################################################################################

if __name__ == "__main__":

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
            type=str, default="DIRECT")
    parser.add_argument("-J", "--jkclass", help="Use JK class for J and K matrix computation", required=False,
            default=False, action="store_true")
    parser.add_argument("--eri", help="Set type of ERI for tensor contraction, nofit|fit (default: None)", required=False,
            default=None)
    parser.add_argument("--rt_HF_iexch", help="(If JKClass is employed) account for the imaginaty-part of HF exchange if the density\
              is complex (for HYBRIDs), either using fitted 3-index intergrals of 4-index integrals;  see --eri option",\
                       required=False,default=False, action="store_true")
    parser.add_argument("-m", "--numpy_mem", help="Set the memeory for the PSI4 driver (default 2 Gib)", required=False,
            default=2, type = int)

    parser.add_argument("-z", "--charge", help="Charge of the whole system",
            default=0, type = int)
    parser.add_argument("-x", "--exmodel", help="Set the exchange model (EX-0 or EX-1) between subsystems (default EX: 0)",
            default=0, type = int)
    parser.add_argument("-a", "--axis", help="Set the boost direction: x:0|y:1|z:2 (default 2)",
            default=2, type = int)

    # options for RT 
    parser.add_argument("--real_time", help="Real time propagation on", required=False,
            default=False, action="store_true")
    parser.add_argument("--exciteA_only", help="Only the A subsystem is excited", 
                       required=False,default=False, action="store_true")
    parser.add_argument("--local_basis", help="Use local basis ", required=False,
                       default=False, action="store_true")
    parser.add_argument("--select", help="Specify the occ-virt MO weighted dipole moment. (-2; occ_list & virt_list).\
                         Occ/virt_list is a list of numbers separated by semicolon. To include all the virtual mainfold\
                          set virt_list=-99 (default: "")",default="", type=str)

    parser.add_argument("--selective_pert", help="Selective perturbation on", required=False,
                       default=False, action="store_true")
    parser.add_argument("-p", "--principal", help="Restrict weighted dipole moment to HOMO->LUMO", required=False,
                       default=False, action="store_true")
    parser.add_argument("-f","--input", help="Set input parameters filename [defaul input.inp]", 
                       required=False, default="input.inp", type=str)
    parser.add_argument("--fde", help="FDE on", required=False,
            default=False, action="store_true")

    # pyemboption goes here
    parser.add_argument("-gB","--geom_env", help="Specify frozen system (Angstrom) geometry (default: geomB.xyz)", required=False, 
            type=str, default="geomB.xyz")
    parser.add_argument("--embthresh", help="set EMB threshold (default = 1.0e-8)", required=False, 
            type=np.float64, default=1.0e-8)
    parser.add_argument("--fde_max", help="Max number of fde iteration for splitSCF scheme  (default: 0)",
            required=False, type=int, default=0)
    parser.add_argument("--modpaths", help="set berthamod and all other modules path [\"path1;path2;...\"] (default = ../src)", 
        required=False, type=str, default="../src")
    parser.add_argument("--env_obs", help="Specify the orbital basis set for the (FDE) enviroment (default: AUG/ADZP)", required=False,
            type=str, default="AUG/ADZP")
    parser.add_argument("--env_func", help="Specify the function for the (FDE) environment density (default: BLYP)", required=False,
            type=str, default="BLYP")
    parser.add_argument("--grid_opts", help="set gridtype (default: 2)",
        required=False, type=int, default=2)
    parser.add_argument("--grid_param", help="set grid parameter i.e grid accuracy in adf (default: '4.0')",
        required=False, type=str, default="4.0")
    parser.add_argument("--jobtype", help="set the adf/psi4 embedding job (default = adf)",
        required=False, type=str, default="adf")
    parser.add_argument("--gridfname", help="set grid filename (default = grid.dat)",
        required=False, type=str, default="grid.dat")
    parser.add_argument("--static_field", help="Add a static field to the SCF (default : False)", required=False, 
            default=False, action="store_true")
    parser.add_argument("--fmax", help="Static field amplitude (default : 1.0e-5)", required=False, 
            type=np.float64, default=1.0e-5)
    parser.add_argument("--fdir", help="External field direction (cartesian)  (default: 2)",
            required=False, type=int, default=2)
    parser.add_argument("-u","--update_offset", help="Embedding potential update offset (potential is static in the prescribed time interval)",
            default=0.1, type = float)
    parser.add_argument("-i","--iterative", help="Set iterative update of the embedding potential", required=False,
            default=False, action="store_true")
   
    args = parser.parse_args()

    if args.fde: 
       pyembopt = embedoption
       
       pyembopt.gridfname = args.gridfname
       
       pyembopt.debug = args.debug
       #pyembopt.verbosity = args.verbosity
       pyembopt.fde_thresh = args.embthresh
       pyembopt.maxit_fde = args.fde_max
       pyembopt.fde_offset = args.update_offset
       pyembopt.iterative = args.iterative
       pyembopt.static_field = args.static_field
       pyembopt.fmax = args.fmax
       pyembopt.fdir = args.fdir
       pyembopt.activefile = args.geomA
       pyembopt.envirofile = args.geom_env
       pyembopt.gtype = args.grid_opts
       pyembopt.jobtype = args.jobtype
       pyembopt.thresh_conv = args.embthresh
       pyembopt.basis = args.env_obs
       pyembopt.excfuncenv = args.env_func
       
       gparam = args.grid_param.split(",")
       if args.jobtype == 'adf':
         gparam = [float(m) for m in gparam]
         if not isinstance(gparam[0],float):
            raise TypeError("adf grid(param) accuracy must be float")
         pyembopt.param = gparam[0]
       else:
         gparam = [int(m) for m in gparam]
         pyembopt.param = tuple(gparam)
    else:
         pyembopt = None

    # call functions here    
    bset,bsetH, molelecule_str, psi4mol, wfn, jkbase = initialize(args.jkclass,args.scf_type,args.obs1,args.obs2,args.geomA,\
                   args.func1,args.func2,args.charge,args.numpy_mem,args.eri,args.rt_HF_iexch,args.exmodel,args.debug)


    res, wfnBO = run(jkbase,psi4mol,bset,bsetH,args.guess,args.func1,args.func2,args.exmodel,wfn,pyembopt)

    if args.local_basis and (not args.real_time):
        from rlmo import regional_localize_MO    
        regional_localize_MO(wfn,wfnBO,cubelist=args.select) # omit '-2' in the string

    import rt_mod_new



    if args.real_time:
       
       fnameinput = args.input
       if not os.path.isfile(fnameinput):
           print("File ", fnameinput, " does not exist")
           exit(1)

       print("Startig real-time tddft computation")
       print()
       # call rt using the four indices I tensor (for small systems due to memory requirements)
       rt_mod_new.run_rt_iterations(fnameinput, bset, bsetH, wfnBO, psi4mol, args.axis, args.select, args.selective_pert, args.local_basis, args.exciteA_only, args.numpy_mem, args.debug,pyembopt)
         
