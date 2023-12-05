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
import cube_util
import rtutil
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
    parser.add_argument("-o1","--obs1", help="Specify the orbital basis set (genbasis;El1:basis1;El2:basis2)", required=False, 
            type=str, default="6-31G*")
    parser.add_argument("--frag_spec", help="Specify the fragment id (separated by semicolon) to be included in the high level portion", required=False, 
            type=str, default="1")

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
    parser.add_argument("--eri", help="Set type of ERI for tensor contraction, nofit|fit (default: None)", required=False,
            default=None)
    parser.add_argument("--rt_HF_iexch", help="(If JKClass is employed) account for the imaginaty-part of HF exchange if the density\
              is complex (for HYBRIDs), either using fitted 3-index intergrals of 4-index integrals;  see --eri option",\
                       required=False,default=False, action="store_true")
    parser.add_argument("-m", "--numpy_mem", help="Set the memeory for the PSI4 driver (default 2 Gib)", required=False,
            default=2, type = int)

    parser.add_argument("-z", "--charge", help="Charge of the bomme subsystem",
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
    parser.add_argument("--print_cube", help="Print the selected MOs", required=False,
                       default=False, action="store_true")

    parser.add_argument("--selective_pert", help="Selective perturbation on", required=False,
                       default=False, action="store_true")
    parser.add_argument("-p", "--principal", help="Restrict weighted dipole moment to HOMO->LUMO", required=False,
                       default=False, action="store_true")
    parser.add_argument("-f","--input", help="Set input parameters filename [defaul input.inp]", 
                       required=False, default="input.inp", type=str)
    parser.add_argument("--use_cap", help="Set Complex Absorbing Potential", required=False,
            default=False, action="store_true")
    parser.add_argument("--fde", help="FDE on", required=False,
            default=False, action="store_true")

    # pyemboption goes here
    parser.add_argument("--env_scf_type", help = "Set scf_type for psi4 run of environment density (default : direct)", required=False,
            type=str, default="direct")
    parser.add_argument("--env_df_guess", help = "Turn on density fitting to rapidly converge before switching \
                to direct scf for enviroment (default: False)",required=False,
            default=False, action="store_true")
    parser.add_argument("--env_df_basis", help = "Set df basis for enviroment calculation (default : def2-universal-jkfit)",required=False,
            type=str, default="def2-universal-jkfit")
    parser.add_argument("-gB","--geom_env", help="Specify frozen system (Angstrom) geometry (default: geomB.xyz)", required=False, 
            type=str, default="geomB.xyz")
    parser.add_argument("-Z", "--charge_tot", help="Charge of the FDE system [bomme(A)+ env(B)]",
            default=0, type = int)
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
    ### restart flag ## 
    parser.add_argument("--restart", help="restart from checkpoint", required=False,
            default=False, action="store_true")
    parser.add_argument("--restart_dumpnum", help="Set the number of iterations between checkpoints", required=False,
            default=100, type=int)
    args = parser.parse_args()

    
    pyembopt = embedoption
    
    pyembopt.gridfname = args.gridfname
    
    pyembopt.debug = args.debug
    pyembopt.nofde = (not args.fde)
    #pyembopt.verbosity = args.verbosity
    pyembopt.fde_thresh = args.embthresh
    pyembopt.tot_charge = args.charge_tot
    pyembopt.core_charge = args.charge
    pyembopt.maxit_fde = args.fde_max
    pyembopt.fde_offset = args.update_offset
    pyembopt.iterative = args.iterative
    pyembopt.static_field = args.static_field
    pyembopt.fmax = args.fmax
    pyembopt.fdir = args.fdir
    pyembopt.activefile = "tmp.xyz"
    pyembopt.envirofile = args.geom_env
    pyembopt.gtype = args.grid_opts
    pyembopt.jobtype = args.jobtype
    pyembopt.thresh_conv = args.embthresh
    pyembopt.basis = args.env_obs
    pyembopt.excfuncenv = args.env_func
    pyembopt.scf_type = args.env_scf_type
    pyembopt.df_basis = args.env_df_basis
    pyembopt.df_guess = args.env_df_guess 
    
    gparam = args.grid_param.split(",")
    if args.jobtype == 'adf':
      gparam = [float(m) for m in gparam]
      if not isinstance(gparam[0],float):
         raise TypeError("adf grid(param) accuracy must be float")
      pyembopt.param = gparam[0]
    else:
      gparam = [int(m) for m in gparam]
      pyembopt.param = tuple(gparam)

    # call functions here    
    bset,bsetH, molelecule_str, psi4mol, wfn, jkbase = initialize(args.jkclass,args.scf_type,args.obs1,args.frag_spec,args.geomA,\
                   args.func1,args.func2,args.numpy_mem,args.eri,args.rt_HF_iexch,args.exmodel,args.debug,args.jobtype)


    res, wfnBO = run(jkbase,psi4mol,bset,bsetH,args.guess,args.func1,args.func2,args.exmodel,wfn,pyembopt)

    if args.print_cube:
       print("    *******************************************************")
       print("    * CUBE PRINT                                           ")
       print("    * Orbitals : %s                                        "\
                               % args.select.replace('-2;' , ''))
       print("    *                                                      ")
       print("    *******************************************************\n")
       # set list for cube printing
       if args.select != "":
          molist = args.select.split("&")
          occlist = molist[0].split(";")
          occlist = [int(m) for m in occlist]
          if occlist[0] < 0 :
              occlist.pop(0)
          virtlist = molist[1].split(";")
          virtlist = [int(m) for m in virtlist]
       Umat = wfnBO["Umat"]
       Ccoeff = wfnBO["Ccoeff"]
       tmp=np.matmul(Umat,Ccoeff)
       #set margin and step
       margin = [4.5,4.5,4.5]
       dstep =  [0.2,0.2,0.2] 
       cube_util.orbtocube(psi4mol,margin,dstep,tmp,occlist,bset,tag="PsiA_occ",path="./")
       cube_util.orbtocube(psi4mol,margin,dstep,tmp,virtlist,bset,tag="PsiA_virt",path="./")

    if args.local_basis and (not args.real_time):
        from rlmo import regional_localize_MO    
        regional_localize_MO(wfn,wfnBO,cubelist=args.select) # omit '-2' in the string

    import rt_mod_new



    if args.real_time:
       
       fnameinput = args.input
       if not os.path.isfile(fnameinput):
           print("File ", fnameinput, " does not exist")
           exit(1)

       field_opts, calc_params = rtutil.set_params(fnameinput)
    

       # get the number of iterations
       dt =  calc_params['delta_t']
       #time_int in atomic unit
       time_int=calc_params['time_int']
       end_iter=int(time_int/dt)
       
       if args.restart:  # in case of restart init_iter != 0
           init_iter = -999
       else:
           init_iter = 0

       iter_opts = (init_iter,end_iter,calc_params)

       print("Startig real-time tddft computation")
       print()
       # call rt using the four indices I tensor (for small systems due to memory requirements)
       import time
       rt_time = time.process_time()
       rt_mod_new.run_rt_iterations(iter_opts, field_opts, bset, bsetH, wfnBO, psi4mol, args.axis, args.select, args.selective_pert, args.local_basis,\
               args.exciteA_only, args.numpy_mem, args.debug,pyembopt, args.use_cap)
       print('Total time of rt_module: %.3f seconds \n\n' % (time.process_time() - rt_time))
