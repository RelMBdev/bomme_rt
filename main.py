import os
import sys
import psi4


sys.path.insert(0, "./common")
modpaths = os.environ.get('MODS_PATH')

if modpaths is not None :
    for path in modpaths.split(";"):
        sys.path.append(path)
from scf_run import run
from init_run import initialize
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
    parser.add_argument("--fitt_HF_exch", help="(If JKClass is employed) fit imaginaty-part HF exchange if the density is complex (for HYBRIDs)",
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
                          set virt_list=-99 (default: 0; 0 & 0)",default="0; 0 & 0", type=str)

    parser.add_argument("--selective_pert", help="Selective perturbation on", required=False,
                       default=False, action="store_true")
    parser.add_argument("-p", "--principal", help="Restrict weighted dipole moment to HOMO->LUMO", required=False,
                       default=False, action="store_true")
    parser.add_argument("-f","--input", help="Set input parameters filename [defaul input.inp]", 
                       required=False, default="input.inp", type=str)

    args = parser.parse_args()

    # call functions here    
    bset,bsetH, molelecule_str, psi4mol, wfn, jkbase = initialize(args.jkclass,args.scf_type,args.obs1,args.obs2,args.geomA,\
                   args.func1,args.func2,args.charge,args.numpy_mem,args.eri,args.fitt_HF_exch)


    res, wfnBO = run(jkbase,psi4mol,bset,bsetH,args.guess,args.func1,args.func2,args.exmodel,wfn)
    import rt_mod_new



    if args.real_time:
       
       fnameinput = args.input
       if not os.path.isfile(fnameinput):
           print("File ", fnameinput, " does not exist")
           exit(1)

       print("Startig real-time tddft computation")
       print()
       # call rt using the four indices I tensor (for small systems due to memory requirements)
       rt_mod_new.run_rt_iterations(fnameinput, bset, bsetH, wfnBO, psi4mol, args.axis, args.select, args.selective_pert, args.local_basis, args.exciteA_only, args.numpy_mem, args.debug)
         
