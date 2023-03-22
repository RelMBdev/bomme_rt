import os.path
import sys
import os 
import numpy as np

import argparse
modpaths = os.environ.get('PYBERTHA_MOD_PATH')

if modpaths is not None :
    for path in modpaths.split(";"):
        sys.path.append(path)

sys.path.insert(0, "../common")
modpaths = os.environ.get('COMMON_PATH')

if modpaths is not None :
    for path in modpaths.split(";"):
        sys.path.append(path)

from dataclasses import dataclass


@dataclass
class embedoption:
    activefile: str
    envirofile :str 
    #dumpfiles: bool
    debug: bool
    thresh: float
    iterative : bool
    tot_charge : int
    core_charge : int  # the core system is the bomme subsystem
    fde_offset: np.float64
    fde_thresh: np.float64 = 1.0e-6
    maxit_fde : int = 0
    inputfile: str = "input.inp"
    gtype: int = 2
    jobtype : str = 'adf'
    nofde: bool = False # remove
    param: tuple = (4.0,)   # these are grid parameters, i.e grid accuracy in adf / radial and angular point number in psi4
    basis: str = 'AUG/ADZP'
    excfuncenv : str = "BLYP"
    static_field : bool = False
    fmax : np.float64 = 1.0e-5
    fdir: int = 2
    densityzero: str = ""
    density : str = ""
    drx: float = 0.1
    dry: float = 0.1
    drz: float = 0.1
    margin: float = 5.5
    gridfname : str = "grid.dat"

import pyembmod
from pyembmod import GridDensityFactory

class GridFuncFactory(GridDensityFactory):
    def matrix_from_grid(self,gfunc):
        nbas = GridDensityFactory.nbf(self)
        phi =  GridDensityFactory.phi(self)
        ws =   GridDensityFactory.points(self)[:,3]  #the weights
        lpos = GridDensityFactory.lpos(self)   # see the psi4numpy tutorial
        #compute  pot matrix
        res = np.zeros((nbas,nbas),dtype=np.complex128) # check
        tmp = np.einsum('pb,p,p,pa->ab', phi, gfunc, ws, phi)
        #to check
        # Add the temporary back to the larger array by indexing, ensure it is symmetric
        res[(lpos[:, None], lpos)] += 0.5 * (tmp + np.conjugate(tmp.T))
        #check Vtmp and V
        er = np.allclose(res,tmp,atol=1.0e-12)
        if  (not er):
          print("Check Vtmp and V")
        #print("N. basis funcs: %i\n" % nbas)
        #check if V has imag part
        if False:
           print("V is real: %s\n" % (np.allclose(res.imag,np.zeros((nbas,nbas),dtype=np.float_),atol=1.0e-12)))
           np.savetxt("vemb.txt", res.real)
        return res.real


class emb_wrapper():
    def __init__(self,mol_obj,pyembopt,basis_act,stdoutprint=True,ovap=None):
       self.__e_field = pyembopt.static_field # Bool
       self.__fmax = pyembopt.fmax 
       self.__fdir = pyembopt.fdir
       self.__nofde = pyembopt.nofde
       self.__ovap = ovap
       self.__mol_obj = mol_obj
       self.__basis_act = basis_act
       self.__grid = None
       self.__debug = pyembopt.debug
       self.__env_obs = pyembopt.basis
       self.__tot_charge = pyembopt.tot_charge
       self.__active_charge = pyembopt.core_charge
       
       activefname = pyembopt.activefile
       if not os.path.isfile(activefname):
           raise Exception("File ", activefname , " does not exist")
       
       envirofname = pyembopt.envirofile
       if not os.path.isfile(envirofname):
           raise Exception("File ", envirofname , " does not exist")
       
       self.__embfactory = pyembmod.pyemb(activefname,envirofname,pyembopt.jobtype) #jobtype='adf' is default
       #grid_param =[50,110] # psi4 grid parameters (see Psi4 grid table), can be set using args.grid_param
       
       self.__embfactory.set_charge(self.__active_charge) 
       self.__embfactory.set_charge(self.__tot_charge,'total') 

       self.__embfactory.set_options(param=pyembopt.param, \
          gtype=pyembopt.gtype, basis=pyembopt.basis) 
       self.__embfactory.set_enviro_func(pyembopt.excfuncenv) 
       self.__embfactory.set_thresh_conv(pyembopt.thresh_conv)
       self.__embfactory.set_grid_filename(pyembopt.gridfname)
       
       if (pyembopt.debug):
          import uuid
          filename = "adfout_" + str(uuid.uuid4()) + ".txt"
          self.__embfactory.set_adf_filenameout(filename)
          filename = "psi4out_" + str(uuid.uuid4()) + ".txt"
          self.__embfactory.set_psi4_filenameout(filename)
       if stdoutprint:
          print("embfactory Options:")
          print(self.__embfactory.get_options())
       
       self.__embfactory.initialize()
       grid = self.__embfactory.get_grid() # locally defined 
       
       if (pyembopt.debug):
          np.savetxt ("grid.txt", grid)
       self.__grid = grid
    def grid(self):
        return self.__grid

    def set_density(self,Cocc,Dmat=None):
        
        activesys = GridDensityFactory(self.__mol_obj,self.__grid,self.__basis_act)  
        
        if not isinstance(Cocc,np.ndarray) and (Dmat is not None):
            if not isinstance(Dmat,np.ndarray):
                raise TypeError("Dmat must be np.ndarray")
            rho = activesys.from_D(Dmat,self.__ovap)
        else:
            if not isinstance(Cocc,np.ndarray):
                raise TypeError("C mat must be np.ndarray")

            rho = activesys.from_Cocc(Cocc)
        self.__rho = rho
        return rho

    def rho_integral(self):
        if not isinstance(self.__grid,np.ndarray):
           raise TypeError("Not a grid array")
        w = self.__grid[:,3]
        sum_rho =np.matmul(self.__rho,w)*2.00
        return sum_rho

    def make_embpot(self,rho_on_grid):      
        
        density=np.zeros((rho_on_grid.shape[0],10))
        density[:,0] = rho_on_grid
        pot = self.__embfactory.get_potential(density) 
        
        if self.__e_field:
          fpot = self.__grid[:,fdir]*self.__fmax 
          fpot = np.ascontiguousarray(fpot, dtype=np.double)
          totpot = pot+fpot
          if nofde:
             totpot = fpot
          totpot = np.ascontiguousarray(totpot, dtype=np.double)
        else:
           totpot = pot 
        if (self.__debug):
            np.savetxt ("initialpot.txt", totpot)
        
        gridfunc = GridFuncFactory(self.__mol_obj,self.__grid,self.__basis_act)  
        vemb = gridfunc.matrix_from_grid(totpot)
         
        return vemb

##########################################################################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-gA","--geom_act", help="Specify active system (Angstrom) geometry (default: geomA.xyz)", required=False, 
            type=str, default="geomA.xyz")
    parser.add_argument("-gB","--geom_env", help="Specify frozen system (Angstrom) geometry (default: geomB.xyz)", required=False, 
            type=str, default="geomB.xyz")
    parser.add_argument("--embthresh", help="set EMB threshold (default = 1.0e-8)", required=False, 
            type=np.float64, default=1.0e-8)
    parser.add_argument("-d", "--debug", help="Debug on, prints debug info to debug_info.txt", required=False, 
            default=False, action="store_true")
    parser.add_argument("--modpaths", help="set berthamod and all other modules path [\"path1;path2;...\"] (default = ../src)", 
        required=False, type=str, default="../src")

    parser.add_argument("--act_obs", \
        help="Specify (Active system) basisset \"atomname1:basisset1,atomname2:basisset2,...\"", \
        required=True, type=str, default="")
    parser.add_argument("--act_func", 
	help="Specify exchange–correlation energy functional for active system available:(default=BLYP)", \
        type=str, default="BLYP")
    parser.add_argument("--env_obs", help="Specify the orbital basis set for the enviroment (default: AUG/ADZP)", required=False,
            type=str, default="AUG/ADZP")
    parser.add_argument("--env_func", help="Specify the function for the environment density (default: BLYP)", required=False,
            type=str, default="BLYP")
    parser.add_argument("--grid_opts", help="set gridtype (default: 2)",
        required=False, type=int, default=2)
    parser.add_argument("--grid_param", help="set grid parameter i.e grid accuracy in adf (default: [4.0,None])",
        required=False, type=str, default='4.0')
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
    

    args = parser.parse_args()
  
    for path in args.modpaths.split(";"):
        sys.path.append(path)

    modpaths = os.environ.get('PYBERTHA_MOD_PATH')

    if modpaths is not None :
        for path in modpaths.split(";"):
            sys.path.append(path)



    pyembopt = embedoption

    pyembopt.gridfname = args.gridfname

    pyembopt.debug = args.debug
    #pyembopt.thresh = args.thresh
    pyembopt.static_field = args.static_field
    pyembopt.fmax = args.fmax
    pyembopt.fdir = args.fdir
    pyembopt.activefile = args.geom_act
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
    
    #print("print grid param")
    #print(gparam)


    # psi4 minimal set up
    import psi4
    from molecule import Molecule
    mol_act = Molecule(args.geom_act)

    psi4.core.set_output_file('psi.out', False)
    
    molobj=psi4.geometry(mol_act.geometry())
    psi4.set_options({'basis': args.act_obs,
                      'puream': 'True',
                      'DF_SCF_GUESS': 'False',
                      'scf_type': 'DIRECT',
                      'dft_radial_scheme' : 'becke',
                      #'dft_radial_points': 49,
                      'dft_spherical_points' : 434,
                      'cubeprop_tasks': ['orbitals'],
                      'e_convergence': 1e-8,
                      'd_convergence': 1e-8})
    ene,wfn = psi4.energy(args.act_func,return_wfn=True)
    Cocc = np.array(wfn.Ca_subset('AO', 'OCC'))
    embed = emb_wrapper(molobj,pyembopt,args.act_obs)
    # here we need the active system density expressed on the grid
           
      
    # use GridDensityFactory
    grid = embed.grid()
    activesys = GridDensityFactory(molobj,grid,args.act_obs)  
        
    if isinstance(Cocc,np.ndarray):
          rho = activesys.from_Cocc(Cocc)
    elif Dmat is not None: #also ovap has to be != None
          rho = activesys.from_D(Dmat,self.__ovap)
    #check number of electron
    nel_ACT = np.matmul(rho,grid[:,3])
    print("n. el act sys: %.8f\n" % (nel_ACT*2.0))
    res = embed.make_embpot(rho)
