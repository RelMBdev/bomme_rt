# bomme_rt
Block-Orthogonalized Manby-Miller mean-field embedding implementation of ground state DFT and RT-TDDFT

Include also Frozen Density Embedding either using ADF or Psi4 as providers of the
fronzen density, through the pyembed (class).


**clone numericaltest branch from https://github.com/BERTHA-4c-DKS/pybertha**
export COMMON_PATH=/home/matteo/bomme_rt/common
export MODS_PATH=/home/matteo/bomme_rt/mods

and

export PYBERTHA_MOD_PATH="$BerthaRootPath/pybertha/pyemb;$BerthaRootPath/xcfun/build/lib/python;$BerthaRootPath/pybertha/src;
$BerthaRootPath/pyadf/src;$BerthaRootPath/berthaingen/pybgen;$BerthaRootPath/pybertha/psi4rt;$BerthaRootPath/pybertha/pyberthaembed;
$BerthaRootPath/xcfun/build/lib/python/xcfun"


see exportvar.sh

For an fde/bommme calculation 

python3.9 ~/bomme_rt/main.py -d -gA H2O.xyz -gB NH3.xyz -f1 blyp -f2 blyp -o1 aug-cc-pvdz -o2 aug-cc-pvdz -z 0 -J --env_obs AUG/ADZP  
--env_func blyp --jobtype adf --grid_opts 2  --grid_param "4.0"  --embthresh  1.0e-8 --fde_max 10 --fde  --real_time --rt_HF_iexch --eri nofit > res.out

-gA : geometry file for the bomme ("core") subsystem. The Bomme subsystem can be partioned in two domains: the high-level theory and low-level theory (defining their associated functionals/basis sets)

-f1 functional defining the high-level theory domain 
-f2 functional defining the low-level theory domain 
-o1 high-level theory domain basis set
-o2 low-level theory domain basis set

-gB : geometry file of the FDE environment
--env_obs Psi4/ADF basis set for the FDE fragment
--env_func functional for the frozen density calculation

bomme geometry file looks like:
3                                            @  <- total nunber of atoms in the "bomme" molecule                            
#comment line                                @
O     1.568501    0.105892    0.000005       @ ]  high-level theory domain
H     0.606736   -0.033962   -0.000628       @ ] 
-frag-                                       @  <- separator
H     1.940519   -0.780005    0.000222       @ ]  low-level theory domain

fde geometry file:

 4
NH3
 N    -1.395591   -0.021564    0.000037
 H    -1.629811    0.961096   -0.106224
 H    -1.862767   -0.512544   -0.755974
 H    -1.833547   -0.330770    0.862307 


To cite this code in all its parts, please refer to the following papers:

M. De Santis, V. Vallet, A. S. P. Gomes, "Environment effects on X-ray absorption spectra with
quantum embedded real-time Time-dependent density functional theory approaches", Front. Chem.
10, 2022, doi:10.3389/fchem.2022.823246

M. De Santis, D. Sorbelli, V. Vallet, A. S. P. Gomes, L. Storchi, L. Belpassi "Frozen-Density Embedding
for including environmental effects in the Dirac-Kohn-Sham theory: an implementation based on
density fitting and prototyping techniques", J. Chem. Theory Comput. 2022, 18, 10, 5992–6009

M. De Santis, L. Belpassi, C. R. Jacob, A. S. P. Gomes, F. Tarantelli, L. Visscher, L. Storchi,
“Environmental effects with Frozen Density Embedding in Real-Time Time-Dependent Density
Functional Theory using localized basis functions”, J. Chem. Theory Comput. 2020, 16, 9, 5695–5711
