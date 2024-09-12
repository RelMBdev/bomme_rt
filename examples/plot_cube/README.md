packages needed 

-pymol
-pycube (https://github.com/BERTHA-4c-DKS/pycubescd)
-psi4
-bomme_rt
-pyadf (please follow instruction from pybertha/pyberthaembedrt repository) (required by bomme_rt)
-xcfun (required by bomme_rt/pyadf)


run a rt simulation setting in the input.inp file a reasonable long interval during which the cube are dumped every X steps i.e:

$ cat input.inp

time_int: 540.0
delta_t : 0.1
calc_flag :1
i_rest : 0
F_max : 0.1
freq_carrier : 0.1
sigma : 64.8
t0 : 393.3
imp_type : gauss_env
r0 : 8.
eta : 9.5
cthresh : 10.
qvec : 0.
prop_id : empc 

#################

export the environment variables (adjust to your needs) : 
$ source ~/bomme_rt/exportvar.sh

run a rt of 540 a.u with dt 0.1 (5400 total
steps), dump the cubes every 50 steps (--rt_cint option)
In this example the gaussian envelop is out of resonance.
The H2 first excitation (at "HF/6-311ppgss) is found roughly at 0.468 a.u, 
thus setting "freq_carrier : 0.1" in the input.inp corresponds to an off-resonant
driving field. The option in square brakets are optional and can be omitted.


python ~/bomme_rt/main.py -a 2 -o1 6-311ppgss -gA geom.xyz --real_time --eri nofit -f1 hf -f2 hf -d [--use_cap] --rt_cdump --rt_cint 50

use run_cub.sh: in the for loop you can tune the starting/final snapshot to be plotted

use plot_occnum.plot to plot some occupation number, i.e the diagonal elements of the td-density matrix (MO basis)

the pymol view can be tested and adjusted using test_cube_view.py
