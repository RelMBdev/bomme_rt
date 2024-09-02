"""
real-time evolution: 
  
"""
__authors__   =  "Matteo De Santis"
__credits__   =  ["Matteo De Santis"]

__copyright__ = "(c) 2020, MDS"
__license__   = "GPL-v3"
__date__      = "2020-03-01"
import os
import sys
import numpy as np
import psi4
from pkg_resources import parse_version

import time
from datetime import datetime

sys.path.insert(0, "../common")
modpaths = os.environ.get('COMMON_PATH')

if modpaths is not None :
    for path in modpaths.split(";"):
        sys.path.append(path)
import bo_helper
from localizer import Localizer 
import rtutil
import fde_utils
import cube_util
##################################################################

# mo_select : string containing two substrings "-2; list_occ & list_vist"
# selective_pert : bool.  Turn on selective perturbation. See Repisky M et al 2015
# local_basis : bool . Use a localized basis instead of ground-state MOs
# exA_only : excite only A subsystem; either using local_bas or selective_pert or numerical approach
# cube_data is a data class that store the cube parameters and the dumping fruquency
def run_rt_iterations(iter_opts, field_opts, bset, bsetH, wfn_bo, embmol, direction, mo_select, selective_pert, local_basis, exA_only,\
                     numpy_mem,debug,pyembopt=None, capflag = False, cube_data=None):
    # TODO: the parameter of the propagation (e.g delta_t, n_iter are derivede locally) can be passed from outside
    numbas = wfn_bo['nbf_tot']
    nbf_A = wfn_bo['nbf_A']
    #ndocc refers to the total number of doubly occupied MO
    ndocc = wfn_bo['ndocc']
    func_h = wfn_bo['func_h']
    func_l = wfn_bo['func_l']
    jkclass = wfn_bo['jkfactory']
    exmodel = wfn_bo['exmodel']

    #a minimalistic RT propagation code
    if mo_select != "":
       molist = mo_select.split("&")
       occlist = molist[0].split(";")
       occlist = [int(m) for m in occlist]
       do_weighted = occlist.pop(0)
       virtlist = molist[1].split(";")
       virtlist = [int(m) for m in virtlist]
    else:
       occlist = None
       virtlist = None
       do_weighted = None

    #for TNO analysis
    #tlist = args.tlist.split("&")
    #mlist = tlist[0].split(";")
    #mlist = [int(m) for m in mlist]
    #plist = tlist[1].split(";")
    #plist = [int(m) for m in plist]

    if (do_weighted == -2):
      if debug:
        print("Selected transitions from %s to %s MOs"% (str(occlist), str(virtlist)))

    ### frequency bin container ###
    #freqlist = args.freq.split(";")
    #freqlist = [np.float_(m) for m in freqlist]
    #if ( (freqlist == np.zeros(len(freqlist))).all() and args.ftden ):
    #         raise Exception("Check list of frequencies for ftden")
    ###

    #field_opts, calc_params = rtutil.set_params(inputfname)
    calc_params = iter_opts[2]  # pick from the tuple
    init_iter = iter_opts[0]
    end_iter = iter_opts[1]

    if field_opts['imp_type'] == 'analytic' :
        analytic = True
    else:
        analytic = False
    
    prop_id = calc_params['prop_id']
    #dt in a.u
    dt =  calc_params['delta_t']
    #time_int in atomic unit
    time_int=calc_params['time_int']
    niter=int(time_int/dt)

    ## get matrices on BO basis
    Htilde = np.array(wfn_bo['Hcore'])
    Ftilde = np.array(wfn_bo['Fock'])
    Dtilde = np.array(wfn_bo['Dmtx'])
    C = np.asarray(wfn_bo['Ccoeff'])
    U = np.array(wfn_bo['Umat'])
    Stilde = np.array(wfn_bo['Ovap'])

    #initialize the mints object
    mints = psi4.core.MintsHelper(bset)
    #intialize fock_factory
    from Fock_helper import fock_factory
    Hcore = np.asarray(mints.ao_potential()) + np.asarray(mints.ao_kinetic())

    fock_base = fock_factory(jkclass,Hcore,Stilde, \
                            funcname=func_l,basisobj=bset,exmodel=exmodel)
    if (not pyembopt.nofde) or pyembopt.static_field or pyembopt.extern_flag:
       
    # initialize the embedding engine
       embed = fde_utils.emb_wrapper(embmol,pyembopt,bset,ovap=np.array(mints.ao_overlap()))
       
       if not pyembopt.nofde:
       # define a temporaty Cocc_AO
          Cocc_tmp = np.matmul(U,C[:,:ndocc])
          # here we need the active system density expressed on the grid
          embed.set_density(Cocc_tmp)
          nel_ACT =embed.rho_integral()
          print("N.el active system : %.8f\n" % nel_ACT)
       
       Vemb = embed.make_embpot()
        
       # set the embedding potential
       fock_base.set_vemb(Vemb)
    else:
       embed = None
    
    #dip_mat is transformed in the BO basis
    dipole=mints.ao_dipole()
    dip_mat=np.matmul(U.T,np.matmul(np.array(dipole[direction]),U))

    dip_dir = [0,1,2]
    dip_dict={'0' : 'x', '1' : 'y', '2' : 'z'}
    dip_dir.pop(direction)
    dipmat_list = []
    numocc_list = []   # to collect the diagonal element of Dp_ti (MO)
    dipmat_list.append(dip_mat)
    # the list contains the dipole matrices in the order: the matrix of the dipole-component in the boost direction,
    # and the additional matrices corresponding to the perpendicular directions to the boost
    for i in dip_dir:
         dipmat_list.append(np.matmul(U.T,np.matmul(np.array(dipole[i]),U)))
    
    dmat_offdiag=dipmat_list[1:]
    
    test=np.matmul(C.T,np.matmul(Stilde,C))
    print("in BO basis C^T Stilde C = 1 : %s" % np.allclose(test,np.eye(C.shape[0])))
   


    outfile  = open("err.txt", "w")

    #initialize the real-time object
    from rt_base import real_time
    rt_prop = real_time(Dtilde, Ftilde, fock_base, ndocc, bset, Stilde, field_opts, dt, C, dipmat_list,\
                           out_file=outfile,  basis_acc = bsetH, func_acc=func_h,U=U,local_basis=local_basis,\
                                             exA_only=exA_only,occlist=occlist, virtlist=virtlist, prop_type=prop_id,debug=debug)
    from operator_ongrid  import function_cap_xyz
    # function_cap_xyz(grid_mtx, ndim,mol, basisobj, r0=9.3, eta=4., thresh=10.,\
    #                                           file_out=sys.stderr, debug=False)

    rt_prop.do_cap(capflag)
    if capflag:
       # get CAP params from pulse_opts dictionary
       cap_r0 = field_opts['r0']
       cap_eta = field_opts['eta']
       cap_thresh = field_opts['cthresh']
       
       # build a grid on the fly
       functional_tmp = psi4.driver.dft.build_superfunctional("svwn", True)[0] # True states that we're working with a restricted system
       Vpot_tmp       = psi4.core.VBase.build(bset, functional_tmp, "RV")         # This object contains different methods associated with DFT methods and the grid.
                                                                               # "RV" states that this is a restricted system consystent with 'functional'
       Vpot_tmp.initialize() # We initialize the object

       # The grid (and weights) can then be extracted from Vpot.
       x, y, z, w = Vpot_tmp.get_np_xyzw()

       xs = psi4.core.Vector.from_array(x)  # as psi4.Vector
       ys = psi4.core.Vector.from_array(y)
       zs = psi4.core.Vector.from_array(z)
       ws = psi4.core.Vector.from_array(w)
       grid_list = (xs,ys,zs,ws)
       
       Vpot_tmp.finalize()
       cap_mtx_AO = function_cap_xyz(grid_list, bset.nbf(), embmol, bset, r0=cap_r0, eta=cap_eta, thresh=cap_thresh)

       capobj = rt_prop.set_CAP(cap_mtx_AO) # AO->MO
    else:   # for debug and testing
       capobj = rt_prop.set_CAP(None)  
     
    if pyembopt is not None:
          rt_prop.embedding_init(embed,pyembopt)

    print("testing initial density matrix\n")
    try :
      C_inv=np.linalg.inv(C)
    except scipy.linalg.LinAlgError:
      print("Error in np.linalg.inv")

    # in an orthonormal basis Dtilde should be diagonal with oo=1, vv=0
    ODtilde=np.matmul(C_inv,np.matmul(Dtilde,C_inv.T))
    Dp_ti = rt_prop.get_Dmat()   # the density matrix (t=-inf) in the basis of MOs (diagonal : [1]_occ , [0]_virt)
    if debug:
       print("D[BO] is diagonal in the orbital basis (tol=1.0e-14): %s" % np.allclose(Dp_ti,ODtilde,atol=1.0e-14))
       diff_test = ODtilde - Dp_ti
       print("max diff |ODtilde - Dtilde| : %.5e\n" % np.max(np.abs(diff_test)) )

    #nuclear dipole for non-homonuclear molecules
    Ndip= embmol.nuclear_dipole()
    Ndip_dir=Ndip[direction]
    #for the time being the nuclear dipole contribution to the dipole and energy(nuclear)
    # is not considered
    Ndip_dir = 0.0

    Enuc_list=[]
    Enuc = embmol.nuclear_repulsion_energy()
    print("N. iterations: %i\n" % niter)

    #set the functional type
    #if (calc_params['func_type'] == 'svwn5'):
    #   func = svwn5_func
    #else:
    #   func=calc_params['func_type']
    
    ###
    
    #if ftden:
    #    #tdmat = np.matmul(U,np.matmul((Dtilde-Dtilde),np.conjugate(U.T)))
    #    tdmat = Dtilde-Dtilde
    #    for n in  binlist:
    #       td_container.append(tdmat * np.exp( -2.0j*np.pi*0*n/Ns))

    ###
    if cube_data.dump:
      # O param and N param  control the cube   
      ovap_AO = np.asarray(mints.ao_overlap())
      os.mkdir("td.ground")
      # legacy ?
      #xp,yp,zp,wp,Nparam,Oparam = cube_util.setgridcube(embmol,cube_data.margin,cube_data.Dx)
      #phi_mat,lpos,dummy=cube_util.phi_builder(embmol,xp,yp,zp,wp,bset)

      cube_util.denstocube(embmol, bset, rt_prop.get_Dmat('AO'),\
                                "density", cube_data.margin, cube_data.Dx)
      
      os.rename("density.cube","td.ground/density.cube")
    ###

    print("analytic : %i" % analytic)
    if (analytic):
       # the returned matrices are not used anyway 
       Dp_init_test , Da_test = rt_prop.init_boost(selective_pert,debug=True) 
    
    #containers
    field_list = []
    weighted_dip = []
    time_list = []
    td_container = []
    
    #for molecules with permanent nuclear dipole add Enuc_list[k] to ene
    #note that : i.e in the case of linear symmetric molecule the nuclear dipole can be zero depending
    #on the choice of reference coordinates system

    #for analysis
    dipmo_mat=np.matmul(np.conjugate(C.T),np.matmul(dip_mat,C))

    
    #weighted dipole setup orbital list
    if (do_weighted == -2) and isinstance(virtlist,list):
      if virtlist[0] ==-99 and ((not local_basis ) or (not selective_pert)) :
        virtlist=[]
        for m in range(ndocc,numbas):
            virtlist.append(m+1)
      else:
        virtlist = rt_prop.virtlist()

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

    print("RT-propagation started at: %s" % dt_string)
    print('Entering in the first step of propagation')
    start =time.time()
    cstart =time.process_time()


    for j in range(init_iter,end_iter+1): # start from i=0
        # for test
        fock_mid_backwd  =  rt_prop.get_midpoint_mtx()

        rt_prop()

        diff_midF=rt_prop.get_midpoint_mtx()-fock_mid_backwd

        norm_diff=np.linalg.norm(diff_midF,'fro')
        outfile.write('||Fock_mid(%i +1/2)- Fock_mid(%i-1/2)||_fro: %.8f\n' % (j,j,norm_diff))

        ###
        #if args.ftden:
        #   #tdmat =np.matmul(U,np.matmul((D_ti-Dtilde),np.conjugate(U.T)))
        #   tdmat =D_ti-Dtilde
        #
        #   count=0
        #   for n in  binlist:
        #      tmp=td_container[count]
        #      tmp+=tdmat * np.exp( -2.0j*np.pi*j*n/Ns)
        #      td_container[count]=tmp
        #      count+=1
        if cube_data.dump:
           if ( ( j % cube_data.int_cdump ) == 0 ) :
              path="td."+str(j).zfill(7)
              os.mkdir(path)
              cube_util.denstocube(embmol, bset, rt_prop.get_Dmat('AO'), "density",\
                             cube_data.margin, cube_data.Dx)
              os.rename("density.cube",path+"/density.cube")

        Ah=np.conjugate(rt_prop.get_midpoint_mtx().T)
        outfile.write('Fock_mid hermitian: %s\n' % np.allclose(rt_prop.get_midpoint_mtx(),Ah))
        
        #for debug
        if debug:
          dipval = rt_prop.get_dipole()[0]
          outfile.write('Dipole  %.8f %.15f\n' % (j*dt, 2.00*dipval[j].real))
        
        numocc_list +=[np.diagonal(Dp_ti).real]

        if (do_weighted == -2):
          #weighted dipole 
          res = rtutil.dipoleanalysis(dipmo_mat,Dp_ti,ndocc,occlist,virtlist,debug)
          weighted_dip.append(res)
        
        func_ti = rt_prop.get_extfield()
        Enuc_list.append(-func_ti*Ndip_dir) #just in case of non-zero nuclear dipole
        field_list.append(func_ti)
       
        #update D_ti and Dp_ti for the next step
        
        D_ti=rt_prop.get_Dmat('AO')
        Dp_ti=rt_prop.get_Dmat()

    outfile.close()
    end=time.time()
    cend = time.process_time()
    ftime  = open("timing.txt", "w")
    ftime.write("time for %i time iterations : (%.3f s, %.3f s)\n" %(niter+1,end-start,cend-cstart))
    ftime.close()
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("RT-propagation ended at: %s" % dt_string)
    t_point=np.linspace(0.0,niter*dt,niter+1)

    dip_list, dip_perp0, dip_perp1 = rt_prop.get_dipole()
    


    dip_t=2.00*np.array(dip_list).real
    ene_t=np.array(rt_prop.get_energy()).real
    imp_t=np.array(field_list)

    if (do_weighted == -2):
      wd_dip=2.00*np.array(weighted_dip).real
      np.savetxt('weighteddip.txt', np.c_[t_point,wd_dip], fmt='%.12e')
    # convert to np.array
    numocc_list = np.array(numocc_list)
    np.savetxt('occnum.txt', np.c_[t_point,numocc_list], fmt='%.5e')
    np.savetxt('dipole-'+2*dip_dict[str(direction)]+'.txt', np.c_[t_point,dip_t], fmt='%.12e')
    # dipole_ij, i denotes the ith component of dipole vector, j denote the direction of the field
    np.savetxt('dipole-'+dip_dict[str(dip_dir[0])]+dip_dict[str(direction)]+'.txt', np.c_[t_point,2.00*np.array(dip_perp0).real], fmt='%.12e')
    np.savetxt('dipole-'+dip_dict[str(dip_dir[1])]+dip_dict[str(direction)]+'.txt', np.c_[t_point,2.00*np.array(dip_perp1).real], fmt='%.12e')
    np.savetxt('imp.txt', np.c_[t_point,imp_t], fmt='%.12e')
    np.savetxt('ene.txt', np.c_[t_point,ene_t], fmt='%.12e')
