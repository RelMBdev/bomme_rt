"""
Aux function for real-time propagation: 
  set_input()
  exp_opmat()
  
"""
__authors__   =  "Matteo De Santis"
__credits__   =  ["Matteo De Santis"]

__copyright__ = "(c) 2020, MDS"
__license__   = "BSD-3-Clause"
__date__      = "2020-03-01"
import os
import sys
import numpy as np
import psi4
from pkg_resources import parse_version


##################################################################

def exp_opmat(mat,dt):
    #first find eigenvectors and eigenvalues of F mat
    try:
       w,v=np.linalg.eigh(mat)
    except np.linalg.LinAlgError:
        print("Error in numpy.linalg.eigh of inputted matrix")
        return None

    diag=np.exp(-1.j*w*dt)

    dmat=np.diagflat(diag)

    # for a general matrix Diag = M^(-1) A M
    # M is v
    #try:
    #   v_i=np.linalg.inv(v)
    #except np.linalg.LinAlgError:
    #   return None

    # transform back
    #tmp = np.matmul(dmat,v_i)
    tmp = np.matmul(dmat,np.conjugate(v.T))

    #in an orthonrmal basis v_inv = v.H

    mat_exp = np.matmul(v,tmp)

    return mat_exp

##################################################################

def set_input(fgeom):
  geomobj = str()
  with open(fgeom,"r") as f:
   numat=f.readline()
   species=f.readline()
   for line in f:
    geomobj +=str(line)
  geomobj += "symmetry c1" +"\n" +"no_reorient" +"\n" +"no_com"
  print("GEOMETRY in angstrom:\n")
  print(geomobj)
  mol =psi4.geometry(geomobj)
  f.close()
  return geomobj, mol,numat,species
###################################################################

def set_params(fname):
    my_dict = {}
    with open(fname) as fileobj:
      for line in fileobj:
        key, value = line.split(":")
        my_dict[key.strip()] = value.strip()
    fileobj.close()
    imp_params = {}
    imp_params['Fmax'] = float(my_dict['F_max'])
    imp_params['w'] = float(my_dict['freq_carrier'])
    imp_params['s'] = float(my_dict['sigma'])
    imp_params['t0'] = float(my_dict['t0']) 
    imp_params['imp_type'] = my_dict['imp_type']
    
    calc_params ={}    
    calc_params['time_int']=float(my_dict['time_int'])
    calc_params['delta_t']=float(my_dict['delta_t'])
    #calc_params['func_type'] =my_dict['func_type'] 
    #calc_params['method']=my_dict['method_type']
    return imp_params,calc_params
##################################################################

def kick (Fmax, w, t, t0=0.0, s=0.0):

    w = 0.0
    func = 0.0
    if t > 0:
      func = 0.0
    elif (t == 0.0):
      func = Fmax
    else:
      return None
      # t out of range

    return func

##################################################################

def gauss_env (Fmax, w, t, t0=3.0, s=0.2):

    #s : sqrt(variance) of gaussian envelop
    #w : the frequency of the carrier
    #Fmax : the maximum amplitude of the field
    #t0 :center of Gaussian envelope (in au time)

    func=Fmax*np.exp(-(t-t0)**2.0/(2.0*s**2.0))*np.sin(w*t)

    return func

##################################################################
#similar to gauss_env, w 
def narrow_gauss (Fmax, w, t, t0=20.0, s=1.0):

    #s : sqrt(variance) of gaussian envelop
    #w : the frequency of the carrier (not needed)
    #Fmax : the maximum amplitude of the field
    #t0 :center of Gaussian envelope (in au time)

    func=Fmax*np.exp(-(t-t0)**2.0/(2.0*s**2.0))

    return func

##################################################################

def envelope (Fmax, w, t, t0=0.0, s=0.0):

   if (t >= 0.0 and t<= 2.00*np.pi/w):
      Amp =(w*t/(2.00*np.pi))*Fmax
   elif (t > 2.00*np.pi/w and t < 4.00*np.pi/w):
      Amp = Fmax
   elif ( t >= 4.00*np.pi/w and t <= 6.00*np.pi/w):
      Amp = (3.00 -w*t/(2.00*np.pi))*Fmax
   elif ( t > 6.00*np.pi/w):
      Amp = 0.0
   else :

      Amp = 0.0

   func = Amp*np.sin(w*t)

   return func

##################################################################

def sin_oc (Fmax, w, t, t0=0.0, s=0.0):

   # 1-oscillation-cycle sinusoid
   if (t >= 0.0 and t<= 2.00*np.pi/w):
      Amp = Fmax
   else:
      Amp = 0.0

   func = Amp*np.sin(w*t)

   return func

##################################################################

def cos_env(Fmax, w, t, t0=0.0, n=20.0):

   #define the period (time for an oscillation cycle)
   #n is the number of oscillation cycles in the
   # envelope
   oc=2.00*np.pi/w
   s=oc*n/2.0
   if (abs(t-s)<= s):
      func=np.sin(w*t)*Fmax*(np.cos(np.pi/2.0/s*(s-t)))**2.0
   else:
      func=0.0

   return func

##################################################################

def lin_cos(Fmax, w, t, t0=0.0, s=0.0):
        
    if ( t <= 2.0*np.pi/w and t>=0.0):
      func=Fmax*w*t/2.0/np.pi*np.cos(w*t)
    else:
      func=Fmax*np.cos(w*t)

    return func
##################################################################

def imp_field(Fmax, w, t, t0=0.1, s=0.0):
    tau=t0/7.0
    func=Fmax*np.sin(w*t)*np.exp(-((t-t0)**2.0)/(2.0*tau*tau))/np.sqrt(2.0*np.pi*tau*tau)

    return func
##################################################################

def analytic(Fmax, w, t, t0=0.0, s=0.0):

   func = 0.0

   return func

##################################################################
##################################################################

funcswitcher = {
    "kick": kick,
    "gauss_env": gauss_env,
    "narrow_gauss": narrow_gauss,
    "envelope": envelope,
    "sin_oc": sin_oc,
    "cos_env": cos_env,
    "lin_cos": lin_cos,
    "analytic": analytic,
    "imp_field": imp_field
     }
   
##################################################################

def mo_fock_mid_forwd_eval(D_ti,fock_mid_ti_backwd,i,delta_t,H,I,U,dipole,\
                               C,C_inv,S,nbf,imp_opts,func_h,func_l,fout,basisset,bsetH,exA=False):

    t_arg=np.float_(i)*np.float_(delta_t)
    
    func = funcswitcher.get(imp_opts['imp_type'], lambda: kick)
    
    pulse = func(imp_opts['Fmax'], imp_opts['w'], t_arg,\
                        imp_opts['t0'], imp_opts['s'])

    #D_ti is in AO basis
    #transform in the MO ref basis
    Dp_ti= np.matmul(C_inv,np.matmul(D_ti,np.conjugate(C_inv.T)))
    
    k=1
    
    Eh_i,Exclow_i,ExcAAhigh_i,ExcAAlow_i,fock_mtx=get_BOFockRT(D_ti,H,I,U,func_h,func_l,basisset,bsetH)
    #DEBUG
    #ExcAAhigh_i=0.0
    #ExcAAlow_i=0.0
    
    #Eh_i,Exclow_i,fock_mtx=get_Fock(D_ti,H,I,func_l,basisset)
    #print('fockmtx s out of loop max diff: %.12e\n' % np.max(tfock_mtx-fock_mtx))
    #add -pulse*dipole
    if exA:
       dimA = bsetH.nbf()
       dmat=np.zeros_like(dipole)
       dmat[:dimA,:dimA] = dipole[:dimA,:dimA]
       dipole = dmat
    fock_ti_ao = fock_mtx - (dipole*pulse)

    #if i==0:
    #    print('F(0) equal to F_ref: %s' % np.allclose(fock_ti_ao,fock_mid_ti_backwd))
    
    #initialize dens_test !useless
    dens_test=np.zeros(Dp_ti.shape)

    # set guess for initial fock matrix
    fock_guess = 2.00*fock_ti_ao - fock_mid_ti_backwd
    #if i==0:
    #   print('Fock_guess for i =0 is Fock_0: %s' % np.allclose(fock_guess,fock_ti_ao))
    #transform fock_guess in MO basis
    while True:
        fockp_guess=np.matmul(np.conjugate(C.T),np.matmul(fock_guess,C))
        u=exp_opmat(fockp_guess,delta_t)
        #u=scipy.linalg.expm(-1.j*fockp_guess*delta_t) ! alternative routine
        test=np.matmul(u,np.conjugate(u.T))
    #print('U is unitary? %s' % (np.allclose(test,np.eye(u.shape[0]))))
        if (not np.allclose(test,np.eye(u.shape[0]))):
            Id=np.eye(u.shape[0])
            diff_u=test-Id
            norm_diff=np.linalg.norm(diff_u,'fro')
            print('from fock_mid:U deviates from unitarity, |UU^-1 -I| %.8f' % norm_diff)
    #evolve Dp_ti using u and obtain Dp_ti_dt (i.e Dp(ti+dt)). u i s built from the guess fock
    #density in the orthonormal basis
        tmpd=np.matmul(Dp_ti,np.conjugate(u.T))
        Dp_ti_dt=np.matmul(u,tmpd)
    #backtrasform Dp_ti_dt
        D_ti_dt=np.matmul(C,np.matmul(Dp_ti_dt,np.conjugate(C.T)))
    #build the correspondig Fock : fock_ti+dt
        
        #DEBUG
        #dum1,dum2,fock_mtx=get_Fock(D_ti_dt,H,I,func_l,basisset)
        dum0,dum1,dum2,dum3,fock_mtx=get_BOFockRT(D_ti_dt,H,I,U,func_h,func_l,basisset,bsetH)
        #print('fockmtx s in loop max diff: %.12e\n' % np.max(tfock_mtx-fock_mtx))
        #update t_arg+=delta_t
        pulse_dt = func(imp_opts['Fmax'], imp_opts['w'], t_arg+delta_t,\
                        imp_opts['t0'], imp_opts['s'])
        fock_ti_dt_ao=fock_mtx -(dipole*pulse_dt)
        fock_inter= 0.5*fock_ti_ao + 0.5*fock_ti_dt_ao
    #update fock_guess
        fock_guess=np.copy(fock_inter)
        if k >1:
        #test on the norm: compare the density at current step and previous step
        #calc frobenius of the difference D_ti_dt_mo_new-D_ti_dt_mo
            diff=D_ti_dt-dens_test
            norm_f=np.linalg.norm(diff,'fro')
            if norm_f<(1e-6):
                tr_dt=np.trace(np.matmul(S,D_ti_dt))
                fout.write('converged after %i interpolations\n' % (k-1))
                fout.write('i is: %d\n' % i)
                fout.write('norm is: %.12f\n' % norm_f)
                fout.write('Trace(D)(t+dt) : %.8f\n' % tr_dt.real)
                break
        dens_test=np.copy(D_ti_dt)
        k+=1
        if k > 20:
         raise Exception("Numember of iterations exceeded (k>20)")
    return Eh_i,Exclow_i,ExcAAhigh_i,ExcAAlow_i,pulse,fock_ti_ao,fock_inter
##################################################################

def run_rt_iterations(inputfname, bset, wfn_bo, embmol, mo_select, debug):
	#a minimalistic RT propagation code
	molist = mo_select.split("&")
	occlist = molist[0].split(";")
	occlist = [int(m) for m in occlist]
	do_weighted = occlist.pop(0)
	virtlist = molist[1].split(";")
	virtlist = [int(m) for m in virtlist]

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

	imp_opts, calc_params = set_params(inputfname)

	if imp_opts['imp_type'] == 'analytic' :
	    analytic = True
	else:
	    analytic = False

	#dt in a.u
	dt =  calc_params['delta_t']
	#time_int in atomic unit
	time_int=calc_params['time_int']
	niter=int(time_int/dt)

	## get matrices on BO basis
	Ftilde = np.array(wfn_bo['Fock'])
	Dtilde = np.array(wfn_bo['Dmtx'])
	C = np.asarray(wfn_bo['Ccoeff']
	U = np.array(wfn_bo['Umat']
	Stilde = np.array(wfn_bo['Ovap']

	#initialize the mints object

	#dip_mat is transformed in the BO basis
	dip_mat=np.matmul(U.T,np.matmul(np.array(dipole[direction]),U))

	dip_dir = [0,1,2]
	dip_dict={'0' : 'x', '1' : 'y', '2' : 'z'}
	dip_dir.pop(direction)
	dmat_offdiag = []

	for i in dip_dir:
	     dmat_offdiag.append(np.matmul(U.T,np.matmul(np.array(dipole[i]),U)))

	test=np.matmul(C.T,np.matmul(Stilde,C))
	print("in BO basis C^T Stilde C = 1 : %s" % np.allclose(test,np.eye(C.shape[0])))

	if args.locbasis:
		localbas=Localizer(Dtilde,np.array(Stilde),counter)
		localbas.localize()
		#unsorted orbitals
		unsorted_orbs=localbas.make_orbitals()
		#the projector P
		Phat=np.matmul(Stilde.np[:,:counter],np.matmul(S11_inv,Stilde.np[:counter,:]))
		#The RLMO are ordered
		# by descending value of the locality parameter.
		sorted_orbs = localbas.sort_orbitals(Phat)
		#the occupation number and the locality measure
		locality,occnum=localbas.locality()
		#save the localization parameters and the occupation numbers  
		np.savetxt('locality_rlmo.out', np.c_[locality,occnum], fmt='%.12e')
		#check the occupation number of ndimA=counter MOs (of A).
		occA = int(np.rint( np.sum(occnum[:counter]) ))
		print("ndocc orbitals (A) after localization: %i\n" % occA)
		#set the sorted as new basis 
		C=sorted_orbs
		#use the density corresponding to (sorted) localized orbitals
		Dp_0 = np.diagflat(occnum)
	else:
	  Dp_0=np.zeros((numbas,numbas))
	  for num in range(int(ndocc)):
	      Dp_0[num,num]=1.0
	try :
	  C_inv=np.linalg.inv(C)
	except scipy.linalg.LinAlgError:
	  print("Error in np.linalg.inv")

	# in an orthonormal basis Dtilde should be diagonal with oo=1, vv=0
	ODtilde=np.matmul(C_inv,np.matmul(Dtilde,C_inv.T))
	print("Dtilde is diagonal in the orbital basis: %s" % np.allclose(Dp_0,ODtilde,atol=1.0e-14))


	#nuclear dipole for non-homonuclear molecules
	Ndip= embmol.nuclear_dipole()
	Ndip_dir=Ndip[direction]
	#for the time being the nuclear dipole contribution to the dipole and energy
	# is not considered
	Ndip_dir = 0.0

	Enuc_list=[]
	print("N. iterations: %i\" % niter)

	#set the functional type
	#if (calc_params['func_type'] == 'svwn5'):
	#   func = svwn5_func
	#else:
	#   func=calc_params['func_type']

	fo  = open("err.txt", "w")
	print("analytic : %i" % analytic)
	if (analytic):
	   print('Perturb density with analytic delta')
	   # set the perturbed density -> exp(-ikP)D_0exp(+ikP)
	   k = imp_opts['Fmax']
	   if   args.exciteA == 1:
	     print("Excite only the A subsystem (High level)\n")
	     #excite only A to prevent leak of the hole/particle density across the AA-BB  boundary
	     #select AA block in BO basis
	     tmpA=np.zeros_like(dip_mat)    
	     tmpA[:counter,:counter]=dip_mat[:counter,:counter]
	     dip_mo=np.matmul(np.conjugate(C.T),np.matmul(tmpA,C))
	   else: 
		#dip_mat is transformed to the reference MO basis
		print("Dipole matrix is transformed to the MO basis\n")
		print("Local basis: %s\n" % args.locbasis)
		dip_mo=np.matmul(np.conjugate(C.T),np.matmul(dip_mat,C))
	   if args.locbasis and (virtlist[0] == -99) and args.selective_pert:
		#use occnum to define a virtlist
		virtlist=[]
		for m in range(numbas):
		  if np.rint(np.abs(occnum))[m] < 1.0: 
		    virtlist.append(m+1)
		dip_mo=util.dipole_selection(dip_mo,-1,ndocc,occlist,virtlist,fo,debug)
	   elif args.selective_pert:
		 dip_mo=util.dipole_selection(dip_mo,virtlist[0],ndocc,occlist,virtlist,fo,debug)
	       
	   u0=util.exp_opmat(dip_mo,np.float_(-k))
	   Dp_init= np.matmul(u0,np.matmul(Dp_0,np.conjugate(u0.T)))
	   func_t0=k
	   #backtrasform Dp_init
	   D_init=np.matmul(C,np.matmul(Dp_init,np.conjugate(C.T)))
	   Dtilde = D_init
	   Dp_0 = Dp_init 
	   
	   #J0p,Exc0p,F_t0=util.get_Fock(D_ti,H,I,func,basisset)
	   #if (func == 'hf'):                                  
	   #    testene = np.trace(np.matmul(D_init,(H+F_t0)))  
	   #else:                                               
	   #    testene = 2.00*np.trace(np.matmul(D_init,H))+J0p+Exc0p
	   #print('trace D(0+): %.8f' % np.trace(Dp_init).real)       
	   #print(testene+Nuc_rep)                                    

	#containers
	ene_list = []
	dip_list = []
	dip_offdiag0 = []
	dip_offdiag1 = []
	imp_list=[]
	weighted_dip = []
	time_list = []
	td_container = []
	dip_list_p = []
	#for molecules with permanent nuclear dipole add Enuc_list[k] to ene
	#note that : i.e in the case of linear symmetric molecule the nuclear dipole can be zero depending
	#on the choice of reference coordinates system

	dipmo_mat=np.matmul(np.conjugate(C.T),np.matmul(dip_mat,C))

	now = datetime.now()
	dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

	print("RT-propagation started at: %s" % dt_string)
	print('Entering in the first step of propagation')
	start =time.time()
	cstart =time.process_time()
	Eh0,Exclow0,ExcAAhigh0,ExcAAlow0,func_t0,F_t0,fock_mid_init=util.mo_fock_mid_forwd_eval(Dtilde,Ftilde,\
				0,np.float_(dt),Htilde,I,U,dip_mat,C,C_inv,Stilde,numbas,imp_opts,func_h,func_l,fo,bset,bsetH,args.exciteA)

	#DEBUG
	#import bo_helper
	#dum1,dum2,dum3,dum4,F_t0=bo_helper.get_BOFockRT(Dtilde, Htilde, I,U, func_h, func_l, bset,bsetH)
	diff_midF=fock_mid_init-Ftilde

	norm_diff=np.linalg.norm(diff_midF,'fro')
	fo.write('||Fock_mid(%i+1/2)-Fock_mid(%i-1/2)||_fro : %.8f\n' % (0,0,norm_diff))

	###
	#if ftden:
	#    #tdmat = np.matmul(U,np.matmul((Dtilde-Dtilde),np.conjugate(U.T)))
	#    tdmat = Dtilde-Dtilde
	#    for n in  binlist:
	#       td_container.append(tdmat * np.exp( -2.0j*np.pi*0*n/Ns))
	#if args.dump:
	#  os.mkdir("td.0000000")
	#  xs,ys,zs,ws,N,O = cube_util.setgridcube(mol,L,D)
	#  phi,lpos,nbas=cube_util.phi_builder(mol,xs,ys,zs,ws,basis_set)
	#  cube_util.denstocube(phi,Da,S,ndocc,mol,"density",O,N,D)
	#  os.rename("density.cube","td.0000000/density.cube")
	###

	### TO BE REWORKED
	#check the Fock
	if debug :
	     print('F_t0 is equal to GS Ftilde : %s' % np.allclose(Ftilde,F_t0,atol=1.0e-12))
	     print('Max of  abs(F_t0 - GS Ftilde) : %.12e\n' % np.max(np.abs(F_t0-Ftilde)))
	#check hermicity of fock_mid_init

	Ah=np.conjugate(fock_mid_init.T)
	fo.write('Fock_mid hermitian: %s\n' % np.allclose(fock_mid_init,Ah))

	#propagate D_t0 -->D(t0+dt)
	#
	#fock_mid_init is transformed in the MO ref basis
	fockp_mid_init=np.matmul(np.conjugate(C.T),np.matmul(fock_mid_init,C))

	#u=scipy.linalg.expm(-1.j*fockp_mid_init*dt)
	u=util.exp_opmat(fockp_mid_init,np.float_(dt))

	temp=np.matmul(Dp_0,np.conjugate(u.T))

	Dp_t1=np.matmul(u,temp)

	#check u if unitary
	test_u=np.matmul(u,np.conjugate(u.T))
	fo.write('U is unitary :%s\n' % np.allclose(test_u,np.eye(u.shape[0])))

	fock_mid_backwd=np.copy(fock_mid_init)

	#backtrasform Dp_t1

	D_t1=np.matmul(C,np.matmul(Dp_t1,np.conjugate(C.T)))

	ene_list.append(Eh0 + Exclow0 + ExcAAhigh0 -ExcAAlow0 + Enuc + 2.0*np.trace(np.matmul(Dtilde,Htilde)) )

	dip_list.append(np.trace(np.matmul(Dtilde,dip_mat)))
	dip_offdiag0.append(np.trace(np.matmul(Dtilde,dmat_offdiag[0])))
	dip_offdiag1.append(np.trace(np.matmul(Dtilde,dmat_offdiag[1])))
	#weighted dipole
	if (do_weighted == -2):
	  if virtlist[0] ==-99 and ((not args.locbasis ) or (not args.selective_pert)) :
	    virtlist=[]
	    for m in range(ndocc,numbas):
		virtlist.append(m+1)
	  res = util.dipoleanalysis(dipmo_mat,Dp_0,ndocc,occlist,virtlist,debug,HL)
	  weighted_dip.append(res)

	fock_mid_backwd=np.copy(fock_mid_init) #prepare the fock at the previous midpint
	D_ti=D_t1
	Dp_ti=Dp_t1
	Enuc_list.append(-func_t0*Ndip_dir) #just in case of non-zero nuclear dipole
	#
	imp_list.append(func_t0)
	if debug :  
	  #trace of D_t1
	  fo.write('%.8f\n' % np.trace(Dp_ti).real)
	  fo.write('Trace of DS %.8f\n' % np.trace(np.matmul(Stilde,D_ti)).real)
	  fo.write('Trace of SD.real %.14f\n' % np.trace(np.matmul(Stilde,D_ti.real)))
	  fo.write('Trace of SD.imag %.14f\n' % np.trace(np.matmul(Stilde,D_ti.imag)))
	  fo.write('Dipole %.8f %.15f\n' % (0.000, 2.00*dip_list[0].real))

	for j in range(1,niter+1):


	    Eh_i,Exclow_i,ExcAAhigh_i,ExcAAlow_i,func_ti,F_ti,fock_mid_tmp=util.mo_fock_mid_forwd_eval(np.copy(D_ti),fock_mid_backwd,\
						    j,np.float_(dt),Htilde,I,U,dip_mat,C,C_inv,Stilde,numbas,imp_opts,func_h,func_l,fo,bset,bsetH,args.exciteA)

	    diff_midF=fock_mid_tmp-fock_mid_backwd

	    norm_diff=np.linalg.norm(diff_midF,'fro')
	    fo.write('||Fock_mid(%i +1/2)- Fock_mid(%i-1/2)||_fro: %.8f\n' % (j,j,norm_diff))

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
	    #if args.dump:
	    #   if ( ( j % args.oint ) == 0 ) :
	    #      path="td."+str(j).zfill(7)
	    #      os.mkdir(path)
	    #      cube_util.denstocube(phi,D_ti,S,ndocc,mol,"density",O,N,D)
	    #      os.rename("density.cube",path+"/density.cube")

	    Ah=np.conjugate(fock_mid_tmp.T)
	    fo.write('Fock_mid hermitian: %s\n' % np.allclose(fock_mid_tmp,Ah))
	    #transform fock_mid_init in MO basis
	    fockp_mid_tmp=np.matmul(np.conjugate(C.T),np.matmul(fock_mid_tmp,C))
	    u=util.exp_opmat(np.copy(fockp_mid_tmp),np.float_(dt))
	    #u=scipy.linalg.expm(-1.0j*fockp_mid_tmp*dt)
	    #check u is unitary
	    test_u=np.matmul(u,np.conjugate(u.T))
	    if (not np.allclose(np.eye(u.shape[0]),test_u)):
		print('U is not unitary\n')
	    
	    #check the trace of density to evolve
	    fo.write('tr of density to evolve: %.8f\n' % np.trace(Dp_ti).real)
	    
	    #evolve the density in orthonormal basis
	    temp=np.matmul(Dp_ti,np.conjugate(u.T))
	    Dp_ti_dt=np.matmul(u,temp)

	    #backtransform Dp_ti_dt
	    D_ti_dt=np.matmul(C,np.matmul(Dp_ti_dt,np.conjugate(C.T)))
	    fo.write('%.8f\n' % np.trace(Dp_ti_dt).real)
	    #dipole expectation for D_ti
	    dip_list.append(np.trace(np.matmul(dip_mat,D_ti)))
	    dip_offdiag0.append(np.trace(np.matmul(D_ti,dmat_offdiag[0])))
	    dip_offdiag1.append(np.trace(np.matmul(D_ti,dmat_offdiag[1])))
	    
	    #for debug
	    if debug:
	      fo.write('Dipole  %.8f %.15f\n' % (j*dt, 2.00*dip_list[j].real))

	    if (do_weighted == -2):
	      #weighted dipole 
	      res = util.dipoleanalysis(dipmo_mat,Dp_ti,ndocc,occlist,virtlist,debug,HL)
	      weighted_dip.append(res)
	    #Energy expectation value at t = t_i 
	    ene_list.append(Eh_i + Exclow_i + ExcAAhigh_i -ExcAAlow_i + Enuc + 2.0*np.trace(np.matmul(D_ti,Htilde)) )
	    Enuc_list.append(-func_ti*Ndip_dir) #just in case of non-zero nuclear dipole
	    imp_list.append(func_ti)
	   
	    #update D_ti and Dp_ti for the next step
	    
	    if debug :
	      fo.write('here I update the matrices Dp_ti and D_ti\n')
	    D_ti=np.copy(D_ti_dt)
	    Dp_ti=np.copy(Dp_ti_dt)
	    #update fock_mid_backwd for the next step
	    fock_mid_backwd=np.copy(fock_mid_tmp)

	fo.close()
	end=time.time()
	cend = time.process_time()
	ftime  = open("timing.txt", "w")
	ftime.write("time for %i time iterations : (%.3f s, %.3f s)\n" %(niter+1,end-start,cend-cstart))
	ftime.close()
	now = datetime.now()
	dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
	print("RT-propagation ended at: %s" % dt_string)
	t_point=np.linspace(0.0,niter*dt,niter+1)
	dip_t=2.00*np.array(dip_list).real
	ene_t=np.array(ene_list).real
	imp_t=np.array(imp_list)

	if (do_weighted == -2):
	  wd_dip=2.00*np.array(weighted_dip).real
	  np.savetxt('weighteddip.txt', np.c_[t_point,wd_dip], fmt='%.12e')

	np.savetxt('dipole-'+2*dip_dict[str(direction)]+'.txt', np.c_[t_point,dip_t], fmt='%.12e')
	# dipole_ij, i denotes the ith component of dipole vector, j denote the direction of the field
	np.savetxt('dipole-'+dip_dict[str(dip_dir[0])]+dip_dict[str(direction)]+'.txt', np.c_[t_point,2.00*np.array(dip_offdiag0).real], fmt='%.12e')
	np.savetxt('dipole-'+dip_dict[str(dip_dir[1])]+dip_dict[str(direction)]+'.txt', np.c_[t_point,2.00*np.array(dip_offdiag1).real], fmt='%.12e')
	np.savetxt('imp.txt', np.c_[t_point,imp_t], fmt='%.12e')
	np.savetxt('ene.txt', np.c_[t_point,ene_t], fmt='%.12e')
