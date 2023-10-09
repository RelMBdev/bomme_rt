import sys
import numpy as np
import os

sys.path.insert(0, "../common")
modpaths = os.environ.get('COMMON_PATH')

if modpaths is not None :
    for path in modpaths.split(";"):
        sys.path.append(path)
import bo_helper
from localizer import Localizer 
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
def exp_opmat_eig(mat,dt):  # do not assume F hermicity
    #first find eigenvectors and eigenvalues of F mat
    try:
       w,v=np.linalg.eig(mat)
    except np.linalg.LinAlgError:
        print("Error in numpy.linalg.eigh of inputted matrix")
        return None

    diag=np.exp(-1.j*w*dt)

    dmat=np.diagflat(diag)

    # for a general matrix Diag = M^(-1) A M
    # M is v
    try:
       v_i=np.linalg.inv(v)
    except np.linalg.LinAlgError:
       return None

    # transform back
    tmp = np.matmul(dmat,v_i)


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
##################################################################
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
    #also include params for a complex absorbing potential
    imp_params['r0'] = float(my_dict['r0'])
    imp_params['eta'] = float(my_dict['eta'])
    imp_params['cthresh'] = float(my_dict['cthresh'])
    #q vector boost-component
    imp_params['qvec'] = float(my_dict['qvec'])
    
    calc_params ={}    
    calc_params['time_int']=float(my_dict['time_int'])
    calc_params['delta_t']=float(my_dict['delta_t'])
    #calc_params['func_type'] =my_dict['func_type'] 
    calc_params['prop_id']=my_dict['prop_id']    # allow to choose the propagator : empc, mmut
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

def mo_fock_mid_forwd_eval(Dp_ti,fock_mid_ti_backwd,i,delta_t,fock_base,dipole,\
                        C,S,imp_opts,U,func_h,bsetH,exA=False,maxiter= 10 ,fout=sys.stderr, cap = None, debug=False):
    t_arg=np.float_(i)*np.float_(delta_t)
    
    func = funcswitcher.get(imp_opts['imp_type'], lambda: kick)
    
    pulse = func(imp_opts['Fmax'], imp_opts['w'], t_arg,\
                        imp_opts['t0'], imp_opts['s'])

    #Dp_ti is in the propgation (orthonormal) basis
    #transform in the AO basis
    D_ti= np.matmul(C,np.matmul(Dp_ti,np.conjugate(C.T)))
    
    k=1
    
    Eh_i,Exclow_i,ExcAAlow_i,ExcAAhigh_i,fock_mtx = fock_base.get_fock(Dmat=D_ti,func_acc=func_h,basis_acc=bsetH,U=U,return_ene=True)
    #DEBUG
    #ExcAAhigh_i=0.0
    #ExcAAlow_i=0.0
    
    #add -pulse*dipole or O_ext*pulse
    O_ext = dipole.get_matrix()
    if exA:
       dimA = bsetH.nbf()
       tmp_mat=np.zeros_like(D_ti)
       tmp_mat[:dimA,:dimA] = O_ext[:dimA,:dimA]
       O_ext = tmp_mat
    fock_ti_ao = fock_mtx + O_ext*pulse

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
        # CAP attempt
        if cap is not None:
          if cap.is_cap_on():
             cap_mtx = cap.cap_matrix()
             u = np.matmul(cap_mtx,np.matmul(u,cap_mtx)) # assuming cap_matrix is in exp(-dt*CAP*0.5) form
        tmpd=np.matmul(Dp_ti,np.conjugate(u.T))
        Dp_ti_dt=np.matmul(u,tmpd)
    #backtrasform Dp_ti_dt
        D_ti_dt=np.matmul(C,np.matmul(Dp_ti_dt,np.conjugate(C.T)))
    #build the correspondig Fock : fock_ti+dt
        
        #DEBUG
        #dum1,dum2,fock_mtx=get_Fock(D_ti_dt,H,I,func_l,basisset)
        fock_mtx = fock_base.get_fock(Dmat=D_ti_dt,func_acc=func_h,basis_acc=bsetH,U=U,return_ene=False)
        
        #print('fockmtx s in loop max diff: %.12e\n' % np.max(tfock_mtx-fock_mtx))
        #update t_arg+=delta_t
        pulse_dt = func(imp_opts['Fmax'], imp_opts['w'], t_arg+delta_t,\
                        imp_opts['t0'], imp_opts['s'])
        fock_ti_dt_ao=fock_mtx + O_ext*pulse_dt
        fock_inter= 0.5*fock_ti_ao + 0.5*fock_ti_dt_ao
    #update fock_guess
        fock_guess=np.copy(fock_inter)
        if k >1:
        #test on the norm: compare the density at current step and previous step
        #calc frobenius of the difference D_ti_dt_mo_new-D_ti_dt_mo
            diff=D_ti_dt-dens_test
            norm_f=np.linalg.norm(diff,'fro')
            if norm_f<(1e-6):
                if debug:
                   tr_dt=np.trace(np.matmul(S,D_ti_dt))
                   fout.write('converged after %i interpolations\n' % (k-1))
                   fout.write('i is: %d\n' % i)
                   fout.write('norm is: %.12f\n' % norm_f)
                   fout.write('Trace(D)(t+dt) : %.8f\n' % tr_dt.real)
                break
        dens_test=np.copy(D_ti_dt)
        k+=1
        if k > maxiter:
         fout.write('norm is: %.12f\n' % norm_f)
         raise Exception("Numember of iterations exceeded maxit = %i)" % maxiter)
    # return energy components , the Fock, the forward midpoint fock and the evolved density matrix 
    return Eh_i,Exclow_i,ExcAAhigh_i,ExcAAlow_i,pulse,fock_ti_ao,fock_inter,Dp_ti_dt
##################################################################
def prop_mmut(Dp_ti,D_midb,i,delta_t,fock_base,dipole,C,S,imp_opts,U,func_h,bsetH,exA=False,maxiter=None,fout=sys.stderr,cap=None,debug=False):
    # if at t_i no  previous midpoint density is available
    # we evolve backward in order to obtain D(t_i-1/2)

    #Dp_ti is in MO basis
    #transform in the AO basis
    D_ti= np.matmul(C,np.matmul(Dp_ti,np.conjugate(C.T)))

    t_arg=np.float_(i)*delta_t
    if imp_opts['imp_type'] == 'null':
      pulse = 0.0
    else:
      func = funcswitcher.get(imp_opts['imp_type'], lambda: kick)
    
      pulse = func(imp_opts['Fmax'], imp_opts['w'], t_arg,\
                        imp_opts['t0'], imp_opts['s'])
    fout.write('Field  %.8f %.15f\n' % (t_arg, pulse))
    
    O_ext = dipole.get_matrix()
    if exA:
       dimA = bsetH.nbf()
       tmp_mat=np.zeros_like(D_ti)
       tmp_mat[:dimA,:dimA] = O_ext[:dimA,:dimA]
       O_ext = tmp_mat



    #print(i,t_arg)
    
    Eh_i,Exclow_i,ExcAAlow_i,ExcAAhigh_i,fock_mtx = fock_base.get_fock(Dmat=D_ti,func_acc=func_h,basis_acc=bsetH,U=U,return_ene=True)
    #add -pulse*dipole
    fock_ti = fock_mtx + O_ext*pulse
    #transform fock_ti in the MO ref basis
    fock_ti_mo=np.matmul(np.conjugate(C.T),np.matmul(fock_ti,C))
    """
    if chebyshev is not None:
       fout.write("j step: %i\n" % i)
       chebyshev.update_params(fock_ti_mo,fout)
    """   
    #calculate u
    #perform some test
    """
    if chebyshev is not None:
       u = chebyshev.exp_mat(fock_ti_mo,)
    else:
       u=exp_opmat(fock_ti_mo,delta_t)
    """   
    u=exp_opmat(fock_ti_mo,delta_t)

    # CAP attempt
    if cap is not None:
      if cap.is_cap_on():
         cap_mtx = cap.cap_matrix()
         u = np.matmul(cap_mtx,np.matmul(u,cap_mtx)) # assuming cap_matrix is in exp(-dt*CAP*0.5) form
    
    # avoid printing
    if debug:
      if np.isreal(t_arg):
         test=np.matmul(u,np.conjugate(u.T))
         #print('U is unitary? %s' % (np.allclose(test,np.eye(u.shape[0]))))
         if (not np.allclose(test,np.eye(u.shape[0]))):
             Id=np.eye(u.shape[0])
             diff_u=test-Id
             norm_diff=np.linalg.norm(diff_u,'fro')
             print('from fock_mid:U is not unitary, |UU^-1 -I| %.8f' % norm_diff)
    ####

    """
    if (chebyshev is not None):
      # input : fock_ti_mo
      #         i (time step index)
      #         half_int flag
       u2 = chebyshev.exp_mat(fock_ti_mo,True)
    else:
       #calculate the new u operator ( for the half-interval propagation)
       u2=exp_opmat(fock_ti_mo,delta_t/2.00)
    """   
    u2=exp_opmat(fock_ti_mo,delta_t/2.00)

    if cap is not None:
      if cap.is_cap_on():
         cap_mtx = cap.cap_matrix(dt_half=True)
         u2 = np.matmul(cap_mtx,np.matmul(u2,cap_mtx)) # assuming cap_matrix is in exp(-dt'*0.5*CAP) form

    if D_midb is None:
        fout.write("\n")
        fout.write("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")
        fout.write("@ D_backw is None at i = %i  @\n" % i)
        fout.write("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")
        
    #calculate D(1-1/2) ->D_midb  (the suffix mo has been dropped: every matrix involved the
    #evolution step is in the mo basis)
    
    D_midb=np.matmul(np.conjugate(u2.T),np.matmul(Dp_ti,u2))
    
    #print('trace of D_midb(%i) : %.8f' % (i,np.trace(D_midb).real))
    # D_midf is the density at the next midpoint (in mo basis)
    
    D_midf=np.matmul(u,np.matmul(D_midb,np.conjugate(u.T)))
    #print('trace of D_midf(%i) : %.8f' % (i,np.trace(D_midf).real))

    #finally get D_ti_dt_mo using u2
    Dp_ti_dt=np.matmul(u2,np.matmul(D_midf,np.conjugate(u2.T)))
    #backtransform D_ti_dt_mo in AO basis
    D_ti_dt=np.matmul(C,np.matmul(Dp_ti_dt,np.conjugate(C.T)))
    #retun densities and energy terms :
    return Eh_i,Exclow_i,ExcAAlow_i,ExcAAhigh_i,pulse,fock_ti,D_midf,Dp_ti_dt
##################################################################

# analysis based on MO-weighted dipole

def dipoleanalysis(dipole,dmat,nocc,occlist,virtlist,debug=False):
    #virtlist can also contain occupied orbitals !check
    tot = len(occlist)*len(virtlist)
    res = np.zeros(tot,dtype=np.complex128)
    count = 0
    for i in occlist:
      for j in virtlist:
         res[count] = dipole[i-1,j-1]*dmat[j-1,i-1] + dipole[j-1,i-1]*dmat[i-1,j-1]
         count +=1
    return res
##################################################################
def dipole_selection(dipole,idlist,nocc,occlist,virtlist,odbg=sys.stderr,debug=False):
 
    if debug:
       odbg.write("Selected occ. Mo: %s \n"% str(occlist))
       odbg.write("Selected virt. Mo: %s \n"% str(virtlist))
    offdiag = np.zeros_like(dipole)
    #diag = numpy.diagonal(tmp)
    #diagonal = numpy.diagflat(diag)
    nvirt = dipole.shape[0]-nocc
    odbg.write("n. virtual orbitals : %i\n" % nvirt)
    if (idlist == -99):
      for b in range(nvirt):
        for j in occlist:
          offdiag[nocc+b,j-1] = dipole[nocc+b,j-1]
    else:
      for b in virtlist:
        for j in  occlist:
          offdiag[b-1,j-1] = dipole[b-1,j-1]
    offdiag=(offdiag+np.conjugate(offdiag.T))
    #offdiag+=diagonal
    res = offdiag

    return res

#############################################################################
# Implementation of fourth order propagator
# -ETRS : exponential time-reversal symmetry propagator
# -CFET4: Fourth-order commutator free exponential-propagator 
#
# -Fock_wrapper
#
#
class Fock_wrapper():
    def __init__(self,fock_base,C,field_operator, imp_opts, U, func_h, bsetH, exA=False, cap=None):
        self.__fock_factory = fock_base
        self.__Umat = U
        self.__ftype_h = func_h 
        self.__basis_h = bsetH
        self.__eri =  None 
        self.__exA_only = exA
        self.__ext_mat = field_operator # the spatial part of the V_ext
        self.__Ccoef = C
        self.__field = funcswitcher.get(imp_opts['imp_type'], lambda: kick)   # beware of the keys
        self.__pulseopt = imp_opts
        self.__Ehartree = None
        self.__Exc_l = None
        self.__Exc_laa = None  #  only considering aa sub-block of the basis
        self.__Exc_haa = None  #
        self.__cap = cap ######
        self.__pulse_list = []

    def Ccoef(self):
        return self.__Ccoef

    def get_ene_terms(self):
        return self.__Ehartree, self.__Exc_l, self.__Exc_haa, self.__Exc_laa, self.__pulse_list

    def Fock(self,D_ti,t_arg,basis='AO',l_bound=False):             # l_bound refers to the lower boundary of the time interval
         #check arguments
         if not isinstance(D_ti,np.ndarray):
             raise TypeError("Check density matrix\n")
         if not isinstance(t_arg,np.float_):
             raise TypeError("t_arg should be float\n")
         if not isinstance(basis,str):
             raise TypeError("basis name is str\n")
         if not isinstance(l_bound,bool):
             raise TypeError("l_bound should be bool\n")

         # to avoid windowing effect the pulse should be zero at t=0
         pulse = self.__field(self.__pulseopt['Fmax'], self.__pulseopt['w'], t_arg,\
                        self.__pulseopt['t0'], self.__pulseopt['s'])

         # store the external field (just for visualization)
         self.__pulse_list = pulse
         if l_bound :
             Eh_i,Exclow_i,ExcAAlow_i,ExcAAhigh_i,fock_mtx = self.__fock_factory.get_fock(Dmat=D_ti, func_acc=self.__ftype_h, basis_acc=self.__basis_h,\
                                                                                U=self.__Umat,return_ene=l_bound)
             self.__Ehartree = Eh_i
             self.__Exc_l    = Exclow_i
             self.__Exc_laa  = ExcAAlow_i
             self.__Exc_haa  = ExcAAhigh_i
         else:
             fock_mtx = self.__fock_factory.get_fock(Dmat=D_ti, func_acc=self.__ftype_h, basis_acc=self.__basis_h, U=self.__Umat,return_ene=l_bound)

         #dipole approximation

         O_ext = self.__ext_mat.get_matrix()
         if self.__exA_only:
            nbfA = self.__basis_h.nbf()
            tmp_mat=np.zeros_like(D_ti)
            tmp_mat[:nbfA,:nbfA] = O_ext[:nbfA,:nbfA]
            O_ext = tmp_mat  # redefined if exA->True
            
         fock_mtx = fock_mtx + O_ext*pulse #implicit conversion
         if basis == 'MO':
             C = self.__Ccoef
             fock_mtx = np.matmul(np.conjugate(C.T),np.matmul(fock_mtx,C))
         return fock_mtx

def ETRS_1(D_0,t_0,t_1,fbase,niter,eps=1.0e-5,test_Fock=None,test_D= None):
    #label [ao] fo the quantities on AO basis
    #if no label present MO basis is assumed
    delta_t = t_1 - t_0
    # start from k-th time
    # set some intial stuff
    C = fbase.Ccoef()
    
    
    # Dao_0[AO] is derivede from D_0[MO]
    Dao_0 = np.matmul(C, np.matmul(D_0,np.conjugate(C.T)))
    
    # explicitly calculated on the MO basis
    F_0 = fbase.Fock(Dao_0,t_0,'MO',True)
    if test_Fock is not None:
       print("test fock: %s\n" % np.allclose(F_0,test_Fock))
       
    if test_D is not None:
       print("test D: %s\n" % np.allclose(Dao_0,test_D))
    u_0 = exp_opmat(F_0,delta_t*0.5)

 
    #evolve D_0 to the time t0+dt/2 (step 1 in algorithm.4)
    #    
    #                  dt/2      dt/2
    #      |-----------------|------------------|
    #      t0              t0+dt/2              t1
    #     F0(t0)          F1[P1](t0)
    # U0(t0+dt/2,t0)      U1(t0,t1)
    #    P0(t0)           P1(t0+dt/2)
    D_1 = np.matmul(u_0, np.matmul(D_0,np.conjugate(u_0.T)) )    
    #print("max diff |D_0-D_1| : %.4e\n" % np.max(np.abs(D_1-D_0)))
    # transform back D_1 to the AO basis
    Dao_1 = np.matmul(C, np.matmul(D_1,np.conjugate(C.T)))

    #print("max diff |D_0-D_1|_AO : %.4e\n" % np.max(np.abs(Dao_1-Dao_0)))
    F_1 = fbase.Fock(Dao_1,t_0,'MO')
    #print("max diff |F_1-F_0| : %.4e\n" % np.max(np.abs(F_0-F_1)))
    #set the new time-evolution operator
    u_1 = exp_opmat(F_1,delta_t)

    D_2=np.matmul(u_1, np.matmul(D_0,np.conjugate(u_1.T)) )    # step 2
    # transform back to the AO basis
    Dao_2 = np.matmul(C, np.matmul(D_2,np.conjugate(C.T)))
    
    # begin loop
    D_in = D_2
    Dao_in = Dao_2
    for i in range(2,niter+1):
        F_i = fbase.Fock(Dao_in,t_1,'MO')

        u_i = exp_opmat(F_i,delta_t*0.5)
        D_out = np.matmul(u_i, np.matmul(D_1,np.conjugate(u_i.T)))
        Dao_out = np.matmul(C,np.matmul(D_out,np.conjugate(C.T))) # C coeffs are real numbers in the case of ground state 
        diff_fro = np.linalg.norm((Dao_out-Dao_in),'fro')                    # molecular orbitals on real basis set
        if diff_fro < D_out.shape[0]*eps:
          break
        if i == niter:
          print("reached max iter")
        Dao_in = np.copy(Dao_out)

    return  Dao_out , D_out    
# ETRS wrap
def ETRS_wrap(Dp_ti,kstep,delta_t,fock_base,dipole,\
                        C,imp_opts,U,func_h,bsetH,exA=False,maxiter= 10 ,fout=sys.stderr, cap = None, thresh=1.0e-5, debug=False):
        # useful quantities
        # use C matrix to get the density on the AO basis from the propagation basis
        # wrap this aux quantities in a container
        fbase = Fock_wrapper(fock_base,C,dipole, imp_opts, U, func_h, bsetH, exA)
        
        # mo subscript is dropped when it can be inferred from the context if MO basis is employed
        # one step
        # do checks
        #######
        #Ccoef = fbase.Ccoef()
        #D_inp = np.matmul(Ccoef,np.matmul(Dp_ti,Ccoef.T))
        #fock = fbase.Fock(D_inp,np.float_(kstep*delta_t),'MO',True)
        #fock_ao = fbase.Fock(D_inp,kstep*delta_t,'AO')
        #Hartree,Exclow, dum1,dum2,Efield = fbase.get_ene_terms()
        #print("two el ene: %.12f\n" % (Hartree+Exclow))
        

        t_arg = kstep*delta_t      # see pseudo code in ALGORITHM 3
        t_a   = t_arg + delta_t
        
        # call ETRS_1 propagator
        # >ETRS_1(Dao_0,t_0,t_1,fbase,niter,eps=1.0e-5)
        # P1(ta) (AO) line 7
        D1_ta, D1_ta_mo = ETRS_1(Dp_ti, t_arg, t_a, fbase, niter=20,test_Fock=None,test_D=None)
        # collect energy terms
        Eh_i, Exclow_i, ExcAAhigh_i, ExcAAlow_i, field_i = fbase.get_ene_terms()
        return Eh_i, Exclow_i, ExcAAhigh_i, ExcAAlow_i, field_i,D1_ta_mo

# Algorithm.1 : Fourth-order commutator free exponential-propagator(CFET4) 
# with two exponential (see J. Chem. Phys. 157, 074106 (2022); doi: 10.1063/5.0106250)
def CFET4(dmat_0,fockm_1,fockm_2,deltat): # the subscripts denote t0,t0+t1,t0+t2 respectively
    # input : dmat_0,fockm_1,fock_2
    # set t1 and t2: in Blanes[2006] t1->c1*h, t2->c2*h; c1 = 1/2-sqrt(3)/6  c2 = 1/2+sqrt(3)/6
    #t1=(0.5-np.sqrt(3.)/6)*deltat
    #t2=(0.5+np.sqrt(3.)/6)*deltat
    u1_arg = (3.+2.*np.sqrt(3))/12.*fockm_1 + (3.-2.*np.sqrt(3))/12.*fockm_2 #! on the MO basis from args
    u1 = exp_opmat(u1_arg,deltat)

    # dmat_0 (MO) could be passed as argument
    #dmat_0 = np.matmul(C_inv, np.matmul(dmat_0,np.conjugate(C_inv.T)) )
    dmat_dt = np.matmul(u1, np.matmul(dmat_0,np.conjugate(u1.T)))
   
    u2_arg = (3.-2.*np.sqrt(3))/12.*fockm_1 + (3.+2.*np.sqrt(3))/12.*fockm_2
    u2 = exp_opmat(u2_arg,deltat)
    
    dmat_dt = np.matmul(u2, np.matmul(dmat_dt,np.conjugate(u2.T)))
    # density returned and expressed in the propagation basis
    return dmat_dt

#aux function

def is_unitary(mat):
    iden = np.eye(mat.shape[0])
    test = np.matmul(mat,np.conjugate(mat.T))
    res = np.allclose(test,iden,atol=1.0e-12)
    return res
def check_trace(mat,file):
    res = np.trace(mat)
    file.write('trace : %.5e\n' % res.real)

def epep2(Dp_ti,kstep,delta_t,fock_base,dipole,\
                        C,imp_opts,U,func_h,bsetH,exA=False,maxiter= 10 ,fout=sys.stderr, cap = None, thresh=1.0e-5, debug=False):
        # useful quantities
        # use C matrix to get the density on the AO basis from the propagation basis
        # wrap this aux quantities in a container
        fbase = Fock_wrapper(fock_base,C,dipole, imp_opts, U, func_h, bsetH, exA)
        D_k = Dp_ti
        # mo subscript is dropped when it can be inferred from the context if MO basis is employed
        # one step
        
        t_arg = kstep*delta_t      # see pseudo code in ALGORITHM 3
        t_a   = t_arg + (0.5 - np.sqrt(3)/6.)*delta_t
        t_b   = t_arg + (0.5 + np.sqrt(3)/6.)*delta_t
        
        # call ETRS_1 propagator
        # >ETRS_1(Dao_0,t_0,t_1,fbase,niter,eps=1.0e-5)
        # P1(ta) (AO) line 7
        D1_ta, D1_ta_mo = ETRS_1(D_k, t_arg, t_a, fbase, niter=20)

        # collect energy terms
        Eh_i, Exclow_i, ExcAAhigh_i, ExcAAlow_i, field_i = fbase.get_ene_terms()


        # get Fock1[D1_ta](t_a) (Fock() takes in input a matrix on AO basis:->output on MO basis)
        F1_ta = fbase.Fock(D1_ta,t_a,'MO')
        # define U_1(t_b,t_a) 

        U1_tatb = exp_opmat(F1_ta, t_b-t_a)
        if debug:
           fout.write("U1_tatb is unitary at tstep %i, %s\n" %(kstep, is_unitary(U1_tatb)))
        
        # D1_tb (on MO basis)
        D1_tb = np.matmul( U1_tatb, np.matmul(D1_ta_mo,np.conjugate(U1_tatb.T) ) )
        if debug:
           fout.write("D1_tb trace at tstep %i\n" % kstep)
           check_trace(D1_tb,fout)

        # get Fock1[D1_tb](tb)
        D1_tb = np.matmul(C, np.matmul(D1_tb,C.T)) # real C coeff, no need for complex conjugation
        F1_tb = fbase.Fock(D1_tb,t_b,'MO')            # step 3
        # P2(tk+1) using CFET4 ->CFET4(dmat_0,fockm_1,fockm_2,deltat,C,C_inv)
        D2_tk_p1_mo = CFET4(D_k,F1_ta,F1_tb,delta_t)
        if debug:
          fout.write("D2_tk_p1 trace at tstep %i\n" % kstep)
          check_trace(D2_tk_p1_mo,fout)
        
        D2_tk_p1 = np.matmul(C,np.matmul(D2_tk_p1_mo,C.T)) # is now on AOs
        
        Dj_tk_p1_mo = D2_tk_p1_mo
        Dj_tk_p1 = D2_tk_p1
        
        # D2_tk_p1 will be updated each jstep  Dj_tk_p1 (where j = 2,..,Nit)
        # PC loop
        # maxiter is predictor-corr max number of iterations
        for jstep in range(2,maxiter+1):  # the predictor - corrector loop as in EMPC
            # get Fockj(tk+1), step 6
            Fj_tk_p1 = fbase.Fock(Dj_tk_p1, t_arg+delta_t, 'MO')
            Uj_tbtk_p1 = exp_opmat(Fj_tk_p1, t_arg+delta_t-t_b)

            if debug:
               fout.write("U[%i]_tatb is unitary at tstep %i, %s\n" %(jstep,kstep, is_unitary(Uj_tbtk_p1)))
            
            Dj_tb = np.matmul( Uj_tbtk_p1, np.matmul(Dj_tk_p1_mo,np.conjugate(Uj_tbtk_p1.T) ) )
            # get Fockj(tb), step 4 (line 17)
            Dj_tb = np.matmul(C, np.matmul(Dj_tb,C.T)) # real C coeff, no need for complex conjugation
            Fj_tb = fbase.Fock(Dj_tb, t_b, 'MO')
            Djp1_tk_p1_mo = CFET4(D_k,F1_ta,Fj_tb,delta_t) #CFET4 and ETRS_1 take in input the density matrix on MO
            Djp1_tk_p1 = np.matmul(C,np.matmul(Djp1_tk_p1_mo,C.T)) # is now on AOs
            # set the convergence check
            diff = Djp1_tk_p1 - Dj_tk_p1
            diff = np.linalg.norm(diff, 'fro')
            if diff < D_k.shape[0]*thresh:
               fout.write('converged\n')
               break
            if jstep == maxiter:
               fout.write("reached maxit %i, fro(diffD) : %.4e\n" % (jstep,diff))
            Dj_tk_p1 = Djp1_tk_p1
            Dj_tk_p1_mo = Djp1_tk_p1_mo
        # D_k is on MO , see line 7 (get D1_ta)
        D_k = Dj_tk_p1_mo
         
        # final results 
        return Eh_i, Exclow_i, ExcAAhigh_i, ExcAAlow_i, field_i,  D_k
        
        #self.__D_final = D_k
        #self.__Dao_final = Dj_tk_p1
