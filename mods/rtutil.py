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
                        C,S,imp_opts,U,func_h,bsetH,exA=False,maxiter= 10 ,Vemb=None,fout=sys.stderr, debug=False):

    t_arg=np.float_(i)*np.float_(delta_t)
    
    func = funcswitcher.get(imp_opts['imp_type'], lambda: kick)
    
    pulse = func(imp_opts['Fmax'], imp_opts['w'], t_arg,\
                        imp_opts['t0'], imp_opts['s'])

    #Dp_ti is in the propgation (orthonormal) basis
    #transform in the AO basis
    D_ti= np.matmul(C,np.matmul(Dp_ti,np.conjugate(C.T)))
    
    k=1
    
    Eh_i,Exclow_i,ExcAAlow_i,ExcAAhigh_i,fock_mtx = fock_base.get_fock(Dmat=D_ti,func_acc=func_h,basis_acc=bsetH,U=U,return_ene=True)
    if isinstance(Vemb,np.ndarray):
        fock_mtx += Vemb
    #DEBUG
    #ExcAAhigh_i=0.0
    #ExcAAlow_i=0.0
    
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
        dum0,dum1,dum2,dum3,fock_mtx = fock_base.get_fock(Dmat=D_ti_dt,func_acc=func_h,basis_acc=bsetH,U=U,return_ene=True)
        if isinstance(Vemb,np.ndarray):
            fock_mtx += Vemb
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
        if k > maxiter:
         raise Exception("Numember of iterations exceeded maxit = %i)" % maxiter)
    # return energy components , the Fock, the forward midpoint fock and the evolved density matrix 
    return Eh_i,Exclow_i,ExcAAhigh_i,ExcAAlow_i,pulse,fock_ti_ao,fock_inter,Dp_ti_dt
