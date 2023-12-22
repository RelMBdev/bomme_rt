import sys
import os
import numpy as np
import rtutil

sys.path.insert(0, "../common")
modpaths = os.environ.get('COMMON_PATH')

if modpaths is not None :
    for path in modpaths.split(";"):
        sys.path.append(path)
from localizer import Localizer 
class abs_pot():
    def __init__(self,cap_mat,Cmat,Umat,dt=0.1,cap_on=False):
        self.__exp_mat = None
        self.__exp_mat_half_ts = None # for delta_t/2
        self.__eigvec = None
        self.__eigval = None
        self.__delta_t = dt
        #self.__do_cap = None
        

        self.__do_cap = cap_on
        if self.__do_cap:
            if isinstance(Umat,np.ndarray):
                cap_mat = np.matmul(cap_mat,Umat)
                cap_mat = np.matmul(Umat.T,cap_mat)
            
            # transform to the propagation basis
            
            tmp = np.matmul(cap_mat,Cmat)
            tmp = np.matmul(Cmat.T,tmp)
            
            #diagonalize so that we can form exp(-A*dt/2) diagonal elements
            try:
               eigval,eigvec=np.linalg.eigh(tmp)
            except np.linalg.LinAlgError:
                print("check CAP\n")
                raise Exception("Error in numpy.linalg.eigh of inputted matrix")
            idx = eigval.argsort()[::-1]
            eigval = eigval[idx]
            eigvec = eigvec[:,idx]
            self.__eigval =eigval
            self.__eigvec =eigvec

    def is_cap_on(self):   # verify fuction
        return self.__do_cap
    def time_step(self):
        return self.__delta_t

    def cap_matrix(self,dt_half=False):
        # the exponential of the CAP is already built-in to fullfill the split-operator form -> i.e exp(-W*Dt/2)
        # -> exp( -i(F(t+Dt/2) -iW) * Dt ) approx exp(-W *Dt/2) exp(-i F(t+Dt/2) * Dt) exp(-W *Dt/2)
        # Further if exp( -iF(t+Dt'/2) * Dt') where Dt' = Dt/2 is needed (in MMUT) we provide the dt_half keyword
        dt = self.__delta_t
        if self.__do_cap:
            if not isinstance(self.__exp_mat_half_ts,np.ndarray): # in case already exists skip
               diag=np.exp(-0.5*self.__eigval*np.float_(dt*0.5) )
             
               dmat=np.diagflat(diag)
               self.__exp_mat_half_ts = np.matmul(self.__eigvec ,np.matmul(dmat,self.__eigvec.T))
               #assign
            # form the exp operator
           
            if not isinstance(self.__exp_mat,np.ndarray): # in case already exists
               diag=np.exp(-0.5*self.__eigval*dt)
             
               dmat=np.diagflat(diag)
               self.__exp_mat = np.matmul(self.__eigvec ,np.matmul(dmat,self.__eigvec.T))
            if dt_half:
                res = self.__exp_mat_half_ts
            else:
                res = self.__exp_mat
            return res
        else:
            return None

class operator_container():
    def __init__(self,dipmat,Cmat,pulse_opts,kind='dipole'):
        self.__res = None

        if isinstance(dipmat,list):
            tmp = dipmat[0]
        else:
            tmp = dipmat

        if kind == 'dipole':
            self.__res = np.float_(-1.00)*tmp #   -> -dipole*Field
        elif kind == 'scatt':
            try :
              C_inv=np.linalg.inv(Cmat)
            except np.linalg.LinAlgError:
              print("Error in np.linalg.inv")

            dip_mo = np.matmul( Cmat.T, np.matmul(tmp,Cmat) )

            q_boost =  pulse_opts['qvec']  # the q vector along boost-direction (to be generalized)
            u0 = rtutil.exp_opmat(dip_mo,np.float_(-q_boost))
            scatt_op = np.matmul( C_inv.T, np.matmul(u0,C_inv) )
            self.__res = scatt_op
        else:
            raise Exception("not implemented\n")
    def get_matrix(self):
        return self.__res

class real_time():
    def __init__(self,Dinit, Fock_init, fock_factory, ndocc, basis, Smat, pulse_opts, delta_t, Cmat, dipmat,\
            out_file=sys.stderr,  basis_acc = None, func_acc=None,U=None,local_basis=False,\
                                             exA_only=False,occlist=None, virtlist=None,i_step=0,prop_type='empc',ext_type='dipole',debug=False):  #default exponential midpoint predictor-corrector
      
      self.__ffactory = fock_factory  
      self.__ndocc    = ndocc
      self.__basisobj = basis
      self.__pulse_param = pulse_opts
      self.__delta_t = delta_t
      self.__basisobj_acc = basis_acc
      self.__func_acc = func_acc
      self.__numbasis = basis.nbf()
      self.__occlist  = occlist
      self.__virtlist = virtlist
      self.__locbasis = local_basis
      self.__localbasobj = None
      self.__exA_only = exA_only
      self.__Dp       = None
      self.__Fock_mid = Fock_init
      self.__Dp_back  = None
      self.__ovapm    = Smat # it can represent either overlap of AO basis functions or BO b. funcs
      self.__Cmat  = Cmat    # Cmat contains MO coefficients (either on AO basis or BO basis)
      self.__dipmat = None
      self.__field_op = None
      self.__perpdip = []
      self.__Umat     = U
      self.__D = Dinit
      self.__outfile = out_file
      self.__step_count = i_step
      self.__embfactory = None
      self.__embopt = None
      self.__ene_list = []
      self.__dip_list = []
      self.__dperp0_list = [] # the expectation values  of perpendicular components (wrt the boost) of dipole
      self.__dperp1_list = []
      self.__field_t = None
      self.__field_list = []
      self.__do_cap = False
      self.__cap_exp = None # exp(-A*dt/2)
      self.__prop_type = prop_type
      self.__ext_type = ext_type # dipole approximation ?
      self.__debug = debug

      if isinstance(dipmat,list):
          self.__dipmat = dipmat[0]
          for mtx in dipmat[1:]:
               self.__perpdip.append(mtx)
      else:
          self.__dipmat= dipmat
        
      #set the field operator (generalize to non dipole-approximation cases)

      self.__field_op = operator_container(dipmat,Cmat,pulse_opts,ext_type)

      if U is not None:
          if not isinstance(U, np.ndarray):
              raise TypeError(" U must be nd.ndarray")
          
      if basis_acc is not None:
          nbf_A = basis_acc.nbf()
      
      if local_basis:
              
              try :
                SAA_inv=np.linalg.inv(Smat[:nbf_A,:nbf_A])
              except scipy.linalg.LinAlgError:
                print("Error in np.linalg.inv")
     
              self.__localbasobj=Localizer(Dinit,Smat,nbf_A)
              self.__localbasobj.localize()
              #unsorted orbitals
              unsorted_orbs=self.__localbasobj.make_orbitals()
              #the projector P
              Phat=np.matmul(Smat[:,:nbf_A],np.matmul(SAA_inv,Smat[:nbf_A,:]))
              #The RLMO are ordered
              # by descending value of the locality parameter.
              sorted_orbs = self.__localbasobj.sort_orbitals(Phat)
              #the occupation number and the locality measure
              locality,occnum=self.__localbasobj.locality()
              #save the localization parameters and the occupation numbers  
              np.savetxt('locality_rlmo.out', np.c_[locality,occnum], fmt='%.12e')
              #check the occupation number of ndimA=nbf_A MOs (of A).
              occA = int(np.rint( np.sum(occnum[:nbf_A]) ))
              print("ndocc orbitals (A) after localization: %i\n" % occA)
              #set the sorted as new basis 
              C=sorted_orbs
              self.__Cmat = C
              #use the density corresponding to (sorted) localized orbitals
              Dp_0 = np.diagflat(occnum)
      else:
        Dp_0=np.zeros((self.__numbasis,self.__numbasis))
        for num in range(int(ndocc)):
            Dp_0[num,num]=1.0
      self.__Dp = Dp_0

    def dipmat(self):
        return self.__dipmat
    def occlist(self):
        return self.__occlist
    def virtlist(self):
        return self.__virtlist
    def i_step(self):
        return self.__step_count

    def init_boost(self,selective_pert=False,debug=False):
       dip_mat  = self.__dipmat
       exA_only = self.__exA_only
       if self.__basisobj_acc is not None:

          nbf_A = self.__basisobj_acc.nbf()

       occlist  = self.__occlist
       virtlist = self.__virtlist
       ndocc = self.__ndocc
       numbas = self.__basisobj.nbf()

       print('Perturb density with analytic delta')
       # set the perturbed density -> exp(-ikP)D_0exp(+ikP)
       k = self.__pulse_param['Fmax']
       if   self.__exA_only:
         print("Excite only the A subsystem (High level)\n")
         #excite only A to prevent leak of the hole/particle density across the AA-BB  boundary
         #select AA block in BO basis
         tmpA=np.zeros_like(dip_mat)    
         tmpA[:nbf_A,:nbf_A]=dip_mat[:nbf_A,:nbf_A]
         dip_mo=np.matmul(np.conjugate(self.__Cmat.T),np.matmul(tmpA,self.__Cmat))
       else: 
            #dip_mat is transformed to the reference MO basis (MO can be 
            # expressed either on AO or BO basis)
            print("Dipole matrix is transformed to the MO basis\n")
            print("Local basis: %s\n" % self.__locbasis)
            dip_mo=np.matmul(np.conjugate(self.__Cmat.T),np.matmul(dip_mat,self.__Cmat))

       if self.__locbasis and selective_pert:
            #use occnum to define a virtlist -> occnum exist only if local_basis has ben used
            if isinstance(self.__virtlist,list):
              if self.__virtlist[0] == -99: 
                 occ_number = self.__localbasobj.occnum()
                 virtlist=[]
                 for m in range(numbas):
                   if np.rint(np.abs(occ_number))[m] < 1.0: 
                     virtlist.append(m+1)
                 dip_mo=rtutil.dipole_selection(dip_mo,-1,ndocc,occlist,virtlist,self.__outfile,debug) # import from ?
                 self.__virtlist = virtlist
       elif selective_pert:
             dip_mo=rtutil.dipole_selection(dip_mo,virtlist[0],ndocc,occlist,virtlist,self.__outfile,debug)
           
       u0 = rtutil.exp_opmat(dip_mo,np.float_(-k))
       Dp_init= np.matmul(u0,np.matmul(self.__Dp,np.conjugate(u0.T)))
       func_t0=k
       #backtrasform Dp_init
       D_init=np.matmul(self.__Cmat,np.matmul(Dp_init,np.conjugate(self.__Cmat.T)))
       Dtilde = D_init
       Dp_0 = Dp_init 
       self.__Dp = Dp_0
       self.__D  = D_init

       return Dp_0, D_init

    def embedding_init(self,embpot,pyembopt):
      self.__embfactory = embpot # this is the same as self.__pyemb in scf_run() class
      self.__embopt = pyembopt
 
    def do_cap(self,flag=True):
        self.__do_cap = flag

    def set_CAP(self,cap_mat):
        # the CAP is provided as a matrix expressed in the AO basis
        # U matrix is needed if transformation to the BO basis is required
        if not self.__do_cap:
            print("CAP is not active\n")
        #init the abs_pot instance 
        tmp = abs_pot(cap_mat,self.__Cmat,self.__Umat, dt=self.__delta_t, cap_on=self.__do_cap) # dt/2 for the cap (split-operator form) is alread accounted
        self.__cap_exp = tmp
        return tmp

    def cap_clear(self):
        self.__cap_exp = None

    def __onestep_prop(self):
        #cap is provided as exp of the cap matrix itself
        #aliases
        i_step = self.__step_count
        C = self.__Cmat # can be generalized to loewdin orthogonalization
        pulse_opts = self.__pulse_param
        dip_mat = self.__dipmat
        ovapm = self.__ovapm
        fo = self.__outfile
        bsetH = self.__basisobj_acc
        func_h = self.__func_acc
        exA_only = self.__exA_only

        #backward_mid is a np.ndarray container for either the fock(i-1/2) or Density(i-1/2)
        backward_mid = np.empty_like(ovapm) 
        
        if self.__prop_type == 'empc':
               Eh,Exclow,ExcAAhigh,ExcAAlow,func_t,F_ti,backward_mid, Dp_ti_dt = rtutil.mo_fock_mid_forwd_eval(self.__Dp,self.__Fock_mid,\
                              i_step,np.float_(self.__delta_t), self.__ffactory, self.__field_op, C, ovapm, pulse_opts, self.__Umat, func_h,\
                              bsetH, exA_only,fout=fo,debug=self.__debug,cap=self.__cap_exp)
        elif self.__prop_type == 'mmut':
               #raise Exception("not available\n")
               Eh,Exclow,ExcAAhigh,ExcAAlow,func_t,F_ti,backward_mid, Dp_ti_dt = rtutil.prop_mmut(self.__Dp,self.__Dp_back,\
                              i_step,np.float_(self.__delta_t), self.__ffactory, self.__field_op, C, ovapm, pulse_opts, self.__Umat,func_h,\
                              bsetH,exA_only, fout=fo, debug=self.__debug, cap=self.__cap_exp)
        elif self.__prop_type == 'epep2':
               #raise Exception("not available\n")
               Eh,Exclow,ExcAAhigh,ExcAAlow,func_t,Dp_ti_dt = rtutil.epep2(self.__Dp,i_step,np.float_(self.__delta_t),self.__ffactory, self.__field_op,\
                        C,pulse_opts,self.__Umat,func_h,bsetH,exA=exA_only,maxiter= 10 ,fout=fo, cap = None, thresh=1.0e-5, debug=self.__debug)
        elif self.__prop_type == 'etrs':       
               Eh,Exclow,ExcAAhigh,ExcAAlow,func_t,Dp_ti_dt = rtutil.ETRS_wrap(self.__Dp,i_step,np.float_(self.__delta_t),self.__ffactory, self.__field_op,\
                        C,pulse_opts,self.__Umat,func_h,bsetH,exA=exA_only,maxiter= 10 ,fout=fo, cap = None, thresh=1.0e-5, debug=self.__debug)
        #propagators go here       
        return (Eh,Exclow,ExcAAhigh,ExcAAlow),func_t,backward_mid,Dp_ti_dt

    def __call__(self):
        i_step = self.__step_count
        dt = self.__delta_t
        fock_base = self.__ffactory
        U = self.__Umat
        dip_mat = self.__dipmat
        C = self.__Cmat 
        pyembopt = self.__embopt
        embfactory = self.__embfactory
        iterative = False # set iterative : False by default
        if pyembopt is not None: 
            iterative = pyembopt.iterative
            nofde = pyembopt.nofde
        ###
        #def mo_fock_mid_forwd_eval(Dp_ti,fock_mid_ti_backwd,i,delta_t,fock_base,dipole,\
        #                        C,S,imp_opts,U,func_h,bsetH,exA=False,maxit= 10 ,Vemb=None,fout=sys.stderr, debug=False)
        ##
        #if iterative embedding is required  feed fock_base with embedding potential here
        if iterative and not nofde:
           if ( ( i_step % int(pyembopt.fde_offset/dt) ) == 0.0 ):
               #if self.__embopt.debug:
               #      print("i step %i, update Vemb\n" % i_step)

               # make new emb potential
               # transform from the progation basis to the atomic basis (either BO or AO)
               # further transform to the AO basis if Umat is not None

               # [not nofde] condition enforces the iterative condition only for relevant cases, e.g for an external static field the iterative procedure would be redundant
               # trasnformation matrices: C matrix (trasform a density matrix from orthonormal basis to non-orthonormal basis)
               #                          U matrix
               embfactory.set_density(None,self.__Dp,self.__ndocc,C,Umat=U) # set electron density on grid through the density matrix (here we use the density represented on the prop. basis,ovapm=Id)
               #if self.__embopt.debug:
                  #check_el_number = embfactory.rho_integral()
                  #print("rho-integral: %.8f\n" % check_el_number)
               Vemb = embfactory.make_embpot()
               fock_base.set_vemb(Vemb)
        
        ene_container,func_t,fock_mid, Dp_ti_dt = self.__onestep_prop()
       
        self.__field_t = func_t     # collect quantities
        self.__field_list.append(func_t)
        Eh       =ene_container[0]
        Exclow   =ene_container[1]
        ExcAAhigh=ene_container[2]
        ExcAAlow =ene_container[3] 
        
        # expressed in AO basis
        Hcore = fock_base.H()
        if U is not None:
            Hcore = np.matmul(U.T,np.matmul(Hcore,U))
        if ExcAAhigh is None:
            ExcAAhigh = 0
            ExcAAlow  = 0
        ene = 2.0*np.trace( np.matmul(Hcore,self.__D) ) + Eh + Exclow + ExcAAhigh -ExcAAlow 
        self.__ene_list.append(ene)
        
        dipole_avg = np.trace(np.matmul(dip_mat,self.__D))

        if len(self.__perpdip) > 0:
           self.__dperp0_list.append(np.trace(np.matmul(self.__D,self.__perpdip[0])))
           self.__dperp1_list.append(np.trace(np.matmul(self.__D,self.__perpdip[1])))


        self.__dip_list.append(dipole_avg)
        
        if self.__prop_type == 'empc': # for DEBUG  or self.__prop_type == 'etrs':
           #update intermidiate (forward) fock matrix
           self.__Fock_mid = fock_mid
        else: 
           self.__Dp_back = fock_mid #alisased
           
        self.__Dp = Dp_ti_dt
        #real coeffs
        self.__D = np.matmul(C,np.matmul(Dp_ti_dt,C.T))
        # update internal iteration counter
        self.__step_count +=1

    def get_Dmat(self,basis='MO'):
        if basis == 'AO':
            res = self.__D
        elif basis == 'MO':
            res = self.__Dp
        else:
            res = None
        return res

    def get_midpoint_mtx(self):
        return self.__Fock_mid

    def iter_num(self):
        return self.__step_count

    def get_dipole(self):
        return self.__dip_list, self.__dperp0_list, self.__dperp1_list
    def get_energy(self):
        return self.__ene_list

    def get_field_list(self):
        return self.__field_list

    def get_extfield(self):     #current value at iteration k
        return self.__field_t

    def __del__(self):  # ? destructor
      return None
    def set_D(self,Dmat,basis='MO'):
        if basis == 'AO':
            self.__D =Dmat
        elif basis == 'MO':
            self.__Dp = Dmat

    def set_Fock(self,Fmat):
        self.__Fock_mid = Fmat
    def clear(self,i_step=0):
        self.__step_count = i_step
        self.__Fock_mid = None
        self.__Dp = None
        self.__D  = None
        self.__ene_list = []
        self.__dip_list = []
        self.__dperp0_list = [] # the expectation values  of perpendicular components (wrt the boost) of dipole
        self.__dperp1_list = []
        self.__field_t = []
