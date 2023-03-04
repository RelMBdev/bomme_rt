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

class real_time():
    def __init__(self,Dinit, Fock_init, fock_factory, ndocc, basis, Smat, pulse_opts, delta_t, Cmat, dipmat,\
            out_file=sys.stderr,  basis_acc = None, func_acc=None,U=None,local_basis=False,\
                                             exA_only=False,occlist=None, virtlist=None):
      
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
      self.__exA_only = exA_only
      self.__occnum  = None
      self.__Dp       = None
      self.__Fock_mid = Fock_init
      self.__ovapm    = Smat # it can represent either overlap of AO basis functions or BO b. funcs
      self.__Cmat  = Cmat    # Cmat contains MO coefficients (either AO or BO)
      self.__dipmat = dipmat
      self.__Umat     = U
      self.__D = Dinit
      self.__outfile = out_file
      self.__step_count = 0
      self.__vemb = None
      self.__ene_list = []
      self.__dip_list = []
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
     
              localbas=Localizer(Dinit,Smat,nbf_A)
              localbas.localize()
              #unsorted orbitals
              unsorted_orbs=localbas.make_orbitals()
              #the projector P
              Phat=np.matmul(Smat[:,:nbf_A],np.matmul(SAA_inv,Smat[:nbf_A,:]))
              #The RLMO are ordered
              # by descending value of the locality parameter.
              sorted_orbs = localbas.sort_orbitals(Phat)
              #the occupation number and the locality measure
              locality,occnum=localbas.locality()
              self.__occnum = occnum
              #save the localization parameters and the occupation numbers  
              np.savetxt('locality_rlmo.out', np.c_[locality,occnum], fmt='%.12e')
              #check the occupation number of ndimA=nbf_A MOs (of A).
              occA = int(np.rint( np.sum(occnum[:nbf_A]) ))
              print("ndocc orbitals (A) after localization: %i\n" % occA)
              #set the sorted as new basis 
              C=sorted_orbs
              #use the density corresponding to (sorted) localized orbitals
              Dp_0 = np.diagflat(occnum)
      else:
        Dp_0=np.zeros((self.__numbasis,self.__numbasis))
        for num in range(int(ndocc)):
            Dp_0[num,num]=1.0
      self.__Dp = Dp_0     

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

       if self.__locbasis and (self.__virtlist[0] == -99) and selective_pert:
            #use occnum to define a virtlist
            virtlist=[]
            for m in range(numbas):
              if np.rint(np.abs(self.__occnum))[m] < 1.0: 
                virtlist.append(m+1)
            dip_mo=dipole_selection(dip_mo,-1,ndocc,occlist,virtlist,self.__outfile,debug)
            self.__virtlist = virtlist
       elif selective_pert:
             dip_mo=dipole_selection(dip_mo,virtlist[0],ndocc,occlist,virtlist,self.__outfile,debug)
           
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

    def set_vemb(self,embpot):
      self.__vemb = embpot

    def __call__(self):
        i_step = self.__step_count
        dt = self.__delta_t
        U = self.__Umat
        C = self.__Cmat # can be generalized to loewdin orthogonalization
        pulse_opts = self.__pulse_param
        fock_base = self.__ffactory
        dip_mat = self.__dipmat
        ovapm = self.__ovapm
        fo = self.__outfile
        bsetH = self.__basisobj_acc
        func_h = self.__func_acc
        exA_only = self.__exA_only

        ###
        #def mo_fock_mid_forwd_eval(Dp_ti,fock_mid_ti_backwd,i,delta_t,fock_base,dipole,\
        #                        C,S,imp_opts,U,func_h,bsetH,exA=False,maxit= 10 ,Vemb=None,fout=sys.stderr, debug=False)
        ##

        Eh,Exclow,ExcAAhigh,ExcAAlow,func_t,F_ti,fock_mid, Dp_ti_dt = rtutil.mo_fock_mid_forwd_eval(self.__Dp,self.__Fock_mid,\
                            i_step,np.float_(dt), fock_base, dip_mat, C, ovapm, pulse_opts, U, func_h, bsetH, exA_only,fout=fo,debug=False, Vemb=self.__vemb)
        
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

        self.__dip_list.append(dipole_avg)

        #update intermidiate (forward) fock matrix
        self.__Fock_mid = fock_mid
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
    def iter_num(self):
        return self.__step_count

    def __del__(self):  # ? destructor
      return None
