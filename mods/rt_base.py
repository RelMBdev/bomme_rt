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
      self.__localbasobj = None
      self.__exA_only = exA_only
      self.__Dp       = None
      self.__Fock_mid = Fock_init
      self.__ovapm    = Smat # it can represent either overlap of AO basis functions or BO b. funcs
      self.__Cmat  = Cmat    # Cmat contains MO coefficients (either on AO basis or BO basis)
      self.__dipmat = None
      self.__perpdip = []
      self.__Umat     = U
      self.__D = Dinit
      self.__outfile = out_file
      self.__step_count = 0
      self.__embfactory = None
      self.__embopt = None
      self.__ene_list = []
      self.__dip_list = []
      self.__dperp0_list = [] # the expectation values  of perpendicular components (wrt the boost) of dipole
      self.__dperp1_list = []
      self.__field_t = []

      if isinstance(dipmat,list):
          self.__dipmat = dipmat.pop(0)
          for mtx in dipmat:
               self.__perpdip.append(mtx)
      else:
          self.__dipmat= dipmat
        
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
      self.__embfactory = embpot
      self.__embopt = pyembopt

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
        pyembopt = self.__embopt
        embfactory = self.__embfactory
        iterative = False # set iterative : False by default
        if pyembopt is not None: 
            iterative = pyembopt.iterative
        ###
        #def mo_fock_mid_forwd_eval(Dp_ti,fock_mid_ti_backwd,i,delta_t,fock_base,dipole,\
        #                        C,S,imp_opts,U,func_h,bsetH,exA=False,maxit= 10 ,Vemb=None,fout=sys.stderr, debug=False)
        ##
        #if iterative embedding is required  feed fock_base with embedding potential here
        if iterative:
           if ( ( i_step % int(pyembopt.fde_offset/dt) ) == 0.0 ):
               # make new emb potential
               # transform from the progation basis to the atomic basis (either BO or AO)
               # further transform to the AO basis if Umat is not None
               D_emb = np.matmul(C,np.matmul(self.__Dp,C.T))
               if U is not None:
                   D_emb = np.matmul(U,np.matmul(D_emb,U.T))
               rho = embfactory.set_density(None,D_emb) # set density through the density matrix
               Vemb = embfactory.make_embpot(rho)
               fock_base.set_vemb(Vemb)
        Eh,Exclow,ExcAAhigh,ExcAAlow,func_t,F_ti,fock_mid, Dp_ti_dt = rtutil.mo_fock_mid_forwd_eval(self.__Dp,self.__Fock_mid,\
                            i_step,np.float_(dt), fock_base, dip_mat, C, ovapm, pulse_opts, U, func_h, bsetH, exA_only,fout=fo,debug=False)
        self.__field_t = func_t
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

    def get_midpoint_mtx(self):
        return self.__Fock_mid

    def iter_num(self):
        return self.__step_count

    def get_dipole(self):
        return self.__dip_list, self.__dperp0_list, self.__dperp1_list
    def get_energy(self):
        return self.__ene_list

    def get_extfield(self):
        return self.__field_t

    def __del__(self):  # ? destructor
      return None
