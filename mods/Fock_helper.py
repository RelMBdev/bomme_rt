import numpy as np
import psi4
import scipy.linalg
from pkg_resources import parse_version

class datacommon():
    def __init__(self,Hcore,ovap,functional,basis,jkbase,dipmat,Ccoef):
        self.__Hcore = Hcore
        self.__ovapm = ovap
        self.__func  = functional
        self.__basis = basis
        self.__jk    = jkbase
        self.__dipmat = dipmat
        self.__Ccoef  = Ccoef #the ground state C coeff matrix
    def Hcore(self):
        return self.__Hcore
    def S(self):
        return self.__ovapm
    def funcname(self):
        return self.__func
    def basisobj(self):
        return self.__basis
    def jk_factory(self):
        return self.__jk
    def eri(self):
        return self.__jk.eri()
    def dipole(self):
        return self.__dipmat
    def Ccoef(self):
        return self.__Ccoef

class jkfactory():
    def __init__(self,bset,molobj,jknative=True,scf_type='DIRECT',eri=None,real_time=False):
        self.__jkflag = jknative
        self.__basisobj = bset
        self.__scftype = scf_type
        self.__eri = eri
        self.__eri_axis = None
        self.__jk = None
        self.__rtfit = real_time # use density fitting the get the correction to the real-time K matrix when jkclass is on
        if jknative:
            #intialize native Psi4 jk object
            if (scf_type=='DIRECT' or scf_type=='PK'): 
               print("using %s scf and JK class\n" % scf_type)
               jk = psi4.core.JK.build(bset)
            elif scf_type == 'MEM_DF' or scf_type == 'DISK_DF':
               print("using %s\n" % scf_type)
               auxb = psi4.core.BasisSet.build(molobj,"DF_BASIS_SCF", "", fitrole="JKFIT",other=psi4.core.get_global_option("BASIS"))
               jk = psi4.core.JK.build_JK(bset,auxb)
            else:
                 print(scf_type)
                 raise Exception("Invalid scf_type.\n")
            jk.set_memory(int(4.0e9))  # 1GB
            jk.set_do_wK(False)
            jk.initialize()
            jk.print_header()
            #assign to a member
            self.__jk = jk

        if (eri is not None):
            print("doing tensor contraction")
            self.__eri_axis  = len(eri.shape)

    def eri(self):
        return self.__eri

    def set_rt_Kfit(self):
        self.__rtfit = True

    def J(self,Cocc,Dmat=None,sum_idx=None,out=None):#Dmat and Cocc are expressed on the Ao basis
           nbf = self.__basisobj.nbf()  # C/Den matrix passed in full dimension, slicing 
                                        # if needed is carried out afterwards
           if not isinstance(Dmat,np.ndarray):
               Dmat = np.matmul(Cocc,np.conjugate(Cocc.T))
           # prepare standard index list
           if (sum_idx is  None) and (out is None):
             Idx= [0,nbf,0,nbf]  # define the subblock nrow,ncol for contraction
             idx = [0,nbf,0,nbf] #slicing the result
           elif (sum_idx is not None):
             Idx = []
             idx = [0,nbf,0,nbf]
             
             for el in sum_idx:
                 
                 for subel in el:
                    Idx.append(subel)
             
           elif (out is not None):
             idx = []  
             Idx = [0,nbf,0,nbf]
             for elm in out:
                 
                 for subel in elm:
                    idx.append(subel)                    
           if self.__jkflag :
              if not isinstance(Cocc,np.ndarray):
                 raise Exception("Cocc is not np.ndarray")
              self.__jk.C_left_add(psi4.core.Matrix.from_array(Cocc))
              self.__jk.compute()
              self.__jk.C_clear()
              Jmat=np.array(self.__jk.J()[0]) #copy into J

           elif (self.__eri is not None):
               #print("nbf: %i\n" % nbf)
               #print(self.__eri.shape)
               #print("idx and Idx")
               #print(idx,Idx)
               if not (self.__eri_axis < 4):
                  eri = self.__eri[idx[0]:idx[1],idx[2]:idx[3],Idx[0]:Idx[1],Idx[2]:Idx[3]]
                  Jmat=np.einsum('pqrs,rs->pq', eri, Dmat[Idx[0]:Idx[1],Idx[2]:Idx[3]])
               else:  
                  X_Q = np.einsum('Qpq,pq->Q', self.__eri[:,Idx[0]:Idx[1],Idx[2]:Idx[3]], Dmat[Idx[0]:Idx[1],Idx[2]:Idx[3]])
                  Jmat = np.einsum('Qpq,Q->pq', self.__eri, X_Q)[idx[0]:idx[1],idx[2]:idx[3]]

           return Jmat      


    def K(self,Cocc,Dmat=None,sum_idx=None,out=None):
            nbf = self.__basisobj.nbf()
            if not isinstance(Dmat,np.ndarray):
                Dmat = np.matmul(Cocc,np.conjugate(Cocc.T))
            # prepare standard index list
            if (sum_idx is  None) and (out is None):
              Idx= [0,nbf,0,nbf]  # define the subblock nrow,ncol for contraction
              idx = [0,nbf,0,nbf] #slicing the result
            elif (sum_idx is not None):
              Idx = []
              idx = [0,nbf,0,nbf]
              
              for elm in sum_idx:
                  
                  for subel in elm:
                     Idx.append(subel)
            elif (out is not None):
              idx = []  
              Idx = [0,nbf,0,nbf]
              for elm in out:
                  
                  for subel in elm:
                     idx.append(subel)
            #print(Idx)
            #print(idx)
            if self.__jkflag :
               if not isinstance(Cocc,np.ndarray):
                  raise Exception("Cocc is not np.ndarray")
               self.__jk.C_left_add(psi4.core.Matrix.from_array(Cocc))
               self.__jk.compute()
               self.__jk.C_clear()
               Kmat=np.array(self.__jk.K()[0])[idx[0]:idx[1],idx[2]:idx[3]] #
               # the native jkclass will use real mol. orbital (nat orbitals of D.real) to compute K, neglecting
               # the imaginary part of D (D.imag)
               if self.__rtfit:
                  #print("debug: fitting the residual K.imag")
                  if not (self.__eri_axis < 4):
                     raise Exception("need fitted 3-index tensor here") 
                  Z_Qqr = np.einsum('Qrs,sq->Qrq', self.__eri[:,idx[2]:idx[3],Idx[2]:Idx[3]], (Dmat.imag)[Idx[0]:Idx[1],Idx[2]:Idx[3]])
                  tmp = np.einsum('Qpq,Qrq->pr', self.__eri[:,idx[0]:idx[1],Idx[0]:Idx[1]], Z_Qqr)
                  Ktmp = np.zeros((nbf,nbf),dtype=np.complex128)
                  Ktmp.real = Kmat
                  Ktmp.imag = tmp
                  Kmat = np.asarray(Ktmp)
            elif (self.__eri is not None):
               if not self.__eri_axis < 4:
                  #'prqs,rs->pq'
                  eri = self.__eri[idx[0]:idx[1],Idx[0]:Idx[1],idx[2]:idx[3],Idx[2]:Idx[3]] 
                  Kmat=np.einsum('prqs,rs->pq', eri, Dmat[Idx[0]:Idx[1],Idx[2]:Idx[3]])
               else:
                  #print("debug fitted_eri Kmat")
                  Z_Qqr = np.einsum('Qrs,sq->Qrq', self.__eri[:,idx[2]:idx[3],Idx[2]:Idx[3]], Dmat[Idx[0]:Idx[1],Idx[2]:Idx[3]])
                  Kmat = np.einsum('Qpq,Qrq->pr', self.__eri[:,idx[0]:idx[1],Idx[0]:Idx[1]], Z_Qqr)
            return Kmat
       
class dft_xc():
    def __init__(self,bset,funcname):
             self.__bset = bset
             self.__alpha = None
             self.__restricted =None
             self.__funcname =funcname
             self.__sup = None
    def get_xc(self,Dmat):         
             if parse_version(psi4.__version__) >= parse_version('1.3a1'):
                  build_superfunctional = psi4.driver.dft.build_superfunctional
             else:
                  build_superfunctional = psi4.driver.dft_funcs.build_superfunctional
                
             # assuming
             
             self.__restricted = True
             self.__sup= build_superfunctional(self.__funcname, self.__restricted)[0]
             self.__sup.set_deriv(2)
             self.__sup.allocate()
             nbf = self.__bset.nbf()
             vname = "RV"
             if not self.__restricted:
                 vname = "UV"
             potential=psi4.core.VBase.build(self.__bset,self.__sup,vname)
             Dm=psi4.core.Matrix.from_array(Dmat[:nbf,:nbf].real)
             potential.initialize()
             potential.set_D([Dm])
             nbf=Dmat.shape[0]
             VxcAAhigh=psi4.core.Matrix(nbf,nbf)
             potential.compute_V([VxcAAhigh])
             potential.finalize()
             # compute the high level XC energy  on Dbo AA
             ExcAAhigh= potential.quadrature_values()["FUNCTIONAL"]
             VxcAAhigh = np.copy(VxcAAhigh)
             del potential
             del  build_superfunctional
             return VxcAAhigh,ExcAAhigh
    def clean(self):
        del self.__sup
        self._alpha = None
        self.__bset = None
        self.__restricted = None
        self.__funcname = None
        self.__restricted = None
    def __del__(self):
        return None
    def is_x_hybrid(self):
        res = self.__sup.is_x_hybrid()
        return res
    def x_alpha(self):
        if self.is_x_hybrid():
               alpha = self.__sup.x_alpha()
        else:
               alpha = None
        self.__alpha = alpha
        return alpha

class fock_factory():
    def __init__(self,jk_fact,Hmat,ovapm,funcname=None,basisobj=None):
          self.__supfunc = None
          self.__Hcore = Hmat
          self.__S = ovapm
          self.__basis = basisobj
          self.__func = funcname
          self.__restricted = True #default
          #self.__basis_h = None # the basis of the 'high-level' subsys
          self.__jkfact = jk_fact
    # just a handle
    def func(self):
        return self.__func
    def H(self):
        return self.__Hcore
    def S(self):
        return self.__S
    def basisobj(self):
        return self.__basis

    def J(self,Cocc,Dmat=None,sum_str=None,out=None,U=None):
          if not isinstance(Cocc,np.ndarray):
              raise TypeError("Cocc must be np.ndarray")
          res = self.__jkfact.J(Cocc,Dmat,sum_str,out)
          if isinstance(U,np.ndarray):
             res = np.matmul(U.T,np.matmul(res,U))
          return res
    def K(self,Cocc,Dmat=None,sum_str=None,out=None,U=None):
          if not isinstance(Cocc,np.ndarray):
              raise TypeError("Cocc must be np.ndarray")
          res = self.__jkfact.K(Cocc,Dmat,sum_str,out)
          if isinstance(U,np.ndarray):
             res = np.matmul(U.T,np.matmul(res,U))
          return res
    def get_xcpot(self,func,bset,Dmat=None,Cocc=None,exmodel=0,return_ene=False):
          # exmodel = 0 and basis=basis_total make the Vxc coincide with the standard supermolecular Vxc matrix
          # basis denote the basis in which the final Fock mtx 
          # will be expressed (total basis or fragment basis)
          # Cocc and Dmat are passed in full dimension (i.e Cocc.shape[0] > basis.nbf()
          if not isinstance(Dmat,np.ndarray):
              Dmat = np.matmul(Cocc,np.conjugate(Cocc.T))
          nbf = bset.nbf()
          # the basis set could either fragment (A or B) basis or total basis (A U B)        
          if func=='hf':
             if not isinstance(Cocc, np.ndarray):
                TypeError("Cocc: input must be np.ndarray")
                
             Cocc_A=np.zeros_like(Cocc)
             Cocc_A[:nbf,:]=np.asarray(Cocc)[:nbf,:]
                
             #DEBUG
             #check=np.matmul(Cocc_A,Cocc_A.T)
             #print("Dbo[:nbf,:nbf] & Dbo[Cocc[:nbf,:]] : %s\n" % np.allclose(Dbo.np[:nbf,:nbf],check[:nbf,:nbf]))
             
             
        
             K = self.__jkfact.K(Cocc_A,Dmat,sum_idx=[[0,nbf],[0,nbf]],out=[[0,nbf],[0,nbf]])    #assuming Exc0 model?
             #Exc_ex0 =  -np.trace(np.matmul(Dbo.np[:na,:na],K))
             #print("ExceAAhigh EX0 mod: %.10e\n" % Exc_ex0)
             #exchange model 1
             if exmodel==1:
                 print("exmodel != 0")
                 nlim = Dmat.shape[0]
                 if nbf >= nlim:
                     raise Exception("Check dimension of sub sys basis set\n")
                 Cocc_B=np.zeros_like(Cocc)
                 Cocc_B[nbf:nlim,:]=np.asarray(CoccAO)[nbf:nlim,:] #see J. Chem. Theory Comput. 2017, 13, 1605-1615
                 
                 #DEBUG
                 #check=np.matmul(Cocc_B,Cocc_B.T)
                 #print("Dbo[na:,na:] & Dbo[Cocc[na:,:]] : %s\n" % np.allclose(Dbo.np[na:,na:],check[na:,na:]))
                 K1 = self.__jkfact.K(Cocc_B,Dmat,sum_idx=[[nbf,nlim],[nbf,nlim]],out=[[0,nbf],[0,nbf]])
                
             
                 #Exc_ex1 =  -np.trace(np.matmul(Dbo.np[:na,:na],K1))
                 #print("EX1 energy: %.10e\n" % Exc_ex1)
                 K += K1
             VxcAAhigh = -K  # or as psi4.core.Matrix object, use .from_array()
             ExcAAhigh = -np.trace(np.matmul(Dmat[:nbf,:nbf],K))
          
          else:
               dft_pot=dft_xc(bset,func) 
               VxcAAhigh, ExcAAhigh = dft_pot.get_xc(Dmat)
          
             
               if dft_pot.is_x_hybrid():
                  alpha = dft_pot.x_alpha()
                  Cocc_A=np.zeros_like(Cocc)
                  Cocc_A[:nbf,:]=np.asarray(Cocc)[:nbf,:]
               
               
                  K = self.__jkfact.K(Cocc_A,out=[[0,nbf],[0,nbf]])    
                  if exmodel==1:
                      nlim = Dmat.shape[0]
                      if nbf >= nlim:
                          raise Exception("Check dimension of sub sys basis set\n")
                      Cocc_B=np.zeros_like(Cocc)
                      Cocc_B[nbf:,:]=np.asarray(CoccAO)[nbf:,:] #see J. Chem. Theory Comput. 2017, 13, 1605-1615
                      
                  
                      K1 = self.__jkfact.K(Cocc_B,sum_idx=[[nbf,nlim],[nbf,nlim]],out=[[0,nbf],[0,nbf]])
                  
                      K += K1
                  if np.iscomplexobj(K):
                    tmp = -alpha*K
                    tmp.real += VxcAAhigh
                    VxcAAhigh = tmp
                  else:
                    VxcAAhigh += -alpha*K
                  ExcAAhigh += -alpha*np.trace(np.matmul(Dmat[:nbf,:nbf],K))
               dft_pot.clean() #clean
               del dft_pot
          if return_ene:

             return VxcAAhigh, ExcAAhigh
          else:
             return VxcAAhigh

    def get_Fock(self,Cocc=None,Dmat=None,return_ene=False):
          if (not isinstance(Cocc, np.ndarray)) and (not isinstance(Dmat, np.ndarray)):
              raise Exception("Cocc and D are both None")
          elif not isinstance(Cocc,np.ndarray):
               ndocc = int(np.rint(np.trace(np.matmul(Dmat,self.__S)).real))
               Cocc = np.zeros((Dmat.shape[0],ndocc))
 #             print(ndocc)
 #             diagonalize to get Cocc
               den_op = np.matmul(self.__S,np.matmul(Dmat.real,self.__S))
               w, v  = scipy.linalg.eigh(den_op,self.__S)
               idx_w = w.argsort()[::-1]
               w = w[idx_w]
               v = v[:,idx_w]
               Cocc = v[:,:ndocc]
               #test_dmat =np.matmul(Cocc,Cocc.T)
               #print("check dmat real: %s\n" % np.allclose(Dmat.real,test_dmat))
               #print(np.linalg.norm(np.abs(Dmat.real-test_dmat),'fro'))
          #else:
          #    Dmat = np.matmul(Cocc,np.conjugate(Cocc.T))
          Vxc = self.get_xcpot(self.__func,self.__basis,Dmat,Cocc,return_ene=return_ene)
          J = self.J(Cocc,Dmat)
          if return_ene:
             # if return_ene=True in get_xcpot() a tuple will be returned
             res = self.__Hcore +2.0*J +Vxc[0]
             Jene = 2.00*np.trace( np.matmul(J,Dmat) )
             return Jene, Vxc[1],res
          else:
             res = self.__Hcore +2.0*J +Vxc
             return res
    
    # a cleaner implementation of the block-orthogonalized Fock
    # for ground state, only the orbital should be provided.
    # basis_acc is the basis set of the small high-level-theory subsys 
    def get_bblock_Fock(self,Cocc=None,Dmat=None,func_acc=None,basis_acc=None,U=None,return_ene=False):
          # Dmat & Cocc are provided in the block-orthogonalized basis (BO)
          if (not isinstance(Cocc, np.ndarray)) and (not isinstance(Dmat, np.ndarray)):
              raise Exception("Cocc and D are both None")
          elif not isinstance(Cocc,np.ndarray):
               ndocc = int(np.rint(np.trace(np.matmul(Dmat,self.__S)).real))
               Cocc = np.zeros((Dmat.shape[0],ndocc))
          #    print(ndocc)
          #    diagonalize to get Cocc
               den_op = np.matmul(self.__S,np.matmul(Dmat.real,self.__S))
               w, v  = scipy.linalg.eigh(den_op,self.__S)
               idx_w = w.argsort()[::-1]
               w = w[idx_w]
               v = v[:,idx_w]
               Cocc = v[:,:ndocc]
          #    test_dmat =np.matmul(Cocc,Cocc.T)
          #    print("check dmat real: %s\n" % np.allclose(Dmat.real,test_dmat))
          #    print(np.linalg.norm(np.abs(Dmat.real-test_dmat),'fro'))
          
          if not isinstance(U,np.ndarray):
             raise TypeError("U must be np.ndarray")
          
          # the two-electron part corresponding to the low-level theory (on the full Dmat/basis)
          # get Dmat/Cocc in the AO basis
          if isinstance(Dmat, np.ndarray):
             Dmat_ao = np.matmul(U,np.matmul(Dmat,U.T)) 
          else:
             Dmat_ao = None

          Cocc_ao = np.matmul(U,Cocc)

          Vxc = self.get_xcpot(self.__func,self.__basis,Dmat_ao,Cocc_ao,return_ene=return_ene)
          J = self.J(Cocc_ao,Dmat_ao)
          # ao/bo subscript omitted when the basis used to express a given quantity 
          # can be inferred from the context
          
          H_bo = np.matmul(U.T,np.matmul(self.__Hcore,U))
 
          # the two-electron part corresponding to the low-level-theory calculated on D_AA
          VxcAA_low = self.get_xcpot(self.__func,basis_acc,Dmat,Cocc,return_ene=return_ene)
          
          # the two-electron part corresponding to the high-level-theory subsys (on D_AA)
          VxcAA_high = self.get_xcpot(func_acc,basis_acc,Dmat,Cocc,return_ene=return_ene)
          
          tmp=np.zeros( (self.__basis.nbf(),self.__basis.nbf()) )
          if return_ene:
             # if return_ene=True in get_xcpot() a tuple will be returned
             tmp[:basis_acc.nbf(),:basis_acc.nbf()] = VxcAA_high[0] - VxcAA_low[0]

             # two_el_bo is the two-electron term (J+Vxc) expressed on the BO basis
             two_el_bo = np.matmul(U.T,np.matmul( ( 2.0*J+Vxc[0] ),U))
             
             res = H_bo + two_el_bo + tmp
             Eh = 2.00*np.trace( np.matmul(J,Dmat_ao) )
             return Eh, Vxc[1], VxcAA_low[1], VxcAA_high[1], res
          else:
             tmp[:basis_acc.nbf(),:basis_acc.nbf()] = VxcAA_high - VxcAA_low

             # two_el_bo is the two-electron term (J+Vxc) expressed on the BO basis
             two_el_bo = np.matmul(U.T,np.matmul( ( 2.0*J+Vxc ),U))
             
             res = H_bo + two_el_bo + tmp
             return res
    
    def __del__(self):
         return None
