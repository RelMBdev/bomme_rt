import scipy.linalg
from scipy.linalg import fractional_matrix_power
import numpy as np

def maxElem(a,idA=[-1],idB=[-1]): # Find largest off-diag. element a[k,l]
    n = len(a)
    aMax = 0.0
    for i in range(n-1):
        for j in range(i+1,n):
            if (j in idB) and (i in idA):
              continue
            if abs(a[i,j]) >= aMax:
                aMax = abs(a[i,j])
                k = i; l = j
    return aMax,k,l




def calc_off(A,m):
  offA=0.0
  for i in range(m):
     for j in range(m):
        if ( i != j ):
            offA+=(A[i,j])**2.0

  offA=np.sqrt(offA)
  return offA



def sym_schur2d(A,p,q):
    if (p >q):
        print("check sym_schur")
    if ( A[p,q] != 0.0 ):
      tau = (A[q,q]- A[p,p])/A[p,q]
      t = np.sign(tau)/(np.abs(tau)+np.sqrt(tau*tau + 1.))
      c = 1.0/(np.sqrt(1.0 + t*t))
      s = c*t
    else:
      c = 1.0
      s = 0.0
    return c,s



class Localizer():
    def __init__(self,density,metric,nbasA,tol=1.0e-8,maxit=1000):

        self.__S = None
        self.__bobasis = None
        self.__nbfa = None
        self.__D = density
        self.__RLMO = None
        self.__locality_par = None
        self.__occnum = None
        self.__maxit= maxit
        self.__tol=tol
        self.__eigvalsA = None
        self.__eigvalsB = None
        self.__numbas = None
        self.__Tmat = None
        self.__DRo = None
        self.__DRo1st = None
        self.__idA = None
        self.__idB = None
        self.__one_Jacobi = None
        self.__fro = None
        self.__fro_init = None
        self.__fro_1stage = None
        self.__offA = None
        self.__res_offA = None
        self.__offA_1stage = None
        self.__UU = None
        
        self.initialize(metric,nbasA)
 
    def add_D(self,density):

        self.__D = density

    def initialize(self,metric,nbasA):

        self.__S = metric
        self.__nbfa = nbasA

        #check if the base is block-orthogonalized
        test=np.zeros_like(metric)
        test[nbasA:,:nbasA] =metric[nbasA:,:nbasA]
        test+=test.T

        if calc_off(test,self.__S.shape[0]) < 1.0e-5:
            self.__bobasis = True
            #replace D -> SDS: the density operator to be diagonalized
            self.__D=np.matmul(self.__S,np.matmul(self.__D,self.__S))
            try:
                   self.__eigvalsA,Taa=scipy.linalg.eigh(self.__D[:self.__nbfa,:self.__nbfa],self.__S[:self.__nbfa,:self.__nbfa],eigvals_only=False)
            except scipy.linalg.LinAlgError:
                   print("Error in scipy.linalg.eigh")
            try:
                   self.__eigvalsB,Tbb=scipy.linalg.eigh(self.__D[self.__nbfa:,self.__nbfa:],self.__S[self.__nbfa:,self.__nbfa:],eigvals_only=False)
            except scipy.linalg.LinAlgError:
                   print("Error in scipy.linalg.eigh")

        else:
            self.__bobasis = False
            O = fractional_matrix_power(np.array(self.__S), 0.5)

            self.__D=np.matmul(O,np.matmul(self.__D,O))
            try:
                   self.__eigvalsA,Taa=np.linalg.eigh(self.__D[:self.__nbfa,:self.__nbfa])
            except np.linalg.LinAlgError:
                   print("Error in np.linalg.eigh")
            try:
                   self.__eigvalsB,Tbb=np.linalg.eigh(self.__D[self.__nbfa:,self.__nbfa:])
            except np.linalg.LinAlgError:
                   print("Error in scipy.linalg.eigh")
        idxA = self.__eigvalsA.argsort()[::-1]
        self.__eigvalsA = self.__eigvalsA[idxA]
        Taa = Taa[:,idxA]
        idxB = self.__eigvalsB.argsort()[::-1]
        self.__eigvalsB = self.__eigvalsB[idxB]
        Tbb = Tbb[:,idxB]
 
        self.__numbas=self.__D.shape[0]
        self.__Tmat=np.zeros((self.__numbas,self.__numbas))
        self.__Tmat[:self.__nbfa,:self.__nbfa]=Taa
        self.__Tmat[self.__nbfa:,self.__nbfa:]=Tbb
        
        self.__DRo=np.matmul(np.conjugate(self.__Tmat.T),np.matmul(self.__D,self.__Tmat))
        self.__idA=[]
        self.__idB=[]
        for k in range(len(self.__eigvalsA)):
            if (self.__eigvalsA[k]*2.0 < 1.3) and (self.__eigvalsA[k]*2.0 > 0.7):
               self.__idA.append(k)
        for k in range(len(self.__eigvalsB)):
            if (self.__eigvalsB[k]*2.0 < 1.3) and (self.__eigvalsB[k]*2.0 > 0.7):
               self.__idB.append(k+self.__nbfa) # ! the index in eivalsB list are shifted. To go back to the proper index, add -nbfa
        if len(self.__idA)==0:
           self.__one_Jacobi=True    
        else:
           self.__one_Jacobi=False  #the second stage of the localization involve singly occupied orbitals

    def localize(self):
        self.__offA=calc_off(self.__DRo,self.__nbfa)
        self.__fro=np.linalg.norm(self.__DRo,'fro')
        self.__fro_init=self.__fro
        
        # residual offA (res_offA) contributes to the total offA
        self.__res_offA=0.0
        for m in self.__idA:
            for n in self.__idB:
               self.__res_offA+=self.__DRo[m,n]**2
        self.__res_offA=np.sqrt(2.*self.__res_offA)

        self.__UU = np.eye(self.__numbas)

        #first stage
        jiter=0
        while True:

            Mmax,p,q = maxElem(self.__DRo,self.__idA,self.__idB)
            c,s=sym_schur2d(self.__DRo,p,q)
            J=np.eye(self.__numbas)
            J[p,p]=c
            J[q,q]=c
            J[p,q]=s
            J[q,p]=-s
            self.__DRo=np.matmul(J.T,np.matmul(self.__DRo,J))
            self.__UU=np.matmul(self.__UU,J)
            self.__offA=calc_off(self.__DRo,self.__numbas)
            if self.__one_Jacobi:
                if ( self.__offA< self.__tol*self.__fro):
                 break
            else:
                if ( abs(self.__offA-self.__res_offA)<self.__tol):
                 break
                elif jiter > self.__maxit:
                 raise Exception("Jacobi diagonalization not converged in %i steps\n" % self.__maxit)
                 

            jiter+=1
            
        #check the (invariant) 2-norm of Dro after the first Jacobi round
        self.__fro_1stage=self.__fro
        self.__offA_1stage=self.__offA

        self.__DRo1st = self.__DRo
        if (not self.__one_Jacobi):
        #second stage
            while True:
       
              Mmax,p,q = maxElem(self.__DRo)
              c,s=sym_schur2d(self.__DRo,p,q)
              J=np.eye(self.__numbas)
              J[p,p]=c
              J[q,q]=c
              J[p,q]=s
              J[q,p]=-s
              self.__DRo=np.matmul(J.T,np.matmul(self.__DRo,J))
              self.__UU=np.matmul(self.__UU,J)
              self.__offA=calc_off(self.__DRo,self.__numbas)
              if ( self.__offA< self.__tol*self.__fro):
               break
        #check: (invariant) 2-norm of Dro
        self.__fro=np.linalg.norm(self.__DRo,'fro')

    def DRo(self):
     
        return self.__DRo
    def DRo1st(self):
     
        return self.__DRo1st

    def get_singly(self):
        indexA = self.__idA
        indexB = []

        for k in self.__idB:
            indexB.append(k-self.__nbfa)

        return indexA,indexB
    def get_singlyval(self):

        indexA = self.__idA
        indexB = []

        for k in self.__idB:
            indexB.append(k-self.__nbfa)
        return self.__eigvalsA[indexA],self.__eigvalsB[indexB]

    def UU(self):
     
        return self.__UU

    def Tmat(self):
     
        return self.__Tmat

    def set_options(self,tolval,maxitval):
        self.__tol   = tolval
        self.__maxit = maxitval

    def eigenvalues(self,label='A'):
        if label == 'A':
          return self.__eigvalsA
        elif label == 'B':
          return self.__eigvalsB
        else:
           raise Exception("Invalid label (A or B)\n")

    def make_orbitals(self):
        if self.__bobasis:
          self.__RLMO =np.matmul(self.__Tmat,self.__UU)
        else:
          B = scipy.linalg.fractional_matrix_power(self.__S, -0.5)
          self.__RLMO =np.matmul(B,np.matmul(self.__Tmat,self.__UU))
        return self.__RLMO

    def sort_orbitals(self,projector):
        occnum = np.diagonal(self.__DRo) 
        loc=[] # on (A) subsys basis
        for i in range(self.__RLMO.shape[1]):
            tmp=np.matmul(self.__RLMO[:,i].T,np.matmul(projector,self.__RLMO[:,i]))
            loc.append(tmp)
        self.__locality_par = np.array(loc)
        self.__occnum = occnum
        idx = self.__locality_par.argsort()[::-1]
        return self.__RLMO[:,idx]

    def locality(self):
        idx = self.__locality_par.argsort()[::-1]
        return self.__locality_par[idx], self.__occnum[idx]
    def occnum(self):
        res = np.diagonal(self.__DRo) 
        return res
    def dump_results(self):
        fo = open('loc_result.txt', "w")
        fo.write("BO basis : ..... %s\n" % str(self.__bobasis))
        fo.write("Initial 2-norm of D_RO : %.8f\n" % self.__fro_init)
        fo.write("After 1st (diagonalization) stage the off-diagonal contribution to the 2-norm was : %.8f\n" % self.__offA_1stage)
        fo.write("After 1st stage the off-diagonal contribution to the 2-norm from SO elements was: %.8f\n" % self.__res_offA)
        fo.write("Singly occupied  orbitals: %s\n" % (not self.__one_Jacobi))
        fo.write("Final off-diagonal contribution to the 2-norm is: %.8f\n" % self.__offA)
        
        
        test=np.matmul(self.__RLMO.T,np.matmul(self.__S,self.__RLMO))
        check=np.allclose(np.eye(self.__numbas),test)
        fo.write("C_RLMO satisy the usual C^T S C relation=1 : %s\n" %check)
        checkdiag = np.matmul(np.conjugate(self.__RLMO.T),np.matmul(self.__D,self.__RLMO))
        fo.write("C_RLMO diagonalize SDS_tilde : % s\n" % np.allclose(checkdiag,self.__DRo))
        fo.write("Trace of D_RLMO : %.8f\n" % np.trace(self.__DRo))
        fo.close()
        


    def make_canonical(self,Fock,projector,threshold):
        locality,occnum = self.locality()
        nAA = self.__nbfa
        occnumA = occnum[:nAA]
        mask = np.abs(locality[:nAA]) > threshold
        occnumA = occnumA[mask]
        lenA = occnumA.shape[0]
        ndoccA = int(np.rint( np.sum(occnumA) ))
        #print("Frag. A docc : %i" % ndoccA)
        occnumB = occnum[lenA:]
        idoccA = occnumA.argsort()[::-1]
        idoccB = occnumB.argsort()[::-1]
        sorted_orbs = self.sort_orbitals(projector)
        C_AA = (sorted_orbs[:,:lenA])[:,idoccA]
        C_BB = (sorted_orbs[:,lenA:])[:,idoccB]
        FpA=np.matmul(np.conjugate(C_AA.T),np.matmul(Fock,C_AA))
        try: 
           epsF,new_C=np.linalg.eigh(FpA)
        except scipy.linalg.LinAlgError:
           print("Error in np.linalg.eigh")
        C_AA_can=np.matmul(C_AA,new_C)
        #
        FpB=np.matmul(np.conjugate(C_BB.T),np.matmul(Fock,C_BB))
        try: 
           epsF,new_C=np.linalg.eigh(FpB)
        except scipy.linalg.LinAlgError:
           print("Error in np.linalg.eigh")
        C_BB_can=np.matmul(C_BB,new_C)
        #test=np.matmul(C_sup_canon.T,np.matmul(Fock,C_sup_canon))
        #print(np.allclose(test,np.diagflat(epsF)))
        C_sup=np.zeros_like(Fock)
        C_sup[:,:lenA]=C_AA_can
        C_sup[:,lenA:]=C_BB_can
        loc=[] # on (A) subsys basis
        for i in range(C_sup.shape[1]):
            tmp=np.matmul(C_sup[:,i].T,np.matmul(projector,C_sup[:,i]))
            loc.append(tmp)
        np.savetxt("locality_can.txt",np.array(loc))
        return lenA,ndoccA,C_AA_can, C_BB_can

class Spade():
    def __init__(self,Cmat,ovap,saa_inv,nbasA):

        self.__S = None
        self.__nbfa = None
        self.__Cocc = Cmat
        self.__Cocc_spade = None
        self.__Saa_inv = None
        self.__Phat = None #the projector in BO basis
        self.__locality_par = None
        self.__sigma = None
        self.initialize(ovap,saa_inv,nbasA)
 
    def set_Cocc(self,Cmat):
        
        self.__Cocc = Cmat

    def initialize(self,ovap,saa_inv,nbasA):

        self.__S = ovap
        self.__Saa_inv = saa_inv
        self.__nbfa = nbasA
        self.__Phat=np.matmul(self.__S[:,:self.__nbfa],np.matmul(self.__Saa_inv,self.__S[:self.__nbfa,:]))

    def localize(self):
        CoccA_bar = np.matmul(self.__Phat,self.__Cocc)
        try:
          u, self.__sigma, vh = np.linalg.svd(CoccA_bar)
        except np.linalg.LinAlgError:
          print("Error in numpy.linalg.svd")


        self.__Cocc_spade= np.matmul(self.__Cocc,np.conjugate(vh.T))
        return self.__Cocc_spade
    def get_locality(self):
        loc=[]
        for i in range(self.__Cocc_spade.shape[1]):
            tmp=np.matmul(self.__Cocc_spade[:,i].T,np.matmul(self.__Phat,self.__Cocc_spade[:,i]))
            loc.append(tmp)
        self.__locality_par = np.array(loc)

        return self.__locality_par

    def get_sigmavals(self):

        return np.diagonal(self.__sigma)

    def get_projector(self):
        return self.__Phat
    def Cocc(self):
        return self.__Cocc_spade
