"""
Aux functions for transition density analysis
"""

__authors__   =  "Matteo De Santis"
__credits__   =  ["Matteo De Santis"]

__copyright__ = "(c) 2020, MDS"
__license__   = "BSD-3-Clause"
__date__      = "2020-03-01"

import numpy as np
import psi4
import scipy.linalg
import os 

#also specified in 'CUBIC_GRID_OVERAGE' 
#spacing
def cubedata(gobj,L = [7.0,7.0,7.0],D = [0.1,0.1,0.1]):

  Xmin = np.zeros(3)
  Xmax = np.zeros(3)
  O = np.zeros(3)
  N = np.zeros(3,dtype=int)
  Xdel = np.zeros(3)

  Xmin[0] = Xmax[0] = gobj.x(0)
  for A in range(gobj.natom()):
    Xmin[0] = (gobj.x(A) if (Xmin[0] > gobj.x(A)) else Xmin[0])
    Xmax[0] = (gobj.x(A) if (Xmax[0] < gobj.x(A)) else Xmax[0])


  Xmin[1] = Xmax[1] = gobj.y(0)
  for A in range(gobj.natom()):
    Xmin[1] = (gobj.y(A) if (Xmin[1] > gobj.y(A)) else Xmin[1])
    Xmax[1] = (gobj.y(A) if (Xmax[1] < gobj.y(A)) else Xmax[1])

  Xmin[2] = Xmax[2] = gobj.z(0)
  for A in range(gobj.natom()):
    Xmin[2] = (gobj.z(A) if (Xmin[2] > gobj.z(A)) else Xmin[2])
    Xmax[2] = (gobj.z(A) if (Xmax[2] < gobj.z(A)) else Xmax[2])

  for k in range(3):
    Xdel[k] = Xmax[k] - Xmin[k]
    N[k] = int((Xmax[k] -Xmin[k] + 2.0*L[k])/D[k])  
    if (D[k]*np.float_(N[k]) <(Xdel[k] +2.0*L[k])) : N[k]+=1
    O[k] = Xmin[k]-(D[k]*np.float_(N[k]) - (Xmax[k] -Xmin[k]))/2.0 
    
  print("X min : %16.3e" % O[0])
  print("Y min : %16.3e" % O[1])
  print("Z min : %16.3e" % O[2])
  print("X max : %16.3e" % (O[0] + D[0]*N[0]))
  print("Y max : %16.3e" % (O[1] + D[1]*N[1]))
  print("Z max : %16.3e" % (O[2] + D[2]*N[2]))
  #conv = 0.5291772108
  print("X points: %i\n" %(N[0]+1))
  print("Y points: %i\n" %(N[1]+1))
  print("Z points: %i\n" %(N[2]+1))
  return O,N

#h2o.print_out_in_bohr()
#basis is the basis object
def phi_builder(mol,xs,ys,zs,ws,basis):
  
  delta = 1.0e-2

  #basis = psi4.core.BasisSet.build(mol, 'ORBITAL',psi4.core.get_global_option('basis')) # or set the basis from input
  #basis = psi4.core.BasisSet.build(mol, 'ORBITAL',basis_set)
  basis_extents = psi4.core.BasisExtents(basis,delta)

  blockopoints = psi4.core.BlockOPoints(xs, ys, zs, ws,basis_extents)
  npoints = blockopoints.npoints()
  print("n points: %i" % npoints)
  #needed?
  lpos = np.array(blockopoints.functions_local_to_global())

  #print("Local basis function mapping")
  #print(lpos)
  #print("lpos shape (%i,)" % (lpos.shape[0]))

  #print some info
  blockopoints.print_out('b_info.txt')

  nbas = basis.nbf() #number of basis functions
  #print("debug! nbasis %i\n" % nbas)

  funcs = psi4.core.BasisFunctions(basis,npoints,nbas)

  funcs.compute_functions(blockopoints)

  phi = np.array(funcs.basis_values()["PHI"])[:npoints, :lpos.shape[0]]
  return phi, lpos,nbas

def orbtocube(mol,L,D,Ca,orblist,basis,tag="tno+",path="./",dens=False):
   O,N=cubedata(mol,L,D)


   nx = N[0]+1
   ny = N[1]+1
   nz = N[2]+1

   ntot= nx*ny*nz
   x=np.zeros(ntot)
   y=np.zeros(ntot)
   z=np.zeros(ntot)
   w=np.ones(ntot)
   #aux variables
   x_ax=np.zeros(nx)
   y_ax=np.zeros(ny)
   z_ax=np.zeros(nz)
   grid = []
   for i in range(nx):
       x_ax[i] = O[0]+D[0]*i
   
   for i in range(ny):
       y_ax[i] = O[1]+D[1]*i
  

   for i in range(nz):
       z_ax[i] = O[2]+D[2]*i


   for i in range(nx):
       for j in range(ny):
           for k  in range(nz):
               tmp = np.array([x_ax[i],y_ax[j],z_ax[k]])
               grid.append(tmp)

   res = np.array(grid)
   xs = psi4.core.Vector.from_array(res[:,0])
   ys = psi4.core.Vector.from_array(res[:,1])
   zs = psi4.core.Vector.from_array(res[:,2])
   ws = psi4.core.Vector.from_array(w)

   phi,lpos,nbas=phi_builder(mol,xs,ys,zs,ws,basis)
   MOnp = np.matmul(phi,Ca)
   if dens:
     rho = np.einsum('pm->p',np.square(MOnp))

   if (not dens):  
      for m in orblist:
         fo =open(path+tag+"_"+str(m)+".cube","w")
         ### write cube header
         fo.write("\n")
         fo.write("\n")
         fo.write("     %i %.6f %.6f %.6f\n" % ( mol.natom(),O[0],O[1],O[2]))
         fo.write("   %i  %.6f %.6f %.6f\n" % (nx, D[0],0.0,0.0))
         fo.write("   %i  %.6f %.6f %.6f\n" % (ny, 0.0,D[1],0.0))
         fo.write("   %i  %.6f %.6f %.6f\n" % (nz, 0.0,0.0,D[2]))
         for A in range(mol.natom()):
             fo.write("  %i %.6f %.6f %.6f %.6f\n" %(mol.charge(A), 0.0000, mol.x(A),mol.y(A),mol.z(A)))
         
         for i in range(ntot):
             fo.write("%.5e\n" % MOnp[i,m-1])
         fo.close
   else:

      fo =open(path+tag+"_"+".cube","w")
      ### write cube header
      fo.write("\n")
      fo.write("\n")
      fo.write("     %i %.6f %.6f %.6f\n" % ( mol.natom(),O[0],O[1],O[2]))
      fo.write("   %i  %.6f %.6f %.6f\n" % (nx, D[0],0.0,0.0))
      fo.write("   %i  %.6f %.6f %.6f\n" % (ny, 0.0,D[1],0.0))
      fo.write("   %i  %.6f %.6f %.6f\n" % (nz, 0.0,0.0,D[2]))
      for A in range(mol.natom()):
          fo.write("  %i %.6f %.6f %.6f %.6f\n" %(mol.charge(A), 0.0000, mol.x(A),mol.y(A),mol.z(A)))
      for i in range(ntot):
          fo.write("%.5e\n" % (rho[i]))
      fo.close
############################################################################
def svd_analysis(tdmat,C,mol,L,D,basis_set,mlist,plist,freqid,debug=False):
   try: 
     u, s, vh = np.linalg.svd(tdmat)
   except np.linalg.LinAlgError:
     print("Error in numpy.linalg.svd")
   if (u.shape[1] == vh.shape[0]):
       Ch = np.matmul(C[:,:],u) 
   
       Cp = np.matmul(C[:,:],np.conjugate(vh.T))
   else:
       ndocc = u.shape[0]
       Ch = np.matmul(C[:,:ndocc],u) 
   
       Cp = np.matmul(C[:,ndocc:],np.conjugate(vh.T))
       #or
       #Cp = np.matmul(C[:,ndocc:],np.conjugate(vh[:ndocc,:].T))
   path="./svd."+freqid+"/"
   os.mkdir(path)
   if debug :
     print("Debug")
     print("U has dim : %i,%i\n" % (u.shape))
     print("V^H has dim : %i,%i\n" % (vh.shape))
     print("C_hole has dim : %i,%i\n" % (Ch.shape))
     print("C_part has dim : %i,%i\n" % (Cp.shape))
     test = np.zeros_like(u)
     print("U real : %s\n" % np.allclose(u.imag,test,atol=1.0e-12))
     test = np.zeros_like(vh)
     print("V^H real : %s\n" % np.allclose(vh.imag,test,atol=1.0e-12))
     np.savetxt(path+"Vh.txt",vh)
   
  
   #in case the orbitals have imaginary part
   test = np.zeros_like(u)
   if (np.allclose(u.imag,test,atol=1.0e-12)):
     orbtocube(mol,L,D,Ch,mlist,basis,"tno-",path)
   else:
     orbtocube(mol,L,D,Ch.real,mlist,basis,"tno.re-",path)
     orbtocube(mol,L,D,Ch.imag,mlist,basis,"tno.im-",path)
     #plot the squared modules of the orbital
     orbtocube(mol,L,D,Ch,mlist,basis,"tno.abs-",path,True)
   test = np.zeros_like(vh)
   if (np.allclose(vh.imag,test,atol=1.0e-12)):
     orbtocube(mol,L,D,Cp,plist,basis,"tno+",path)
   else:
     orbtocube(mol,L,D,Cp.real,plist,basis,"tno.re+",path)
     orbtocube(mol,L,D,Cp.imag,plist,basis,"tno.im+",path)
     #plot the squared modules of the orbital
     orbtocube(mol,L,D,Cp,plist,basis,"tno.abs+",path,True)
   np.savetxt(path+"seigv.txt",s)
   #consistency check
   try: 
      lambd1,uvec=np.linalg.eigh(np.matmul(tdmat,np.conjugate(tdmat.T)))
   except scipy.linalg.LinAlgError:
       print("Error in scipy.linalg.eigh")
   try: 
      lambd2,vvec=np.linalg.eigh(np.matmul(np.conjugate(tdmat.T),tdmat))
   except scipy.linalg.LinAlgError:
       print("Error in scipy.linalg.eigh")
   print("Sum of lambda eigenvalues")
   print(np.sum(lambd1))
   np.savetxt(path+"lambd1.txt",lambd1)
   np.savetxt(path+"lambd2.txt",lambd2)
##############################################################################
def nocv_analysis(tdmat,S,npairs,wfn,freqid):

 temp = np.matmul(S,np.matmul(tdmat,S))
 try: 
    eigvals,eigvecs=scipy.linalg.eig(temp,S)
 except scipy.linalg.LinAlgError:
     print("Error in scipy.linalg.eigh for SDS")
 
 path="./nocv."+freqid+"/"
 os.mkdir(path)
 np.savetxt(path+"eiglist.txt",eigvals)
 for i in range(npairs):
   j = i + 1
   label = "pair."+str(j)
   tmp =  eigvecs[:,i]
   d1 = np.outer(tmp,np.conjugate(tmp))
   #check if d1 sum to 1
   trace = np.trace(np.matmul(d1,S))
   print("trace of nocv_-%i : %.8f\n" % (j,trace.real))
   tmp =  eigvecs[:,-i-1]
   d2 = np.outer(tmp,np.conjugate(tmp))
   #check if d2 sum to 1
   trace = np.trace(np.matmul(d2,S))
   print("trace of nocv_+%i : %.8f\n" % (j,trace.real))
   deltadens = eigvals[i]*(d1 - d2)
 
   wfn.Da().copy(psi4.core.Matrix.from_array(d1.real))
   wfn.Db().copy(psi4.core.Matrix.from_array(d1.real)) 
   psi4.cubeprop(wfn)
   os.rename("Da.cube",path+"nocv-"+str(j)+".cube")
   os.remove("Db.cube")
   os.remove("Ds.cube")
   os.remove("Dt.cube")
   ######
   wfn.Da().copy(psi4.core.Matrix.from_array(d2.real))
   wfn.Db().copy(psi4.core.Matrix.from_array(d2.real)) 
   psi4.cubeprop(wfn)
   os.rename("Da.cube", path+"nocv+"+str(j)+".cube")
   os.remove("Db.cube")
   os.remove("Ds.cube")
   os.remove("Dt.cube")
   ######
   wfn.Da().copy(psi4.core.Matrix.from_array(deltadens.real))
   wfn.Db().copy(psi4.core.Matrix.from_array(deltadens.real))
   psi4.cubeprop(wfn)
   os.rename("Da.cube",path+label+".cube")
   os.remove("Db.cube")
   os.remove("Ds.cube")
   os.remove("Dt.cube")
##############################################################################
# custom aux funcs
def setgridcube(mol,basis_set,L=[4.0,4.0,4.0],D=[0.2,0.2,0.2]):
   O,N=cubedata(mol,L,D)


   nx = N[0]+1
   ny = N[1]+1
   nz = N[2]+1

   ntot= nx*ny*nz
   x=np.zeros(ntot)
   y=np.zeros(ntot)
   z=np.zeros(ntot)
   w=np.ones(ntot)
   #aux variables
   x_ax=np.zeros(nx)
   y_ax=np.zeros(ny)
   z_ax=np.zeros(nz)
   grid = []
   for i in range(nx):
       x_ax[i] = O[0]+D[0]*i
   
   for i in range(ny):
       y_ax[i] = O[1]+D[1]*i
  

   for i in range(nz):
       z_ax[i] = O[2]+D[2]*i


   for i in range(nx):
       for j in range(ny):
           for k  in range(nz):
               tmp = np.array([x_ax[i],y_ax[j],z_ax[k]])
               grid.append(tmp)

   res = np.array(grid)
   xs = psi4.core.Vector.from_array(res[:,0])
   ys = psi4.core.Vector.from_array(res[:,1])
   zs = psi4.core.Vector.from_array(res[:,2])
   ws = psi4.core.Vector.from_array(w)

   return xs,ys,zs,ws,N,O
def denstocube(mol,basis,Dens,S,ndocc,tag,L,D=[0.2,0.2,0.2]):
   
    O,N=cubedata(mol,L,D)
    
    
    nx = N[0]+1
    ny = N[1]+1
    nz = N[2]+1
    
    ntot= nx*ny*nz
    x=np.zeros(ntot)
    y=np.zeros(ntot)
    z=np.zeros(ntot)
    w=np.ones(ntot)
    #aux variables
    x_ax=np.zeros(nx)
    y_ax=np.zeros(ny)
    z_ax=np.zeros(nz)
    grid = []
    for i in range(nx):
        x_ax[i] = O[0]+D[0]*i
    
    for i in range(ny):
        y_ax[i] = O[1]+D[1]*i
   
    
    for i in range(nz):
        z_ax[i] = O[2]+D[2]*i
    
    
    for i in range(nx):
        for j in range(ny):
            for k  in range(nz):
                tmp = np.array([x_ax[i],y_ax[j],z_ax[k]])
                grid.append(tmp)
    
    res = np.array(grid)
    xs = psi4.core.Vector.from_array(res[:,0])
    ys = psi4.core.Vector.from_array(res[:,1])
    zs = psi4.core.Vector.from_array(res[:,2])
    ws = psi4.core.Vector.from_array(w)
    
    phi,lpos,nbas=phi_builder(mol,xs,ys,zs,ws,basis)
    
    
    temp = np.matmul(S,np.matmul(Dens.real,S))
    try: 
       eigvals,eigvecs=scipy.linalg.eigh(temp,S,eigvals_only=False)
    except scipy.linalg.LinAlgError:
        print("Error in scipy.linalg.eigh of inputted matrix")
        return None 
    Rocc = eigvecs[:,-ndocc:]
    MO = np.matmul(phi,Rocc)
    MOs = np.square(MO)
    rho = np.einsum('pm->p',MOs)
    fo =open(tag+".cube","w")
    ### write cube header
    fo.write("\n")
    fo.write("\n")
    fo.write("     %i %.6f %.6f %.6f\n" % ( mol.natom(),O[0],O[1],O[2]))
    fo.write("   %i  %.6f %.6f %.6f\n" % (nx, D[0],0.0,0.0))
    fo.write("   %i  %.6f %.6f %.6f\n" % (ny, 0.0,D[1],0.0))
    fo.write("   %i  %.6f %.6f %.6f\n" % (nz, 0.0,0.0,D[2]))
    for A in range(mol.natom()):
        fo.write("  %i %.6f %.6f %.6f %.6f\n" %(mol.charge(A), 0.0000, mol.x(A),mol.y(A),mol.z(A)))
    for i in range(ntot):
        fo.write("%.5e\n" % (2.0*rho[i]))
    fo.close

def esptocube(xs,ys,zs,wfnobj,mol,O,N,D=[0.2,0.2,0.2]):
    points[:,0] = xs
    points[:,1] = ys
    points[:,2] = zs
    nx = N[0]+1
    ny = N[1]+1
    nz = N[2]+1
    
    ntot= nx*ny*nz
    fo =open("eps.cube","w")
    ### write cube header
    fo.write("\n")
    fo.write("\n")
    fo.write("     %i %.6f %.6f %.6f\n" % ( mol.natom(),O[0],O[1],O[2]))
    fo.write("   %i  %.6f %.6f %.6f\n" % (nx, D[0],0.0,0.0))
    fo.write("   %i  %.6f %.6f %.6f\n" % (ny, 0.0,D[1],0.0))
    fo.write("   %i  %.6f %.6f %.6f\n" % (nz, 0.0,0.0,D[2]))
    for A in range(mol.natom()):
        fo.write("  %i %.6f %.6f %.6f %.6f\n" %(mol.charge(A), 0.0000, mol.x(A),mol.y(A),mol.z(A)))
    for i in range(ntot):
        myepc = psi4.core.ESPPropCalc(wfn_scf)
        psi4_matrix = psi4.core.Matrix.from_array(points)
        esp = np.array(myepc.compute_esp_over_grid_in_memory(psi4_matrix))

        fo.write("%.5e\n" % (2.0*rho[i]))
    fo.close
