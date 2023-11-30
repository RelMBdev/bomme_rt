import sys
import os
import psi4
#import numpy as np
##################################################################

def isinteger(x):
    isint = True
    try:
        int(x)
    except ValueError:
       isint = False 
    return isint

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


def isinteger(x):
    isint = True
    try:
        int(x)
    except ValueError:
       isint = False 
    return isint

def isdouble(x):
    isdouble = True
    try:
        float(x)
    except ValueError:
       isdouble = False 
    return isdouble

class Molecule():

  def __init__(self,fgeom=None):
      self.__geometry = None
      self.__res = str()
      self.__natom = None
      self.__labels = None
      self.__status_on = None
      if fgeom is not None:
         self.set_geometry(fgeom)

  def set_geometry(self,fgeom):

      self.__geometry=str()
      with open(fgeom,'r') as data:
         self.__natom = int(data.readline()) # natom is the 'proper' 
                                           # number of atoms in the 
                                           # (active) molecule
         next(data)
         for line in data:
            self.__geometry += str(line)

  def finalize(self):
      self.__res += self.__geometry
      self.__geometry = None

  def set_label(self): #can be used on a raw geometry string (symb X Y Z)
         labeled = str()
         tmp=self.__geometry.split('\n')
         opts = tmp.pop(0).strip()
         check = opts.strip().split()
         if not isinteger(check[0]):
             opts = '0 0' #psi4 by default assign the ch/mult
         tmp.pop()
         for gline in tmp:
             labeled += str(gline.split()[0]) +'1' + '   ' + str(gline.split()[1])  + '   ' + str(gline.split()[2]) + '   ' + str(gline.split()[3])+'\n'
         self.__geometry = opts + '\n' + labeled

  def get_labels(self,fgeom,label):

      self.__labels=[]
      with open(fgeom,'r') as data:
         next(data)
         next(data)
         for line in data:
           tmp = line.split()
           if label :
             self.__labels.append(tmp[0]+'1')
           else:
             self.__labels.append(tmp[0])
         self.__labels = list(dict.fromkeys(self.__labels))

  def from_string(self,geomstr,natom=None):
      self.__natom = natom
      if self.__geometry is None:
         self.__geometry = geomstr
      else:
         self.__geometry += geomstr

  def get_ghost(self,idx1,idx0=0):
      gdummy = str()
      tmp=self.__geometry.split('\n')
      tmp.pop()
      for m in tmp[idx0:idx1]:
        gdummy +="@"+m.strip()+'\n'
      return gdummy
  def append(self,options):
      # options is a string like "symmetry c1"+"\n" or a 
      # string containing moelcular coordinates
      self.__res += options

  def set_charge_mul(self,charge,mult=1):
      if self.__geometry is None:
         self.__geometry=str(charge)+' '+str(mult) + '\n'
      else:
         self.__geometry = str(charge)+' '+str(mult) + '\n' +self.__geometry

  def display_xyz(self):
      print(self.__res)

  def natom(self):
      return self.__natom

  def set_natom(self):
      self.__res = str(self.__natom) +'\n' + self.__res

  def geometry(self):
      return self.__res

  def labels(self):
      return self.__labels
  def core_geom(self,idx1,idx0=0):

         res = str()
         tmp=self.__res.split('\n')

         for m in tmp[idx0:idx1]:
            res += m+'\n'
         return res

def gparser(frag_spec,geom_file):


    f_id = []

    if isinstance(frag_spec,str):
            tmp = frag_spec.split(';')
            test = isinteger(tmp[0])
            if test:
               f_id = [int(x) for x in tmp]
    #print(f_id)

    # get the molecular geometry from file 
    bp_mol = Molecule(geom_file)
    bp_mol.finalize()
    #append some options
    bp_mol.append("symmetry c1" + "\n" + "no_reorient" + "\n" + "no_com")
    #print(bp_mol.geometry())

    # set psi4 object
    tot_sys = psi4.geometry(bp_mol.geometry())


    bp_str = bp_mol.geometry()

    
    # do the labeling
    tot_sys.set_active_fragments(f_id)
    nfrag = tot_sys.nfragments()
    lowfrag_list = [x for x in range(1,nfrag+1)]

    for elm in f_id:
        lowfrag_list.remove(elm)
    


    #loop on frags
    new = Molecule()
    
    for idx in f_id:
        frag = tot_sys.extract_subsets(idx)
        #frag_geom =frag.geometry().np
        geom_str =  frag.save_string_xyz()
        new.from_string(geom_str)
        #set label
        new.set_label()
        #append the sepator
        new.finalize()
        new.append('--\n')

    low = Molecule()
    if len(lowfrag_list)>0:
      for idx in lowfrag_list:
          frag = tot_sys.extract_subsets(idx)
          #frag_geom =frag.geometry().np
          geom_str =  frag.save_string_xyz()
          low.from_string(geom_str)
          #set label
          low.finalize()
          low.append('--\n')
      new.append(low.geometry())
    new.append("symmetry c1" + "\n" + "no_reorient" + "\n" + "no_com")
    

    #new.display_xyz()
    return new, f_id
