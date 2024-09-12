#!/bin/bash
export PYCUBE="$HOME/BERTHA/pycubescd/PY3"
export REFDEN="../td.ground"
#id_fig=1
#incr=1
mkdir fig_data
for i in {0003500..0005400..50} 
do 
  fdir='td.'$i	
  echo entering $fdir
  cd $fdir
  python $PYCUBE/pysub_cube.py -f1 density.cube -f2 $REFDEN/density.cube -o diff.cube
  fname='snap_'$i
  cp diff.cube ./this.cube
  cp ../geom.xyz ./
  sed -e "s/_xyzfile/geom/g" \
  -e "s/_colorp/blue/g" \
  -e "s/_colorm/red/g" \
  -e "s/_label/ /g" \
  -e "s/_iso/1.e-5/g" ../mode.py > cube.py
  pymol cube.py
  mv this.png  ../fig_data/$fname.png 
  rm this.cube
  cd ..
#  id_fig=$(($id_fig+$incr))
done

