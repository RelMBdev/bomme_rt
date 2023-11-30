#!/bin/bash
export BerthaRootPath=/home/matteo/BERTHA
export BERTHA_API_PATH=$BerthaRootPath/pybertha/src
export BommePath=$HOME/bomme_rt

export COMMON_PATH=$BommePath/common
export MODS_PATH=$BommePath/mods

export PYBERTHA_MOD_PATH="$BerthaRootPath/pybertha/pyemb;$BerthaRootPath/xcfun/build/lib/python;$BerthaRootPath/pybertha/src;$BerthaRootPath/pyadf/src;$BerthaRootPath/berthaingen/pybgen;$BerthaRootPath/pybertha/psi4rt;$BerthaRootPath/pybertha/pyberthaembed;$BerthaRootPath/xcfun/build/lib/python/xcfun"

ulimit -s unlimited
