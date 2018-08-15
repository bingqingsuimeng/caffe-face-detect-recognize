# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2017/08/03 14:09
#project: Face recognize
#company: Senscape
#rversion: 0.1
#tool:   python 2.7
#modified:
####################################################
import os
import sys

path=sys.argv[1]
out_path=sys.argv[2]
out_file=open(out_path,'w')
files=os.listdir(path)
count=0
for file_ in files:
    out_file.write(file_.strip()+'\n')
    count+=1
print count
