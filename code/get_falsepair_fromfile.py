# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2017/08/04 14:09
#project: Face recognize
#company: Senscape
#rversion: 0.1
#tool:   python 2.7
#modified:
####################################################
import os
import sys
import numpy as np

input_path1=sys.argv[1]
out_path2=sys.argv[2]
in_file=open(input_path1,'r')
out_file=open(out_path2,'w')
files=in_file.readlines()
in_len=len(files)
print in_len
for i in range(in_len):
    temp_str=files[i].split(',')
    out_file.write(temp_str[0].strip())
    count=0
    for j in range(in_len):
        ran_idx=np.random.randint(in_len,size=1)
        count+=1
        if i != ran_idx[0]:
            break;
    #print count
    #print ran_idx
    photo=files[ran_idx[0]].split(',')
    out_file.write(',')
    out_file.write(photo[1].strip())
    out_file.write('\n')
in_file.close()
out_file.close()
