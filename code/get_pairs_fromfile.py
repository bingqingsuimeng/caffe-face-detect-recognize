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

path1=sys.argv[1]
path2=sys.argv[2]
path3=sys.argv[3]
id_files=open(path1,'r')
photo_files=open(path2,'r')
out_file=open(path3,'w')
id_images=id_files.readlines()
photo_images=photo_files.readlines()
count=0
for id_img in id_images:
    id_img=id_img.strip()
    str_temp=id_img[:-4]
    lg=len(str_temp)
    out_file.write(id_img)
    #print str_temp
    for photo_img in photo_images:
        photo_img=photo_img.strip()
        #print photo_img[:lg]
        if str_temp== photo_img[:lg] :
            out_file.write(',')
            out_file.write(photo_img)
            count+=1
    out_file.write('\n')
print count
id_files.close()
photo_files.close()
out_file.close()
