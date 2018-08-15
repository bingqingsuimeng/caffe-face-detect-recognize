# -*-coding-*-:utf-8
import re
import numpy as np
import sys
import os
import shutil

path_name=sys.argv[1]
path1=sys.argv[2]
path2=sys.argv[3]
path3=sys.argv[4]
path4=sys.argv[5]
parten_i=re.compile(r'[a-z]+')
parten_I=re.compile(r'[A-W]|[Y-Z]+')
count1=0
count2=0
count3=0
count4=0
#img_path=os.path.abspath(path_name)+os.sep
for sub_path in os.listdir(path_name):
    if os.path.isdir(path_name+os.sep+sub_path):
        files= os.listdir(path_name+os.sep+sub_path)
        for file_ in files:
            if parten_i.search(file_[:-4]):
                shutil.copyfile(path_name+os.sep+sub_path+os.sep+file_,path1+os.sep+file_)
                count1+=1
            elif parten_I.search(file_[:-4]):
                shutil.copyfile(path_name+os.sep+sub_path+os.sep+file_,path2+os.sep+file_)
                count2+=1
            elif(len(file_[:-4])<14):
                shutil.copyfile(path_name+os.sep+sub_path+os.sep+file_,path3+os.sep+file_)
                count3+=1
            else:
                shutil.copyfile(path_name+os.sep+sub_path+os.sep+file_,path4+os.sep+file_)
                count4+=1
        print (count1,count2,count3,count4)
