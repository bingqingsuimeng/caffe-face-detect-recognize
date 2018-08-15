# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2017/08/04 14:09
#project: Face recognize
#company: Senscape
#rversion: 0.1
#tool:   python 2.7
#modified:2018/06/12 09:24
#description  generate txt file,for example: img1.png  img1.png img2.png  ... img_n.png
####################################################
import numpy as np 
import os
import sys
import argparse

parser = argparse.ArgumentParser(description='Compare 2 images using LBPH-uniform')
parser.add_argument('--image_dir',type=str,default='/home/lxy/Pictures',help='the directory should include 2 more Picturess')
parser.add_argument('--holdvaule',type=float,default=0.02,help='select images below the vaule')
args=parser.parse_args()

def get():
    out_file = open('highway.txt','w')
    directory = args.image_dir
    filename = []
    for root,dirs,files in os.walk(directory):
        for file_1 in files:
            print("file:", file_1)
            filename.append(file_1)
    filename = np.sort(filename)
    print(len(filename))
    print(filename)
    for i in range(0,len(filename)):
        #print(len(filename))
        #out_file.write("{},".format(filename[i]))
        if i < len(filename):
            out_file.write("{},".format(filename[i]))
            #print(filename)
            for j in range(0,len(filename)):
                if j < len(filename)-1:
                    out_file.write("{},".format(filename[j]))
                else:
                    out_file.write("{}".format(filename[j]))
            out_file.write("\n")

if __name__ == "__main__":
    get()