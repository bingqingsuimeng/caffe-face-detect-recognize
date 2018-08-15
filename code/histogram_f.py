# -*- coding: utf-8 -*-
###############################################
#created by :  lxy 
#Time:  2017/07/12 14:09
#project: Face recognize
#company: Senscape
#rversion: 0.1
#tool:   python 2.7
#modified: 2018/05/22 16:09
####################################################
import matplotlib.pyplot as plt
import numpy as np
import string
import argparse

def args():
    parser = argparse.ArgumentParser(description="get txt date")
    parser.add_argument('--input_path',default='./distence_high.txt',type=str,\
                        help="the txt file input")
    parser.add_argument('--out_path',default='./hist_out.txt',type=str,\
                        help="the txt file output")
    return parser.parse_args()

def get_hist(input_path,output_path):
    input_file=open(input_path,'r')
    out_file=open(output_path,'w')
    data_arr=[]
    lines_ = input_file.readlines()
    for line in lines_:
        line = string.strip(line)
        line_s = line.split(" ")
        for i in range(len(line_s)):
            temp=string.atof(line_s[i])
            data_arr.append(temp)
    data_in=np.asarray(data_arr)
    max_dist = np.max(data_in)
    print("max ",max_dist)
    num_bins=10
    a,b,c=plt.hist(data_in,num_bins,range=(0,max_dist),normed=1,color='blue',cumulative=1)
    plt.title('histogram')
    fig = plt.gcf()
    img_path = input_path[9:-4]
    img_path = "hist-"+img_path +"-1.png"
    plt.savefig(img_path, format='png')
    plt.show()
    for i in range(26):
        out_file.write(str(a[i])+'\t'+str(b[i])+'\n')
    input_file.close()
    out_file.close()

if __name__ =="__main__":
    parm = args()
    get_hist(parm.input_path,parm.out_path)

