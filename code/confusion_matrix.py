import numpy as np  
import matplotlib.pyplot as plt  
from pylab import *
import argparse
import string

def args():
    parser = argparse.ArgumentParser(description="get txt date")
    parser.add_argument('--file_in',default='./distence.txt',type=str,\
                        help="the txt file input")
    parser.add_argument('--norm',default=False,type=bool,\
                        help="norm the column mat")
    return parser.parse_args()

def get_mat(file_p):
    f = open(file_p,'r')
    lines = f.readlines()
    mat_o = []
    for line_1 in lines:
        line_1 = string.strip(line_1)
        line_s = line_1.split(" ")
        tmp_row = [] 
        print("line ",line_s)
        for i in range(len(line_s)):
            tmp_row.append(string.atof(line_s[i]))
        mat_o.append(tmp_row)
    return mat_o

def get_mat_txt(file_p):
    f = open(file_p,'r')
    lines = f.readlines()
    mat_o = []
    key_ = ['lxy0', 'lxy1', 'lxy2' ,'lxy3',
        'lxy4', 'lxy5', 'lxy6', 'lxy7',
        'shj0' ,'shj1', 'shj2', 'shj3',
        'shj4', 'shj5', 'shj6', 'shj7',
        'shj8', 'zz0', 'zz1', 'zz2',
        'zz3', 'zz4', 'zz5', 'zz6',
        'zz7', 'zz8']
    for line_1 in lines:
        line_1 = string.strip(line_1)
        line_s = line_1.split(" ")
        tmp_row = []
        #print("line ",line_s)
        for i in range(len(line_s)):
            tmp_row.append(string.atof(line_s[i]))
        mat_o.append(tmp_row)

def plt_mat(conf_arr,norm):
    if norm :
        norm_conf = []  
        for i in conf_arr:  
            a = 0  
            tmp_arr = []  
            a = sum(i, 0)  
            for j in i:  
                tmp_arr.append(float(j)/float(a))  
            norm_conf.append(tmp_arr)  
    else:
        norm_conf = conf_arr
    fig = plt.figure()  
    plt.clf()  
    ax = fig.add_subplot(111)  
    ax.set_aspect(1)  
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet,  
                    interpolation='nearest')  
    width = len(conf_arr)  
    height = len(conf_arr[0])  
    ##for x in xrange(width):  
    ##    for y in xrange(height):  
    ##        ax.annotate(str(conf_arr[x][y]), xy=(y, x),  
    ##                    horizontalalignment='center',  
    ##                    verticalalignment='center')  
    cb = fig.colorbar(res)  
    alphabet = ['lxy0', 'lxy1', 'lxy2' ,'lxy3',
        'lxy4', 'lxy5', 'lxy6', 'lxy7',
        'shj0' ,'shj1', 'shj2', 'shj3',
        'shj4', 'shj5', 'shj6', 'shj7',
        'shj8', 'zz0', 'zz1', 'zz2',
        'zz3', 'zz4', 'zz5', 'zz6',
        'zz7', 'zz8'] 
    plt.xticks(fontsize=7)  
    plt.yticks(fontsize=7)  
    locs, labels = plt.xticks(range(width), alphabet[:width])  
    for t in labels:  
        t.set_rotation(90)  
    #plt.xticks('orientation', 'vertical')  
    #locs, labels = xticks([1,2,3,4], ['Frogs', 'Hogs', 'Bogs', 'Slogs'])  
    #setp(alphabet, 'rotation', 'vertical')  
    plt.yticks(range(height), alphabet[:height])
    arg = args()
    out_name = arg.file_in
    out_name = "confusion_matrix-"+out_name[10:-4]+".png"
    plt.savefig(out_name, format='png')
    plt.show()

if __name__ == '__main__':
    parm = args()
    mat_dis = get_mat(parm.file_in)
    plt_mat(mat_dis,parm.norm)
