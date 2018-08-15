#include <iostream>
#include <fstream>
#include "FACE.h"
#include "opencv2/opencv.hpp"
#include <cv.h>
#include <string>
#include <ctime>
//#include "havon_ffd.h"
#define HEIGHT_ 480
#define WIDTH_ 640

using namespace std;
using namespace cv;

//int main(int argc, char const *argv[])
int main(int argc, char** argv)
{
     // ifstream infile;
     // infile.open("lfw_list.txt");
     // int i=0;
     // string a[14000];
     // while(!infile.eof())
     // {
     //     getline(infile,a[i],'\n');
     //     i++;
     // }
     // for(int ii=0;ii<i;ii++)
     // {
     //     cout<<a[ii]<<endl;
     // }
     // infile.close();



    string pt[5] = {
            "../model/step1.prototxt",
            "../model/step2.prototxt",
            "../model/step3.prototxt",
            "../model/deploy_quarter.prototxt",
            "../model/deploy_ms.prototxt"
    };

    string cm[5] = {
            "../model/step1.caffemodel",
            "../model/step2.caffemodel",
            "../model/step3.caffemodel",
            "../model/quarter.caffemodel",
            "../model/ms.caffemodel"
    };

    cv::Mat img1,img2;
    vector<Mat> cropFace1,cropFace2;
    vector<float> feature1,feature2,feature3,feature4;
    float distance,distance1,distance2;


    vector<string> model_file(pt, pt + 5);
    vector<string> trained_file(cm, cm + 5);
    FACE face(model_file, trained_file);
    clock_t  start, finish;
    float total_time;
    float height_b,width_b,ratio_b;
    int resize_b_x,resize_b_y;
    cv::Mat img2_resize;
    string result_path1,result_path2;
    cv::Rect face_box,ave_box;
    cv::Mat img_roi;
    ave_box.x=17;
    ave_box.y=21;
    ave_box.width=75;
    ave_box.height=65;

    img1 = cv::imread(argv[1],1);
    img2 = cv::imread(argv[2],1);
    result_path1=argv[3];
    result_path2=argv[4];

    ofstream result_file1(result_path1);
    ofstream result_file2(result_path2);

    start=clock();
    height_b=img2.rows;
    width_b=img2.cols;
    ratio_b=min(height_b/HEIGHT_, width_b/WIDTH_);
    resize_b_x=std::floor(width_b/ratio_b);
    resize_b_y=std::floor(height_b/ratio_b);
    if(height_b>HEIGHT_ || width_b>WIDTH_){
      cout<<"the image2 will be resized"<<endl;
      //cout<<"begin to detect image2"<<endl;
      cv::resize(img2,img2_resize,cv::Size(resize_b_x,resize_b_y));
      face.detectFromOriImg(img2_resize,img2,cropFace2);
    }
    else{
      //cout<<"begin to detect image2"<<endl;
      face.detect(img2,cropFace2);
    }
    face.detect(img1,cropFace1);


    //face.detect(img2,cropFace2);
    cv::Mat img1_resize=cv::Mat::zeros(112, 96, CV_8UC3);
    face_box=face.Get_boundingbox();
    if(cropFace1.size()==0){
      cv::Mat srcROI=img1(ave_box);
      srcROI.copyTo(img_roi);
      cv::resize(img_roi,img1_resize,img1_resize.size());
      cropFace1.push_back(img1_resize);
    }
    cv::imshow("img_win",cropFace1[0]);
    cv::waitKey(0);
    cv::imshow("img_win2",cropFace2[0]);
    cv::waitKey(0);
     if(cropFace1.size()>0 && cropFace2.size()>0)
     {
         face.extractFeature(cropFace1[0],3,feature1);
         face.extractFeature(cropFace2[0],3,feature2);
         face.extractFeature(cropFace1[0],4,feature3);
         face.extractFeature(cropFace2[0],4,feature4);
     }
    // cout<<feature1.size()<<endl;
    face.calculateDistance(feature1,feature2,3,distance1);
    face.calculateDistance(feature3,feature4,4,distance2);
    distance=0.85*distance1+1.15*distance2;
    cout<<"distance1:"<<distance1<<'\r'<<endl;
    cout<<"distance2:"<<distance2<<'\r'<<endl;
    cout<<"distance:"<<distance<<'\r'<<endl;
    cout<<face_box.x<<'\t'<<face_box.y<<'\t'<<face_box.width<<'\t'<<face_box.height<<endl;

    for(int i=0; i<feature3.size(); i++){
         result_file1<<feature3[i]<<endl;
       }
    result_file1<<"*********************************************************"<<endl;
    for(int j=0;j<feature4.size();j++){
      result_file2<<feature4[j]<<endl;
    }
     if(distance>1.1672)
         cout<<"different pair"<<endl;
     else
        cout<<"same pair"<<endl;
     finish=clock();
     total_time=float((finish-start))/CLOCKS_PER_SEC;
   	cout<<"the total time is: "<<total_time<<" s"<<endl;

    cv::VideoCapture cap(0);
    return 0;
}
