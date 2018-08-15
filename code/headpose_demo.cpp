//###############################################
//#created by :  lxy
//#Time:  2017/08/23 10:09
//#project: Face recognize
//#company: Senscape
//#rversion: 0.1
//#tool:  gcc
//#modified:
//####################################################
//#define CPU_ONLY
#include <iostream>
#include <fstream>
#include "FACE.h"
#include "pose_estimate.h"
#include <cv.h>
#include <ctime>

#define HEIGHT_ 480
#define WIDTH_ 640

using namespace std;
using namespace cv;


int main(int argc, char** argv){
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

  ::google::InitGoogleLogging(" ");
  cv::Mat img1,img2;

  vector<string> model_file(pt, pt + 5);
  vector<string> trained_file(cm, cm + 5);
  FACE face(model_file, trained_file);
  cout<<"load face success"<<endl;
  string pose_prototxt_file = "../pose_model/deploy.prototxt";
  //string pose_prototxt_file = "../model/step1.prototxt";
  string pose_model_file = "../pose_model/68point_dlib_with_pose.caffemodel";
  //string pose_model_file = "../model/step1.caffemodel";
  const char*  mean_filename = "../pose_model/VGG_mean.binaryproto";
  HeadPose head_pose(pose_prototxt_file,pose_model_file,mean_filename);
  cout<<"load head model success"<<endl;
  vector<vector<float>> face_angles;

  clock_t  start, finish;
  float total_time;
  float height_b,width_b,ratio_b;
  int resize_b_x,resize_b_y;
  cv::Mat img2_resize;
  vector<cv::Mat> cropFace2;
  vector<cv::Mat> cropFace1;

  img2 = cv::imread(argv[1],1);
  cout<<"read is success "<<img2.rows<<'\t'<<img2.cols<<endl;
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
  //cout<<"detect is success"<<endl;
  //cv::imshow("img_win2",cropFace2[0]);
  //cv::waitKey(0);
  if(cropFace2.size()>0){
    for(int j=0; j<cropFace2.size(); j++){
      if(cropFace2[j].rows !=224 || cropFace2[j].cols != 224){
        cv::Mat crop_resize;
        cv::resize(cropFace2[j],crop_resize,head_pose.input_geometry_);
        cropFace1.push_back(crop_resize);
      }
      else{
        cropFace1.push_back(cropFace2[j]);
      }
    }
  }
  start=clock();
   if(cropFace1.size()>0)
   {
     for(int k=0; k< cropFace1.size(); k++){
       vector<float> angle;
       head_pose.Predict(cropFace1[k],angle);
       face_angles.push_back(angle);
       //angle.clear();
     }
       //face.extractFeature(cropFace2[0],3,feature2);
   }
   else{
     cout<<"can not detect face"<<endl;
   }
   cropFace2.clear();
   cropFace1.clear();
   finish=clock();
   cout<<"head pose time is: "<< float((finish-start))/CLOCKS_PER_SEC<<endl;
   cout<<"the number faces are: "<<face_angles.size()<<endl;
   cout<<"the angle number are: "<<face_angles[0].size()<<endl;
   cout<<"the face angle are Pitch, Yaw, Roll: "<<endl;
   if(face_angles.size()>0){
     for(int i=0; i<face_angles.size(); i++){
       cout<<face_angles[i][0]<<'\t'<<face_angles[i][1]<<'\t'<<face_angles[i][2]<<endl;
       cout<<face_angles[i][0]*50<<'\t'<<face_angles[i][1]*50<<'\t'<<face_angles[i][2]*50<<endl;
     }
   }
   face_angles.clear();
   return 0;
}
