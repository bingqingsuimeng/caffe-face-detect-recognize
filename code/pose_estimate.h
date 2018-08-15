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
#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <opencv2/core/core.hpp>               //【2】OpenCv中的核心功能模块头文件
#include <opencv2/highgui/highgui.hpp>         //【3】高层GUI图形用户界面头文件
#include <opencv2/imgproc/imgproc.hpp>         //【4】OpenCv的图像处理模块头文件
#include <algorithm>                           //【5】STL中的头文件
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <iostream>

using namespace cv;
using namespace caffe;


class HeadPose {
public:
  HeadPose(const string &prototxt_fil, const string  &model_file,const char* mean_file);
  ~HeadPose();
  int Set_mean(const char* mean_file);
  void WrapInputLayer(std::vector<cv::Mat>* input_channels);
  void Preprocess(const cv::Mat& img,std::vector<cv::Mat>* input_channels);
  unsigned int Get_blob_index(const string &query_blob_name);
  void Predict(const cv::Mat &image, vector<float> &angles);

  //public Parameters
  boost::shared_ptr< Net<float> >   net_;
  cv::Size                          input_geometry_;
  int                               num_channels_;
  cv::Mat                           mean_;
  int                               mean_fg_;
};
