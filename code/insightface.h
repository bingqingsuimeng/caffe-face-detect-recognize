#include <stdio.h>
#include <sys/time.h>
#include <opencv2/opencv.hpp>
#include <caffe/caffe.hpp>
#include <vector>
using namespace std;
using namespace cv;
using namespace caffe;

class InsightFace
{
    public:
        static InsightFace *GetInstance();
        void ExtractFeatures(Mat cropface,vector<float>& feature);
        void CalculateDistance(std::vector<float> feature1, std::vector<float> feature2, float& l2);
        int feature_length_ = 512;
    private:
        static InsightFace *instance_;
        int Init();
        int Set_mean(const char* mean_file);
        void WrapInputLayer(std::vector<cv::Mat>* input_channels);
        void Preprocess(const cv::Mat& img,std::vector<cv::Mat>* input_channels);
        unsigned int Get_blob_index(const string &query_blob_name);
        //void Predict(const cv::Mat &image, vector<float> &angles);

         //public Parameters
        boost::shared_ptr< Net<float> >  net_;
        cv::Size input_geometry_;
        int num_channels_;
        cv::Mat  mean_;
        int mean_fg_;
};