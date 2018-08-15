#include "pose_estimate.h"
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

using namespace std;

HeadPose::HeadPose(const string &prototxt_fil, const string  &model_file, const char* mean_file){
  /*
  #ifdef CPU_ONLY
      Caffe::set_mode(Caffe::CPU);
  #else
      Caffe::set_mode(Caffe::GPU);
  #endif */
  Caffe::set_mode(Caffe::GPU);
  boost::shared_ptr<Net<float> > net;
  Phase phase = TEST;
  net.reset(new Net<float>(prototxt_fil, phase));
  net->CopyTrainedLayersFrom(model_file);
  //cout<<"Head class load success"<<endl;
  Blob<float>* input_layer = net->input_blobs()[0];
  num_channels_ = input_layer->channels();
  //num_channels_ =3;
  //cout<<"get input layer channel "<<num_channels_<<endl;
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
  //input_geometry_ = cv::Size(224,224);
  //cout<<"Head class init over"<<endl;
  net_=net;
  cout<<"head pose class net has input "<<net_->num_inputs()<<endl;
  cout<<"head pose class net has output "<<net_->num_outputs()<<endl;
  mean_fg_=Set_mean(mean_file);
}

HeadPose::~HeadPose(){ }

int HeadPose::Set_mean(const char* mean_file){
  Blob<float> mean_blob;
  BlobProto blob_proto;
  unsigned int num_pixel;
  vector<cv::Mat> channels;
  bool succeed = caffe::ReadProtoFromBinaryFile(mean_file, &blob_proto);
  if (succeed){
    mean_blob.FromProto(blob_proto);
    float* data = mean_blob.mutable_cpu_data();
    for (int i=0; i<num_channels_; i++){
      cv::Mat channel(mean_blob.height(),mean_blob.width(),CV_32FC1,data);
      channels.push_back(channel);
      data+=mean_blob.height()*mean_blob.width();
    }
    cv::Mat mean;
    cv::merge(channels,mean);
    cv::Scalar channel_mean = cv::mean(mean);
    cout<<"the mean value is "<<channel_mean.val[0]<<channel_mean.val[1]<<channel_mean.val[2]<<channel_mean.val[3]<<endl;
    mean_ = cv::Mat(input_geometry_,mean.type(),channel_mean);
    return 0;
  }
  else{
    return 1;
  }
}

void HeadPose::WrapInputLayer(std::vector<cv::Mat>* input_channels){
  Blob<float>* input_layer = net_->input_blobs()[0];
  int width  = input_layer->width();                 //【2】得到网络指定的输入图像的宽
  int height = input_layer->height();                //【3】得到网络指定的输入图像的高
                                                       //【4】input_data指向网络的输入blob
  float* input_data = input_layer->mutable_cpu_data();

  for (int i = 0; i < input_layer->channels(); ++i){
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);            //【6】将上面的Mat同input_channels关联起来
    input_data += width * height;                  //【7】一个一个通道地操作
  }
}

void HeadPose::Preprocess(const cv::Mat &img, std::vector<cv::Mat>* input_channels){
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
      cv::cvtColor(img, sample, CV_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
      cv::cvtColor(img, sample, CV_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
      cv::cvtColor(img, sample, CV_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
      cv::cvtColor(img, sample, CV_GRAY2BGR);
  else
      sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)                //【1】将输入图像的尺寸强制转化为网络规定的输入尺寸
      cv::resize(sample, sample_resized, input_geometry_);
  else
      sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)                              //【2】将输入图像转化成为网络前传合法的数据规格
      sample_resized.convertTo(sample_float, CV_32FC3);
  else
      sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);//【3】将图像减去均值
  //将减去均值的图像分散在input_channels中，由于在WrapInputLayer函数中，
  //     input_channels已经和网络的输入blob关联起来了，因此在这里实际上是把
  //     图像送入了网络的输入blob
  cv::split(sample_normalized, *input_channels);
}

unsigned int HeadPose::Get_blob_index(const string &query_blob_name){
  std::string str_query(query_blob_name);
  vector< string > const & blob_names = net_->blob_names();
  if(net_->has_blob(str_query)){
    for( unsigned int i = 0; i != blob_names.size(); ++i )
    {
      if( str_query == blob_names[i] )
        {
          return i;
        }
    }
   }
   else{
     cout<<"In the model has no "<<str_query<<" blob name"<<endl;
     return 0;
   }
}

void HeadPose::Predict(const cv::Mat &image, vector<float> &angles){
  string Blob_name;
  unsigned int blob_id;
  std::vector<float> f_68points;
  //cout<<"head pose class net has input "<<net_->num_inputs()<<endl;
  //cout<<"head pose class net has output "<<net_->num_outputs()<<endl;
  Blob<float>* input_layer = net_->input_blobs()[0];
  //【1】input_layer是网络的输入blob
  //【2】表示网络只输入一张图像，图像的通道数是num_channels_，高为
  //  input_geometry_.height，宽为input_geometry_.width
  input_layer->Reshape(1, num_channels_,input_geometry_.height, input_geometry_.width);
  net_->Reshape();                                  //【3】初始化网络的各层

  std::vector<cv::Mat> input_channels;              //【4】存储输入图像的各个通道
  WrapInputLayer(&input_channels);                  //【5】将存储输入图像的各个通道的input_channels放入网络的输入blob中
  Preprocess(image, &input_channels);                 //【6】将img的各通道分开并存储在input_channels中
  net_->ForwardPrefilled();                         //【7】进行网络的前向传输
  //【8】Copy the output layer to a std::vector
  //【9】output_layer指向网络输出的数据，存储网络输出数据的blob的规格是(1,c,1,1)
  /*
  Blob_name="68point";
  blob_id=Get_blob_index(Blob_name);
  boost::shared_ptr<Blob<float> > blob_68points = net_->blobs()[blob_id];
  //Blob<float>* output_layer = net_->output_blobs()[0];
  const float* begin = blob_68points->cpu_data();
  const float* end = begin + blob_68points->channels();
  f_68points = std::vector<float>(begin, end);
  */
  //Blob_name="poselayer";
  //blob_id=Get_blob_index(Blob_name);
  //boost::shared_ptr<Blob<float> > blob_angle = net_->blobs()[blob_id];
  Blob<float>* blob_angle = net_->output_blobs()[0];
  const float* angle_begin = blob_angle->cpu_data();
  const float* angle_end = angle_begin + blob_angle->channels();
  angles = std::vector<float>(angle_begin, angle_end);
}
