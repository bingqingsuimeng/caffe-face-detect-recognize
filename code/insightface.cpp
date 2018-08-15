#include "insightface.h"
//###############################################
//#created by :  lxy
//#Time:  2018/05/24 10:09
//#project: Face detect
//#company: Senscape
//#rversion: 0.1
//#tool:  gcc
//#modified:
//####################################################
#define CPU_ONLY 0

InsightFace *InsightFace::instance_ = NULL;
InsightFace *InsightFace::GetInstance()
{
    if(instance_ == NULL){
        instance_ = new InsightFace;
        instance_->Init();
    }
    return instance_;
}

int InsightFace::Init()
{
    #ifdef CPU_ONLY
        Caffe::set_mode(Caffe::CPU);
    #else
        Caffe::set_mode(Caffe::GPU);
    #endif
    //prototxt saved path
    std::string prototxt_fil = "../model/model_15ms.prototxt";
    std::string model_file = "../model/model_15ms.caffemodel";
    //Caffe::set_mode(Caffe::GPU);
    boost::shared_ptr<Net<float> > net;
    Phase phase = TEST;
    net.reset(new Net<float>(prototxt_fil, phase));
    net->CopyTrainedLayersFrom(model_file);
    //cout<<"Head class load success"<<endl;
    Blob<float>* input_layer = net->input_blobs()[0];
    num_channels_ = input_layer->channels();
    //cout<<"get input layer channel "<<num_channels_<<endl;
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
    //input_geometry_ = cv::Size(224,224);
    //cout<<"Head class init over"<<endl;
    net_=net;
    cout<<"insightfacee class net has input "<<net_->num_inputs()<<endl;
    cout<<"insightface class net has output "<<net_->num_outputs()<<endl;
    //mean_fg_=Set_mean(mean_file);
    return 0;
}

int InsightFace::Set_mean(const char* mean_file)
{
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

void InsightFace::WrapInputLayer(std::vector<cv::Mat>* input_channels)
{
    Blob<float>* input_layer = net_->input_blobs()[0];
    int width  = input_layer->width();                 //get the width of the net_input
    int height = input_layer->height();                //get the height of the net_input
                                                        //put the input_data to blob of net_input
    float* input_data = input_layer->mutable_cpu_data();

    for (int i = 0; i < input_layer->channels(); ++i){
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);            //connect the input_data with the net_input_channels
        input_data += width * height;                  //deal with the input_data one channel by one
    }
}

void InsightFace::Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels)
{
    cv::Mat sample;
    if (img.channels() == 3 && num_channels_ == 1)
        cv::cvtColor(img, sample, CV_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cv::cvtColor(img, sample, CV_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cv::cvtColor(img, sample, CV_BGRA2RGB);
    else if (img.channels() == 1 && num_channels_ == 3)
        cv::cvtColor(img, sample, CV_GRAY2RGB);
    else
        cv::cvtColor(img, sample, CV_BGR2RGB);
        //sample = img;

    //cv::Mat sample_resized;
    if (sample.size() != input_geometry_)                //convert the input_mat to the net_input size
        cv::resize(sample, sample, input_geometry_);
    //else
        //sample_resized = sample;

    //cv::Mat sample_float;
    if (num_channels_ == 3)                              //norm the (input_image-127.5)/127.5
        sample.convertTo(sample, CV_32FC3, 0.00784314,-1);
    else
        sample.convertTo(sample, CV_32FC1, 0.00784314,-1);

    //cv::Mat sample_normalized;
    //cv::subtract(sample_float, mean_, sample_normalized);//the input_image subtract the mean_data
    cv::split(sample, *input_channels); //split the input_image to 3 channels
    sample.release();
}

unsigned int InsightFace::Get_blob_index(const string &query_blob_name)
{
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

void InsightFace::ExtractFeatures(cv::Mat cropface,vector<float>& feature)
{
    string Blob_name;
    //unsigned int blob_id;
    //std::vector<float> f_68points;
    //cout<<"insightface net has input "<<net_->num_inputs()<<endl;
    //cout<<"insightface net has output "<<net_->num_outputs()<<endl;
    Blob<float>* input_layer = net_->input_blobs()[0];
    //input_layer is the blob of the net_input
    //the net will input one image,and the num_channels_ of the image，height is
    //  input_geometry_.height，width is input_geometry_.width
    input_layer->Reshape(1, num_channels_,input_geometry_.height, input_geometry_.width);
    net_->Reshape();                                  //【initial the net 

    std::vector<cv::Mat> input_channels;              //saved the channels of the input image
    WrapInputLayer(&input_channels);                  //put the channels of the input image to the input_blobs of the net 
    Preprocess(cropface, &input_channels);                 //split the input image and save data into input_channels
    net_->ForwardPrefilled();                         //forward the net
    //Copy the output layer to a std::vector
    //output_layer point to the output_data of the net，saved the output_data blob shape is (1,c,1,1)
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
    Blob<float>* blob_feature = net_->output_blobs()[0];
    const float* feature_begin = blob_feature->cpu_data();
    const float* feature_end = feature_begin + blob_feature->channels();
    feature = std::vector<float>(feature_begin, feature_end);
}

void InsightFace::CalculateDistance(std::vector<float> feature1, std::vector<float> feature2, float& l2)
{

		    float sqrtsum1 = 0;
		    float sqrtsum2 = 0;

		    for(int i = 0; i <feature_length_; i++)//for sum
		    {
		        sqrtsum1 += (feature1[i] * feature1[i]);
		        sqrtsum2 += (feature2[i] * feature2[i]);
		    }

		    sqrtsum1 = sqrt(sqrtsum1);//for square root
		    sqrtsum2 = sqrt(sqrtsum2);

		    for(int i = 0; i < feature_length_; i++)//for normalization
		    {
		        feature1[i] /= sqrtsum1;
		        feature2[i] /= sqrtsum2;
		    }

		    float distance_ = 0;
		    for(int i = 0; i < feature_length_; i++)
		    {
		        distance_ += ((feature1[i] - feature2[i]) * (feature1[i] - feature2[i]));
		    }
		    l2= sqrt(distance_);

}