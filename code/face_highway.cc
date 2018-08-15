#include "FACE_mt.h"
#include <iostream>
#include <string>
#include "opencv2/opencv.hpp"
#include <fstream>
#include <vector>
#include <boost/algorithm/string.hpp>
#include <sys/stat.h>
#include <sys/types.h>
#include <ctime>
#define HEIGHT_ 480
#define WIDTH_ 640
//#define THRESHOLD 0.75
#define RES_D 1
#define MIN_INI 30
using namespace std;
using namespace cv;

void SaveImage(std::string img_path, std::string name,cv::Mat img){
	
	mkdir(img_path.c_str(),S_IRUSR | S_IWUSR | S_IXUSR | S_IRWXG | S_IRWXO);
	int temp_len=img_path.length();
	img_path.insert(temp_len,name);
	string pic_mat=".jpg";
	temp_len=img_path.length();
	img_path.insert(temp_len,pic_mat);
	cout<<"the save phot is:"<<img_path<<endl;
	cv::imwrite( img_path , img );
}


int main(int argc, char **argv)
{
    
    //if(RES_D){
        string pt[4] = {
		"../model_512/step1.prototxt",
		"../model_512/step2.prototxt",
		"../model_512/step3.prototxt",
		"../model_512/face.prototxt"
	};

	string cm[4] = {
		"../model_512/step1.caffemodel",
		"../model_512/step2.caffemodel",
		"../model_512/step3.caffemodel",
		"../model_512/face.caffemodel"
	};

	vector<string> model_file(pt, pt + 4);
	vector<string> trained_file(cm, cm + 4);
	FACE face_net(model_file, trained_file);
  //  }
  //  else{    
  //      string model_file   ="../model/crypt_model";
 //       FACE face_net(model_file);    
  //  }
    string input_file_path;
		string img_src_path;
		string output_file_path;
		string result_param_path;
	string threshold_adj;
	float THRESHOLD;

    input_file_path=argv[1];
		output_file_path = argv[2];
    threshold_adj=argv[3];
    result_param_path=argv[4];
   // out_dir=argv[5];
   
    //mkdir(out_dir.c_str(),S_IRUSR | S_IWUSR | S_IXUSR | S_IRWXG | S_IRWXO);

    THRESHOLD=std::stof(threshold_adj);
    

    ifstream files_in(input_file_path.c_str());
		ofstream output_file(output_file_path.c_str());
		ofstream result_file(result_param_path.c_str());
    int read_flag=0 ;
    cv::Mat img_src1,img_src2;
    cv::Mat img_s1;
    cv::Mat img1_resize,img2_resize;
    float height_a,width_a;
		float height_b,width_b;
    float ratio_a,ratio_b;
    int resize_a_x,resize_a_y;
		int resize_b_x,resize_b_y;
    vector<cv::Mat> crop_a_faces,crop_b_faces;
		int area1,area2;
		cv::Mat tem_face;
		vector<float> feature_a,feature_b;
		float Euclidean_distance;
		int right_num=0,total_num=0;
		int success_match=0;
		int extruct_fg_a=0,extruct_fg_b=0;
		vector<string> detect_failed_img;
		vector<string> no_detect_photo;
		int num_failed=0;
		int num_photo_failed=0;
		float min_a_distance=MIN_INI,max_a_distance=-1;

		string id_path="/data/common/HighRailway/id/";
		string photo_path="/data/common/HighRailway/photo/";
		
		int id_length = id_path.length();
		int photo_length = photo_path.length();
		int error_match_num=0;
		int photo_total_num=0;
	float total_time;
   	 clock_t start,finish;

    while ( std::getline(files_in,img_src_path)){
	start=clock();
 			total_num+=1;
			  vector<string> img_b_paths;
				string img_a_path;
				string img_b_path;
   	vector<string> strVec;
        boost::trim(img_src_path);
        boost::split(strVec,img_src_path,boost::is_any_of(","));
	string img_name=strVec[0];
        int name_length= img_name.length();
        cout<<"the size is: " <<strVec.size()<<endl;
				for(int i=0; i<strVec.size();i++){
					if(i==0){
						img_a_path=id_path;
						img_a_path.insert(id_length,strVec[0]);
					}
					else{
						img_b_path=photo_path;
						img_b_path.insert(photo_length,strVec[i]);
						img_b_paths.push_back(img_b_path);
					}
				}
				//cout<<"the path id is: "<<img_a_path<<endl;
				
        img_s1=cv::imread(img_a_path);
        //height_a=img_s1.rows;
	//width_a=img_s1.cols;
        //cv::resize(img_s1,img_src1,cv::Size(2*width_a,2*height_a));
	img_src1=img_s1;
        if(!img_src1.empty()){
          height_a=img_src1.rows;
          width_a=img_src1.cols;
          ratio_a=min(height_a/HEIGHT_, width_a/WIDTH_);
          resize_a_x=std::floor(width_a/ratio_a);
          resize_a_y=std::floor(height_a/ratio_a);
          if(height_a>HEIGHT_ || width_a>WIDTH_){
            cv::resize(img_src1,img1_resize,cv::Size(resize_a_x,resize_a_y));
            face_net.detectFromOriImg(img1_resize,img_src1,crop_a_faces);
          }
          else{
						face_net.detect(img_src1,crop_a_faces);
          }
					if(crop_a_faces.size()>0 && face_net.bounding_box_.size()>0){
						for(int i=0;i<face_net.bounding_box_.size()-1;i++){
						    area1=face_net.bounding_box_[i].height *face_net.bounding_box_[i].width;
								for(int j=i+1;j<face_net.bounding_box_.size();j++){
									 area2=face_net.bounding_box_[j].height *face_net.bounding_box_[j].width;
									 if(area1<area2){
										   tem_face=crop_a_faces[i];
											 crop_a_faces[i]=crop_a_faces[j];
											 crop_a_faces[j]=tem_face;
									 }
								}
						 }
						face_net.extractFeature(crop_a_faces[0],feature_a);
						/*
						string name=img_name.substr(0,name_length-4);
						string subdir = "id_f_face/";
						mkdir(subdir.c_str(),S_IRUSR | S_IWUSR | S_IXUSR | S_IRWXG | S_IRWXO);
						int temp_len=subdir.length();
						subdir.insert(temp_len,name);
						string pic_mat=".jpg";
						temp_len=subdir.length();
						subdir.insert(temp_len,pic_mat);
						*/
						//cout<<"the save id is:"<<subdir<<endl;
						//cv::imwrite( subdir , crop_a_faces[0] );
						extruct_fg_a=1;
					 }
					 else{
					     cout<<"the image1 failed extracting" <<endl;
							 detect_failed_img.push_back(img_a_path);
							 num_failed+=1;
						 }
         }
				 else {
					   cout<<"the image1 failed reading:"<<endl;
				 }

				for(int j=0;j<img_b_paths.size();j++){
           //cout<<"the image2 path is: "<<img_b_paths[j]<<endl;
           img_src2=cv::imread(img_b_paths[j]);
				   if(!img_src2.empty()){
							photo_total_num+=1;
						 	height_b=img_src2.rows;
						 	width_b=img_src2.cols;
						 	ratio_b=min(height_b/HEIGHT_, width_b/WIDTH_);
						 	resize_b_x=std::floor(width_b/ratio_b);
						 	resize_b_y=std::floor(height_b/ratio_b);
						 	if(height_b>HEIGHT_ || width_b>WIDTH_){
								cout<<"the image2 will be resized"<<endl;
								//cout<<"begin to detect image2"<<endl;
						 		cv::resize(img_src2,img2_resize,cv::Size(resize_b_x,resize_b_y));
						 		face_net.detectFromOriImg(img2_resize,img_src2,crop_b_faces);
						 	}
						 	else{
								//cout<<"begin to detect image2"<<endl;
						 		face_net.detect(img_src2,crop_b_faces);
						 	}
							//cout<<"image2 success detect and size is: "<<crop_b_faces.size()<<endl;
						 	if(crop_b_faces.size()>0 && face_net.bounding_box_.size()>0){
						 		for(int i=0;i<face_net.bounding_box_.size()-1;i++){
						 				area1=face_net.bounding_box_[i].height *face_net.bounding_box_[i].width;
						 				for(int j=i+1;j<face_net.bounding_box_.size();j++){
						 					 area2=face_net.bounding_box_[j].height *face_net.bounding_box_[j].width;
						 					 if(area1<area2){
						 							 tem_face=crop_b_faces[i];
						 							 crop_b_faces[i]=crop_b_faces[j];
						 							 crop_b_faces[j]=tem_face;
						 					 }
						 				}
						 		 }
						 		face_net.extractFeature(crop_b_faces[0],feature_b);
								extruct_fg_b=1;
						/*
						string img_name=strVec[j+1];
						int  name_length= img_name.length();
						string name=img_name.substr(0,name_length-4);
								string subdir = "photo_f_face/";
						mkdir(subdir.c_str(),S_IRUSR | S_IWUSR | S_IXUSR | S_IRWXG | S_IRWXO);
						int temp_len=subdir.length();
						subdir.insert(temp_len,name);
						string pic_mat=".jpg";
						temp_len=subdir.length();
						subdir.insert(temp_len,pic_mat);
						cout<<"the save phot is:"<<subdir<<endl;
						cv::imwrite( subdir , crop_b_faces[0] );
						*/
						 	 }
						 	 else{
						 			 cout<<"the image2 failed extracting" <<endl;
									 no_detect_photo.push_back(img_b_paths[j]);
									 num_photo_failed+=1;
								 }
						  }
						  else
						 		 cout<<"the image2 failed reading:"<<endl;

				if( extruct_fg_a && extruct_fg_b){
					face_net.calculateDistance(feature_a,feature_b,Euclidean_distance);
					result_file<<Euclidean_distance<<endl;
					if(Euclidean_distance > THRESHOLD){
						     success_match+=1;
					}
					else{
							error_match_num+=1;
							//output_file<<"The match dataset error image is: "<<img_a_path<<'\t';
							//output_file<<"and"<<img_b_paths[j]<<endl;
					}

					if(Euclidean_distance <min_a_distance){
							min_a_distance=Euclidean_distance;
					}
					if(Euclidean_distance >max_a_distance){
							max_a_distance=Euclidean_distance;
					}
				}

				extruct_fg_b=0;
				crop_b_faces.clear();
				face_net.bounding_box_.clear();
			}
			if(success_match){
				right_num+=1;
			}
			extruct_fg_a=0;
			success_match=0;
			crop_a_faces.clear();
 	 		//if(total_num==100){break;}
	finish=clock();
	total_time=float((finish-start))/CLOCKS_PER_SEC;
	cout<<"the total time is: "<<total_time<<" s"<<endl;
    }

		output_file<<"******************************************************************"<<endl;
		output_file<<"the match dataset  error number are: "<<error_match_num<<endl;
		output_file<<"the match dataset  total number are: "<<photo_total_num<<endl;
    		output_file<<"the threshold is: "<<THRESHOLD<<endl<<endl;
		output_file<<"the id match success number is "<<right_num<<endl;
                output_file<<"the id total images  are "<<total_num<<endl;
		output_file<<"the dataset probability is "<<float(right_num)/total_num<<endl;
		output_file<<"******************************************************************"<<endl;

			output_file<<"match dataset min distance is: "<<min_a_distance<<endl;
			output_file<<"match dataset max distance is: "<<max_a_distance<<endl;
		output_file<<"******************************************************************"<<endl;
			output_file<<"the dataset id failed extract image num is: "<<num_failed<<endl;
			output_file<<"the dataset photo failed extract image num is: "<<num_photo_failed<<endl;

		output_file<<"******************************************************************"<<endl;
		output_file<<"the image id faied detect are below"<<endl;
		for(int i=0;i< detect_failed_img.size();i++){
			output_file<<detect_failed_img[i]<<endl;
		}
		output_file<<"******************************************************************"<<endl;
		output_file<<"the image photo faied detect are below"<<endl;
		for(int i=0;i< no_detect_photo.size();i++){
			output_file<<no_detect_photo[i]<<endl;
		}
		files_in.close();
		output_file.close();
		result_file.close();
    return 0;
}
