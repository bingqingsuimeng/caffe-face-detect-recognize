/*###############################################
#created by :  lxy
#Time:  2017/08/03 14:09
#project: Face recognize
#company: Senscape
#rversion: 0.1
#tool:
#modified:
####################################################*/
#include "FACE.h"
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
#define IMG_RS 1
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

void GetName(const std::string &s, std::string &n) {
		std::size_t found = s.find_last_of("/");
		//std::string tmp_str = s.substr(0, found);
		//found = tmp_str.find_last_of("/");
		//n = tmp_str.substr(found + 1);
    std::size_t pose=s.find(".bmp");
    n=s.substr(found+1, pose-found-1);
	}


int main(int argc, char **argv)
{

    //if(RES_D){
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

	vector<string> model_file(pt, pt + 5);
	vector<string> trained_file(cm, cm + 5);
	FACE face_net(model_file, trained_file);
  //  }
  //  else{
  //      string model_file   ="../model/crypt_model";
 //       FACE face_net(model_file);
  //  }
    string input_file_path,input_photo_path;
		string img_src_path,photo_src_path;
		string output_file_path;
		string result_param_path;
		string result_param_path2;
		string threshold_adj;
		float THRESHOLD;

    input_file_path=argv[1];
		input_photo_path=argv[2];
		output_file_path = argv[3];
    threshold_adj=argv[4];
    result_param_path=argv[5];
    //result_param_path2=argv[6];

    THRESHOLD=std::stof(threshold_adj);


    ifstream files_in(input_file_path.c_str());
		ifstream photo_in(input_photo_path.c_str());
		ofstream output_file(output_file_path.c_str());
		ofstream result_file(result_param_path.c_str());
		//ofstream result_file2(result_param_path2.c_str());
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
		vector<float> feature_a1,feature_b1,feature_a2,feature_b2;
		float Euclidean_distance,Euclidean_distance1,Euclidean_distance2;
		int right_num=0,total_num=0;
		int success_match=0;
		int extruct_fg_b=0;
		vector<int> extruct_fg_a;
		vector<string> detect_failed_img;
		vector<string> no_detect_photo;
		vector<string> nomatch_photos;
		int num_failed=0,num_failed_a=0;
		int num_photo_failed=0;
		float min_a_distance=MIN_INI,max_a_distance=-1;

		string id_path="/data/common/HighRailway/id/";
		string photo_path="/data/common/HighRailway/photo/";

		int id_length = id_path.length();
		int photo_length = photo_path.length();
		int error_match_num=0;
		int photo_total_num=0;
		double total_time=0,total_time_a=0,total_ave_time1=0,total_ave_time2=0,total_time_b=0;
   	 clock_t start,finish,start_a,finish_a,start_b,finish_b;
		int num_b_total=0;

		cv::Rect ave_box,face_box;
    cv::Mat img_roi;
		cv::Mat src_resize=cv::Mat::zeros(112, 96, CV_8UC3);
    ave_box.x=17;
    ave_box.y=22;
    ave_box.width=70;
    ave_box.height=50;
		vector<vector<float>> feature_b1_vec,feature_b2_vec;
		vector<vector<float>> feature_a1_vec;
		vector<string> name_a_vec;
		int min_x=1000,min_y=1000,max_wid=0,max_high=0;
		int real_r_num=0;

		while(std::getline(files_in,img_src_path)){
			string img_a_path;
			string name;

			start_a=clock();
			boost::trim(img_src_path);
			img_a_path=img_src_path;
			img_src1=cv::imread(img_a_path);
			if(!img_src1.empty()){
				height_a=img_src1.rows;
				width_a=img_src1.cols;
				ratio_a=min(height_a/HEIGHT_, width_a/WIDTH_);
				resize_a_x=std::floor(width_a/ratio_a);
				resize_a_y=std::floor(height_a/ratio_a);
				if(IMG_RS && (height_a>HEIGHT_ || width_a>WIDTH_)){
					cv::resize(img_src1,img1_resize,cv::Size(resize_a_x,resize_a_y));
					face_net.detectFromOriImg(img1_resize,img_src1,crop_a_faces);
				}
				else{
					face_net.detect(img_src1,crop_a_faces);
				}
				//if(crop_a_faces.size()>0 && face_net.bounding_box_.size()>0){
				/*
				if(crop_a_faces.size()==0){
					cv::Mat srcROI=img_src1(ave_box);
					srcROI.copyTo(img_roi);
					cv::resize(img_roi,src_resize,src_resize.size());
					crop_a_faces.push_back(src_resize);
				}
				*/
				if(crop_a_faces.size()>0 ){
					if(crop_a_faces.size()>=2){
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
					 }
					face_net.extractFeature(crop_a_faces[0],3,feature_a1);
					//face_net.extractFeature(crop_a_faces[0],4,feature_a2);
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
					feature_a1_vec.push_back(feature_a1);
					GetName(img_a_path,name);
					cout<<"the id input name is: "<<name<<endl;
					name_a_vec.push_back(name);
					extruct_fg_a.push_back(1);
				 }
				 else{
						 cout<<"the image1 failed extracting" <<endl;
						 detect_failed_img.push_back(img_a_path);
						 num_failed_a+=1;
					 }
			 }
			 else {
					 cout<<"the image1 failed reading:"<<endl;
			 }
			finish_a=clock();
			total_time_a=float((finish_a-start_a))/CLOCKS_PER_SEC;
			total_ave_time1+=total_time_a;
			crop_a_faces.clear();
		}

    while ( std::getline(photo_in,photo_src_path)){
	      start=clock();
 				total_num+=1;
			  vector<string> img_b_paths;
				string img_a_path;
				string img_b_path;
   			vector<string> strVec;
        boost::trim(photo_src_path);
        boost::split(strVec,photo_src_path,boost::is_any_of(","));
				string img_name=strVec[0];
        int name_length= img_name.length();
				string name=img_name.substr(0,name_length-4);
        //cout<<"the size is: " <<strVec.size()<<endl;
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


				for(int j=0;j<img_b_paths.size();j++){
           cout<<"the image2 path is: "<<img_b_paths[j]<<endl;
					 cout<<"the input photo name is: "<<name<<endl;
					 start_b=clock();
           img_src2=cv::imread(img_b_paths[j]);
				   if(!img_src2.empty()){
							photo_total_num+=1;
						 	height_b=img_src2.rows;
						 	width_b=img_src2.cols;
						 	ratio_b=min(height_b/HEIGHT_, width_b/WIDTH_);
						 	resize_b_x=std::floor(width_b/ratio_b);
						 	resize_b_y=std::floor(height_b/ratio_b);
						 	if( IMG_RS &&(height_b>HEIGHT_ || width_b>WIDTH_)){
								cout<<"the image2 will be resized"<<endl;
								//cout<<"begin to detect image2"<<endl;
						 		cv::resize(img_src2,img2_resize,cv::Size(resize_b_x,resize_b_y));
						 		face_net.detectFromOriImg(img2_resize,img_src2,crop_b_faces);
						 	}
						 	else{
								//cout<<"begin to detect image2"<<endl;
						 		face_net.detect(img_src2,crop_b_faces);
						 	}
							//cout<<"the crop size is: " <<crop_b_faces.size()<<endl;
							//cout<<"image2 success detect and size is: "<<crop_b_faces.size()<<endl;
						 	if(crop_b_faces.size()>0 && face_net.bounding_box_.size()>0){
								if(crop_b_faces.size()>=2){
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
								 }
							  for(int k=0;k<crop_b_faces.size();k++){
									face_net.extractFeature(crop_b_faces[k],3,feature_b1);
									//face_net.extractFeature(crop_b_faces[k],4,feature_b2);
									feature_b1_vec.push_back(feature_b1);
									//feature_b2_vec.push_back(feature_b2);
									extruct_fg_b=1;
								}
						 	 }
						 	 else{
						 			 cout<<"the image2 failed extracting" <<endl;
									 no_detect_photo.push_back(img_b_paths[j]);
									 num_photo_failed+=1;
								 }
						  }
						  else
						 		 cout<<"the image2 failed reading:"<<endl;

				if( extruct_fg_a[0] && extruct_fg_b){
					for(int k=0;k<feature_b1_vec.size();k++){
					  face_net.calculateDistance(feature_a1_vec[0],feature_b1_vec[k],3,Euclidean_distance1);
					  //face_net.calculateDistance(feature_a2,feature_b2_vec[k],4,Euclidean_distance2);
					  //Euclidean_distance=0.85*Euclidean_distance1+1.15*Euclidean_distance2;
					  if(Euclidean_distance1 <= THRESHOLD){
						   success_match+=1;
							num_b_total+=1;
					  }
						result_file<<Euclidean_distance1<<endl;
						//result_file2<<Euclidean_distance2<<endl;
				  }

					finish_b=clock();
					total_time_b=float((finish_b-start_b))/CLOCKS_PER_SEC;
					total_ave_time2+=total_time_b;

					if(Euclidean_distance1 <min_a_distance){
							min_a_distance=Euclidean_distance;
					}
					if(Euclidean_distance1 >max_a_distance){
							max_a_distance=Euclidean_distance;
					}
				}

				if(success_match){
					right_num+=1;
					if(name.compare(name_a_vec[0])==0){
						real_r_num+=1;
					}
					else{
						nomatch_photos.push_back(img_b_paths[j]);
					}
				}
				else{
					error_match_num+=1;
				}

				extruct_fg_b=0;
				crop_b_faces.clear();
				face_net.bounding_box_.clear();
				feature_b1_vec.clear();
				success_match=0;
				//feature_b2_vec.clear();
			}
 	 //if(total_num==100){break;}
	finish=clock();
	total_time=float((finish-start))/CLOCKS_PER_SEC;
	cout<<"the total time is: "<<total_time<<" s"<<endl;
    }

		output_file<<"******************************************************************"<<endl;
		output_file<<"photo could extract feature but  can't match number are: "<<error_match_num<<endl;
		output_file<<"the  photo dataset  total number are:    "<<photo_total_num<<endl;
    output_file<<"the threshold is: "<<THRESHOLD<<endl<<endl;
		output_file<<"the id match success number is "<<right_num<<endl;
    output_file<<"the id real match name images  are "<<real_r_num<<endl;
		output_file<<"the dataset probability is "<<float(right_num)/photo_total_num<<endl;
		output_file<<"******************************************************************"<<endl;

			output_file<<"match dataset min distance is: "<<min_a_distance<<endl;
			output_file<<"match dataset max distance is: "<<max_a_distance<<endl;
		output_file<<"******************************************************************"<<endl;
			output_file<<"the dataset id failed extract image num is: "<<num_failed_a<<endl;
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
		output_file<<"******************************************************************"<<endl;
		output_file<<"the extracted photo mismatch are below"<<endl;
		for(int i=0;i< nomatch_photos.size();i++){
			output_file<<nomatch_photos[i]<<endl;
		}
		files_in.close();
		photo_in.close();
		output_file.close();
		result_file.close();
		//result_file2.close();
    return 0;
}
