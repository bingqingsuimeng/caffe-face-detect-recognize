#include "FACE.h"
#include <iostream>
#include <string.h>
#include "opencv2/opencv.hpp"
#include <fstream>
#define HEIGHT_ 480
#define WIDTH_ 640
#define MIN_INI 30
using namespace std;
using namespace cv;

/*
void GetName(const std::string &s, std::string &n) {
		std::size_t found = s.find_last_of("/");
		std::string tmp_str = s.substr(0, found);
		found = tmp_str.find_last_of("/");
		n = tmp_str.substr(found + 1);
                //std::size_t pose=s.find(".jpg");
               // n=s.substr(found+1, pose-found-1);
	}
*/
int main(int argc, char **argv)
{
    string model_file   ="../model/crypt_model";
    FACE face(model_file);
    cout<<"load model success"<<endl;
    string input_file_a_path, img_a_path;
    string input_file_b_path, img_b_path;
    string output_file_path;
    string threshold_adj;
    float THRESHOLD,DIFF;

    input_file_a_path=argv[1];
    input_file_b_path=argv[2];
		output_file_path = argv[3];
    threshold_adj=argv[4];

    THRESHOLD=std::stof(threshold_adj);
    DIFF=std::stof(threshold_adj);

    ifstream file_a_in(input_file_a_path.c_str());
    ifstream file_b_in(input_file_b_path.c_str());
    ofstream output_file(output_file_path.c_str());
    int read_flag=0 ;
    cv::Mat img_src1,img_src2;
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
		int extruct_fg_a=0,extruct_fg_b=0;
		float prob,prob_num=0,prob_ave=0;
		vector<float> probs;
		vector<string> detect_failed_img;
		vector<int> num_failed_extract;
                int num_failed=0;
		vector<float> max_a_distances,min_a_distances;
		vector<float> max_b_distances,min_b_distances;
		float min_a_distance=MIN_INI,max_a_distance=-1;
		float min_b_distance=MIN_INI,max_b_distance=-1;
   int error_match_num=0,error_mismatch_num=0;
   //cout<<"begin to read file"<<endl;
    while ( std::getline(file_a_in,img_a_path)){
        std::getline(file_b_in,img_b_path);
				total_num+=1;
        read_flag=read_flag+1;
	//cout<<"the read num is: "<<total_num<<endl;
       // cout<<" the img_a path is: "<<img_a_path<<endl;
        img_src1=cv::imread(img_a_path);
       // cout<<" the img_b path is: "<<img_b_path<<endl;
        img_src2=cv::imread(img_b_path);
        if(!img_src1.empty()){
          height_a=img_src1.rows;
          width_a=img_src1.cols;
          ratio_a=min(height_a/HEIGHT_, width_a/WIDTH_);
          resize_a_x=std::floor(width_a/ratio_a);
          resize_a_y=std::floor(height_a/ratio_a);
          if(height_a>HEIGHT_ || width_a>WIDTH_){
	      cout<<" the image1 is resized"<<endl;
            cv::resize(img_src1,img1_resize,cv::Size(resize_a_x,resize_a_y));
            face.detectFromOriImg(img1_resize,img_src1,crop_a_faces);
          }
          else{
						face.detect(img_src1,crop_a_faces);
		//cout<<"image1 detect is successful"<<endl;
          }
	 // cout<<"image1 crop size is:"<<crop_a_faces.size()<<endl;
					if(crop_a_faces.size()>0 && face.bounding_box_.size()>0){
						//cout<<" the image1 bounding box size is: "<<face.bounding_box_.size()<<endl;
						for(int i=0;i<face.bounding_box_.size()-1;i++){
						    area1=face.bounding_box_[i].height *face.bounding_box_[i].width;
								for(int j=i+1;j<face.bounding_box_.size();j++){
									 area2=face.bounding_box_[j].height *face.bounding_box_[j].width;
									 if(area1<area2){
										   tem_face=crop_a_faces[i];
											 crop_a_faces[i]=crop_a_faces[j];
											 crop_a_faces[j]=tem_face;
									 }
								}
						 }
						//cout<<"image1 begin to extract"<<endl;
						face.extractFeature(crop_a_faces[0],feature_a);
						//cout<<"imag1 extract successful"<<endl;
						extruct_fg_a=1;
					 }
					 else{
					     cout<<"the image1 failed extracting" <<endl;
					     detect_failed_img.push_back(img_a_path);
					     num_failed+=1;
					 }
         }
				 else {
					   cout<<"the image1 failed reading: "<<img_a_path<<endl;
				 }


				 if(!img_src2.empty()){
						 	height_b=img_src2.rows;
						 	width_b=img_src2.cols;
						 	ratio_b=min(height_b/HEIGHT_, width_b/WIDTH_);
						 	resize_b_x=std::floor(width_b/ratio_b);
						 	resize_b_y=std::floor(height_b/ratio_b);
						 	if(height_b>HEIGHT_ || width_b>WIDTH_){
							   // cout<<" the image2 will be resized"<<endl;
						 		cv::resize(img_src2,img2_resize,cv::Size(resize_b_x,resize_b_y));
						 		face.detectFromOriImg(img2_resize,img_src2,crop_b_faces);
						 	}
						 	else{
								//cout<<"begin to detect img2"<<endl;	
						 		face.detect(img_src2,crop_b_faces);
								//cout<<"the image2 detect is successful"<<endl;
						 	}
						 	if(crop_b_faces.size()>0 && face.bounding_box_.size()>0){
						 		for(int i=0;i<face.bounding_box_.size()-1;i++){
						 				area1=face.bounding_box_[i].height *face.bounding_box_[i].width;
						 				for(int j=i+1;j<face.bounding_box_.size();j++){
						 					 area2=face.bounding_box_[j].height *face.bounding_box_[j].width;
						 					 if(area1<area2){
						 							 tem_face=crop_b_faces[i];
						 							 crop_b_faces[i]=crop_b_faces[j];
						 							 crop_b_faces[j]=tem_face;
						 					 }
						 				}
						 		 }
						 		face.extractFeature(crop_b_faces[0],feature_b);
								extruct_fg_b=1;
						 	 }
						 	 else{
						 			 cout<<"the image2 failed extracting" <<endl;
									detect_failed_img.push_back(img_b_path);
									num_failed+=1;
							}
						  }
						  else
						 		 cout<<"the image2 failed reading: "<<img_b_path<<endl;

				if( extruct_fg_a && extruct_fg_b){
					face.calculateDistance(feature_a,feature_b,Euclidean_distance);
					//cout<<"the distance is: "<<Euclidean_distance<<endl;
					//cv::imshow("img_a",crop_a_faces[0]);
					//cv::imshow("img_b",crop_b_faces[0]);
					//cv::waitKey(0);
					crop_a_faces.clear();
					crop_b_faces.clear();
				  if(read_flag <=300){
					    if(Euclidean_distance < THRESHOLD){
						     right_num+=1;
					    }
							else{
								error_match_num+=1;
								output_file<<"The match dataset error image is: "<<img_a_path<<'\t';
								output_file<<"another is: "<<img_b_path<<endl;
							}
					   if(Euclidean_distance <min_a_distance){
								min_a_distance=Euclidean_distance;
							}
					   if(Euclidean_distance >max_a_distance){
								max_a_distance=Euclidean_distance;
							}
				  }
					if(read_flag>300 && read_flag <=600){
						  if(Euclidean_distance >=DIFF){
								right_num+=1;
							}
							else{   
								error_mismatch_num+=1;
								output_file<<"**********************************************************************"<<endl;
								output_file<<"The mismatch dataset error image is: "<<img_a_path<<'\t';
								output_file<<"another is: "<<img_b_path<<endl;
								output_file<<"**********************************************************************"<<endl;
							}
						if(Euclidean_distance <min_b_distance){
								min_b_distance=Euclidean_distance;
							}
						if(Euclidean_distance >max_b_distance){
								max_b_distance=Euclidean_distance;
							}
					}
				}
				//if(float(total_num)/600.0 >9){
				  //  cout<<total_num<<endl;
				//}
				if(read_flag==600){
					read_flag=0;
					prob=float(right_num)/600.0;
					probs.push_back(prob);
					prob_num+=prob;
					cout<<"the dataset num:"<<(total_num/600)<<" probability is "<<prob<<endl;
					cout<<min_a_distance<<'\t'<<max_a_distance<<'\t'<<min_b_distance<<'\t'<<max_b_distance<<endl;
					right_num=0;
					num_failed_extract.push_back(num_failed);
					num_failed=0;
					min_a_distances.push_back(min_a_distance);
					min_b_distances.push_back(min_b_distance);
					max_a_distances.push_back(max_a_distance);
					max_b_distances.push_back(max_b_distance);
					min_a_distance=MIN_INI;
					min_b_distance=MIN_INI;
					max_a_distance=-1;
					max_b_distance=-1;
				}
				extruct_fg_a=0;
				extruct_fg_b=0;
    }
   
    output_file<<"****************************************************************************"<<endl;
    output_file<<"the match dataset and mismatch dataset error number are: "<<error_match_num<<'\t'<<error_mismatch_num<<endl;
    output_file<<"the threshold is: "<<THRESHOLD<<endl<<endl;
    for(int j=0;j<probs.size();j++){
        output_file<<"the dataset "<<j<<" probability is: "<<probs[j]<<endl;
    }
    prob_ave=prob_num/10.0;
		cout<<"the average probability is: "<<prob_ave<<endl;
    output_file<<"the average probability is: "<<prob_ave<<endl<<endl;
    output_file<<"******************************************************************"<<endl;
		for(int i=0;i<min_a_distances.size();i++){
			output_file<<"dataset "<<i<<"  match set and mismatch set  min and max distances are:   ";
			output_file<<min_a_distances[i]<<'\t'<<max_a_distances[i]<<'\t'<<min_b_distances[i]<<'\t'<<max_b_distances[i]<<endl;
			
		}
    output_file<<"****************************************************************************"<<endl;
    for(int j=0;j<num_failed_extract.size();j++){
	output_file<<"the dataset "<<j<<" failed extract image num is: "<<num_failed_extract[j]<<endl;
    }
    output_file<<"****************************************************************************"<<endl;
    output_file<<"the failed detect images are: "<<endl<<endl;
    for(int i=0; i< detect_failed_img.size();i++){
	output_file<<detect_failed_img[i]<<endl;
    }
    file_a_in.close();
    file_b_in.close();
    output_file.close();
    return 0;
}
