#ifndef FACE_INFO_H
#define FACE_INFO_H
#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <ctime>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <functional>
#include <ctime>
#include "list.h"

#define MAX_PERSON 100

#define FRAMEINTERVAL 10
#define GRID_X 5
#define GRID_Y 5
#define PATTERNNUM 256  //pow(2, (NEIGHBORS))
#define LBPHSIZE PATTERNNUM*GRID_X*GRID_Y
#define RECT_WIDTH 50
#define RECT_HEIGHT 50


struct CamInfo {
    int camera_id;
    std::string ip_addr;
    std::string location;
    std::string mac;
    std::string name;
};

struct FaceDetectInfo{
    cv::Rect bounding_box;   ///人脸在原图中对应box
    float score;
    cv::Mat bbox_face;
    std::time_t time_stamp;
};

struct FaceTrackInfo{
    FaceDetectInfo raw_face;
    cv::Rect bounding_box;   ///人脸在原图对应box
    int32_t face_id;
    bool is_detect;
};

struct FaceQuality{
    float blur;
    float rotate;
    float up_down;
    float left_right;
};

// private structs

struct square_rect {
    uint32_t cx, cy, size;
    float score;
};

struct SquareRectNode {
        square_rect record;
        SquareRectNode* pnext;
        SquareRectNode* ppre;
        uint8_t trackingflag;
        uint8_t reserve[3];
};

typedef struct _movpara {
float k,b,vx,vy;//y=kx+b, vx is velocity,ax is accelerate
}MovPara;

typedef struct record{
        uint32_t IDindex;
        float distance;
}Record;


typedef struct _LBPBlock{
        float LBPH[LBPHSIZE];
        uint32_t LBPHmatsize;
        uint32_t Label;
}LBPBlock;

typedef struct _LBPNode{
        struct list_head list;
        LBPBlock savedLBPBlock;
}LBPNode;

typedef struct _LBPHFeatureConfig{
    int _grid_x;
    int _grid_y;
    int _radius;
    int _neighbors;
    double _threshold;
}LBPHFeatureConfig;

struct idrecord {
    MovPara moveStatePara;
    SquareRectNode* pArrayRecord;
    uint32_t agingCount;
    //CvKalman* kalman;
    uint8_t recordNum;
    uint8_t recordNumHistory;
    uint8_t trackerInitFlag;
    //KCFTracker tracker;
    uint8_t LBPAddedFlag;
    uint32_t cxLastLBP, cyLastLBP,sizeLastLBP,idx;
    LBPNode* LBPRecord[5];
    uint32_t LBPRecordCurIndex;
    float scoreHighest;
    uint8_t reuseIDFlag;
    uint8_t Reserve[3];
};


#endif
