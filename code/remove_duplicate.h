#ifndef REMOVE_DUPLICATE
#define REMOVE_DUPLICATE

#include "stdio.h"
#include <stdint.h>
//#include "face_tracker.h"
#include "list.h"
#include "face_info.h"
///
#define MAX_PERSON 100

#define FRAMEINTERVAL 10
#define GRID_X 5
#define GRID_Y 5
#define PATTERNNUM 256  //pow(2, (NEIGHBORS))
#define LBPHSIZE PATTERNNUM*GRID_X*GRID_Y
#define RECT_WIDTH 50
#define RECT_HEIGHT 50
/////////////////////////
#define RADIUS 2
#define NEIGHBORS 8
#define THRESHHOLD 25//47
#define LBPH_LIST_LENGTH 100


#define HISTOCMP_CHISQR_ALT 4
#define HISTOCMP_CHISQR  1
#define LBPH_CELL_HEIGHT (RECT_HEIGHT-2*RADIUS)/GRID_Y
#define LBPH_CELL_WIDTH  (RECT_WIDTH-2*RADIUS)/GRID_X

#define PI   3.1415926535897932384626433832795
#define Epsilon 1.0E-6
#define ID_REUSE_FRAMENUM 100


class RemoveDuplicate{
public:
    static RemoveDuplicate *GetInstance();

    int LBPHTrain(uint8_t *arraySrc, uint32_t labels,idrecord *IDrect,uint32_t frameNum,idrecord *record_all,int camera_id,int lbp_thres=42);
    int CheckDuplicate(cv::Mat, idrecord *IDRecord,uint32_t frameNum,square_rect rects,int camera_id,int lbp_thres = 42);
    int CheckDuplicate1(cv::Mat face, idrecord *IDRecordAll,uint32_t frameNum,square_rect rects,int camera_id);
    double CheckDuplicate2(cv::Mat face, cv::Mat face2);

private:
    static RemoveDuplicate *instance_;
    int Init();
    float CalcDistance(float x1,float y1,float x2,float y2);
    void ELBP(uint8_t *arraySrc, int *arrayDst);
    void CalcHist(const float *_ptrs, int* histArray, const double* _uniranges);
    void CalcHistogram( float *imageArray, float *histArray, const int* histSize, const float** ranges);
    void CopyArray2D(float *src,float *dst,int startHeight,int endHeight,int startWidth,int endWidth);
    void SpatialHistogram(int *arraySrc, float *arrayQuery,int numPatterns,int grid_x, int grid_y, bool /*normed*/);
    void Histogram(uint8_t *arraySrc, float *arrayQuery, int _neighbors, int _grid_x, int _grid_y);
    double CompareHist( float *h1, float *h2, int method );
    uint32_t DeleteLBPHNodebyID(idrecord *IDrect,uint32_t index);
    uint32_t SaveLBPHNodebyID(idrecord *IDrect,LBPNode *pLBPNode,idrecord *record_all,int camera_id);
    LBPNode* LBPHPredict(uint8_t *arraySrc,int &minClass, double &minDist,int camera_id,int lbp_thres = 42);
    double LBPHPredict2(uint8_t *arraySrc,uint8_t *arraySrc2);
    LBPNode* LBPHPredict1(uint8_t *arraySrc,int &minClass, double &minDist,idrecord *IDRecordAll,uint32_t frameNum,square_rect *rects_end,int camera_id);

    float LBPHResult_[LBPHSIZE];
    float LBPHResult_det[LBPHSIZE];
    float LBPHResult1_[LBPHSIZE];
    int g_deleteLBPHNodeCnt_;
    int g_tagLBPHNodeNum_[4]={0};
    LBPNode g_LBPNodeListHead[4];
    LBPHFeatureConfig g_LBPConfig = {GRID_X,GRID_Y,RADIUS,NEIGHBORS,THRESHHOLD};
    cv::Mat recog_face_;// = Mat::zeros(RECT_WIDTH, RECT_HEIGHT, CV_32FC1);

};

#endif
