#include "remove_duplicate.h"
#include <stdlib.h>
//#include "face_tracker.h"
//#ifdef __cplusplus
//extern "C" {
//#endif
RemoveDuplicate *RemoveDuplicate::instance_ = NULL;
RemoveDuplicate *RemoveDuplicate::GetInstance()
{
    if(instance_ == NULL){
        instance_ = new RemoveDuplicate;
        instance_->Init();
    }
    return instance_;
}

int RemoveDuplicate::Init()
{
    g_deleteLBPHNodeCnt_ = 0;
    //g_tagLBPHNodeNum_ = 0;
    recog_face_ = cv::Mat::zeros(RECT_WIDTH, RECT_HEIGHT, CV_32FC1);
    for(int i=0;i<4;i++)
        INIT_LIST_HEAD(&g_LBPNodeListHead[i].list);
    return 0;
}
//计算每个cell的elbp值
void RemoveDuplicate::ELBP(uint8_t *arraySrc, int *arrayDst)
{
    int neighbors = NEIGHBORS;
    int radius = RADIUS;

    for(int n=0; n<neighbors; n++)
    {
        // sample points
        float x = (float)(radius * cos(2.0*PI*n/(float)(neighbors)));
        float y = (float)(-radius * sin(2.0*PI*n/(float)(neighbors)));
        // relative indices
        int fx = (int)(floor(x));
        int fy = (int)(floor(y));
        int cx = (int)(ceil(x));
        int cy = (int)(ceil(y));
        // fractional part
        float ty = y - fy;
        float tx = x - fx;
        // set interpolation weights
        float w1 = (1 - tx) * (1 - ty);
        float w2 =      tx  * (1 - ty);
        float w3 = (1 - tx) *      ty;
        float w4 =      tx  *      ty;

        for(int i=radius; i < RECT_HEIGHT-radius;i++) {
            for(int j=radius;j < RECT_WIDTH-radius;j++) {
                // calculate interpolated value
                float t = (float)(w1*arraySrc[(i+fy)*RECT_WIDTH +(j+fx)] + w2*arraySrc[(i+fy)*RECT_WIDTH + (j+cx)] + w3*arraySrc[(i+cy)*RECT_WIDTH + (j+fx)] + w4*arraySrc[(i+cy)*RECT_WIDTH+(j+cx)]);
                // floating point precision, so check some machine-dependent epsilon
                arrayDst[(i-radius)*(RECT_WIDTH - 2*RADIUS) + (j-radius)] += ((t > arraySrc[i*RECT_WIDTH + j]) || (fabs(t-arraySrc[i*RECT_WIDTH + j]) < Epsilon)) << n;
            }
        }
    }

    return;
}


void RemoveDuplicate::CalcHist(const float *_ptrs, int* histArray, const double* _uniranges)
{
    int* H = (int *)histArray;
    int x;

    const double* uniranges = &_uniranges[0];

    double a = uniranges[0], b = uniranges[1];
    int sz = PATTERNNUM;
    const float* p0 = _ptrs;

    for( x = 0; x < LBPH_CELL_HEIGHT*LBPH_CELL_WIDTH; x++, p0++ )//imsize.width = 9*9
    {
        int idx = floor(*p0*a + b);
        if( (unsigned)idx < (unsigned)sz )
        {
            ((int*)H)[idx]++;
        }
    }

    return;

}

void RemoveDuplicate::CalcHistogram( float *imageArray, float *histArray,
                    const int* histSize, const float** ranges)
{
    float *ptrs = imageArray;
    double uniranges[2];

    double low = ranges[0][0], high = ranges[0][1];
    double t = histSize[0]/(high - low);

    uniranges[0] = t;
    uniranges[1] = -t*low;

    const double* _uniranges = &uniranges[0];
    int TemphistArray[PATTERNNUM] = {0};

    CalcHist(ptrs, TemphistArray, _uniranges);

    for(int i = 0; i < PATTERNNUM; i++) {
        histArray[i] = (float)TemphistArray[i];
    }

    return;
}


void RemoveDuplicate::CopyArray2D(float *src,float *dst,int startHeight,int endHeight,int startWidth,int endWidth)
{
    for(int i = 0; i < endHeight - startHeight; i++)
    {
        for(int j = 0; j < endWidth - startWidth; j++)
        {
            dst[i*LBPH_CELL_WIDTH + j] = src[(startHeight + i)*(RECT_WIDTH - 2*RADIUS) + (startWidth + j)];
        }
    }
}

void RemoveDuplicate::SpatialHistogram(int *arraySrc, float *arrayQuery,int numPatterns,
                      int grid_x, int grid_y, bool /*normed*/)
{
    float arraySrctemp[(RECT_HEIGHT-2*RADIUS)*(RECT_WIDTH-2*RADIUS)];
    float arraySrcCell[LBPH_CELL_HEIGHT*LBPH_CELL_WIDTH];

    // calculate LBP patch size
    int width = LBPH_CELL_WIDTH;
    int height = LBPH_CELL_HEIGHT;

    // allocate memory for the spatial histogram
    int resultRowIdx = 0;

    for(int i = 0; i < RECT_HEIGHT-2*RADIUS; i++) {
        for(int j = 0; j < RECT_WIDTH-2*RADIUS; j++) {
            arraySrctemp[i*(RECT_WIDTH-2*RADIUS) + j] = (float)arraySrc[i*(RECT_WIDTH - 2*RADIUS) + j];
        }
    }

    for(int i = 0; i < grid_y; i++) {
        for(int j = 0; j < grid_x; j++) {

            CopyArray2D(arraySrctemp,arraySrcCell,i*height,(i+1)*height,j*width,(j+1)*width);
            float cellHistArray[PATTERNNUM] = {0};

            // Establish the number of bins.
            int histSize = numPatterns;
            // Set the ranges.
            float range[] = { (float)(0), (float)(numPatterns-1) };
            const float* histRange = { range };
            // calc histogram
            CalcHistogram(arraySrcCell, cellHistArray, &histSize, &histRange);

            // normalize
            for(int i = 0; i < PATTERNNUM; i++)
            {
                cellHistArray[i] /= (int)(LBPH_CELL_HEIGHT * LBPH_CELL_WIDTH);
            }
            memcpy(&arrayQuery[resultRowIdx*PATTERNNUM],&cellHistArray[0],sizeof(float)*PATTERNNUM);
            resultRowIdx++;
        }
    }
    return;
}

void RemoveDuplicate::Histogram(uint8_t *arraySrc, float *arrayQuery, int _neighbors, int _grid_x, int _grid_y)
{
    int arrayDst[(RECT_HEIGHT - 2*RADIUS)*(RECT_WIDTH - 2*RADIUS)] = {0};

    ELBP(arraySrc,arrayDst);

    SpatialHistogram(
                arrayDst, /* lbp_image */
                arrayQuery,
                (int)(pow(2.0, (double)(_neighbors))), /* number of possible patterns */
                _grid_x, /* grid size x */
                _grid_y, /* grid size y */
                true /* normed histograms */);
}

double RemoveDuplicate::CompareHist( float *h1, float *h2, int method )
{
    double result = 0;

    if( (method == HISTOCMP_CHISQR) || (method == HISTOCMP_CHISQR_ALT))
    {
        for(int j = 0 ; j < LBPHSIZE; j++ )
        {
            double a = h1[j] - h2[j];
            double b = (method == HISTOCMP_CHISQR) ? h1[j] : h1[j] + h2[j];
            if( fabs(b) > DBL_EPSILON )
            {
                result += a*a/b;
            }
        }
    }

    if( method == HISTOCMP_CHISQR_ALT )
    {
        result *= 2;
    }

    return result;
}

uint32_t RemoveDuplicate::DeleteLBPHNodebyID(idrecord *IDrect,uint32_t index)
{
    //printf("------------~~!!!!DeleteLBPHNodebyID:enter!node to be delete:0x%x\n",IDrect->LBPRecord[index]);
#ifdef DEBUG
    fprintf(stderr,"in DeleteLBPHNodebyID \n");
#endif
    if(IDrect->LBPRecord[index] == NULL){
        //fprintf(stderr,"IDrect->LBPRecord[index] == NULL \n");
        //list_del_init(&(IDrect->LBPRecord[index]->list));
        return 0;
    }
    else{
        list_del_init(&(IDrect->LBPRecord[index]->list));
        if(IDrect->LBPRecord[index] !=NULL ){
            free(IDrect->LBPRecord[index]);
            IDrect->LBPRecord[index] = NULL;
        }

        g_deleteLBPHNodeCnt_++;
    }
    return IDrect->LBPRecordCurIndex;
}

uint32_t RemoveDuplicate::SaveLBPHNodebyID(idrecord *IDrect,LBPNode *pLBPNode,idrecord *record_all,int camera_id)
{
    uint32_t index = (++IDrect->LBPRecordCurIndex)%5;
    struct list_head *pos;
#ifdef DEBUG
    fprintf(stderr,"in SaveLBPHNodebyID index = %d \n",index);
#endif
    //printf("~~~~~~~~~SaveLBPHNodebyID:entry!g_tagLBPHNodeNum = \n");

    if(NULL != IDrect->LBPRecord[index])
    {
        //fprintf(stderr,"IDrect->LBPRecord[index]==null to DeleteLBPHNodebyID \n");
        DeleteLBPHNodebyID(IDrect,index);
        g_tagLBPHNodeNum_[camera_id]--;
    }

    if(g_tagLBPHNodeNum_[camera_id] >= LBPH_LIST_LENGTH)
    {
        LBPNode *TempLBPNode = NULL;
        pos = g_LBPNodeListHead[camera_id].list.prev;
        TempLBPNode = list_entry(pos, LBPNode, list);

        //printf("SaveLBPHNodebyID:TempLBPNode = \n");

        if(0 != record_all[TempLBPNode->savedLBPBlock.Label].recordNum)
        {
            for(int i = 0; i < 5; i++)
            {
                if(TempLBPNode == record_all[TempLBPNode->savedLBPBlock.Label].LBPRecord[i])
                {
                    //printf("SaveLBPHNodebyID: One LBP in IDrecord to be deleted:label \n");
                    DeleteLBPHNodebyID(&record_all[TempLBPNode->savedLBPBlock.Label],i);
                    g_tagLBPHNodeNum_[camera_id]--;
                    TempLBPNode = NULL;
                    break;
                }
            }
        }

        if(NULL != TempLBPNode)
        {
            list_del_init(&(TempLBPNode->list));
            g_tagLBPHNodeNum_[camera_id]--;
            free(TempLBPNode);
        }
    }
    list_add(&(pLBPNode->list), &(g_LBPNodeListHead[camera_id].list));//add to head
    IDrect->LBPRecord[index] = pLBPNode;
    IDrect->LBPRecordCurIndex = index;
    g_tagLBPHNodeNum_[camera_id]++;

    return index;
}

int RemoveDuplicate::LBPHTrain(uint8_t *arraySrc, uint32_t labels,idrecord *IDrect,uint32_t frameNum,idrecord *record_all,int camera_id,int lbp_thres)
{
    LBPNode *pLBPNode;
    LBPNode *pTempLBPNode = NULL;

    //printf("LBPHTrain:enter!labels = %d,IDrect->LBPRecordCurIndex = %d\n",
    //	labels,IDrect->LBPRecordCurIndex);
    Histogram(arraySrc, LBPHResult1_, g_LBPConfig._neighbors,g_LBPConfig._grid_x,g_LBPConfig._grid_y);

    if(NULL != IDrect->LBPRecord[IDrect->LBPRecordCurIndex])
    {
        //printf("LBPHTrain:LBPrecord is more than 5\n");
        for(int i = 0;i< 5;i++){
             pTempLBPNode = IDrect->LBPRecord[i];
             if(pTempLBPNode != NULL){
                 double dist = CompareHist(pTempLBPNode->savedLBPBlock.LBPH, LBPHResult1_, HISTOCMP_CHISQR_ALT);
//                 printf("compare dist : %lf\n",dist);
                 if(dist < (float)(lbp_thres *0.85 ))
                     return 0;
             }
             else{
                 break;
             }
        }
//        pTempLBPNode = IDrect->LBPRecord[IDrect->LBPRecordCurIndex];
//        double dist = CompareHist(pTempLBPNode->savedLBPBlock.LBPH, LBPHResult1_, HISTOCMP_CHISQR_ALT);

//        if(dist < (float)(THRESHHOLD*6)/7)
//        {
//            return 0;
//        }
    }
    pLBPNode = (LBPNode*)malloc(sizeof(LBPNode));
    pLBPNode->savedLBPBlock.Label = labels;
    pLBPNode->savedLBPBlock.LBPHmatsize = LBPHSIZE;
    memcpy(pLBPNode->savedLBPBlock.LBPH,LBPHResult1_,sizeof(float)*pLBPNode->savedLBPBlock.LBPHmatsize);
    SaveLBPHNodebyID(IDrect,pLBPNode,record_all,camera_id);

    return 1;
}

LBPNode* RemoveDuplicate::LBPHPredict(uint8_t *arraySrc,int &minClass, double &minDist,int camera_id,int lbp_thres)
{

    LBPNode *pTempLBPNode,*retLBPNode;
    struct list_head *pos, *next;

    //printf("LBPHPredict: enter!\n");

    Histogram(arraySrc, LBPHResult_, g_LBPConfig._neighbors,g_LBPConfig._grid_x,g_LBPConfig._grid_y);
    Histogram(arraySrc, LBPHResult_, g_LBPConfig._neighbors,g_LBPConfig._grid_x,g_LBPConfig._grid_y);

    minDist = DBL_MAX;
    minClass = -1;

    list_for_each_safe(pos, next, &g_LBPNodeListHead[camera_id].list)
    {
        pTempLBPNode = list_entry(pos, LBPNode, list);
        double dist = CompareHist((pTempLBPNode->savedLBPBlock.LBPH), LBPHResult_, HISTOCMP_CHISQR_ALT);
        //printf("LBPHPredict: dist = %.3f,Label = %d,size = %d\n",
        //        dist,pTempLBPNode->savedLBPBlock.Label,pTempLBPNode->savedLBPBlock.LBPHmatsize);
//        if((dist < minDist) && (dist < g_LBPConfig._threshold))
        if((dist < minDist) && (dist < lbp_thres))
        {
            minDist = dist;
            minClass = pTempLBPNode->savedLBPBlock.Label;
            retLBPNode = pTempLBPNode;
        }
    }
    //printf("LBPHPredict: mindist = %.3f\n",minDist);
    return retLBPNode;
}

double RemoveDuplicate::LBPHPredict2(uint8_t *arraySrc,uint8_t *arraySrc2)
{
    //printf("LBPHPredict: enter!\n");

    Histogram(arraySrc, LBPHResult_, g_LBPConfig._neighbors,g_LBPConfig._grid_x,g_LBPConfig._grid_y);
    Histogram(arraySrc2, LBPHResult_det, g_LBPConfig._neighbors,g_LBPConfig._grid_x,g_LBPConfig._grid_y);

    double dist = CompareHist( LBPHResult_,LBPHResult_det, HISTOCMP_CHISQR_ALT);
    //printf("LBPHPredict: mindist = %.3f\n",minDist);
    return dist;
}

float RemoveDuplicate::CalcDistance(float x1,float y1,float x2,float y2)
{
    float result = sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2));
    return result;
}

int RemoveDuplicate::CheckDuplicate(cv::Mat face, idrecord *IDRecordAll,uint32_t frameNum,square_rect rects,int camera_id,int lbp_thres)
{
    int label = -1;
    double confidence = 0;
    if(0 != g_tagLBPHNodeNum_[camera_id])
    {
            uint8_t arraySrc[RECT_HEIGHT*RECT_WIDTH] = {0};
            if (face.isContinuous())
            {
               memcpy(arraySrc,face.data,sizeof(uint8_t)*RECT_HEIGHT*RECT_WIDTH);
            }

            LBPHPredict(arraySrc,label,confidence,camera_id);
            //printf("-------predict result:g_tagLBPHNodeNum = %d,g_deleteLBPHNodeCnt = %d,label = %d,confidence = %.3f\n",g_tagLBPHNodeNum,g_deleteLBPHNodeCnt,label,confidence);

            ///找到了LBPH记录
            if(-1 != label)
            {
                    float distancetoLBP = CalcDistance(IDRecordAll[label].cxLastLBP,IDRecordAll[label].cyLastLBP,rects.cx,rects.cy);

                    ///判断是否与当前帧间隔太久，或者中心距离太远
//                    if(distancetoLBP > IDRecordAll[label].sizeLastLBP*3 ||
//                            (frameNum > IDRecordAll[label].agingCount) && (frameNum - IDRecordAll[label].agingCount > ID_REUSE_FRAMENUM))
//                    {
//                            label = -1;
                            //g_reuseIDFailCnt++;
//                    }
                    ///找到的ID控制块已经被删除，重建ID控制块
//                    else
                    if(0 == IDRecordAll[label].recordNum)
                    {
                            printf("!!!----founded ID has been deleted.\n");
                            //rebuild the ID list
                   //InitList(&IDRecordAll[label].pArrayRecord);
                            IDRecordAll[label].reuseIDFlag = 1;
                            //initKalman(&IDRecordAll[label],rects[idx].cx,rects[idx].cy);
                            //trackingProcessKalman(&IDRecordAll[label], &rects[idx].cx, &rects[idx].cy, 0,&rects[idx]);
                    }
                    //printf("!!!----label = %d\n",label);
            }
    }
    return label;
}



double RemoveDuplicate::CheckDuplicate2(cv::Mat face,cv::Mat face2)
{
    double confidence = 0;
    uint8_t arraySrc[RECT_HEIGHT*RECT_WIDTH] = {0};
    uint8_t arraySrc2[RECT_HEIGHT*RECT_WIDTH] = {0};
    if (face.isContinuous())
      {
        memcpy(arraySrc,face.data,sizeof(uint8_t)*RECT_HEIGHT*RECT_WIDTH);
      }
    if (face2.isContinuous()){
        memcpy(arraySrc2,face2.data,sizeof(uint8_t)*RECT_HEIGHT*RECT_WIDTH);
    }

    confidence = LBPHPredict2(arraySrc,arraySrc2);
    return confidence;
}




LBPNode* RemoveDuplicate::LBPHPredict1(uint8_t *arraySrc,int &minClass, double &minDist,idrecord *IDRecordAll,uint32_t frameNum,square_rect *rects_end,int camera_id)
{
    int labelTemp;
    double dummy;

    LBPNode *pTempLBPNode,*retLBPNode;
    struct list_head *pos, *next;
    double distancetoLBP;

    //printf("LBPHPredict: enter!\n");

    Histogram(arraySrc, LBPHResult_, g_LBPConfig._neighbors,g_LBPConfig._grid_x,g_LBPConfig._grid_y);

    minDist = DBL_MAX;
    minClass = -1;

    list_for_each_safe(pos, next, &g_LBPNodeListHead[camera_id].list)
    {
        pTempLBPNode = list_entry(pos, LBPNode, list);

        labelTemp = pTempLBPNode->savedLBPBlock.Label;

        distancetoLBP = CalcDistance(IDRecordAll[labelTemp].cxLastLBP,IDRecordAll[labelTemp].cyLastLBP,rects_end->cx,rects_end->cy);

        //判断是否与当前帧间隔太久，或者中心距离太远
        if(distancetoLBP > IDRecordAll[labelTemp].sizeLastLBP*3 ||
                (frameNum > IDRecordAll[labelTemp].agingCount) && (frameNum - IDRecordAll[labelTemp].agingCount > ID_REUSE_FRAMENUM))
        {
            //g_reuseIDFailCnt++;
            printf("LBPHPredict:Rect center is far from LBP,continue.LBP(%d,%d),rect(%d,%d)\n",
                   IDRecordAll[labelTemp].cxLastLBP,IDRecordAll[labelTemp].cyLastLBP,rects_end->cx,rects_end->cy);
        }
        else
        {
            double dist = CompareHist((pTempLBPNode->savedLBPBlock.LBPH), LBPHResult_, HISTOCMP_CHISQR_ALT);
            printf("LBPHPredict: dist = %.3f,Label = %d,size = %d\n",
                   dist,pTempLBPNode->savedLBPBlock.Label,pTempLBPNode->savedLBPBlock.LBPHmatsize);
            if((dist < minDist) && (dist < g_LBPConfig._threshold)) {
                minDist = dist;
                minClass = pTempLBPNode->savedLBPBlock.Label;
                retLBPNode = pTempLBPNode;
            }
        }
    }

    return retLBPNode;
}

int RemoveDuplicate::CheckDuplicate1(cv::Mat face, idrecord *IDRecordAll,uint32_t frameNum,square_rect rects,int camera_id)
{
    int label = -1;
    double confidence = 0;
    if(0 != g_tagLBPHNodeNum_[camera_id])
    {
            uint8_t arraySrc[RECT_HEIGHT*RECT_WIDTH] = {0};
            if (face.isContinuous())
            {
               memcpy(arraySrc,face.data,sizeof(uint8_t)*RECT_HEIGHT*RECT_WIDTH);
            }

            LBPHPredict1(arraySrc,label,confidence,IDRecordAll,frameNum,&rects,camera_id);

            //找到了LBPH记录
            if(-1 != label)
            {
                //g_reuseIDSuccessCnt++;

                //找到的ID控制块已经被删除，重建ID控制块
                if(0 == IDRecordAll[label].recordNum)
                {
                    printf("!!!----founded ID has been deleted.\n");
                    //rebuild the ID list
                    //InitList(&IDRecordAll[label].pArrayRecord);
                    IDRecordAll[label].reuseIDFlag = 1;
                    //initKalman(&IDRecordAll[label],rects[idx].cx,rects[idx].cy);
                    //trackingProcessKalman(&IDRecordAll[label], &rects[idx].cx, &rects[idx].cy, 0,&rects[idx]);
                }
                printf("!!!----call LBPHPredict,success: Reuse ID label = %d\n",label);
            }
    }
    return label;
}
