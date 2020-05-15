
#pragma once

// clang-format off

#include <regex>
#include <string>
#include <vector>
#include "ofMain.h"
#include "ofThread.h"
#include "ofUtils.h"
#include "ofxCv.h"
#include "ofxOpenCv.h"



//#include <tesseract/baseapi.h>

// clang-format on
//
using namespace ofxCv;
using namespace cv;

class ThreadedOcr : public ofThread
{
    Mat m_ocr;
    string m_plate_number = {};

  public:
    ThreadedOcr();
    void threadedFunction();
    void update(const cv::Mat ocrimage);
    vector<int> m_platedb;
    // void update(const cv::Mat ocrimage);
};
