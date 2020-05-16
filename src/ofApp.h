#pragma once
// clang-format off

#include "ofMain.h"
#include "ofxCv.h"
#include "ofxOpenCv.h"

#ifdef PI_CAM
#include "ofxCvPiCam.h"
#endif

#include <thread>
#include <queue>          // std::queue
// clang-format on
#include "thread_safe_queue.h"
using namespace ofxCv;
using namespace cv;

//#define PI_CAM

class ofApp : public ofBaseApp
{
  public:
    void setup();
    void update();
    void draw();

    void keyPressed(int key);
    void keyReleased(int key);
    void mouseMoved(int x, int y);
    void mouseDragged(int x, int y, int button);
    void mousePressed(int x, int y, int button);
    void mouseReleased(int x, int y, int button);
    void mouseEntered(int x, int y);
    void mouseExited(int x, int y);
    void windowResized(int w, int h);
    void dragEvent(ofDragInfo dragInfo);
    void gotMessage(ofMessage msg);

    void createMask();
    void updateMask();
    static void process_tesseract();
    void detect_ocr(Rect rect);
    static bool compare_entry(const Rect& e1, const Rect& e2);
    static void remove_producer(std::thread::id id);
    static void remove_consumer(std::thread::id id);
    void wait_sensor();
    std::string exec(const char* cmd);
    ///
    ofVideoPlayer m_video;

#ifdef PI_CAM
    ofxCvPiCam cam;
#else
    ofVideoGrabber cam;
#endif

    bool m_isVideoMode = false;
    int m_match_counter = -1;
    long m_frameNumber = 0;
    cv::Mat m_frame;
    int m_viewMode = 1;
    Mat m_frameGray;
    Mat m_matGrayBg;
    Mat m_cannyOutput;
    ofxCvGrayscaleImage m_grayBg, m_grayDiff;
    ofxCvGrayscaleImage m_grayImage;
    ofxCvGrayscaleImage m_grayFrame;

    vector<Vec4i> m_hierarchy;
    vector<vector<Point>> m_contours;

    Rect m_plate_size_max;
    Rect m_plate_size_min;
    vector<Rect> m_rect_found;
    static ofImage m_ocr;
    ofTrueTypeFont m_font;
    static string m_plate_number;
    Rect m_mask_rect;
    //  ofxHttpUtils m_httpUtils;
    vector<Point> m_maskPoints;
    cv::Mat m_mask;
    cv::Mat m_maskOutput;

    static vector<int> m_platedb;
    //  ThreadedOcr m_threadOcr;
    std::thread* m_ocrthread1 = nullptr;
    std::thread* m_ocrthread2 = nullptr;
    std::thread* m_ocrthread3 = nullptr;

    static std::queue<ofImage> m_ocrQueue;
    static vector<ofImage*> m_ocrList;

    int m_search_time = 0;
};
