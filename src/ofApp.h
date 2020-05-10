#pragma once
// clang-format off

#include "ofMain.h"
#include "ofxCv.h"
#include "ofxOpenCv.h"

// clang-format on

using namespace ofxCv;
using namespace cv;

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

    void detect_ocr(Rect rect);
    std::string exec(const char* cmd);
    ///
    ofVideoPlayer m_video;
    ofVideoGrabber cam;
    bool m_isVideoMode = false;
    int m_match_counter = -1;
    long m_frameNumber = 0;
    cv::Mat m_frame;

    Mat m_frameGray;
    Mat m_matGrayBg;
    Mat m_cannyOutput;
    ofxCvGrayscaleImage m_grayBg, m_grayDiff;
    ofxCvGrayscaleImage m_grayImage;
    ofxCvGrayscaleImage m_grayFrame;

    vector<Vec4i> m_hierarchy;
    vector<vector<Point>> m_contours;

    Rect m_plate_size;
    vector<Rect> m_rect_found;
    ofImage m_ocr;
    ofTrueTypeFont m_font;
    string m_plate_number = "";
    Rect m_mask_rect;
};
