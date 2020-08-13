#pragma once
// clang-format off

#include "ofMain.h"
#include "ofxCv.h"
#include "ofxOpenCv.h"
#include "camera.h"

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

    void createMask();
    void updateMask();
    void img_processor();

    bool is_ocr_detection_found(const string& text);
    bool process_tesseract();

    float m_framerateMult;
    static bool compare_entry(const Rect& e1, const Rect& e2);
    std::string exec(const char* cmd);

    Camera m_camera;
    cv::Mat m_frame;
    cv::Mat m_image;
    cv::Mat m_gray;
    cv::Mat m_mask_image;
    cv::Mat m_canny_image;
    cv::Mat m_resized_image;
    cv::Mat m_lightenMat;
    cv::Mat m_gray_masked;
    int m_view_mode = 1;
    int m_blur_value = 3;

    bool m_plate_rectangle_set = false;
    bool m_key_control_set = false;
    vector<Point> m_maskPoints;

    unsigned long previousMillis = 0;

    size_t m_size = 0;

    void regulate_framerate();
    vector<Vec4i> m_hierarchy;
    vector<vector<Point>> m_contours;
    bool is_duplicate(Rect rect);
    static bool m_start_processing;
    bool m_found;
    bool m_isVideoMode = false;
    int m_match_counter = -1;
    long m_frameNumber = 0;
    // cv::Mat m_frame;
    int m_viewMode = 1;
    Mat m_frameGray;
    Mat m_matGrayBg;
    Mat m_cannyOutput;
    ofxCvGrayscaleImage m_grayBg, m_grayDiff;
    ofxCvGrayscaleImage m_grayImage;
    ofxCvGrayscaleImage m_grayFrame;

    Rect m_plate_size_max;
    Rect m_plate_size_min;
    vector<Rect> m_rect_found;
    vector<Rect> m_rect_duplicates;
    ofImage m_ocr;
    ofTrueTypeFont m_font;
    string m_plate_number;
    Rect m_mask_rect;
    //  ofxHttpUtils m_httpUtils;
    cv::Mat m_mask;
    //    cv::Mat m_maskOutput;

    vector<int> m_platedb;

    int m_search_time = 0;
    int m_lighten_value = 0;
};
