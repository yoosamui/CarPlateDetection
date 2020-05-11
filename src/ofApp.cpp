#include "ofApp.h"
#include <algorithm>
#include <locale>
#include <opencv2/text/ocr.hpp>
#include <regex>
#include <string>
#include <vector>

/*
#include <stdlib.h>
#include <stdexcept>
#include "math.h"
// dd#include "ofApp.h"
#include "ofAppRunner.h"
#include "ofGraphics.h"
#include "ofPolyline.h"
#include "ofRectangle.h"
#include "stdio.h"
//#include "ofxBaseGui.h"
//#include "ofxGuiGroup.h"
#include <time.h>
*/
#define CAM_WIDTH 640
#define CAM_HEIGHT 480

// work best without preprocessing
//#define OCR_PROCESS_IMAGE 1

// https://github.com/tesseract-ocr/tessdoc/blob/master/Compiling-%E2%80%93-GitInstallation.md
//--------------------------------------------------------------
void ofApp::setup()
{
    ofSetVerticalSync(true);
    // needed for tesseract
    setlocale(LC_ALL, "C");
    setlocale(LC_CTYPE, "C");
    setlocale(LC_NUMERIC, "C");

    int h = 300;
    int w = 450;
    int centerX = (CAM_WIDTH / 2) - w / 2;
    int centerY = (CAM_HEIGHT / 2) - h / 2;

    m_mask_rect = Rect(centerX, centerY, w, h);

    h = 100;
    w = 200;
    centerX = (m_mask_rect.width / 2);  //- (w / 2);
    centerY = ((m_mask_rect.height / 2) + h / 2);

    m_plate_size_max = Rect(centerX, centerY, w, h);
    m_plate_size_min = Rect(centerX, centerY, 20, 20);

    // load default video
    m_isVideoMode = m_video.load("videos/default.mov");

    // ofxCvColorImage colorImg;
    // colorImg.allocate(img.getWidth(), img.getHeight());
    // colorImg.setFromPixels(img);

    // m_grayBg.allocate(img.getWidth(), img.getHeight());
    // m_grayBg = colorImg;

    m_grayImage.allocate(CAM_WIDTH, CAM_HEIGHT);
    m_grayDiff.allocate(CAM_WIDTH, CAM_HEIGHT);

    m_font.load(OF_TTF_SANS, 64, true, true);
    if (m_isVideoMode) {
        //        video.setLoopState(OF_LOOP_NORMAL);
        //        video.setSpeed(VIDEOPLAYSPEED);
        m_video.play();
    } else {
        // setup camera (w,h,color = true,gray = false);
        cam.setup(CAM_WIDTH, CAM_HEIGHT, true);
    }

    this->updateMask();
}

//--------------------------------------------------------------
void ofApp::update()
{
    if (m_isVideoMode) {
        m_video.update();
        m_frame = toCv(m_video);
    } else {
        cam.update();
        m_frame = toCv(cam);
    }

    if (!m_frame.empty()) {
        m_frameNumber++;
        m_frame.copyTo(m_maskOutput, m_mask);

        // Mat matgray;
        //        convertColor(m_frame, m_frameGray, CV_RGB2GRAY);
        convertColor(m_maskOutput, m_frameGray, CV_RGB2GRAY);

        ofImage gray;
        toOf(m_frameGray, gray);

        m_grayImage.setFromPixels(gray.getPixels());

        m_rect_found.clear();

        if (!m_plate_number.empty()) {
            //   m_ocr.clear();
            return;
        }

        // Perform Edge detection
        Canny(m_frameGray, m_cannyOutput, 160, 160, 3);
        findContours(m_cannyOutput, m_contours, m_hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
        if (m_contours.size() == 0) {
            return;
        }

        int approx_size[2]{4, 8};
        int n = m_contours.size();
        vector<Point> approx;

        // find rectangle or squarei

        for (int a = 0; a < 2; a++) {
            for (int i = 0; i < n; i++) {
                approxPolyDP(Mat(m_contours[i]), approx, arcLength(Mat(m_contours[i]), true) * 0.01,
                             true);
                if (approx.size() == approx_size[a]) {
                    Rect r = boundingRect(m_contours[i]);
                    // clang-format off
                    //
                    if (r.width > m_plate_size_min.width && r.height > m_plate_size_min.height &&
                        r.height <= m_plate_size_max.height && r.width <= m_plate_size_max.width &&
                        r.height <= r.width && r.width >= r.height
                        /*
                        r.x > m_mask_rect.x && r.x + r.width  < m_mask_rect.x + m_mask_rect.width &&
                        r.y > m_mask_rect.y && r.y + r.height < m_mask_rect.y + m_mask_rect.height
*/
                        ) {

                       // pusch the rectangle
                        m_rect_found.push_back(r);

                    }
                    // clang-format on
                }
            }
        }

        bool m_accurate = false;
        n = m_rect_found.size();
        if (n > 0) {
            std::sort(m_rect_found.begin(), m_rect_found.end(), ofApp::compare_entry);
            for (int i = 0; i < n; i++) {
                Rect r = m_rect_found[i];

                if (m_accurate) {
                    vector<int> lastfound;
                    for (int z = 0; z < 4; z++) {
                        this->detect_ocr(r);
                        if (!m_plate_number.empty()) {
                            int number = std::stoi(m_plate_number);
                            vector<int>::iterator it =
                                find(lastfound.begin(), lastfound.end(), number);

                            if (it != lastfound.end()) {
                                return;
                                // found it
                            } else {
                                lastfound.push_back(std::stoi(m_plate_number));
                                m_plate_number = {};
                                // doesn't exist
                            }
                        }
                    }

                    n = lastfound.size();
                    for (int z = 0; z < n; z++) {
                        printf("-> %d\n", lastfound[i]);
                    }

                } else {
                    // start ocr detection
                    this->detect_ocr(r);
                }
            }
        }
    }
}

//--------------------------------------------------------------
void ofApp::draw()
{
    ofBackground(ofColor::black);

    switch (m_viewMode) {
        case 1:
            drawMat(m_frame, 0, 0);
            break;
        case 2:
            drawMat(m_maskOutput, 0, 0);
            break;
        case 3:
            drawMat(m_frameGray, 0, 0);
            break;
        case 4:
            drawMat(m_cannyOutput, 0, 0);
            break;

        default:
            break;
    }

    //    m_video.draw(0, 0, 800, 600);
    //   drawMat(matgray, 0, 0);
    //    drawMat(m_cannyOutput, 0, 0);
    //  drawMat(m_maskOutput, 0, 0);

    // m_grayImage.draw(0, 0);

    // m_grayBg.draw(0, 0);
    // frameGrayImg.draw(0, 0);
    //    m_grayDiff.draw(0, 0);
    //
    //

    ofNoFill();
    ofSetLineWidth(2);
    ofSetColor(ofColor::white);
    ofDrawRectangle(m_mask_rect.x, m_mask_rect.y, m_mask_rect.width, m_mask_rect.height);

    // show the plate size
    ofDrawRectangle(m_plate_size_max.x, m_plate_size_max.y, m_plate_size_max.width,
                    m_plate_size_max.height);
    char lbl_rectbuf[128];
    sprintf(lbl_rectbuf, "%d %d  %d %d", m_plate_size_max.x, m_plate_size_max.y,
            m_plate_size_max.width, m_plate_size_max.height);
    ofDrawBitmapStringHighlight(lbl_rectbuf, m_plate_size_max.x, m_plate_size_max.y);

    ofDrawRectangle(m_plate_size_min.x, m_plate_size_min.y, m_plate_size_min.width,
                    m_plate_size_min.height);
    sprintf(lbl_rectbuf, "%d %d  %d %d", m_plate_size_min.x, m_plate_size_min.y,
            m_plate_size_min.width, m_plate_size_min.height);
    ofDrawBitmapStringHighlight(lbl_rectbuf, m_plate_size_min.x, m_plate_size_min.y);

    if (m_plate_number.empty() /* && m_match_counter != 0 */) {
        //  Rect r = m_rect_found[0];
        //  ofNoFill();
        //  ofSetLineWidth(4);
        //  ofSetColor(ofColor::white);
        //        ofSetColor(yellowPrint);
        //  ofDrawRectangle(r.x, r.y, r.width, r.height);
        // if (m_match_counter++ > 1000) {
        // m_plate_number = {};
        //  m_match_counter = 0;
        //  std::this_thread::sleep_for(std::chrono::milliseconds(100));
        // }
    }
    ofPushStyle();

    int n = m_rect_found.size();
    for (int i = 0; i < n; i++) {
        Rect r = m_rect_found[i];
        ofNoFill();
        ofSetLineWidth(2);
        // ofSetColor(ofColor::white);
        ofSetColor(yellowPrint);
        ofDrawRectangle(r.x, r.y, r.width, r.height);

        char lbl_rectbuf[128];
        sprintf(lbl_rectbuf, "%d %d  %d %d", r.x, r.y, r.width, r.height);
        ofDrawBitmapStringHighlight(lbl_rectbuf, r.x, r.y - 10);
        //        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        //   string dimension = to_string(r.y);
        //   ofDrawBitmapStringHighlight(dimension, r.y, r.x);
    }

    ofPopStyle();

    if (m_ocr.getPixels().size() > 1) {
        m_ocr.draw(0, 610);
        m_font.drawString(m_plate_number, 400, 680);

        //  sprintf(lbl_rectbuf, "size  %d %d", m_ocr.getWidth(), m_ocr.getHeight());
        //  ofDrawBitmapStringHighlight(lbl_rectbuf, 1, CAM_HEIGHT - 24);
    }
}

void ofApp::updateMask()
{
    // clang-format off

    m_maskPoints.clear();
    m_maskOutput.release();

    m_maskPoints.push_back(cv::Point(m_mask_rect.x, m_mask_rect.y));
    m_maskPoints.push_back(cv::Point(m_mask_rect.x + m_mask_rect.width, m_mask_rect.y));
    m_maskPoints.push_back(cv::Point(m_mask_rect.x + m_mask_rect.width, m_mask_rect.y + m_mask_rect.height));
    m_maskPoints.push_back(cv::Point(m_mask_rect.x, m_mask_rect.y + m_mask_rect.height));
    m_maskPoints.push_back(cv::Point(m_mask_rect.x, m_mask_rect.y));

    // clang-format on

    this->createMask();
}

void ofApp::createMask()
{
    if (m_maskPoints.size() == 0) {
        m_maskPoints.push_back(cv::Point(2, 2));
        m_maskPoints.push_back(cv::Point(CAM_WIDTH - 2, 2));
        m_maskPoints.push_back(cv::Point(CAM_WIDTH - 2, CAM_HEIGHT - 2));
        m_maskPoints.push_back(cv::Point(2, CAM_HEIGHT - 2));
        m_maskPoints.push_back(cv::Point(2, 2));
    }

    CvMat* matrix = cvCreateMat(CAM_HEIGHT, CAM_WIDTH, CV_8UC1);
    m_mask = cvarrToMat(matrix);  // OpenCV provided this function instead of Mat(matrix).

    for (int i = 0; i < m_mask.cols; i++) {
        for (int j = 0; j < m_mask.rows; j++) m_mask.at<uchar>(cv::Point(i, j)) = 0;
    }

    vector<cv::Point> polyright;
    approxPolyDP(m_maskPoints, polyright, 1.0, true);
    fillConvexPoly(m_mask, &polyright[0], polyright.size(), 255, 8, 0);

    /*
        if (config.maskPoints.size() == 0) {

            config.maskPoints.push_back(cv::Point(2, 2));
            config.maskPoints.push_back(cv::Point(CAMERAWIDTH - 2, 2));
            config.maskPoints.push_back(cv::Point(CAMERAWIDTH - 2, CAMERAHEIGHT - 2));
            config.maskPoints.push_back(cv::Point(2, CAMERAHEIGHT - 2));
            config.maskPoints.push_back(cv::Point(2, 2));
        }

        mask = cvCreateMat(CAMERAHEIGHT, CAMERAWIDTH, CV_8UC1);
        for (int i = 0; i < mask.cols; i++)

            for (int j = 0; j < mask.rows; j++)
                mask.at<uchar>(cv::Point(i, j)) = 0;

        vector<cv::Point> polyright;
        approxPolyDP(config.maskPoints, polyright, 1.0, true);
        fillConvexPoly(mask, &polyright[0], polyright.size(), 255, 8, 0);
        */
}

bool ofApp::compare_entry(const Rect& e1, const Rect& e2)
{
    // bool myfunction (my_data i, my_data j) { return ( i.data_one < j.data_one); }

    return e1.width < e2.width;

    //  return e1.width != e2.width ? e1.width < e2.width : e1.height < e2.height;
    //    if (e1.width != e2.width) return (e1.width < e2.width);
    //    return (e1.height < e2.height);
}

unsigned long previousMillis = 0;

/**
 *
 *
 *
 */
void ofApp::detect_ocr(Rect rect)
{
    if (!m_plate_number.empty()) {
        // m_ocr.clear();
        return;
    }

    m_ocr.setFromPixels(m_grayImage.getPixels());
    m_ocr.crop(rect.x, rect.y, rect.width, rect.height);

#ifdef OCR_PROCESS_IMAGE
    // preprocess the image
    Mat gray = toCv(m_ocr);
    Mat matblur;

    GaussianBlur(gray, matblur, Size(3, 3), 0);

    Mat thres;
    threshold(matblur, thres, 0, 255, cv::THRESH_BINARY_INV + cv::THRESH_OTSU);

    Mat opening;

    Mat element = getStructuringElement(MORPH_RECT, Size(1, 1));
    morphologyEx(thres, opening, MORPH_OPEN, element);

    Mat m_invert = 255 - opening;
    toOf(m_invert, m_ocr);

#endif

    m_ocr.resize(m_ocr.getWidth() + 20, m_ocr.getHeight() + 20);
    m_ocr.update();

    uint64_t currentMillis = ofGetElapsedTimeMillis();
    // if ((int)(currentMillis - previousMillis) >= 100)
    {
        string filename = "result_image.jpg";
        //  m_ocr.save(filename);

        //  printf("ocr-image -->  %f %f\n", m_ocr.getWidth(), m_ocr.getHeight());
        // https://docs.opencv.org/3.4/d7/ddc/classcv_1_1text_1_1OCRTesseract.html
        // be sure that the export var has the eng.training
        static auto ocrp = cv::text::OCRTesseract::create(NULL, "eng", "0123456789", 3, 6);

        Mat img;
        img = toCv(m_ocr);
        string text = ocrp->run(img, 40, cv::text::OCR_LEVEL_TEXTLINE);
        string pnumber = std::regex_replace(text, std::regex("([^0-9])"), "");

        if (pnumber.empty() || pnumber.length() == 1) {
            ocrp = cv::text::OCRTesseract::create(NULL, "eng", "0123456789", 3, 9);
            text = ocrp->run(img, 10, cv::text::OCR_LEVEL_TEXTLINE);
            pnumber = std::regex_replace(text, std::regex("([^0-9])"), "");
        }

        if (pnumber.length() > 1) {
            printf(pnumber.c_str());
            printf("\n");

            string filename = "ocr_image_" + ofGetTimestampString() + ".jpg";
            //      m_ocr.save(filename);

            m_plate_number = pnumber;
        }

        // save the last time we was here
        previousMillis = currentMillis;
    }
}

std::string ofApp::exec(const char* cmd)
{
    const int MAX_BUFFER = 255;

    char buffer[MAX_BUFFER];
    std::string result = "";
    FILE* pipe = popen(cmd, "r");
    if (!pipe) throw std::runtime_error("popen() failed!");
    try {
        while (!feof(pipe)) {
            if (fgets(buffer, MAX_BUFFER, pipe) != NULL) result.append(buffer);
        }

    } catch (...) {
        pclose(pipe);
        throw;
    }
    pclose(pipe);
    return result;
}
/*
OF_KEY_BACKSPACE,
OF_KEY_RETURN,
OF_KEY_PRINTSCR,
OF_KEY_F1 - OF_KEY_F12,
OF_KEY_LEFT,
OF_KEY_UP,
OF_KEY_RIGHT,
OF_KEY_DOWN,
OF_KEY_PAGE_UP,
OF_KEY_PAGE_DOWN,
OF_KEY_HOME,
OF_KEY_END,
OF_KEY_INSERT
*/

bool m_plate_rectangle_set = false;
bool m_key_control_set = false;
const int m_increment = 8;
//--------------------------------------------------------------
void ofApp::keyPressed(int key)
{
    if (key == 'c') {
        m_plate_number = {};
    }

    if (key == '1') {
        m_viewMode = 1;
        return;
    }

    if (key == '2') {
        m_viewMode = 2;
        return;
    }

    if (key == '3') {
        m_viewMode = 3;
        return;
    }

    if (key == '4') {
        m_viewMode = 4;
        return;
    }

    if (m_isVideoMode) {
        if (key == 32) {
            if (!m_video.isPaused()) {
                m_video.setPaused(true);
            } else {
                m_video.setPaused(false);
            }
            return;
        }
    }

    // TAB
    if (key == 9) {
        m_plate_rectangle_set = !m_plate_rectangle_set;
    }

    Rect* dispacher = m_plate_rectangle_set ? &m_plate_size_max : &m_mask_rect;

    //
    //    printf("Key = %d\n", key);

    //
    if (OF_KEY_ALT == key) {
        m_key_control_set = !m_key_control_set;
    }

    if (m_key_control_set) {
        // resize mask
        if (m_key_control_set && key == OF_KEY_UP) {
            dispacher->height -= m_increment;

            if (dispacher == &m_mask_rect) this->updateMask();
        }

        if (m_key_control_set && key == OF_KEY_DOWN) {
            dispacher->height += m_increment;

            if (dispacher == &m_mask_rect) this->updateMask();
        }

        if (m_key_control_set && key == OF_KEY_LEFT) {
            dispacher->width -= m_increment;

            if (dispacher == &m_mask_rect) this->updateMask();
        }

        if (m_key_control_set && key == OF_KEY_RIGHT) {
            dispacher->width += m_increment;

            if (dispacher == &m_mask_rect) this->updateMask();
        }

        return;
    }

    // move mask
    if (OF_KEY_UP == key) {
        dispacher->y -= m_increment;

        if (dispacher == &m_mask_rect) this->updateMask();
    }
    if (OF_KEY_DOWN == key) {
        dispacher->y += m_increment;

        if (dispacher == &m_mask_rect) this->updateMask();
    }

    if (OF_KEY_LEFT == key) {
        dispacher->x -= m_increment;

        if (dispacher == &m_mask_rect) this->updateMask();
    }
    if (OF_KEY_RIGHT == key) {
        dispacher->x += m_increment;

        if (dispacher == &m_mask_rect) this->updateMask();
    }
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key) {}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y) {}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button) {}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button) {}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button) {}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y) {}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y) {}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h) {}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg) {}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo) {}
