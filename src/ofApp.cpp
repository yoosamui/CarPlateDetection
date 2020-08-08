#include "ofApp.h"
#include <algorithm>
#include <condition_variable>
#include <iostream>
#include <locale>
#include <mutex>
#include <opencv2/text/ocr.hpp>
#include <queue>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

//#define CAM_WIDTH 320   // 640
//#define CAM_HEIGHT 240  // 480

#define CAM_WIDTH 640
#define CAM_HEIGHT 480
Mat masked;

// work best without preprocessing
//#define OCR_PROCESS_IMAGE 1

unsigned long previousMillis = 0;

ofImage ofApp::m_ocr;
vector<int> ofApp::m_platedb;
string ofApp::m_plate_number;

std::vector<std::thread> producers, consumers;
static int thread_counter;
ThreadSafeQueue<int /*std::thread::id*/> ts_queue(1000);
static std::map<int /*std::thread::id*/, cv::Mat> m_ocrMap;

vector<Rect> ofApp::m_rect_duplicates;

int max_producers = 1000;
int max_consumers = 1;

std::mutex tmutex;

#ifdef IP_CAM
cv::VideoCapture capture("rtsp://admin:admin@192.168.1.88:554/11",
                         cv::CAP_ANY);  // 0 = autodetect default API
#endif

//--------------------------------------------------------------
void ofApp::setup()
{
    thread_counter = 0;

    ofSetVerticalSync(true);
    ofSetWindowTitle("Number recognition v1.0");

    // needed for tesseract
    setlocale(LC_ALL, "C");
    setlocale(LC_CTYPE, "C");
    setlocale(LC_NUMERIC, "C");

    m_plate_number = "0";

    int h = 300;
    int w = 450;
    int centerX = (CAM_WIDTH / 2) - w / 2;
    int centerY = (CAM_HEIGHT / 2) - h / 2;

    m_mask_rect = Rect(centerX, centerY, w, h);
    m_mask_rect = Rect(0, 0, CAM_WIDTH, CAM_HEIGHT);

    h = 100;
    w = 100;
    centerX = (m_mask_rect.width / 2);  //- (w / 2);
    centerY = ((m_mask_rect.height / 2) + h / 2);

    m_plate_size_max = Rect(centerX, centerY, w, h);
    m_plate_size_min = Rect(centerX, centerY, 10, 10);

    m_platedb.push_back(200);

    m_platedb.push_back(1402);
    m_platedb.push_back(396);
    m_platedb.push_back(96);
    m_platedb.push_back(356);
    m_platedb.push_back(149);
    m_platedb.push_back(357);
    m_platedb.push_back(146);
    m_platedb.push_back(470);   // sup moto
    m_platedb.push_back(7095);  // ford sup
    m_platedb.push_back(4349);
    m_platedb.push_back(207);
    m_platedb.push_back(9311);  // alex
    m_platedb.push_back(7079);  // lars

    m_platedb.push_back(3786);   // phone
    m_platedb.push_back(93459);  // phone

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
#ifndef IP_CAM
        // setup camera (w,h,color = true,gray = false);
        cam.setup(CAM_WIDTH, CAM_HEIGHT, true);
#endif
    }

    this->updateMask();
}

bool ofApp::is_ocr_detection_found(const string& text)
{
    if (text.empty() || text.length() < 2) return false;

    try {
        long pnumber = std::stol(text);

        printf("---------->%ld\n", pnumber);
        // checks if exitst in database
        vector<int>::iterator it = find(m_platedb.begin(), m_platedb.end(), pnumber);

        if (it != m_platedb.end()) {
            m_plate_number = to_string(pnumber);
            printf("-----FOUND\n");
            m_rect_duplicates.clear();
            return true;
        }
    } catch (const std::invalid_argument& ia) {
        std::cerr << "Invalid argument: " << ia.what() << '\n';
    }

    return false;
}

bool ofApp::process_tesseractX(int id)
{
    if (!m_plate_number.empty()) {
        return true;
    }

    if (m_ocrMap.count(id) != 0) {
        /*
            std::ostringstream sid;
            sid << id;
            std::string idstr = sid.str();

            printf("FOUND THREAD %s \n", idstr.c_str());
    */
        printf("FOUND THREAD %d \n", id);

        Mat img = m_ocrMap[id];

        auto ocrp = cv::text::OCRTesseract::create(NULL, "eng", "0123456789", 3, 6);

        string text = ocrp->run(img, 40, cv::text::OCR_LEVEL_TEXTLINE);
        string pnumber = std::regex_replace(text, std::regex("([^0-9])"), "");
        //  printf("---------->%s\n", text.c_str());
        if (is_ocr_detection_found(pnumber)) return true;

        ocrp = cv::text::OCRTesseract::create(NULL, "eng", "0123456789", 3, 9);
        text = ocrp->run(img, 10, cv::text::OCR_LEVEL_TEXTLINE);
        pnumber = std::regex_replace(text, std::regex("([^0-9])"), "");
        //  printf("---------->%s\n", text.c_str());

        if (is_ocr_detection_found(pnumber)) return true;
    } else {
        printf("image not found\n");
    }

    return false;
}

bool ofApp::process_tesseract()
{
    if (m_ocr.getPixels().size() > 0 && m_plate_number.empty()) {
        auto ocrp = cv::text::OCRTesseract::create(NULL, "eng", "0123456789", 3, 6);

        Mat img;
        img = toCv(m_ocr);

        string text = ocrp->run(img, 10, cv::text::OCR_LEVEL_TEXTLINE);
        string pnumber = std::regex_replace(text, std::regex("([^0-9])"), "");
        printf("[1]---------->%s %s\n", text.c_str(), pnumber.c_str());

        if (is_ocr_detection_found(pnumber)) return true;

        ocrp = cv::text::OCRTesseract::create(NULL, "eng", "0123456789", 3, 9);
        text = ocrp->run(img, 10, cv::text::OCR_LEVEL_TEXTLINE);
        pnumber = std::regex_replace(text, std::regex("([^0-9])"), "");
        printf("[2]---------->%s %s\n", text.c_str(), pnumber.c_str());

        if (is_ocr_detection_found(pnumber)) return true;
    }

    return false;
}

static bool isSet = false;
Mat lightenMat;
//--------------------------------------------------------------
void ofApp::update()
{
    if (m_isVideoMode) {
        m_video.update();
        m_frame = toCv(m_video);
    } else {
#ifdef IP_CAM
        // m_frame = cam.grab();
        capture.read(m_frame);
#else
        cam.update();
        m_frame = toCv(cam);
#endif
    }

    if (!m_frame.empty()) {
        m_frameNumber++;

        //#ifdef IP_CAM
        // Mat frame_resized(CAM_WIDTH, CAM_HEIGHT, CV_8UC4);
        Mat frame_resized;
        Size size(CAM_WIDTH, CAM_HEIGHT);
        resize(m_frame, frame_resized, size);  // resize image
        // resize(m_frame, m_maskOutput, size);  // resize image
        //  Mat lightenMat;
        lightenMat = frame_resized + cv::Scalar(m_lighten_value, m_lighten_value, m_lighten_value);

        //#e    ndif
        // printf("%d\n", m_mask_rect.height);
        // terminate();

        //#ifndef IP_CAM
        // m_frame.copyTo(m_maskOutput, m_mask);
        //        m_maskOutput.copyTo(masked, m_mask);//works
        // frame_resized.copyTo(m_maskOutput, m_mask);  // works
        lightenMat.copyTo(m_maskOutput, m_mask);  // not works

        //   m_maskOutput.copyTo(m_mask);
        //#endif

        convertColor(m_maskOutput, m_frameGray, CV_RGB2GRAY);
        // convertColor(resized_frame, m_frameGray, CV_RGB2GRAY);

        ofImage gray;
        toOf(m_frameGray, gray);
        m_grayImage.setFromPixels(gray.getPixels());

        if (isSet) return;
        m_rect_found.clear();

        if (!m_plate_number.empty()) {
            //    m_ocr.clear();

            return;
        } else {
            uint64_t currentMillis = ofGetElapsedTimeMillis();
            if ((int)(currentMillis - previousMillis) >= 1000) {
                m_search_time++;
                previousMillis = currentMillis;
            }
        }

        if (m_search_time >= 30) {
            m_plate_number = "not found.";
            m_rect_duplicates.clear();
        }

        // Noise Reduction Since edge detection is susceptible to noise in the image, first step is
        // to remove the noise in the image with a 5x5 Gaussian filter
        blur(m_frameGray, 1);
        //  dilate(m_frameGray);

        // Perform Canny Edge Detection.
        //
        // Canny(m_frameGray, m_cannyOutput, 160, 160, 3);
        Canny(m_frameGray, m_cannyOutput, 200, 50,
              3);  // Apperture size 3-7

        findContours(m_cannyOutput, m_contours, m_hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
        if (m_contours.size() == 0) {
            return;
        }

        int approx_size[2]{4, 8};
        int n = m_contours.size();
        vector<Point> approx;
        // find rectangle or square
        for (int a = 0; a < 2; a++) {
            for (int i = 0; i < n; i++) {
                approxPolyDP(Mat(m_contours[i]), approx, arcLength(Mat(m_contours[i]), true) * 0.01,
                             true);
                if (approx.size() == approx_size[a]) {
                    Rect r = boundingRect(m_contours[i]);
                    // clang-format off

                    if (r.width > m_plate_size_min.width && r.height > m_plate_size_min.height &&
                        r.height <= m_plate_size_max.height && r.width <= m_plate_size_max.width //&&
                //        r.height <= r.width && r.width >= r.height
                        ) {

                        m_rect_found.push_back(r);
                    }

                    // clang-format on
                }
            }
        }
        //    isSet = true;

        // sort asc and process image
        if (m_rect_found.size()) {
            std::sort(m_rect_found.begin(), m_rect_found.end(), ofApp::compare_entry);
            img_processor();
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
            //  drawMat(lightenMat, 0, 0);
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
    // ofNoFill();
    // ofSetLineWidth(2);
    // int n = m_rect_found.size();
    // for (int i = 0; i < n; i++) {
    // Rect r = m_rect_found[i];
    //// ofSetColor(ofColor::white);
    // ofSetColor(yellowPrint);
    // ofDrawRectangle(r.x, r.y, r.width, r.height);
    //}
    //#ifdef AAAAA
    ofNoFill();
    ofSetLineWidth(1.5);
    ofSetColor(ofColor::white);
    ofDrawRectangle(m_mask_rect.x, m_mask_rect.y, m_mask_rect.width, m_mask_rect.height);

    // show the plate size
    ofDrawRectangle(m_plate_size_max.x, m_plate_size_max.y, m_plate_size_max.width,
                    m_plate_size_max.height);

    char lbl_rectbuf[512];
    /*
    sprintf(lbl_rectbuf, "%d %d  %d %d", m_plate_size_max.x, m_plate_size_max.y,
            m_plate_size_max.width, m_plate_size_max.height);
    ofDrawBitmapStringHighlight(lbl_rectbuf, m_plate_size_max.x, m_plate_size_max.y);

    ofDrawRectangle(m_plate_size_min.x, m_plate_size_min.y, m_plate_size_min.width,
                    m_plate_size_min.height);
    sprintf(lbl_rectbuf, "%d %d  %d %d", m_plate_size_min.x, m_plate_size_min.y,
            m_plate_size_min.width, m_plate_size_min.height);
    ofDrawBitmapStringHighlight(lbl_rectbuf, m_plate_size_min.x, m_plate_size_min.y);
    */

    // show mode
    sprintf(lbl_rectbuf, "mode: %d", m_viewMode);
    ofDrawBitmapStringHighlight(lbl_rectbuf, 1, CAM_HEIGHT + 12);

    // show threads
    sprintf(lbl_rectbuf, "consumers: %d", consumers.size());
    ofDrawBitmapStringHighlight(lbl_rectbuf, 80, CAM_HEIGHT + 12);

    if (m_plate_number.empty()) {
        ofDrawBitmapStringHighlight("searching...", 150, CAM_HEIGHT + 12);
    }

    sprintf(lbl_rectbuf, "producers: %d consumers %d ", producers.size(), consumers.size());
    ofDrawBitmapStringHighlight(lbl_rectbuf, 1, 12);

    sprintf(lbl_rectbuf, "%d:%d:%d", ofGetHours(), ofGetMinutes(), ofGetSeconds());
    ofDrawBitmapStringHighlight(lbl_rectbuf, 280, 12);

    sprintf(lbl_rectbuf, "%d secs. lighten_value: %d", m_search_time, m_lighten_value);

    ofDrawBitmapStringHighlight(lbl_rectbuf, 380, 12);

    ofPushStyle();

    int n = m_rect_found.size();
    ofNoFill();
    ofSetLineWidth(2);
    ofSetColor(yellowPrint);
    for (int i = 0; i < n; i++) {
        Rect r = m_rect_found[i];
        // ofSetColor(ofColor::white);
        ofDrawRectangle(r.x, r.y, r.width, r.height);
        /*
                char lbl_rectbuf[128];
                sprintf(lbl_rectbuf, "%d %d  %d %d", r.x, r.y, r.width, r.height);
                ofDrawBitmapStringHighlight(lbl_rectbuf, r.x, r.y - 10);
        */
        //  std::this_thread::sleep_for(std::chrono::milliseconds(1150));

        //   string dimension = to_string(r.y);
        //   ofDrawBitmapStringHighlight(dimension, r.y, r.x);
    }

    ofPopStyle();

    if (m_ocr.getPixels().size() > 1) {
        m_ocr.draw(0, 610);
        m_font.drawString(m_plate_number, 400, 680);
    }
    //#endif
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
}

bool ofApp::compare_entry(const Rect& e1, const Rect& e2)
{
    // bool myfunction (my_data i, my_data j) { return ( i.data_one < j.data_one); }

    return e1.y > e2.y;

    //  return e1.width != e2.width ? e1.width < e2.width : e1.height < e2.height;
    //    if (e1.width != e2.width) return (e1.width < e2.width);
    //    return (e1.height < e2.height);
}
/*
void removeThread(std::thread::id id)
{
    std::lock_guard<std::mutex> lock(threadMutex);
    auto iter = std::find_if(threadList.begin(), threadList.end(), [=](std::thread &t) { return
(t.get_id() == id); }); if (iter != threadList.end())
    {
        iter->detach();
        threadList.erase(iter);
    }
}https://stackoverflow.com/questions/46187414/remove-thread-from-vector-upon-thread-completion
*/
void ofApp::wait_sensor()
{
    m_search_time = 0;
    m_plate_number = "Wait...";
}
void ofApp::remove_producer(std::thread::id id)
{
    std::lock_guard<std::mutex> lock(tmutex);
    auto iter = std::find_if(producers.begin(), producers.end(),
                             [=](std::thread& t) { return (t.get_id() == id); });
    if (iter != producers.end()) {
        iter->detach();
        producers.erase(iter);
    }
}

void ofApp::remove_consumer(std::thread::id id)
{
    std::lock_guard<std::mutex> lock(tmutex);
    auto iter = std::find_if(consumers.begin(), consumers.end(),
                             [=](std::thread& t) { return (t.get_id() == id); });
    if (iter != consumers.end()) {
        iter->detach();
        consumers.erase(iter);
    }
}

bool ofApp::is_duplicate(Rect rect)
{
    for (auto r : m_rect_duplicates) {
        if (r.y == rect.y && r.x == rect.x && r.width == rect.width && r.height == rect.height) {
            return true;
        }
    }

    m_rect_duplicates.push_back(rect);
    return false;
}

void ofApp::img_processor()
{
    if (!m_plate_number.empty()) {
        return;
    }

    auto n = m_rect_found.size();
    if (!n) return;

    for (int i = 0; i < n; i++) {
        Rect rect = m_rect_found[i];

        if (is_duplicate(rect)) {
            //  continue;
        }

        m_ocr.setFromPixels(m_grayImage.getPixels());
        m_ocr.crop(rect.x, rect.y, rect.width, rect.height);

        m_ocr.resize(m_ocr.getWidth() + 4, m_ocr.getHeight() + 4);
        m_ocr.update();

        string filename = "ocr_image_" + to_string(i) + "_" + ofGetTimestampString() + ".jpg";
        // m_ocr.save(filename);
        //  printf(" %d \n", i);

        // start ocr detection
        if (ofApp::process_tesseract()) break;

        // if (producers.size() < 1000)
        // producers.push_back(std::thread([&, thread_counter]() {
        // std::thread::id id = std::this_thread::get_id();
        // std::lock_guard<std::mutex> lock(tmutex);
        // ts_queue.push(thread_counter [>id<]);

        // m_ocrMap[thread_counter] = toCv(m_ocr).clone();
        /////

        ////        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        ////        string filename = "ocr_image_" + ofGetTimestampString() + ".jpg";
        ////      m_ocr.save(filename);
        //}));

        // thread_counter++;
    }

    // static bool consumerset = false;
    // if (!consumerset) {
    // consumerset = true;
    //// Create consumers.
    // consumers.push_back(std::thread([&, thread_counter]() {
    //// std::thread::id id;
    // int id;

    // while (true) {
    // ts_queue.pop(id);
    ////  if (ofApp::process_tesseract()) break;
    // ofApp::process_tesseractX(id);
    ////                    remove_producer(id);
    //}
    //}));
    //}
}

void ofApp::ocr_detection(Rect rect)
{
#ifdef AAAA
    if (!m_plate_number.empty()) {
        return;
    }

    // crop the image
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

    // resize
    m_ocr.resize(m_ocr.getWidth() + 8, m_ocr.getHeight() + 8);
    m_ocr.update();
    string filename = "ocr_image_" + ofGetTimestampString() + ".jpg";
    m_ocr.save(filename);

    /*
        uint64_t currentMillis = ofGetElapsedTimeMillis();
        if ((int)(currentMillis - previousMillis) >= 100) {
            previousMillis = currentMillis;
        }
        */

    ofApp::process_tesseract();
    return;

    // Create producers.
    if (producers.size() < 1000)
        producers.push_back(std::thread([&, thread_counter]() {
            std::thread::id id = std::this_thread::get_id();
            std::lock_guard<std::mutex> lock(tmutex);
            ts_queue.push(id);

            //        std::this_thread::sleep_for(std::chrono::milliseconds(100));
            //        string filename = "ocr_image_" + ofGetTimestampString() + ".jpg";
            //      m_ocr.save(filename);
        }));

    if (thread_counter < max_consumers)
        // Create consumers.
        consumers.push_back(std::thread([&, thread_counter]() {
            std::thread::id id;
            while (true) {
                ts_queue.pop(id);
                ofApp::process_tesseract();
                remove_producer(id);
            }
        }));

    thread_counter++;
#endif
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
    if (key == 'q') {
        terminate();
        return;
    }
    if (key == 'c') {
        m_plate_number = {};
        m_rect_duplicates.clear();
        m_ocrMap.clear();
        m_search_time = 0;
        return;
    }
    if (key == 's') {
        wait_sensor();
        return;
    }

    if (key == '+') {
        m_lighten_value++;
        return;
    }
    if (key == '-') {
        m_lighten_value--;
        return;
    }

    m_plate_number = {};
    if (key == '1') {
        m_search_time = 0;
        m_plate_number = "";
        m_viewMode = 1;
        return;
    }

    if (key == '2') {
        isSet = false;
        m_search_time = 0;
        m_plate_number = "";
        m_viewMode = 2;
        return;
    }

    if (key == '3') {
        m_search_time = 0;
        m_plate_number = "";
        m_viewMode = 3;
        return;
    }

    if (key == '4') {
        m_search_time = 0;
        m_plate_number = "";
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
