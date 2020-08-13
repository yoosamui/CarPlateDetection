#include "ofApp.h"
#include <algorithm>
#include <condition_variable>
#include <iostream>
#include <locale>
//#include <mutex>
#include <opencv2/text/ocr.hpp>
#include <queue>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

static const int m_increment = 8;
static const int RESOLUTION_WIDTH = 1280;
static const int RESOLUTION_HEIGHT = 728;
static const int OCR_IMAGE_RESIZE = 16;
static const int CANNY_LOWTHRESHOLD = 100;
static const int CANNY_RATIO = 3;
static const int CANNY_KERNELSIZE = 3;
static const int SEARCH_TIMEOUT = 30;
static const string ONVIV_RTSP =
    "rtsp://admin:master!31416Pi@192.168.1.89:554/Streaming/channels/101";

bool ofApp::m_start_processing;
float framerateMult;
//--------------------------------------------------------------
void ofApp::setup()
{
    // initialize static members
    m_start_processing = false;
    m_found = false;

    // set of options
    ofSetVerticalSync(true);
    framerateMult = 1.0f;
    ofSetFrameRate(60);
    ofSetWindowTitle("Licence plate recognition v4.0");

    // needed for tesseract
    setlocale(LC_ALL, "C");
    setlocale(LC_CTYPE, "C");
    setlocale(LC_NUMERIC, "C");

    cout << "Start Cammera stream connection...\n";

    // connect to the cammera;
    if (!m_camera.connect(ONVIV_RTSP)) {
        terminate();
    }

    cout << "Cammera stream connected\n";

    // define default mask size
    int h = RESOLUTION_HEIGHT - 400;
    int w = RESOLUTION_WIDTH - 450;
    int centerX = (RESOLUTION_WIDTH / 2) - w / 2;
    int centerY = (RESOLUTION_HEIGHT / 2) - h / 2;

    m_mask_rect = Rect(centerX, centerY, w, h);

    h = 100;
    w = 120;
    centerX = (m_mask_rect.width / 2);
    centerY = ((m_mask_rect.height / 2) + h / 2);

    m_plate_size_max = Rect(centerX, centerY, w, h);
    m_plate_size_min = Rect(centerX, centerY, 10, 10);

    this->updateMask();

    m_font.load(OF_TTF_SANS, 16, true, true);

    // holds the plate number for show it in the UI.
    m_plate_number = "0";

    // TODO:  valid plate database
    m_platedb.push_back(470);   // sup moto
    m_platedb.push_back(7095);  // ford sup
    m_platedb.push_back(371);   // post man

    m_grayImage.allocate(RESOLUTION_WIDTH, RESOLUTION_HEIGHT);
}

bool ofApp::is_ocr_detection_found(const string& text)
{
    if (text.empty() || text.length() < 2) return false;

    try {
        long pnumber = std::stol(text);

        // checks if exitst in database
        vector<int>::iterator it = find(m_platedb.begin(), m_platedb.end(), pnumber);

        if (it != m_platedb.end()) {
            m_plate_number = to_string(pnumber);
            m_rect_duplicates.clear();
            m_start_processing = false;
            m_found = true;

            return true;
        }
    } catch (const std::invalid_argument& ia) {
        std::cerr << "Invalid argument: " << ia.what() << '\n';
    }

    return false;
}

bool ofApp::process_tesseract()
{
    if (m_ocr.getPixels().size() > 0 && m_plate_number.empty()) {
        Mat img;
        img = toCv(m_ocr);

        auto ocrp = cv::text::OCRTesseract::create(NULL, "eng", "0123456789", 3, 6);

        string text = ocrp->run(img, 10, cv::text::OCR_LEVEL_TEXTLINE);
        string pnumber = std::regex_replace(text, std::regex("([^0-9])"), "");
        // printf("[6]---------->%s %s\n", text.c_str(), pnumber.c_str());

        if (is_ocr_detection_found(pnumber)) return true;
    }

    return false;
}

//--------------------------------------------------------------
void ofApp::update()
{
    framerateMult =
        60.0f / (1.0f / ofGetLastFrameTime());  // changed as per Arturo correction..thanks

    if (ofGetElapsedTimef() < 10.0f)
        ofSetFrameRate(120);
    else if (ofGetElapsedTimef() < 20.0f)
        ofSetFrameRate(60);
    else if (ofGetElapsedTimef() < 30.0f)
        ofSetFrameRate(30);
    else if (ofGetElapsedTimef() < 40.0f)
        ofSetFrameRate(10);

    if (!m_camera.get_object().read(m_frame)) return;
    // check if we succeeded

    // m_camera.get_object() >> m_frame;
    if (m_frame.empty()) return;
    //
    m_frameNumber++;

    // resize the cammera frame
    Size size(RESOLUTION_WIDTH, RESOLUTION_HEIGHT);
    resize(m_frame, m_resized_image, size);

    m_lightenMat = m_resized_image + cv::Scalar(m_lighten_value, m_lighten_value, m_lighten_value);

    // convert to gray
    // cvtColor(m_resized_image, m_gray, COLOR_BGR2GRAY);
    cvtColor(m_lightenMat, m_gray, COLOR_BGR2GRAY);

    // create the mask
    m_gray.copyTo(m_mask_image, m_mask);

    // convert to ofImage for faster drawing
    ofImage gray;
    toOf(m_gray, gray);
    m_grayImage.setFromPixels(gray.getPixels());

    if (m_start_processing && !m_found) {
        if (m_search_time < 10) m_blur_value = 2;
        if (m_search_time >= 10) m_blur_value = 1;
        if (m_search_time >= 20) m_blur_value = 3;
    }

    // Noise Reduction Since edge detection is susceptible to noise
    // in the image, first step is to remove the noise with a Gaussian filter
    m_mask_image.copyTo(m_gray_masked);

    blur(m_gray_masked, m_gray_masked, Size(m_blur_value, m_blur_value));

    // Use the OpenCV function cv::Canny to implement the Canny Edge Detector.
    // If a pixel gradient is higher than the upper threshold, the pixel is accepted as an edge
    // If a pixel gradient value is below the lower threshold, then it is rejected.
    // If the pixel gradient is between the two thresholds, then it will be accepted only if it is
    // connected to a pixel that is above the upper threshold. Canny recommended a upper:lower ratio
    // between 2:1 and 3:1.
    Canny(m_gray_masked, m_canny_image, CANNY_LOWTHRESHOLD, CANNY_LOWTHRESHOLD * CANNY_RATIO,
          CANNY_KERNELSIZE);

    if (!m_start_processing) return;

    if (m_search_time >= SEARCH_TIMEOUT) {
        m_plate_number = {};
        m_start_processing = false;
        m_found = false;
        return;
    }

    uint64_t currentMillis = ofGetElapsedTimeMillis();
    if ((int)(currentMillis - previousMillis) >= 1000) {
        m_search_time++;
        previousMillis = currentMillis;
    }

    // Finds contours in the canny image.
    findContours(m_canny_image, m_contours, m_hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    m_size = m_contours.size();
    if (!m_size) {
        return;
    }

    cout << "Frame: " << to_string(m_frameNumber) << " contours : " << to_string(m_size) << "\n";

    m_rect_found.clear();
    int approx_size[2]{4, 8};
    vector<Point> approx;

    // debug
    //   printf("Contours size: %d\n", m_size);
    int counter = 0;

    // find rectangle or square
    for (int a = 0; a < 2; a++) {
        for (size_t i = 0; i < m_size; i++) {
            approxPolyDP(Mat(m_contours[i]), approx, arcLength(Mat(m_contours[i]), true) * 0.01,
                         true);
            if (approx.size() == approx_size[a]) {
                Rect r = boundingRect(m_contours[i]);
                if (is_duplicate(r)) {
                    continue;
                }

                if (r.width > m_plate_size_min.width && r.height > m_plate_size_min.height &&
                    r.height <= m_plate_size_max.height && r.width <= m_plate_size_max.width &&
                    r.height <= r.width && r.width >= r.height) {
                    m_rect_found.push_back(r);
                    //    printf("[%2d] %d %d %d %d\n", counter, r.y, r.x, r.width, r.height);
                    counter++;
                }
            }
        }
    }
    //  isSet = true;
    if (m_rect_found.size()) {
        std::sort(m_rect_found.begin(), m_rect_found.end(), ofApp::compare_entry);
        img_processor();
    }
}

//--------------------------------------------------------------
void ofApp::draw()
{
    ofBackground(ofColor::black);

    switch (m_view_mode) {
        case 1:
            m_grayImage.draw(0, 0);
            break;
        case 2:
            drawMat(m_gray_masked, 0, 0);
            break;
        case 3:
            drawMat(m_canny_image, 0, 0);
            break;
    }

    // show resolution and mask
    ofNoFill();
    ofSetLineWidth(1.5);
    ofSetColor(ofColor::white);
    ofDrawRectangle(0, 0, RESOLUTION_WIDTH, RESOLUTION_HEIGHT);
    ofDrawRectangle(m_mask_rect.x, m_mask_rect.y, m_mask_rect.width, m_mask_rect.height);

    // show mode
    char buffer[512];
    sprintf(buffer,
            "Time: %.2d:%.2d:%.2d | View: %d | Blur: %d | Brightness: %d | Elapsed: "
            "%2d | Dup: %d | FPS: %2.2f",
            ofGetHours(), ofGetMinutes(), ofGetSeconds(), m_view_mode, m_blur_value,
            m_lighten_value, m_search_time, (int)m_rect_duplicates.size(), ofGetFrameRate());

    ofDrawBitmapString(buffer, 300, RESOLUTION_HEIGHT + 22);

    // Draw scann rectangle
    m_size = m_rect_found.size();
    if (m_size) {
        ofPushStyle();
        ofSetLineWidth(1.5);
        ofSetColor(yellowPrint);
        for (size_t i = 0; i < m_size; i++) {
            Rect r = m_rect_found[i];
            ofDrawRectangle(r.x, r.y, r.width, r.height);
        }

        ofPopStyle();
    }

    // show the plate size
    ofDrawRectangle(m_plate_size_max.x, m_plate_size_max.y, m_plate_size_max.width,
                    m_plate_size_max.height);
    ofPushStyle();
    // Show the result if something found
    if (m_found && !m_plate_number.empty()) {
        //   m_ocr.draw(0, 610);

        string message("Licence detected");
        if (m_frameNumber % 4) message = {};

        ofSetColor(ofColor::white);
        m_font.drawString(m_plate_number, 2, RESOLUTION_HEIGHT + 24);

        ofSetColor(yellowPrint);
        m_font.drawString(message, 60, RESOLUTION_HEIGHT + 24);

    } else {
        if (m_start_processing) {
            string message("Processing: " + to_string(m_search_time) + " secs.");
            //            if (m_frameNumber % 6) message = {};
            m_font.drawString(message, 2, RESOLUTION_HEIGHT + 24);
        } else {
            ofSetColor(ofColor::red);
            m_font.drawString("License plate not found", 2, RESOLUTION_HEIGHT + 24);
        }
    }

    ofPopStyle();
}

void ofApp::updateMask()
{
    // clang-format off

    m_maskPoints.clear();
    m_mask_image.release();

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
        m_maskPoints.push_back(cv::Point(RESOLUTION_WIDTH - 2, 2));
        m_maskPoints.push_back(cv::Point(RESOLUTION_WIDTH - 2, RESOLUTION_HEIGHT - 2));
        m_maskPoints.push_back(cv::Point(2, RESOLUTION_HEIGHT - 2));
        m_maskPoints.push_back(cv::Point(2, 2));
    }

    CvMat* matrix = cvCreateMat(RESOLUTION_HEIGHT, RESOLUTION_WIDTH, CV_8UC1);
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
    return e1.y > e2.y;
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
    m_size = m_rect_found.size();
    if (!m_size) return;

    for (size_t i = 0; i < m_size; i++) {
        Rect rect = m_rect_found[i];

        m_ocr.setFromPixels(m_grayImage.getPixels());
        m_ocr.crop(rect.x, rect.y, rect.width, rect.height);

        m_ocr.resize(m_ocr.getWidth() + OCR_IMAGE_RESIZE, m_ocr.getHeight() + OCR_IMAGE_RESIZE);

        if (ofApp::process_tesseract()) {
            //        string filename = "ocr_" + to_string(i) + "_" + ofGetTimestampString() +
            //        ".jpg"; m_ocr.save(filename);
            break;
        }
    }
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

//--------------------------------------------------------------
void ofApp::keyPressed(int key)
{
    if (key == 's') {
        m_start_processing = true;
        m_plate_number = {};
        m_rect_duplicates.clear();
        m_search_time = 0;
        m_found = false;
        m_frameNumber = 0;
        return;
    }
    if (key == OF_KEY_F1) {
        m_blur_value = 1;
        return;
    }

    if (key == OF_KEY_F2) {
        m_blur_value = 2;
        this->updateMask();
        return;
    }

    if (key == OF_KEY_F3) {
        m_blur_value = 3;
        this->updateMask();
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

    if (key == '1') {
        m_view_mode = 1;
        return;
    }

    if (key == '2') {
        m_view_mode = 2;
        return;
    }

    if (key == '3') {
        m_view_mode = 3;
        return;
    }

    if (!m_start_processing) {
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
