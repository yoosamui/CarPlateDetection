#include "threadedocr.h"
#include <algorithm>
#include <locale>
#include <opencv2/text/ocr.hpp>

ThreadedOcr::ThreadedOcr()
{
    m_platedb.push_back(1402);
    m_platedb.push_back(396);
    m_platedb.push_back(96);
    m_platedb.push_back(356);
    m_platedb.push_back(149);
    m_platedb.push_back(357);
    m_platedb.push_back(146);
    m_platedb.push_back(470);
    m_platedb.push_back(7095);
    m_platedb.push_back(4349);
}

void ThreadedOcr::update(const cv::Mat ocrimage)
{
    m_ocr = ocrimage.clone();
}

void ThreadedOcr::threadedFunction()
{
    // string filename = "ocr_image_" + ofGetTimestampString() + ".jpg";
    // auto ocrp = cv::text::OCRTesseract::create(NULL, "eng", "0123456789", 1, 6);
    // auto text = ocrp->run(m_ocr, 40, cv::text::OCR_LEVEL_TEXTLINE);

    //    displayInfo("Start Running IN Thread!");
    while (isThreadRunning()) {
        //        printf("Thread\n");
        //    ocrp->SetImage(m_ocr);
        //   auto ocrp = cv::text::OCRTesseract::create(NULL, "eng", "0123456789", 1, 6);
        //   string text = ocrp->run(m_ocr, 40, cv::text::OCR_LEVEL_TEXTLINE);
        //        string pnumber = std::regex_replace(text, std::regex("([^0-9])"), "");
        /*
                        if (pnumber.empty() == false) {
                            printf(pnumber.c_str());
                            printf("\n");
                        }

                        if (pnumber.empty()) {
                            auto ocrp = cv::text::OCRTesseract::create(NULL, "eng", "0123456789", 3,
           9); text = ocrp->run(m_ocr, 10, cv::text::OCR_LEVEL_TEXTLINE); pnumber =
           std::regex_replace(text, std::regex("([^0-9])"), "");
                        }

                        if (!pnumber.empty()) {
                            printf(pnumber.c_str());
                            printf("\n");

                            string filename = "ocr_image_" + ofGetTimestampString() + ".jpg";
                            //      m_ocr.save(filename);

                            int number = stoi(pnumber);
                            vector<int>::iterator it = find(m_platedb.begin(), m_platedb.end(),
           number);

                            if (it != m_platedb.end()) {
                                m_plate_number = pnumber;
                                printf("Thread---------------------------------->\n");
                            }
                        }
                        */
        ofSleepMillis(100);
    }
}

