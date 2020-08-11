#pragma once

#include <ctype.h>
#include <iostream>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/videoio.hpp"

using namespace cv;
using namespace std;

class Camera
{
  public:
    Camera() {}

    // the camera will be deinitialized automatically
    // in VideoCapture destructor
    ~Camera() { cout << "Shutdown\n"; }

    bool connect(const string& rstp_stream);
    VideoCapture get_object() const;

  private:
    VideoCapture m_camera;
};
