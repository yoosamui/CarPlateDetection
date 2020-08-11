#include "camera.h"

bool Camera::connect(const string& rstp_stream)
{
    if (!m_camera.open(rstp_stream)) return false;
    if (!m_camera.isOpened()) {
        cout << "Could not initialize camera capturing..." << rstp_stream << "\n";
        return false;
    }

    return true;
}

VideoCapture Camera::get_object() const
{
    return m_camera;
}
