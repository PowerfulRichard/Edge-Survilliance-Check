#include <opencv2/opencv.hpp>
#include <iostream>
int main() {
    cv::Mat img = cv::imread("/root/face_rpi_py/snapshot.jpg", 1);
    std::cout << "图片是否为空: " << img.empty() << std::endl;
    std::cout << "图片尺寸: " << img.cols << "x" << img.rows << std::endl;
    return 0;
}