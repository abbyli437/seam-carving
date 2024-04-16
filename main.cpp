// main.cpp
#include <opencv2/opencv.hpp>
#include "invert_colors.cuh"

int main() {
    cv::Mat h_img = cv::imread("path_to_image.jpg");
    if (h_img.empty()) {
        std::cerr << "Failed to load image." << std::endl;
        return -1;
    }

    cv::cuda::GpuMat d_img;
    d_img.upload(h_img);

    cudaInvertColors(d_img.ptr<unsigned char>(), d_img.cols, d_img.rows);

    cv::Mat processed_img;
    d_img.download(processed_img);

    cv::imshow("Original Image", h_img);
    cv::imshow("Processed Image", processed_img);
    cv::waitKey(0);

    return 0;
}
