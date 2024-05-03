#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#include <chrono>
#include <omp.h>
#include <Magick++.h>

using namespace cv;
using namespace std;
using namespace Magick;

enum SeamDirection { VERTICAL, HORIZONTAL };

float energy_image_time = 0;
float cumulative_energy_map_time = 0;
float find_seam_time = 0;
float reduce_time = 0;

bool demo;
bool debug;
bool save_gif;

Mat createEnergyImage(Mat &image) {
    auto start = std::chrono::high_resolution_clock::now();
    Mat image_blur, image_gray;
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    Mat grad, energy_image;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;

    // apply a gaussian blur to reduce noise
    GaussianBlur(image, image_blur, Size(3,3), 0, 0, BORDER_DEFAULT);
    
    // convert to grayscale
    cvtColor(image_blur, image_gray, COLOR_RGB2GRAY);
    
    // use Sobel to calculate the gradient of the image in the x and y direction
    //Sobel(image_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
    //Sobel(image_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
    
    // use Scharr to calculate the gradient of the image in the x and y direction
    Scharr(image_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT);
    Scharr(image_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT);

    // convert gradients to abosulte versions of themselves
    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);
    
    // total gradient (approx)
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
    
    // convert the default values to double precision
    grad.convertTo(energy_image, CV_64F, 1.0/255.0);
    
    // create and show the newly created energy image
    if (demo) {
        namedWindow("Energy Image", WINDOW_AUTOSIZE); imshow("Energy Image", energy_image);
    }
    
    // calculate time taken
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    energy_image_time += elapsed.count();
    
    return energy_image;
}

Mat createCumulativeEnergyMap(Mat &energy_image, SeamDirection seam_direction) {
    auto start = std::chrono::high_resolution_clock::now();
    double a,b,c;
    
    // get the numbers of rows and columns in the image
    int rowsize = energy_image.rows;
    int colsize = energy_image.cols;
    
    // initialize the map with zeros
    Mat cumulative_energy_map = Mat(rowsize, colsize, CV_64F, double(0));
    
    // copy the first row
    if (seam_direction == VERTICAL) energy_image.row(0).copyTo(cumulative_energy_map.row(0));
    else if (seam_direction == HORIZONTAL) energy_image.col(0).copyTo(cumulative_energy_map.col(0));
    
    // take the minimum of the three neighbors and add to total, this creates a running sum which is used to determine the lowest energy path
    if (seam_direction == VERTICAL) {
        const int JUMP = 90; // choose something even
        const int BASE = (JUMP - 1) * 2;
        // for (int row = 1; row < rowsize; row++) {
        //     #pragma omp parallel for private(a, b, c)
        //     for (int col = 0; col < colsize; col++) {
        //         a = cumulative_energy_map.at<double>(row - 1, max(col - 1, 0));
        //         b = cumulative_energy_map.at<double>(row - 1, col);
        //         c = cumulative_energy_map.at<double>(row - 1, min(col + 1, colsize - 1));
                
        //         cumulative_energy_map.at<double>(row, col) = energy_image.at<double>(row, col) + std::min(a, min(b, c));
        //     }
        // }
        int row_start;
        for (row_start = 1; row_start + JUMP - 1 < rowsize; row_start += JUMP) {
            #pragma omp parallel for private(a, b, c)
            for (int top_triangle = 0; top_triangle < (colsize + BASE - 1) / BASE; top_triangle++) {
                for (int row = row_start; row < row_start + JUMP - 1; row++) {
                    int first = top_triangle * BASE + row - row_start;
                    int last = (top_triangle + 1) * BASE - row + row_start;
                    for (int col = first; col < min(last, colsize); col++) {
                        a = cumulative_energy_map.at<double>(row - 1, max(col - 1, 0));
                        b = cumulative_energy_map.at<double>(row - 1, col);
                        c = cumulative_energy_map.at<double>(row - 1, min(col + 1, colsize - 1));

                        cumulative_energy_map.at<double>(row, col) = energy_image.at<double>(row, col) + std::min(a, min(b, c));
                    }
                }
            }
            #pragma omp parallel for private(a, b, c)
            for (int bottom_triangle = 0; bottom_triangle < (colsize + BASE - 1 + JUMP - 1) / BASE; bottom_triangle++) {
                for (int row = row_start + 1; row < row_start + JUMP; row++) {
                    int first = bottom_triangle * BASE - (row - row_start);
                    int last = bottom_triangle * BASE + (row - row_start);
                    for (int col = max(0, first); col < min(last, colsize); col++) {
                        a = cumulative_energy_map.at<double>(row - 1, max(col - 1, 0));
                        b = cumulative_energy_map.at<double>(row - 1, col);
                        c = cumulative_energy_map.at<double>(row - 1, min(col + 1, colsize - 1));

                        cumulative_energy_map.at<double>(row, col) = energy_image.at<double>(row, col) + std::min(a, min(b, c));
                    }
                }
            }
        }
        for (int row = row_start; row < rowsize; row++) {
            #pragma omp parallel for private(a, b, c)
            for (int col = 0; col < colsize; col++) {
                a = cumulative_energy_map.at<double>(row - 1, max(col - 1, 0));
                b = cumulative_energy_map.at<double>(row - 1, col);
                c = cumulative_energy_map.at<double>(row - 1, min(col + 1, colsize - 1));

                cumulative_energy_map.at<double>(row, col) = energy_image.at<double>(row, col) + std::min(a, min(b, c));
            }
        }
    }
    else if (seam_direction == HORIZONTAL) {
        const int JUMP = 90; // choose something even
        const int BASE = (JUMP - 1) * 2;
        // for (int col = 1; col < colsize; col++) {
        //     #pragma omp parallel for private(a, b, c)
        //     for (int row = 0; row < rowsize; row++) {
        //         a = cumulative_energy_map.at<double>(max(row - 1, 0), col - 1);
        //         b = cumulative_energy_map.at<double>(row, col - 1);
        //         c = cumulative_energy_map.at<double>(min(row + 1, rowsize - 1), col - 1);
                
        //         cumulative_energy_map.at<double>(row, col) = energy_image.at<double>(row, col) + std::min(a, min(b, c));
        //     }
        // }
        int col_start;
        for (col_start = 1; col_start + JUMP - 1 < colsize; col_start += JUMP) {
            #pragma omp parallel for private(a, b, c)
            for (int top_triangle = 0; top_triangle < (colsize + BASE - 1) / BASE; top_triangle++) {
                for (int col = col_start; col < col_start + JUMP - 1; col++) {
                    int first = top_triangle * BASE + col - col_start;
                    int last = (top_triangle + 1) * BASE - col + col_start;
                    for (int row = first; row < min(last, colsize); row++) {
                        a = cumulative_energy_map.at<double>(row - 1, max(col - 1, 0));
                        b = cumulative_energy_map.at<double>(row - 1, col);
                        c = cumulative_energy_map.at<double>(row - 1, min(col + 1, colsize - 1));

                        cumulative_energy_map.at<double>(row, col) = energy_image.at<double>(row, col) + std::min(a, min(b, c));
                    }
                }
            }
            #pragma omp parallel for private(a, b, c)
            for (int bottom_triangle = 0; bottom_triangle < (colsize + BASE - 1 + JUMP - 1) / BASE; bottom_triangle++) {
                for (int col = col_start + 1; col < col_start + JUMP; col++) {
                    int first = bottom_triangle * BASE - (col - col_start);
                    int last = bottom_triangle * BASE + (col - col_start);
                    for (int row = max(0, first); row < min(last, colsize); row++) {
                        a = cumulative_energy_map.at<double>(row - 1, max(col - 1, 0));
                        b = cumulative_energy_map.at<double>(row - 1, col);
                        c = cumulative_energy_map.at<double>(row - 1, min(col + 1, colsize - 1));

                        cumulative_energy_map.at<double>(row, col) = energy_image.at<double>(row, col) + std::min(a, min(b, c));
                    }
                }
            }
        }
        for (int col = col_start; col < colsize; col++) {
            #pragma omp parallel for private(a, b, c)
            for (int row = 0; row < rowsize; row++) {
                a = cumulative_energy_map.at<double>(row - 1, max(col - 1, 0));
                b = cumulative_energy_map.at<double>(row - 1, col);
                c = cumulative_energy_map.at<double>(row - 1, min(col + 1, colsize - 1));

                cumulative_energy_map.at<double>(row, col) = energy_image.at<double>(row, col) + std::min(a, min(b, c));
            }
        }
    }
    
    // create and show the newly created cumulative energy map converting map into color (similar to MATLAB's imagesc())
    if (demo) {
        Mat color_cumulative_energy_map;
        double Cmin;
        double Cmax;
        cv::minMaxLoc(cumulative_energy_map, &Cmin, &Cmax);
        float scale = 255.0 / (Cmax - Cmin);
        cumulative_energy_map.convertTo(color_cumulative_energy_map, CV_8UC1, scale);
        applyColorMap(color_cumulative_energy_map, color_cumulative_energy_map, cv::COLORMAP_JET);
        
        namedWindow("Cumulative Energy Map", WINDOW_AUTOSIZE); imshow("Cumulative Energy Map", color_cumulative_energy_map);
    }
    
    // calculate time taken
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    cumulative_energy_map_time += elapsed.count();
    
    return cumulative_energy_map;
}

vector<int> findOptimalSeam(Mat &cumulative_energy_map, SeamDirection seam_direction) {
    auto start = std::chrono::high_resolution_clock::now();
    double a,b,c;
    int offset = 0;
    vector<int> path;
    double min_val, max_val;
    cv::Point min_pt, max_pt;
    
    // get the number of rows and columns in the cumulative energy map
    int rowsize = cumulative_energy_map.rows;
    int colsize = cumulative_energy_map.cols;
    
    if (seam_direction == VERTICAL) {
        // copy the data from the last row of the cumulative energy map
        Mat row = cumulative_energy_map.row(rowsize - 1);

        std::vector<double> vec;
        row.copyTo(vec);
        int firstMinIndex = 0;
        int secondMinIndex = 1;
        double firstMin = vec[0];
        double secondMin = vec[1];

        if (vec[1] < vec[0]) {
            firstMinIndex = 1;
            secondMinIndex = 0;
            firstMin = vec[1];
            secondMin = vec[0];
        }

        for (size_t i = 2; i < vec.size(); ++i) {
            if (vec[i] < firstMin) {
                secondMin = firstMin;
                secondMinIndex = firstMinIndex;
                firstMin = vec[i];
                firstMinIndex = i;
            } else if (vec[i] < secondMin) {
                secondMin = vec[i];
                secondMinIndex = i;
            }
        }
    
        // get min and max values and locations
        // minMaxLoc(row, &min_val, &max_val, &min_pt, &max_pt);
        
        // initialize the path vector
        path.resize(rowsize);
        int min_index = firstMinIndex; //min_pt.x;
        path[rowsize - 1] = min_index;
        
        // starting from the bottom, look at the three adjacent pixels above current pixel, choose the minimum of those and add to the path
        for (int i = rowsize - 2; i >= 0; i--) {
            a = cumulative_energy_map.at<double>(i, max(min_index - 1, 0));
            b = cumulative_energy_map.at<double>(i, min_index);
            c = cumulative_energy_map.at<double>(i, min(min_index + 1, colsize - 1));
            
            if (min(a,b) > c) {
                offset = 1;
            }
            else if (min(a,c) > b) {
                offset = 0;
            }
            else if (min(b, c) > a) {
                offset = -1;
            }
            
            min_index += offset;
            min_index = min(max(min_index, 0), colsize - 1); // take care of edge cases
            path[i] = min_index;
        }
    }
    
    else if (seam_direction == HORIZONTAL) {
        // copy the data from the last column of the cumulative energy map
        Mat col = cumulative_energy_map.col(colsize - 1);
        
        // get min and max values and locations
        minMaxLoc(col, &min_val, &max_val, &min_pt, &max_pt);
        
        // initialize the path vector
        path.resize(colsize);
        int min_index = min_pt.y;
        path[colsize - 1] = min_index;
        
        // starting from the right, look at the three adjacent pixels to the left of current pixel, choose the minimum of those and add to the path
        for (int i = colsize - 2; i >= 0; i--) {
            a = cumulative_energy_map.at<double>(max(min_index - 1, 0), i);
            b = cumulative_energy_map.at<double>(min_index, i);
            c = cumulative_energy_map.at<double>(min(min_index + 1, rowsize - 1), i);
            
            if (min(a,b) > c) {
                offset = 1;
            }
            else if (min(a,c) > b) {
                offset = 0;
            }
            else if (min(b, c) > a) {
                offset = -1;
            }
            
            min_index += offset;
            min_index = min(max(min_index, 0), rowsize - 1); // take care of edge cases
            path[i] = min_index;
        }
    }
    
    // calculate time taken
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    find_seam_time += elapsed.count();
    
    return path;
}

void reduce(Mat &image, vector<int> path, SeamDirection seam_direction) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // get the number of rows and columns in the image
    int rowsize = image.rows;
    int colsize = image.cols;
    
    // create a 1x1x3 dummy matrix to add onto the tail of a new row to maintain image dimensions and mark for deletion
    Mat dummy(1, 1, CV_8UC3, Vec3b(0, 0, 0));
    
    if (seam_direction == VERTICAL) { // reduce the width
        for (int i = 0; i < rowsize; i++) {
            // take all pixels to the left and right of marked pixel and store them in appropriate subrow variables
            Mat new_row;
            Mat lower = image.rowRange(i, i + 1).colRange(0, path[i]);
            Mat upper = image.rowRange(i, i + 1).colRange(path[i] + 1, colsize);
            
            // merge the two subrows and dummy matrix/pixel into a full row
            if (!lower.empty() && !upper.empty()) {
                hconcat(lower, upper, new_row);
                hconcat(new_row, dummy, new_row);
            }
            else {
                if (lower.empty()) {
                    hconcat(upper, dummy, new_row);
                }
                else if (upper.empty()) {
                    hconcat(lower, dummy, new_row);
                }
            }
            // take the newly formed row and place it into the original image
            new_row.copyTo(image.row(i));
        }
        // clip the right-most side of the image
        image = image.colRange(0, colsize - 1);
    }
    else if (seam_direction == HORIZONTAL) { // reduce the height
        for (int i = 0; i < colsize; i++) {
            // take all pixels to the top and bottom of marked pixel and store the in appropriate subcolumn variables
            Mat new_col;
            Mat lower = image.colRange(i, i + 1).rowRange(0, path[i]);
            Mat upper = image.colRange(i, i + 1).rowRange(path[i] + 1, rowsize);
            
            // merge the two subcolumns and dummy matrix/pixel into a full row
            if (!lower.empty() && !upper.empty()) {
                vconcat(lower, upper, new_col);
                vconcat(new_col, dummy, new_col);
            }
            else {
                if (lower.empty()) {
                    vconcat(upper, dummy, new_col);
                }
                else if (upper.empty()) {
                    vconcat(lower, dummy, new_col);
                }
            }
            // take the newly formed column and place it into the original image
            new_col.copyTo(image.col(i));
        }
        // clip the bottom-most side of the image
        image = image.rowRange(0, rowsize - 1);
    }
   
    if (demo) {
        namedWindow("Reduced Image", WINDOW_AUTOSIZE); imshow("Reduced Image", image);
    }
    
    // calculate time taken
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    reduce_time += elapsed.count();
}

void showPath(Mat &energy_image, vector<int> path, SeamDirection seam_direction) {
    // loop through the image and change all pixels in the path to white
    if (seam_direction == VERTICAL) {
        for (int i = 0; i < energy_image.rows; i++) {
            energy_image.at<double>(i,path[i]) = 1;
        }
    }
    else if (seam_direction == HORIZONTAL) {
        for (int i = 0; i < energy_image.cols; i++) {
            energy_image.at<double>(path[i],i) = 1;
        }
    }
    
    // display the seam on top of the energy image
    namedWindow("Seam on Energy Image", WINDOW_AUTOSIZE); imshow("Seam on Energy Image", energy_image);
}

void driver(Mat &image, SeamDirection seam_direction, int iterations) {
    auto start = std::chrono::high_resolution_clock::now();
    
    namedWindow("Original Image", WINDOW_AUTOSIZE); imshow("Original Image", image);
    
    // set up gif resources
    InitializeMagick("");
    vector<Image> frames;
    
    // perform the specified number of reductions
    for (int i = 0; i < iterations; i++) {
        Mat energy_image = createEnergyImage(image);
        Mat cumulative_energy_map = createCumulativeEnergyMap(energy_image, seam_direction);
        vector<int> path = findOptimalSeam(cumulative_energy_map, seam_direction);
        reduce(image, path, seam_direction);
        if (demo) {
            showPath(energy_image, path, seam_direction);
        }
        if (save_gif) {
            imwrite("result.jpg", image);
            Image tmp;
            tmp.read("result.jpg");
            tmp.animationDelay(5);
            frames.push_back(tmp);
        }
    }
    
    // calculate and output time taken
    if (debug) {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        float total_time = elapsed.count();
        cout << "Final image size: " << image.rows << "x" << image.cols << endl;
        cout << "energy image time taken: "; cout << fixed; cout << setprecision(7); cout << energy_image_time << endl;
        cout << "cumulative energy map time taken: "; cout << fixed; cout << setprecision(7); cout << cumulative_energy_map_time << endl;
        cout << "find seam time taken: "; cout << fixed; cout << setprecision(7); cout << find_seam_time << endl;
        cout << "reduce time taken: "; cout << fixed; cout << setprecision(7); cout << reduce_time << endl;
        cout << "total time taken: "; cout << fixed; cout << setprecision(7); cout << total_time << endl;
    }
    
    namedWindow("Reduced Image", WINDOW_AUTOSIZE); imshow("Reduced Image", image); waitKey(0);
    imwrite("result.jpg", image);

    if (save_gif) {
        writeImages(frames.begin(), frames.end(), "output.gif");
    }
}

int main() {
    string filename, reduce_direction, width_height, s_iterations, gif, threads;
    SeamDirection seam_direction;
    int iterations;
    int num_threads = 0;

    cout << "Please enter a filename: ";
    cin >> filename;
    
    Mat image = imread(filename);
    if (image.empty()) {
        cout << "Unable to load image, please try again." << endl;
        exit(EXIT_FAILURE);
    }
    
    cout << "Reduce width or reduce height? (0 to reduce width | 1 to reduce height): ";
    cin >> reduce_direction;
    
    if (reduce_direction == "0" || reduce_direction == "1") {
        if (reduce_direction == "0") {
            width_height = "width";
            seam_direction = VERTICAL;
        }
        else if (reduce_direction == "1") {
            width_height = "height";
            seam_direction = HORIZONTAL;
        }
    }
    else {
        cout << "Invalid choice, please re-run and try again" << endl;
        return 0;
    }
    
    cout << "Reduce " << width_height << " how many times? ";
    cin >> s_iterations;

    cout << "Would you like to save as gif? (1 for yes | 0 for no) ";
    cin >> gif;
    save_gif = (gif == "1");

    cout << "Enter the number of threads: ";
    cin >> threads;
    num_threads = stoi(threads);
    omp_set_num_threads(num_threads);
    
    iterations = stoi(s_iterations);
    int rowsize = image.rows;
    int colsize = image.cols;
    
    // check that inputted number of iterations doesn't exceed the image size
    if (seam_direction == VERTICAL) {
        if (iterations > colsize) {
            cout << "Input is greater than image's width, please try again." << endl;
            return 0;
        }
    }
    else if (seam_direction == HORIZONTAL) {
        if (iterations > rowsize) {
            cout << "Input is greater than image's height, please try again." << endl;
            return 0;
        }
    }

    demo = false;
    debug = true;
    
    if (demo) iterations = 1;
    
    driver(image, seam_direction, iterations);

    return 0;
}