#include <iostream>
#include <dirent.h>
#include <sys/stat.h>
#include <fstream>
#include <math.h>
#include <assert.h>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/photo/photo.hpp"
#include "opencv2/imgcodecs.hpp"


#define DEBUG_FLAG  0   // Debug flag for image channels


/* Channel type */
enum class ChannelType : unsigned char {
    BLUE = 0,
    GREEN,
    RED
};

/* Hierarchy type */
enum class HierarchyType : unsigned char {
    INVALID_CNTR = 0,
    CHILD_CNTR,
    PARENT_CNTR
};

/* Enhance the image */
bool enhanceImage(cv::Mat src, ChannelType channel_type, 
                            cv::Mat *norm, cv::Mat *dst) {

    // Split the image
    std::vector<cv::Mat> channel(3);
    cv::split(src, channel);
    cv::Mat img = channel[0];

    // Normalize the image
    cv::Mat normalized;
    cv::normalize(img, normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    *norm = normalized;

    // Enhance the image using Gaussian blur and thresholding
    cv::Mat enhanced;
    switch(channel_type) {
        case ChannelType::BLUE: {
            // Enhance the blue channel

            // Create the mask
            cv::Mat src_gray;
            cv::threshold(normalized, src_gray, 10, 255, cv::THRESH_TOZERO);
            bitwise_not(src_gray, src_gray);
            cv::GaussianBlur(src_gray, enhanced, cv::Size(3,3), 0, 0);
            cv::threshold(enhanced, enhanced, 150, 255, cv::THRESH_BINARY);

            // Invert the mask
            bitwise_not(enhanced, enhanced);
        } break;

        case ChannelType::GREEN: {
            // Enhance the green channel

            // Create the mask
            cv::Mat src_gray;
            cv::threshold(normalized, src_gray, 10, 255, cv::THRESH_TOZERO);
            bitwise_not(src_gray, src_gray);
            cv::GaussianBlur(src_gray, enhanced, cv::Size(3,3), 0, 0);
            cv::threshold(enhanced, enhanced, 240, 255, cv::THRESH_BINARY);

            // Invert the mask
            bitwise_not(enhanced, enhanced);
        } break;

        case ChannelType::RED: {
            // Enhance the red channel

            // Create the mask
            cv::Mat src_gray;
            cv::threshold(normalized, src_gray, 5, 255, cv::THRESH_TOZERO);
            bitwise_not(src_gray, src_gray);
            cv::GaussianBlur(src_gray, enhanced, cv::Size(3,3), 0, 0);
            cv::threshold(enhanced, enhanced, 250, 255, cv::THRESH_BINARY);

            // Invert the mask
            bitwise_not(enhanced, enhanced);
        } break;

        default: {
            std::cerr << "Invalid channel type" << std::endl;
            return false;
        }
    }
    *dst = enhanced;
    return true;
}

/* Find the contours in the image */
void contourCalc(cv::Mat src, ChannelType channel_type, 
                    double min_area, cv::Mat *dst, 
                    std::vector<std::vector<cv::Point>> *contours, 
                    std::vector<cv::Vec4i> *hierarchy, 
                    std::vector<HierarchyType> *validity_mask, 
                    std::vector<double> *parent_area) {

    cv::Mat temp_src;
    src.copyTo(temp_src);
    switch(channel_type) {
        case ChannelType::BLUE :
        case ChannelType::GREEN : {
            findContours(temp_src, *contours, *hierarchy, cv::RETR_EXTERNAL, 
                                                        cv::CHAIN_APPROX_SIMPLE);
        } break;

        case ChannelType::RED : {
            findContours(temp_src, *contours, *hierarchy, cv::RETR_CCOMP, 
                                                        cv::CHAIN_APPROX_SIMPLE);
        } break;

        default: return;
    }

    *dst = cv::Mat::zeros(temp_src.size(), CV_8UC3);
    if (!contours->size()) return;
    validity_mask->assign(contours->size(), HierarchyType::INVALID_CNTR);
    parent_area->assign(contours->size(), 0.0);

    // Keep the contours whose size is >= than min_area
    cv::RNG rng(12345);
    for (int index = 0 ; index < (int)contours->size(); index++) {
        if ((*hierarchy)[index][3] > -1) continue; // ignore child
        auto cntr_external = (*contours)[index];
        double area_external = fabs(contourArea(cv::Mat(cntr_external)));
        if (area_external < min_area) continue;

        std::vector<int> cntr_list;
        cntr_list.push_back(index);

        int index_hole = (*hierarchy)[index][2];
        double area_hole = 0.0;
        while (index_hole > -1) {
            std::vector<cv::Point> cntr_hole = (*contours)[index_hole];
            double temp_area_hole = fabs(contourArea(cv::Mat(cntr_hole)));
            if (temp_area_hole) {
                cntr_list.push_back(index_hole);
                area_hole += temp_area_hole;
            }
            index_hole = (*hierarchy)[index_hole][0];
        }
        double area_contour = area_external - area_hole;
        if (area_contour >= min_area) {
            (*validity_mask)[cntr_list[0]] = HierarchyType::PARENT_CNTR;
            (*parent_area)[cntr_list[0]] = area_contour;
            for (unsigned int i = 1; i < cntr_list.size(); i++) {
                (*validity_mask)[cntr_list[i]] = HierarchyType::CHILD_CNTR;
            }
            cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0,255), 
                                            rng.uniform(0,255));
            drawContours(*dst, *contours, index, color, cv::FILLED, cv::LINE_8, *hierarchy);
        }
    }
}

/* Process the images inside each directory */
bool processImage(std::string path, std::string blue_image, std::string green_image, 
                                            std::string red_image, std::string metrics_file) {

    /* Create the data output file for images that were processed */
    std::ofstream data_stream;
    data_stream.open(metrics_file, std::ios::app);
    if (!data_stream.is_open()) {
        std::cerr << "Could not open the data output file." << std::endl;
        return false;
    }

    // Create the output directory
    std::string out_directory = path + "result/";
    struct stat st = {0};
    if (stat(out_directory.c_str(), &st) == -1) {
        mkdir(out_directory.c_str(), 0700);
    }

    // Extract the Blue stream for each input image
    std::string blue_path = path + "original/" + blue_image;
    cv::Mat blue = cv::imread(blue_path.c_str(), cv::IMREAD_COLOR | cv::IMREAD_ANYDEPTH);
    if (blue.empty()) {
        std::cerr << "Invalid blue input filename" << std::endl;
        return false;
    }

    // Extract the Green stream for each input image
    std::string green_path = path + "original/" + green_image;
    cv::Mat green = cv::imread(green_path.c_str(), cv::IMREAD_COLOR | cv::IMREAD_ANYDEPTH);
    if (green.empty()) {
        std::cerr << "Invalid green input filename" << std::endl;
        return false;
    }

    // Extract the Red stream for each input image
    std::string red_path = path + "original/" + red_image;
    cv::Mat red = cv::imread(red_path.c_str(), cv::IMREAD_COLOR | cv::IMREAD_ANYDEPTH);
    if (red.empty()) {
        std::cerr << "Invalid red input filename" << std::endl;
        return false;
    }

    /** Gather BGR channel information needed for feature extraction **/
    cv::Mat blue_normalized, blue_enhanced, green_normalized, green_enhanced, 
                                                    red_normalized, red_enhanced;
    if(!enhanceImage(blue, ChannelType::BLUE, &blue_normalized, &blue_enhanced)) {
        return false;
    }
    if(!enhanceImage(green, ChannelType::GREEN, &green_normalized, &green_enhanced)) {
        return false;
    }
    if(!enhanceImage(red, ChannelType::RED, &red_normalized, &red_enhanced)) {
        return false;
    }

    // Blue channel
    std::string out_blue = out_directory + blue_image;
    out_blue.insert(out_blue.find_last_of("."), "_blue_enhanced", 14);
    if (DEBUG_FLAG) cv::imwrite(out_blue.c_str(), blue_enhanced);

    cv::Mat blue_segmented;
    std::vector<std::vector<cv::Point>> contours_blue;
    std::vector<cv::Vec4i> hierarchy_blue;
    std::vector<HierarchyType> blue_contour_mask;
    std::vector<double> blue_contour_area;
    contourCalc(blue_enhanced, ChannelType::BLUE, 1.0, &blue_segmented, 
                &contours_blue, &hierarchy_blue, &blue_contour_mask, &blue_contour_area);
    out_blue.insert(out_blue.find_last_of("."), "_segmented", 10);
    if (DEBUG_FLAG) cv::imwrite(out_blue.c_str(), blue_segmented);

    // Green channel
    std::string out_green = out_directory + green_image;
    out_green.insert(out_green.find_last_of("."), "_green_enhanced", 15);
    if (DEBUG_FLAG) cv::imwrite(out_green.c_str(), green_enhanced);

    cv::Mat green_segmented;
    std::vector<std::vector<cv::Point>> contours_green;
    std::vector<cv::Vec4i> hierarchy_green;
    std::vector<HierarchyType> green_contour_mask;
    std::vector<double> green_contour_area;
    contourCalc(green_enhanced, ChannelType::GREEN, 1.0, &green_segmented, 
                &contours_green, &hierarchy_green, &green_contour_mask, &green_contour_area);
    out_green.insert(out_green.find_last_of("."), "_segmented", 10);
    if (DEBUG_FLAG) cv::imwrite(out_green.c_str(), green_segmented);

    // Red channel
    std::string out_red = out_directory + red_image;
    out_red.insert(out_red.find_last_of("."), "_red_enhanced", 13);
    if (DEBUG_FLAG) cv::imwrite(out_red.c_str(), red_enhanced);

    cv::Mat red_segmented;
    std::vector<std::vector<cv::Point>> contours_red;
    std::vector<cv::Vec4i> hierarchy_red;
    std::vector<HierarchyType> red_contour_mask;
    std::vector<double> red_contour_area;
    contourCalc(red_enhanced, ChannelType::RED, 1.0, &red_segmented, 
                &contours_red, &hierarchy_red, &red_contour_mask, &red_contour_area);
    out_red.insert(out_red.find_last_of("."), "_segmented", 10);
    if (DEBUG_FLAG) cv::imwrite(out_red.c_str(), red_segmented);

    /* Common image name */
    std::string common_image = blue_image;
    common_image[common_image.length()-5] = 'x';

    /* Normalized image */
    std::vector<cv::Mat> merge_normalized;
    merge_normalized.push_back(blue_normalized);
    merge_normalized.push_back(green_normalized);
    merge_normalized.push_back(red_normalized);
    cv::Mat color_normalized;
    cv::merge(merge_normalized, color_normalized);
    std::string out_normalized = out_directory + common_image;
    out_normalized.insert(out_normalized.find_last_of("."), "_a_normalized", 13);
    cv::imwrite(out_normalized.c_str(), color_normalized);

    /* Enhanced image */
    std::vector<cv::Mat> merge_enhanced;
    merge_enhanced.push_back(blue_enhanced);
    merge_enhanced.push_back(green_enhanced);
    merge_enhanced.push_back(red_enhanced);
    cv::Mat color_enhanced;
    cv::merge(merge_enhanced, color_enhanced);
    std::string out_enhanced = out_directory + common_image;
    out_enhanced.insert(out_enhanced.find_last_of("."), "_b_enhanced", 11);
    cv::imwrite(out_enhanced.c_str(), color_enhanced);

    data_stream << blue_image << "," << green_image << "," << red_image;

    data_stream << std::endl;
    data_stream.close();
    return true;
}

/* Main - create the threads and start the processing */
int main(int argc, char *argv[]) {

    /* Check for argument count */
    if (argc != 2) {
        std::cerr << "Invalid number of arguments." << std::endl;
        return -1;
    }

    /* Read the path to the data */
    std::string path(argv[1]);

    /* Read the list of directories to process */
    std::string image_list_filename = path + "image_list.dat";
    std::vector<std::string> input_images;
    FILE *file = fopen(image_list_filename.c_str(), "r");
    if (!file) {
        std::cerr << "Could not open 'image_list.dat' inside '" << path << "'." << std::endl;
        return -1;
    }
    char line[128];
    while (fgets(line, sizeof(line), file) != NULL) {
        line[strlen(line)-1] = 0;
        std::string temp_str(line);
        input_images.push_back(temp_str);
    }
    fclose(file);

    if (input_images.size()%3) {
        std::cerr << "Image count incorrect." << std::endl;
        return -1;
    }

    /* Create and prepare the file for metrics */
    std::string metrics_file = path + "computed_metrics.csv";
    std::ofstream data_stream;
    data_stream.open(metrics_file, std::ios::out);
    if (!data_stream.is_open()) {
        std::cerr << "Could not create the metrics file." << std::endl;
        return -1;
    }

    data_stream << "Blue image,Green image,Red image";

    data_stream << std::endl;
    data_stream.close();

    /* Process each image */
    for (unsigned int index = 0; index < input_images.size(); index += 3) {
        std::cout << "Processing " << input_images[index] 
                    << ", " << input_images[index+1] 
                    << " and " << input_images[index+2] << std::endl;
        if (!processImage(path, input_images[index], input_images[index+1], 
                                                input_images[index+2], metrics_file)) {
            std::cout << "ERROR !!!" << std::endl;
            return -1;
        }
    }

    return 0;
}

