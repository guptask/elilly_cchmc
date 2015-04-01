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


#define DEBUG_FLAG              0   // Debug flag for image channels
#define NUM_AREA_BINS           21  // Number of bins
#define BIN_AREA                25  // Bin area
#define ROI_FACTOR              3   // ROI of cell = ROI factor x mean diameter
#define MIN_CELL_ARC_LENGTH     250 // Cell arc length

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
            cv::threshold(enhanced, enhanced, 250, 255, cv::THRESH_BINARY);

            // Invert the mask
            bitwise_not(enhanced, enhanced);
        } break;

        case ChannelType::GREEN: {
            // Enhance the green channel
            cv::threshold(normalized, enhanced, 25, 255, cv::THRESH_BINARY);
        } break;

        case ChannelType::RED: {
            // Enhance the red channel
            cv::threshold(normalized, enhanced, 5, 255, cv::THRESH_BINARY);
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

/* Filter out ill-formed or small cells */
void filterCells(   std::vector<std::vector<cv::Point>> blue_contours,
                    std::vector<HierarchyType> blue_contour_mask,
                    std::vector<std::vector<cv::Point>> *filtered_contours ) {

    for (size_t i = 0; i < blue_contours.size(); i++) {
        if (blue_contour_mask[i] != HierarchyType::PARENT_CNTR) continue;

        // Eliminate small contours via contour arc calculation
        if ((arcLength(blue_contours[i], true) >= MIN_CELL_ARC_LENGTH) && 
                                            (blue_contours[i].size() >= 5)) {
            filtered_contours->push_back(blue_contours[i]);
        }
    }
}

/* Separation metrics */
void separationMetrics( std::vector<std::vector<cv::Point>> contours, 
                        float *mean_diameter,
                        float *stddev_diameter,
                        float *mean_aspect_ratio,
                        float *stddev_aspect_ratio,
                        float *mean_proximity_cnt,
                        float *stddev_proximity_cnt ) {

    // Compute the normal distribution parameters of cells
    std::vector<cv::Point2f> mc(contours.size());
    std::vector<float> dia(contours.size());
    std::vector<float> aspect_ratio(contours.size());

    for (size_t i = 0; i < contours.size(); i++) {
        cv::Moments mu = moments(contours[i], true);
        mc[i] = cv::Point2f(static_cast<float>(mu.m10/mu.m00), 
                                            static_cast<float>(mu.m01/mu.m00));
        cv::RotatedRect min_area_rect = minAreaRect(cv::Mat(contours[i]));
        dia[i] = (float) sqrt(pow(min_area_rect.size.width, 2) + 
                                                pow(min_area_rect.size.height, 2));
        aspect_ratio[i] = float(min_area_rect.size.width)/min_area_rect.size.height;
        if (aspect_ratio[i] > 1.0) {
            aspect_ratio[i] = 1.0/aspect_ratio[i];
        }
    }
    cv::Scalar mean_dia, stddev_dia;
    cv::meanStdDev(dia, mean_dia, stddev_dia);
    *mean_diameter = static_cast<float>(mean_dia.val[0]);
    *stddev_diameter = static_cast<float>(stddev_dia.val[0]);

    cv::Scalar mean_ratio, stddev_ratio;
    cv::meanStdDev(aspect_ratio, mean_ratio, stddev_ratio);
    *mean_aspect_ratio = static_cast<float>(mean_ratio.val[0]);
    *stddev_aspect_ratio = static_cast<float>(stddev_ratio.val[0]);

    float roi = (ROI_FACTOR * mean_dia.val[0])/2;
    std::vector<float> count(contours.size(), 0.0);
    for (size_t i = 0; i < contours.size(); i++) {
        for (size_t j = 0; j < contours.size(); j++) {
            if (i == j) continue;
            if (cv::norm(mc[i]-mc[j]) <= roi) {
                count[i]++;
            }
        }
    }
    cv::Scalar mean_count, stddev_count;
    cv::meanStdDev(count, mean_count, stddev_count);
    *mean_proximity_cnt = static_cast<float>(mean_count.val[0]);
    *stddev_proximity_cnt = static_cast<float>(stddev_count.val[0]);
}

/* Group contour areas into bins */
void binArea(std::vector<HierarchyType> contour_mask, 
                std::vector<double> contour_area, 
                std::string *contour_bins,
                unsigned int *contour_cnt) {

    std::vector<unsigned int> count(NUM_AREA_BINS, 0);
    *contour_cnt = 0;
    for (size_t i = 0; i < contour_mask.size(); i++) {
        if (contour_mask[i] != HierarchyType::PARENT_CNTR) continue;
        unsigned int area = static_cast<unsigned int>(round(contour_area[i]));
        unsigned int bin_index = 
            (area/BIN_AREA < NUM_AREA_BINS) ? area/BIN_AREA : NUM_AREA_BINS-1;
        count[bin_index]++;
    }

    for (size_t i = 0; i < count.size(); i++) {
        *contour_cnt += count[i];
        *contour_bins += std::to_string(count[i]) + ",";
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
    data_stream << common_image << ",";


    /** Extract multi-dimensional features for analysis **/

    // Blue-red channel intersection
    cv::Mat blue_red_intersection;
    bitwise_and(blue_enhanced, red_enhanced, blue_red_intersection);

    // Blue-green channel intersection
    cv::Mat blue_green_intersection;
    bitwise_and(blue_enhanced, green_enhanced, blue_green_intersection);

    // Green-red channel intersection
    cv::Mat green_red_intersection;
    bitwise_and(green_enhanced, red_enhanced, green_red_intersection);

    // Filter the blue contours
    std::vector<std::vector<cv::Point>> contours_blue_filtered;
    filterCells(contours_blue, blue_contour_mask, &contours_blue_filtered);
    data_stream << contours_blue_filtered.size() << ",";

    // Separation metrics
    float mean_dia = 0.0, stddev_dia = 0.0;
    float mean_aspect_ratio = 0.0, stddev_aspect_ratio = 0.0;
    float mean_proximity_cnt = 0.0, stddev_proximity_cnt = 0.0;
    separationMetrics(  contours_blue_filtered, 
                        &mean_dia, &stddev_dia, 
                        &mean_aspect_ratio, &stddev_aspect_ratio, 
                        &mean_proximity_cnt, &stddev_proximity_cnt  );
    data_stream << mean_dia << "," << stddev_dia << "," 
                << mean_aspect_ratio << "," << stddev_aspect_ratio << ","
                << mean_proximity_cnt << "," << stddev_proximity_cnt << ",";

    // Segment the green-red intersection
    cv::Mat green_red_segmented;
    std::vector<std::vector<cv::Point>> contours_green_red;
    std::vector<cv::Vec4i> hierarchy_green_red;
    std::vector<HierarchyType> green_red_contour_mask;
    std::vector<double> green_red_contour_area;
    contourCalc(green_red_intersection, ChannelType::RED, 1.0, &green_red_segmented, 
                        &contours_green_red, &hierarchy_green_red, &green_red_contour_mask, 
                        &green_red_contour_area);

    // Characterize the synapse
    std::string synapse_bins;
    unsigned int synapse_cnt;
    binArea(green_red_contour_mask, green_red_contour_area, 
                    &synapse_bins, &synapse_cnt);
    data_stream << synapse_cnt << "," << synapse_bins;


    data_stream << std::endl;
    data_stream.close();

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

    /* Analyzed image */
    cv::Mat drawing_blue  = blue_enhanced;
    cv::Mat drawing_green = cv::Mat::zeros(green_enhanced.size(), CV_8UC1);
    cv::Mat drawing_red   = cv::Mat::zeros(red_enhanced.size(), CV_8UC1);

    // Draw cell boundaries
    for (size_t i = 0; i < contours_blue_filtered.size(); i++) {
        cv::RotatedRect min_ellipse = fitEllipse(cv::Mat(contours_blue_filtered[i]));
        ellipse(drawing_blue, min_ellipse, 255, 4, 8);
        ellipse(drawing_green, min_ellipse, 255, 4, 8);
        ellipse(drawing_red, min_ellipse, 0, 4, 8);
    }

    // Merge the modified red, blue and green layers
    std::vector<cv::Mat> merge_analyzed;
    merge_analyzed.push_back(drawing_blue);
    merge_analyzed.push_back(drawing_green);
    merge_analyzed.push_back(drawing_red);
    cv::Mat color_analyzed;
    cv::merge(merge_analyzed, color_analyzed);
    std::string out_analyzed = out_directory + common_image;
    out_analyzed.insert(out_analyzed.find_last_of("."), "_c_analyzed", 11);
    cv::imwrite(out_analyzed.c_str(), color_analyzed);

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

    data_stream << "Image,Cell Count,Cell Diameter (mean),Cell Diameter (std. dev.),\
                    Cell Aspect Ratio (mean),Cell Aspect Ratio (std. dev.),\
                    ROI Cell Count (mean),ROI Cell Count (std. dev.),";

    data_stream << "Synapse Count,";
    for (unsigned int i = 0; i < NUM_AREA_BINS-1; i++) {
        data_stream << i*BIN_AREA 
                    << " <= Synapse area < " 
                    << (i+1)*BIN_AREA << ",";
    }
    data_stream << "Synapse area >= " 
                << (NUM_AREA_BINS-1)*BIN_AREA << ",";

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

