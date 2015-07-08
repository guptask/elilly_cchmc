#include <iostream>
#include <dirent.h>
#include <sys/stat.h>
#include <fstream>
#include <cmath>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/photo/photo.hpp"
#include "opencv2/imgcodecs.hpp"


#define DEBUG_FLAG              0     // Debug flag for image channels
#define NUM_AREA_BINS           21    // Number of bins
#define BIN_AREA                25    // Bin area
#define ROI_FACTOR              3     // ROI of cell = ROI factor x mean diameter
#define MIN_CELL_ARC_LENGTH     20    // Cell arc length
#define SOMA_FACTOR             1.5   // Soma factor
#define COVERAGE_RATIO          0.4   // Coverage ratio lower threshold for neural soma
#define PI                      3.14  // Approximate value of pi

/* Channel type */
enum class ChannelType : unsigned char {
    BLUE = 0,
    GREEN,
    RED,
    RED_HIGH
};

/* Hierarchy type */
enum class HierarchyType : unsigned char {
    INVALID_CNTR = 0,
    CHILD_CNTR,
    PARENT_CNTR
};

/* Enhance the image */
bool enhanceImage(  cv::Mat src,
                    ChannelType channel_type,
                    cv::Mat *norm,
                    cv::Mat *dst    ) {

    // Split the image
    std::vector<cv::Mat> channel(3);
    cv::split(src, channel);
    cv::Mat img = channel[0];

    // Normalize the image
    cv::Mat normalized;
    cv::normalize(img, normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    // Enhance the image using Gaussian blur and thresholding
    cv::Mat enhanced;
    switch(channel_type) {
        case ChannelType::BLUE: {
            // Enhance the blue channel
            cv::threshold(normalized, enhanced, 5, 255, cv::THRESH_BINARY);
        } break;

        case ChannelType::GREEN: {
            // Enhance the green channel
            cv::threshold(normalized, enhanced, 20, 255, cv::THRESH_BINARY);
        } break;

        case ChannelType::RED: {
            // Enhance the red channel
            cv::threshold(normalized, enhanced, 5, 255, cv::THRESH_BINARY);
        } break;

        case ChannelType::RED_HIGH: {
            // Enhance the red high channel
            cv::threshold(normalized, enhanced, 90, 255, cv::THRESH_BINARY);
        } break;

        default: {
            std::cerr << "Invalid channel type" << std::endl;
            return false;
        }
    }
    *norm = normalized;
    *dst = enhanced;
    return true;
}

/* Find the contours in the image */
void contourCalc(   cv::Mat src, ChannelType channel_type, 
                    double min_area, cv::Mat *dst, 
                    std::vector<std::vector<cv::Point>> *contours, 
                    std::vector<cv::Vec4i> *hierarchy, 
                    std::vector<HierarchyType> *validity_mask, 
                    std::vector<double> *parent_area    ) {

    cv::Mat temp_src;
    src.copyTo(temp_src);
    switch(channel_type) {
        case ChannelType::BLUE :
        case ChannelType::GREEN : {
            findContours(temp_src, *contours, *hierarchy, cv::RETR_EXTERNAL, 
                                                        cv::CHAIN_APPROX_SIMPLE);
        } break;

        case ChannelType::RED : 
        case ChannelType::RED_HIGH : {
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
                    std::vector<std::vector<cv::Point>> *filtered_contours  ) {

    for (size_t i = 0; i < blue_contours.size(); i++) {
        if (blue_contour_mask[i] != HierarchyType::PARENT_CNTR) continue;

        // Eliminate small contours via contour arc calculation
        if ((arcLength(blue_contours[i], true) >= MIN_CELL_ARC_LENGTH) && 
                                            (blue_contours[i].size() >= 5)) {
            filtered_contours->push_back(blue_contours[i]);
        }
    }
}

/* Classify cells as neural cells or astrocytes */
void classifyCells( std::vector<std::vector<cv::Point>> filtered_blue_contours,
                    cv::Mat blue_green_intersection,
                    std::vector<std::vector<cv::Point>> *neural_contours,
                    std::vector<std::vector<cv::Point>> *astrocyte_contours ) {

    for (size_t i = 0; i < filtered_blue_contours.size(); i++) {

        // Determine whether neural cell by calculating blue-green coverage area
        cv::Mat drawing = cv::Mat::zeros(blue_green_intersection.size(), CV_8UC1);

        // Calculate radius and center of the nucleus
        cv::Moments mu = moments(filtered_blue_contours[i], true);
        cv::Point2f mc = cv::Point2f(   static_cast<float>(mu.m10/mu.m00), 
                                        static_cast<float>(mu.m01/mu.m00)   );

        float actual_area = contourArea(filtered_blue_contours[i]);
        float radius = sqrt(actual_area / PI);
        cv::circle(drawing, mc, SOMA_FACTOR*radius, 255, -1, 8);
        int initial_score = countNonZero(drawing);

        cv::Mat contour_intersection;
        bitwise_and(drawing, blue_green_intersection, contour_intersection);
        int final_score = countNonZero(contour_intersection);

        float coverage_ratio = ((float) final_score) / initial_score;
        if (coverage_ratio < COVERAGE_RATIO) {
            astrocyte_contours->push_back(filtered_blue_contours[i]);
        } else {
            neural_contours->push_back(filtered_blue_contours[i]);
        }
    }
}

/* Separation metrics */
void separationMetrics( std::vector<std::vector<cv::Point>> contours, 
                        float *aggregate_diameter,
                        float *aggregate_aspect_ratio   ) {

    *aggregate_diameter = 0;
    *aggregate_aspect_ratio = 0;
    for (size_t i = 0; i < contours.size(); i++) {
        auto min_area_rect = minAreaRect(cv::Mat(contours[i]));
        float aspect_ratio = float(min_area_rect.size.width)/min_area_rect.size.height;
        if (aspect_ratio > 1.0) aspect_ratio = 1.0/aspect_ratio;
        *aggregate_aspect_ratio += aspect_ratio;
        *aggregate_diameter += 2 * sqrt(contourArea(contours[i]) / PI);
    }
}

/* Group contour areas into bins */
void binArea(   std::vector<HierarchyType> contour_mask, 
                std::vector<double> contour_area, 
                std::string *contour_output ) {

    std::vector<unsigned int> count(NUM_AREA_BINS, 0);
    for (size_t i = 0; i < contour_mask.size(); i++) {
        if (contour_mask[i] != HierarchyType::PARENT_CNTR) continue;
        unsigned int area = static_cast<unsigned int>(round(contour_area[i]));
        unsigned int bin_index = 
            (area/BIN_AREA < NUM_AREA_BINS) ? area/BIN_AREA : NUM_AREA_BINS-1;
        count[bin_index]++;
    }

    unsigned int contour_cnt = 0;
    std::string area_binned;
    for (size_t i = 0; i < count.size(); i++) {
        area_binned += "," + std::to_string(count[i]);
        contour_cnt += count[i];
    }
    *contour_output = std::to_string(contour_cnt) + area_binned;
}

/* Process each image */
bool processImage(  std::string path,
                    std::string blue_image,
                    std::string green_image,
                    std::string red_image,
                    std::string *result     ) {

    // Create the output directory
    std::string out_directory = path + "result/";
    struct stat st = {0};
    if (stat(out_directory.c_str(), &st) == -1) {
        mkdir(out_directory.c_str(), 0700);
    }

    // Extract the Blue stream for each input image
    std::string blue_path = path + "original/" + blue_image;
    std::string cmd = "convert -quiet -quality 100 " + blue_path + " /tmp/img.jpg";
    system(cmd.c_str());
    cv::Mat blue = cv::imread("/tmp/img.jpg", cv::IMREAD_COLOR | cv::IMREAD_ANYDEPTH);
    if (blue.empty()) {
        std::cerr << "Invalid blue input filename" << std::endl;
        return false;
    }
    system("rm /tmp/img.jpg");

    // Extract the Green stream for each input image
    std::string green_path = path + "original/" + green_image;
    cmd = "convert -quiet -quality 100 " + green_path + " /tmp/img.jpg";
    system(cmd.c_str());
    cv::Mat green = cv::imread("/tmp/img.jpg", cv::IMREAD_COLOR | cv::IMREAD_ANYDEPTH);
    if (green.empty()) {
        std::cerr << "Invalid green input filename" << std::endl;
        return false;
    }
    system("rm /tmp/img.jpg");

    // Extract the Red stream for each input image
    std::string red_path = path + "original/" + red_image;
    cmd = "convert -quiet -quality 100 " + red_path + " /tmp/img.jpg";
    system(cmd.c_str());
    cv::Mat red = cv::imread("/tmp/img.jpg", cv::IMREAD_COLOR | cv::IMREAD_ANYDEPTH);
    if (red.empty()) {
        std::cerr << "Invalid red input filename" << std::endl;
        return false;
    }
    system("rm /tmp/img.jpg");

    /** Gather BGR channel information needed for feature extraction **/
    cv::Mat blue_normalized, blue_enhanced, green_normalized, green_enhanced, 
            red_normalized, red_enhanced, red_high_normalized, red_high_enhanced;
    if(!enhanceImage(blue, ChannelType::BLUE, &blue_normalized, &blue_enhanced)) {
        return false;
    }
    if(!enhanceImage(green, ChannelType::GREEN, &green_normalized, &green_enhanced)) {
        return false;
    }
    if(!enhanceImage(red, ChannelType::RED, &red_normalized, &red_enhanced)) {
        return false;
    }
    if(!enhanceImage(red, ChannelType::RED_HIGH, &red_high_normalized, &red_high_enhanced)) {
        return false;
    }

    // Blue channel
    cv::Mat blue_segmented;
    std::vector<std::vector<cv::Point>> contours_blue;
    std::vector<cv::Vec4i> hierarchy_blue;
    std::vector<HierarchyType> blue_contour_mask;
    std::vector<double> blue_contour_area;
    contourCalc(blue_enhanced, ChannelType::BLUE, 1.0, &blue_segmented, 
                &contours_blue, &hierarchy_blue, &blue_contour_mask, &blue_contour_area);

    // Green channel
    cv::Mat green_segmented;
    std::vector<std::vector<cv::Point>> contours_green;
    std::vector<cv::Vec4i> hierarchy_green;
    std::vector<HierarchyType> green_contour_mask;
    std::vector<double> green_contour_area;
    contourCalc(green_enhanced, ChannelType::GREEN, 1.0, &green_segmented, 
                &contours_green, &hierarchy_green, &green_contour_mask, &green_contour_area);

    // Red channel
    cv::Mat red_segmented;
    std::vector<std::vector<cv::Point>> contours_red;
    std::vector<cv::Vec4i> hierarchy_red;
    std::vector<HierarchyType> red_contour_mask;
    std::vector<double> red_contour_area;
    contourCalc(red_enhanced, ChannelType::RED, 1.0, &red_segmented, 
                &contours_red, &hierarchy_red, &red_contour_mask, &red_contour_area);

    // Red High channel
    cv::Mat red_high_segmented;
    std::vector<std::vector<cv::Point>> contours_red_high;
    std::vector<cv::Vec4i> hierarchy_red_high;
    std::vector<HierarchyType> red_high_contour_mask;
    std::vector<double> red_high_contour_area;
    contourCalc(red_high_enhanced, ChannelType::RED_HIGH, 1.0, 
                &red_high_segmented, &contours_red_high, 
                &hierarchy_red_high, &red_high_contour_mask, 
                &red_high_contour_area);


    /* Common image name */
    std::string common_image = blue_image;
    common_image[common_image.length()-5] = 'x';


    /** Extract multi-dimensional features for analysis **/

    // Blue-green channel intersection
    cv::Mat blue_green_intersection;
    bitwise_and(blue_enhanced, green_enhanced, blue_green_intersection);

    // Filter the blue contours
    std::vector<std::vector<cv::Point>> contours_blue_filtered;
    filterCells(contours_blue, blue_contour_mask, &contours_blue_filtered);
    *result += std::to_string(contours_blue_filtered.size()) + ",";

    // Classify the filtered cells as neural cells or astrocytes
    std::vector<std::vector<cv::Point>> neural_contours, astrocyte_contours;
    classifyCells(contours_blue_filtered, blue_green_intersection, 
                                        &neural_contours, &astrocyte_contours);

    // Separation metrics for neural cells
    float aggregate_dia = 0.0, aggregate_aspect_ratio = 0.0;
    separationMetrics(neural_contours, &aggregate_dia, &aggregate_aspect_ratio);
    *result +=  std::to_string(neural_contours.size())  + "," +
                std::to_string(aggregate_dia)           + "," +
                std::to_string(aggregate_aspect_ratio)  + ",";

    // Separation metrics for astrocytes
    aggregate_dia = aggregate_aspect_ratio = 0.0;
    separationMetrics(astrocyte_contours, &aggregate_dia, &aggregate_aspect_ratio);
    *result +=  std::to_string(astrocyte_contours.size())   + "," +
                std::to_string(aggregate_dia)               + "," +
                std::to_string(aggregate_aspect_ratio)      + ",";

    /* Green-red channel intersection */
    cv::Mat green_red_intersection;
    bitwise_and(green_enhanced, red_enhanced, green_red_intersection);

    // Segment the green-red intersection
    cv::Mat green_red_segmented;
    std::vector<std::vector<cv::Point>> contours_green_red;
    std::vector<cv::Vec4i> hierarchy_green_red;
    std::vector<HierarchyType> green_red_contour_mask;
    std::vector<double> green_red_contour_area;
    contourCalc(green_red_intersection, ChannelType::RED, 1.0, &green_red_segmented, 
                        &contours_green_red, &hierarchy_green_red, &green_red_contour_mask, 
                        &green_red_contour_area);

    // Characterize the green-red intersection
    std::string green_red_output;
    binArea(green_red_contour_mask, green_red_contour_area, &green_red_output);
    *result += green_red_output + ",";

    /* Green-red high channel intersection */
    cv::Mat green_red_high_intersection;
    bitwise_and(green_enhanced, red_high_enhanced, green_red_high_intersection);

    // Segment the green-red high intersection
    cv::Mat green_red_high_segmented;
    std::vector<std::vector<cv::Point>> contours_green_red_high;
    std::vector<cv::Vec4i> hierarchy_green_red_high;
    std::vector<HierarchyType> green_red_high_contour_mask;
    std::vector<double> green_red_high_contour_area;
    contourCalc(green_red_high_intersection, ChannelType::RED_HIGH, 1.0, 
                &green_red_high_segmented, &contours_green_red_high, 
                &hierarchy_green_red_high, &green_red_high_contour_mask, 
                &green_red_high_contour_area);

    // Characterize the green-red high intersection
    std::string green_red_high_output;
    binArea(green_red_high_contour_mask, green_red_high_contour_area, &green_red_high_output);
    *result += green_red_high_output;


    /* Normalized image */
    std::vector<cv::Mat> merge_normalized;
    merge_normalized.push_back(blue_normalized);
    merge_normalized.push_back(green_normalized);
    merge_normalized.push_back(red_normalized);
    cv::Mat color_normalized;
    cv::merge(merge_normalized, color_normalized);
    std::string out_normalized = out_directory + common_image;
    out_normalized.insert(out_normalized.find_last_of("."), "_a_normalized", 13);
    if (DEBUG_FLAG) {
        cv::imwrite("/tmp/img.jpg", color_normalized);
        cmd = "convert -quiet /tmp/img.jpg " + out_normalized;
        system(cmd.c_str());
        system("rm /tmp/img.jpg");
    }

    /* Enhanced image */
    std::vector<cv::Mat> merge_enhanced;
    merge_enhanced.push_back(blue_enhanced);
    merge_enhanced.push_back(green_enhanced);
    merge_enhanced.push_back(red_enhanced);
    cv::Mat color_enhanced;
    cv::merge(merge_enhanced, color_enhanced);
    std::string out_enhanced = out_directory + common_image;
    out_enhanced.insert(out_enhanced.find_last_of("."), "_b_enhanced", 11);
    if (DEBUG_FLAG) {
        cv::imwrite("/tmp/img.jpg", color_enhanced);
        cmd = "convert -quiet /tmp/img.jpg " + out_enhanced;
        system(cmd.c_str());
        system("rm /tmp/img.jpg");
    }

    /* Analyzed image */
    cv::Mat drawing_blue  = 2*blue_normalized;
    cv::Mat drawing_green = green_normalized;
    cv::Mat drawing_red   = red_normalized;

    // Draw neural cell boundaries
    for (size_t i = 0; i < neural_contours.size(); i++) {
        cv::RotatedRect min_ellipse = fitEllipse(cv::Mat(neural_contours[i]));
        ellipse(drawing_blue, min_ellipse, 255, 2, 8);
        ellipse(drawing_green, min_ellipse, 255, 2, 8);
        ellipse(drawing_red, min_ellipse, 255, 2, 8);
    }

    // Draw astrocyte boundaries
    for (size_t i = 0; i < astrocyte_contours.size(); i++) {
        cv::RotatedRect min_ellipse = fitEllipse(cv::Mat(astrocyte_contours[i]));
        ellipse(drawing_blue, min_ellipse, 255, 2, 8);
        ellipse(drawing_green, min_ellipse, 255, 2, 8);
        ellipse(drawing_red, min_ellipse, 0, 2, 8);
    }

    // Merge the modified red, blue and green layers
    std::vector<cv::Mat> merge_analyzed;
    merge_analyzed.push_back(drawing_blue);
    merge_analyzed.push_back(drawing_green);
    merge_analyzed.push_back(drawing_red);
    cv::Mat color_analyzed;
    cv::merge(merge_analyzed, color_analyzed);
    std::string out_analyzed = out_directory + common_image;
    if (DEBUG_FLAG) out_analyzed.insert(out_analyzed.find_last_of("."), "_c_analyzed", 11);
    cv::imwrite("/tmp/img.jpg", color_analyzed);
    cmd = "convert -quiet /tmp/img.jpg " + out_analyzed;
    system(cmd.c_str());
    system("rm /tmp/img.jpg");

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

    /* Read the image index */
    std::string img_index_filename = path + "ImageIndex.ColumbusIDX.csv";
    std::ifstream img_index_stream;
    img_index_stream.open(img_index_filename);
    if (!img_index_stream.is_open()) {
        std::cerr << "Invalid file " << img_index_filename << std::endl;
    }

    std::vector<std::string> well_name;
    std::string buffer;
    getline(img_index_stream, buffer); // ignore the header
    while (getline(img_index_stream, buffer)) {
        std::vector<std::string> tokens;
        std::istringstream iss(buffer);
        std::string token;
        while (getline(iss, token, '\t')) {
            tokens.push_back(token);
        }
        well_name.push_back(tokens[tokens.size()-5]);
    }

    /* Create and prepare the file for metrics */
    std::string metrics_file = path + "computed_metrics.csv";
    std::ofstream data_stream;
    data_stream.open(metrics_file, std::ios::out);
    if (!data_stream.is_open()) {
        std::cerr << "Could not create the metrics file." << std::endl;
        return -1;
    }

    data_stream << "Well Name,";
    data_stream << "Cell Count,";
    data_stream << "Neural Cell Count,";
    data_stream << "Neural Cell Diameter (mean),";
    data_stream << "Neural Cell Aspect Ratio (mean),";
    data_stream << "Astrocyte Count,";
    data_stream << "Astrocyte Diameter (mean),";
    data_stream << "Astrocyte Aspect Ratio (mean),";

    data_stream << "Green-Red Contour Count,";
    for (unsigned int i = 0; i < NUM_AREA_BINS-1; i++) {
        data_stream << i*BIN_AREA << " <= Green-Red Contour Area < " << (i+1)*BIN_AREA << ",";
    }
    data_stream << "Green-Red Contour Area >= " << (NUM_AREA_BINS-1)*BIN_AREA << ",";

    data_stream << "Green-Red High Contour Count,";
    for (unsigned int i = 0; i < NUM_AREA_BINS-1; i++) {
        data_stream << i*BIN_AREA << " <= Green-Red High Contour Area < " << (i+1)*BIN_AREA << ",";
    }
    data_stream << "Green-Red High Contour Area >= " << (NUM_AREA_BINS-1)*BIN_AREA << ",";

    data_stream << std::endl;

    /* Process each image */
    for (unsigned int index = 0; index < input_images.size(); index += 3) {
        std::cout   << "Processing "
                    << input_images[index]      << ", "
                    << input_images[index+1]    << ", "
                    << input_images[index+2]    << std::endl;

        std::string result = well_name[index] + ",";
        if (!processImage(  path,
                            input_images[index],
                            input_images[index+1],
                            input_images[index+2],
                            &result )) {
            std::cout << "ERROR !!!" << std::endl;
            return -1;
        }
        data_stream << result << std::endl;
    }
    data_stream.close();

    return 0;
}

