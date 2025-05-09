#ifndef TRAVEL_UTILS_H
#define TRAVEL_UTILS_H

#include <vector>
#include <map>
#include <iostream>
#include <sstream>
#include <signal.h>
#include <ros/ros.h>
#include <geometry_msgs/Pose.h>
#include <pcl/common/common.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/centroid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <boost/format.hpp>
#include "utils/nanoflann.hpp"
#include "utils/nanoflann_utils.hpp"


#define SENSOR_HEIGHT 1.73

#define UNLABELED 0
#define OUTLIER 1
#define NUM_ALL_CLASSES 34
#define ROAD 40
#define PARKING 44
#define SIDEWALKR 48
#define OTHER_GROUND 49
#define BUILDING 50
#define FENSE 51
#define LANE_MARKING 60
#define VEGETATION 70
#define TERRAIN 72

#define TRUEPOSITIVE 3
#define TRUENEGATIVE 2
#define FALSEPOSITIVE 1
#define FALSENEGATIVE 0
using namespace std;
int NUM_ZEROS = 5;
double VEGETATION_THR = - SENSOR_HEIGHT * 3 / 4;

#define INVALID_IDX -1
struct PointXYZILID
{
  PCL_ADD_POINT4D;                    // quad-word XYZ
  float    intensity;                 ///< laser intensity reading
  uint16_t label;                     ///< point label
  uint16_t id;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW     // ensure proper alignment
} EIGEN_ALIGN16;

// Register custom point struct according to PCL
POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZILID,
                                  (float, x, x)
                                  (float, y, y)
                                  (float, z, z)
                                  (float, intensity, intensity)
                                  (uint16_t, label, label)
                                  (uint16_t, id, id))

using PointT = PointXYZILID;
using num_t = float;

void PointXYZILID2XYZI(pcl::PointCloud<PointXYZILID>& src,
                       pcl::PointCloud<pcl::PointXYZI>::Ptr dst){
  dst->points.clear();
  for (const auto &pt: src.points){
    pcl::PointXYZI pt_xyzi;
    pt_xyzi.x = pt.x;
    pt_xyzi.y = pt.y;
    pt_xyzi.z = pt.z;
    pt_xyzi.intensity = pt.intensity;
    dst->points.push_back(pt_xyzi);
  }
}

template <typename PointType>
void saveLabels(const std::string abs_dir, const int frame_num, const pcl::PointCloud<PointType> &cloud_in,
                                             const pcl::PointCloud<PointType> &labeled_pc) {
    // Save labels as a .label file
    // It is relevant to 3DUIS benchmark
    // https://codalab.lisn.upsaclay.fr/competitions/2183?secret_key=4763e3d2-1f22-45e6-803a-a862528426d2
    // Labels are set in the intensity of `labeled_pc`, which is larger than 0. i.e. > 0
    const float SQR_EPSILON = 0.00001;

    int num_cloud_in = cloud_in.points.size();
    std::vector<uint32_t> labels(num_cloud_in, 0); // 0: not interested

    int N = labeled_pc.points.size();
    PointCloud<num_t> cloud;
    cloud.pts.resize(N);
    for (size_t i = 0; i < N; i++)
    {
        cloud.pts[i].x = labeled_pc.points[i].x;
        cloud.pts[i].y = labeled_pc.points[i].y;
        cloud.pts[i].z = labeled_pc.points[i].z;
    }

    // construct a kd-tree index:
    using my_kd_tree_t = nanoflann::KDTreeSingleIndexAdaptor<
            nanoflann::L2_Simple_Adaptor<num_t, PointCloud<num_t>>,
    PointCloud<num_t>, 3 /* dim */
                       >;

    my_kd_tree_t index(3 /*dim*/, cloud, {10 /* max leaf */});

    int num_valid = 0;
    for (int j = 0; j < cloud_in.points.size(); ++j) {
        const auto query_pcl = cloud_in.points[j];
        const num_t query_pt[3] = {query_pcl.x, query_pcl.y, query_pcl.z};
        {
            size_t num_results = 1;
            std::vector<uint32_t> ret_index(num_results);
            std::vector<num_t> out_dist_sqr(num_results);

            num_results = index.knnSearch(
                    &query_pt[0], num_results, &ret_index[0], &out_dist_sqr[0]);

            // In case of less points in the tree than requested:
            ret_index.resize(num_results);
            out_dist_sqr.resize(num_results);
            if(out_dist_sqr[0] < SQR_EPSILON) { // it is the same point!
                labels[j] = labeled_pc.points[ret_index[0]].intensity;
                ++num_valid;
            }
        }
    }
    // Must be equal to the # of above-ground points
    std::cout << "# of valid points: " << num_valid << std::endl;

     //  To follow the KITTI format, # of zeros are set to 6
    const int NUM_ZEROS = 6;

    std::string count_str = std::to_string(frame_num);
    std::string count_str_padded = std::string(NUM_ZEROS - count_str.length(), '0') + count_str;
    std::string abs_label_path = abs_dir + "/" + count_str_padded + ".label";

    // Semantics are just set to be zero
    for (int i = 0; i < labels.size(); ++i) {
        uint32_t shifted = labels[i] << 16;
        labels[i] = shifted;
    }

    std::cout << "\033[1;32m" << abs_label_path << "\033[0m" << std::endl;
    std::ofstream output_file(abs_label_path, std::ios::out | std::ios::binary);
    output_file.write(reinterpret_cast<char*>(&labels[0]), num_cloud_in * sizeof(uint32_t));
}

std::vector<int> outlier_classes = {UNLABELED, OUTLIER};
std::vector<int> ground_classes = {ROAD, PARKING, SIDEWALKR, OTHER_GROUND, LANE_MARKING, VEGETATION, TERRAIN};
std::vector<int> ground_classes_except_terrain = {ROAD, PARKING, SIDEWALKR, OTHER_GROUND, LANE_MARKING};
std::vector<int> traversable_ground_classes = {ROAD, PARKING, LANE_MARKING, OTHER_GROUND};


template <typename PointType>
int count_num_ground(const pcl::PointCloud<PointType>& pc){
  int num_ground = 0;

  std::vector<int>::iterator iter;

  for (auto const& pt: pc.points){
    iter = std::find(ground_classes.begin(), ground_classes.end(), pt.label);
    if (iter != ground_classes.end()){ // corresponding class is in ground classes
      if (pt.label == VEGETATION){
        if (pt.z < VEGETATION_THR){
           num_ground ++;
        }
      }else num_ground ++;
    }
  }
  return num_ground;
}

template <typename PointType>
int count_num_ground_without_vegetation(const pcl::PointCloud<PointType>& pc){
  int num_ground = 0;

  std::vector<int>::iterator iter;

  std::vector<int> classes = {ROAD, PARKING, SIDEWALKR, OTHER_GROUND, LANE_MARKING, TERRAIN};

  for (auto const& pt: pc.points){
    iter = std::find(classes.begin(), classes.end(), pt.label);
    if (iter != classes.end()){ // corresponding class is in ground classes
      num_ground ++;
    }
  }
  return num_ground;
}

std::map<int, int> set_initial_gt_counts(std::vector<int>& gt_classes){
  map<int, int> gt_counts;
  for (int i = 0; i< gt_classes.size(); ++i){
    gt_counts.insert(pair<int,int>(gt_classes.at(i), 0));
  }
  return gt_counts;
}

template <typename PointType>
std::map<int, int> count_num_each_class(const pcl::PointCloud<PointType>& pc){
  int num_ground = 0;
  auto gt_counts = set_initial_gt_counts(ground_classes);
  std::vector<int>::iterator iter;

  for (auto const& pt: pc.points){
    iter = std::find(ground_classes.begin(), ground_classes.end(), pt.label);
    if (iter != ground_classes.end()){ // corresponding class is in ground classes
      if (pt.label == VEGETATION){
        if (pt.z < VEGETATION_THR){
           gt_counts.find(pt.label)->second++;
        }
      }else gt_counts.find(pt.label)->second++;
    }
  }
  return gt_counts;
}

template <typename PointType>
int count_num_outliers(const pcl::PointCloud<PointType>& pc){
  int num_outliers = 0;

  std::vector<int>::iterator iter;

  for (auto const& pt: pc.points){
    iter = std::find(outlier_classes.begin(), outlier_classes.end(), pt.label);
    if (iter != outlier_classes.end()){ // corresponding class is in ground classes
      num_outliers ++;
    }
  }
  return num_outliers;
}

template <typename PointType>
void discern_ground(const pcl::PointCloud<PointType>& src, pcl::PointCloud<PointType>& ground, pcl::PointCloud<PointType>& non_ground){
  ground.clear();
  non_ground.clear();
  std::vector<int>::iterator iter;
  for (auto const& pt: src.points){
    if (pt.label == UNLABELED || pt.label == OUTLIER) continue;
    iter = std::find(ground_classes.begin(), ground_classes.end(), pt.label);
    if (iter != ground_classes.end()){ // corresponding class is in ground classes
      if (pt.label == VEGETATION){
        if (pt.z < VEGETATION_THR){
          ground.push_back(pt);
        }else non_ground.push_back(pt);
      }else  ground.push_back(pt);
    }else{
      non_ground.push_back(pt);
    }
  }
}

template <typename PointType>
void discern_ground_without_vegetation(const pcl::PointCloud<PointType>& src, pcl::PointCloud<PointType>& ground, pcl::PointCloud<PointType>& non_ground){
  ground.clear();
  non_ground.clear();
  std::vector<int>::iterator iter;
  for (auto const& pt: src.points){
    if (pt.label == UNLABELED || pt.label == OUTLIER) continue;
    iter = std::find(ground_classes.begin(), ground_classes.end(), pt.label);
    if (iter != ground_classes.end()){ // corresponding class is in ground classes
      if (pt.label != VEGETATION) ground.push_back(pt);
    }else{
      non_ground.push_back(pt);
    }
  }
}

template <typename PointType>
void calculate_precision_recall(const pcl::PointCloud<PointType>& pc_curr,
                                pcl::PointCloud<PointType>& ground_estimated,
                                double & precision,
                                double& recall,
                                double& accuracy,
                                bool consider_outliers=true){

  int num_ground_est = ground_estimated.points.size();
  int num_ground_gt = count_num_ground(pc_curr);
  // int num_TP = count_num_ground(ground_estimated);
  int num_inliers_ground = count_num_ground(ground_estimated);
  int num_TP, num_FP, num_FN, num_TN;
  if (consider_outliers){
    int num_outliers_est = count_num_outliers(ground_estimated);
    num_TP = num_inliers_ground + num_outliers_est;
    num_FP = num_ground_est - num_TP;
    num_FN = num_ground_gt - num_inliers_ground;
    num_TN = pc_curr.points.size() - (num_TP + num_FP + num_FN);
    precision = (double)(num_TP)/(num_TP + num_FP) * 100;
    recall = (double)(num_TP)/(num_TP + num_FN) * 100;
    accuracy = (double)(num_TP + num_TN)/(num_TP + num_TN + num_FP + num_FN) * 100;
  }else{
    precision = (double)(num_TP)/num_ground_est * 100;
    recall = (double)(num_TP)/num_ground_gt * 100;
  }
}

template <typename PointType>
int count_num_outliers_veg(const pcl::PointCloud<PointType>& pc){
    int num_outliers = 0;
    std::vector<int> outlier_classes_veg = {UNLABELED, OUTLIER, VEGETATION};

    std::vector<int>::iterator iter;
    for (auto const& pt: pc.points){
        iter = std::find(outlier_classes_veg.begin(), outlier_classes_veg.end(), pt.label);
        if (iter != outlier_classes_veg.end()){ // corresponding class is in ground classes
            num_outliers ++;
        }
    }
    return num_outliers;
}

template <typename PointType>
void calculate_precision_recall_without_vegetation(const pcl::PointCloud<PointType>& pc_curr,
                                                   pcl::PointCloud<PointType>& ground_estimated,
                                                   double & precision,
                                                   double& recall,
                                                   double& accuracy,
                                                   bool reject_num_of_outliers=true){
  int num_ground_est = ground_estimated.size();// - num_veg_est; // num. positives
  int num_ground_gt = count_num_ground_without_vegetation(pc_curr);

  int num_inliers_ground = count_num_ground_without_vegetation(ground_estimated);
  int num_TP, num_FP, num_FN, num_TN;
  if (reject_num_of_outliers){
      int num_outliers_est = count_num_outliers_veg(ground_estimated);
      int num_outliers_gt = count_num_outliers_veg(pc_curr);
      num_TP = num_inliers_ground + num_outliers_est;
      num_FP = num_ground_est - num_TP;
      num_FN = num_ground_gt - num_inliers_ground;
      num_TN = pc_curr.points.size() - (num_TP + num_FP + num_FN);
//        num_FP = (num_ground_est - num_outliers_est) - num_TP;
//        num_FN = num_ground_gt - num_TP;
//        num_TF = (pc_curr.points.size() - num_outliers_gt - num_veg_gt) - (num_TP + num_FP + num_FN);
      precision = (double)(num_TP)/(num_TP + num_FP) * 100;
      recall = (double)(num_TP)/(num_TP + num_FN) * 100;
      accuracy = (double)(num_TP + num_TN)/(num_TP + num_TN + num_FP + num_FN) * 100;
  }else{
      // Not recommended
      num_TP = num_inliers_ground;
      num_FP = num_ground_est - num_TP;
      num_FN = num_ground_gt - num_TP;
      num_TN = pc_curr.points.size()  - (num_TP + num_FP + num_FN);
//        num_TF = (pc_curr.points.size() - num_veg_gt) - (num_TP + num_FP + num_FN);
      precision = (double)(num_TP)/num_ground_est * 100;
      recall = (double)(num_TP)/num_ground_gt * 100;
      accuracy = (double)(num_TP + num_TN)/(num_TP + num_TN + num_FP + num_FN) * 100;
  }
}


// bool isSameCluster(const pcl::PointXYZINormal &p1, const pcl::PointXYZINormal &p2, float sqr_dist) {
//     if (!debug) {
//         if (sqr_dist < gt_clustering_tol*gt_clustering_tol && 
//             p1.curvature == p2.curvature &&
//             p1.intensity == p2.intensity) {  // intensity: label, curvature: id
//             // printf("Label: %.1f, id: %.1f, dist: %.2f, thres: %.2f\n", p1.intensity, p1.curvature, sqr_dist, gt_clustering_tol*gt_clustering_tol);
//             // printf("p1: %f;%f;%f,  p2: %f;%f;%f\n", p1.x, p1.y, p1.z, p2.x, p2.y, p2.z);
//             return true;
//         }
//     } else {
//         if (sqr_dist < gt_clustering_tol*gt_clustering_tol && 
//             p1.intensity == p2.intensity) {  // intensity: label, curvature: id
//             // printf("Label: %.1f, id: %.1f, dist: %.2f, thres: %.2f\n", p1.intensity, p1.curvature, sqr_dist, gt_clustering_tol*gt_clustering_tol);
//             // printf("p1: %f;%f;%f,  p2: %f;%f;%f\n", p1.x, p1.y, p1.z, p2.x, p2.y, p2.z);
//             return true;
//         }
//     }
//     return false;
// } 

// void copyPointCloudCEC(pcl::PointCloud<PointType> &src, 
//         pcl::PointCloud<pcl::PointXYZINormal> &dst) {
//     dst.points.resize(src.points.size());
//     for (size_t i = 0; i < src.points.size(); i++) {
//         dst[i].x = src[i].x;
//         dst[i].y = src[i].y;
//         dst[i].z = src[i].z;
//         dst[i].intensity = src[i].label;
//     }
// }

//function for USE
template <typename PointType>
float evalUSE(typename pcl::PointCloud<PointType>::Ptr cloud_in) {
    float USE = 0.0;
    std::map<uint32_t, std::map<uint32_t, int>> pred_to_tgt;
    for (auto &pt : cloud_in->points) {
        uint32_t pred_label = pt.id;
        uint32_t tgt_label = pt.label;
        std::cout << "id: " << pred_label << " label: " << tgt_label << std::endl;
        if (pred_label == 0)  // Unlabelled due to minimum cluster size
            continue;
        if (pred_to_tgt.find(pred_label) == pred_to_tgt.end()) {
            pred_to_tgt.emplace(pred_label, std::map<uint32_t, int>());
        }
        if (pred_to_tgt[pred_label].find(tgt_label) == pred_to_tgt[pred_label].end()) {
            pred_to_tgt[pred_label][tgt_label] = 1;
        } else {
            pred_to_tgt[pred_label][tgt_label]++;
        }
    }

    for (const auto &element : pred_to_tgt) {
        std::map<uint32_t, int> targets = element.second;
        int total = 0;
        float entropy = 0.0;
        for (auto label : targets) {
            total += label.second;
        }
        if (total > 0) {
            for (auto label : targets) {
                if (label.second > 0) {
                    entropy -= label.second / (1.0 * total) * log(label.second / (1.0 * total));
                }
            }
        }
        USE += entropy;
    }
    return USE;
}
//

// function for OSE

// float evalOSE(pcl::PointCloud<PointType>::Ptr cloud_in,
//              std::vector<pcl::PointIndices> &cluster_indices) {
//     float OSE = 0.0;
//     float gt_clustering_tol = 1.0;
//     float min_cluster_size = 10;
//     float max_cluster_size= 300000;
//     pcl::PointCloud<pcl::PointXYZINormal>::Ptr cec_tgt(new pcl::PointCloud<pcl::PointXYZINormal>());
//     copyPointCloudCEC(*cloud_in, *cec_tgt);
//     cluster_extractor_.setInputCloud(cec_tgt);
//     cluster_extractor_.setConditionFunction(&isSameCluster);
//     cluster_extractor_.setMinClusterSize(min_cluster_size);
//     cluster_extractor_.setMaxClusterSize(max_cluster_size);
//     cluster_extractor_.setClusterTolerance(gt_clustering_tol);
//     cluster_extractor_.segment(cluster_indices);

//     // compute OSE
//     std::map<uint32_t, int> pred_count;
//     for (auto cluster : cluster_indices) {
//         pred_count.clear();
//         size_t cluster_size = cluster.indices.size();
//         float entropy = 0.0;
//         for (auto idx : cluster.indices) {
//             uint32_t pred_label = cloud_in->points[idx].id;
//             if (pred_count.find(pred_label) == pred_count.end()) {
//                 pred_count[pred_label] = 1;
//             } else {
//                 pred_count[pred_label]++;
//             }               
//         }
//         for (auto element : pred_count) {
//             // printf("element second: %d, cluster_size: %d\n", element.second, (int)cluster_size);
//             entropy -= element.second / (1.0*cluster_size) * log(element.second / (1.0*cluster_size));
//         }
//         // printf("Per-cluster OSE entropy: %f\n", entropy);
//         OSE += entropy;
//     }

//     // Checking clustering result
//     printf("GT: # of clusters: %d\n", (int)cluster_indices.size());
//     pcl::PointCloud<pcl::PointXYZL>::Ptr test_pcl(new pcl::PointCloud<pcl::PointXYZL>());
//     test_pcl->points.resize(cec_tgt->points.size());
//     for (auto cluster : cluster_indices) {
//         uint label = rand() % 1000;
//         for (auto idx : cluster.indices) {
//             // visualization
//             test_pcl->points[idx].x = cec_tgt->points[idx].x;
//             test_pcl->points[idx].y = cec_tgt->points[idx].y;
//             test_pcl->points[idx].z = cec_tgt->points[idx].z;
//             test_pcl->points[idx].label = label;
//         }
//     }
//     if (cloud_test_pub.getNumSubscribers() > 0) {
//         sensor_msgs::PointCloud2 test_cloud;
//         pcl::toROSMsg(*test_pcl, test_cloud);
//         test_cloud.header = cloud_header;
//         cloud_test_pub.publish(test_cloud);    
//     }
//     return OSE;
// }


#endif
