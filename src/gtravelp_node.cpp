#define PCL_NO_PRECOMPILE

#include "gtravelp/aos.hpp"
#include "gtravelp/tgs.hpp"
#include "gtravelp/node.h"
#include "utils/utils.hpp"  // for kitti
#include "utils/utils_rellis.hpp" // for rellis

ros::Publisher pub_nonground_cloud;
ros::Publisher pub_ground_cloud;
ros::Publisher pub_labeled_cloud;
ros::Publisher pub_raw_cloud;
ros::Publisher pub_TP_cloud;
ros::Publisher pub_FN_cloud;
ros::Publisher pub_FP_cloud;


boost::shared_ptr<gtravelp::gTravelpGroundSeg<PointT>> gtravelp_ground_seg;
boost::shared_ptr<gtravelp::ObjectCluster<PointT>> gtravelp_object_seg;

pcl::PointCloud<PointT>::Ptr cloud_in;
pcl::PointCloud<PointT>::Ptr filtered_pc;
pcl::PointCloud<PointT>::Ptr ground_pc;
pcl::PointCloud<PointT>::Ptr nonground_pc;
pcl::PointCloud<PointT>::Ptr outlier_pc;
pcl::PointCloud<PointT>::Ptr labeled_pc;

pcl::PointCloud<PointT>  TP;
pcl::PointCloud<PointT>  FP;
pcl::PointCloud<PointT>  FN;
pcl::PointCloud<PointT>  TN;

float  min_range_, max_range_;
string abs_save_dir_;
bool   save_labels_ = false;
double total_precision = 0.0;
double total_recall = 0.0;
double total_accuracy = 0.0;
double total_precision_wo_veg = 0.0;
double total_recall_wo_veg = 0.0;
double total_accuracy_wo_veg = 0.0;
double total_time = 0.0;
int frame_count = 0;

void callbackNode(const gtravelp::node::ConstPtr &msg) {

    gtravelp::node node_msg = *msg;
    std::cout << "Seq: " << node_msg.header.seq << std::endl;
    // std::cout << "Frame: " << node_msg.header.frame_id << std::endl;
    std_msgs::Header node_header = node_msg.header;
    cloud_in->clear();
    filtered_pc->clear();
    ground_pc->clear();
    nonground_pc->clear();
    labeled_pc->clear();
    TP.clear();
    FP.clear();
    FN.clear();
    // Convert to PCL
    node_msg.lidar.header = node_header;
    pcl::fromROSMsg(node_msg.lidar, *cloud_in);
    // std::cout<< "Cloud size: " << cloud_in->size() << std::endl;
    pub_raw_cloud.publish(node_msg.lidar);

    // Filter nan points and points out of range
    filtered_pc->header = cloud_in->header;
    filtered_pc->points.reserve(cloud_in->points.size());  // this is not good for a filter
    for (auto &point : cloud_in->points){
        bool is_nan = std::isnan(point.x) || std::isnan(point.y) || std::isnan(point.z);
        double pt_range = 0.0;
        if (is_nan){
            continue;
        }    
        // pt_range = sqrt(pow(point.x, 2) + pow(point.y, 2) + pow(point.z, 2));
        // if (pt_range <= min_range_ || pt_range >= max_range_){
        //     continue;
        // }
        filtered_pc->push_back(point);
    }
    // Apply traversable ground segmentation
    double ground_seg_time = 0.0;
    double time_taken_embed = 0.0;
    double time_taken_modelnode = 0.0;
    double time_taken_BFS = 0.0;
    double time_taken_refine = 0.0;
    double time_taken_merging = 0.0;
    double time_taken_segment = 0.0;
    double time_taken_rgnr = 0.0;
    gtravelp_ground_seg->estimateGround(*filtered_pc, *ground_pc, *nonground_pc, ground_seg_time, time_taken_embed, time_taken_modelnode, time_taken_BFS, time_taken_refine, time_taken_merging, time_taken_segment, time_taken_rgnr);
    std::cout << "\033[1;35m Traversable-Ground Seg: " << filtered_pc->size() << " -> Ground: " << ground_pc->size() << ", NonGround: " << nonground_pc->size() << "\033[0m" << std::endl;
    std::cout << "\033[1;33m Traversable-Ground Seg time: " << ground_seg_time << "\033[0m" << std::endl;
    std::cout << "\033[1;33m Embed time: " << time_taken_embed << "\033[0m" << std::endl;
    std::cout << "\033[1;33m PCA time: " << time_taken_modelnode << "\033[0m" << std::endl;
    std::cout << "\033[1;33m BFS time: " << time_taken_BFS << "\033[0m" << std::endl;
    std::cout << "\033[1;33m Merging time: " << time_taken_merging << "\033[0m" << std::endl;
    std::cout << "\033[1;33m RGNR time: " << time_taken_rgnr << "\033[0m" << std::endl;
    std::cout << "\033[1;33m Refine time: " << time_taken_refine << "\033[0m" << std::endl;
    std::cout << "\033[1;33m Segment time: " << time_taken_segment << "\033[0m" << std::endl;
    double precision, recall, accuracy, precision_wo_veg, recall_wo_veg, accuracy_wo_veg;
    calculate_precision_recall(*filtered_pc, *ground_pc, precision, recall, accuracy);
    calculate_precision_recall_without_vegetation(*filtered_pc, *ground_pc, precision_wo_veg, recall_wo_veg, accuracy_wo_veg);

    cout << "\033[1;32m P: " << precision_wo_veg << " | R: " << recall_wo_veg << "\033[0m" << endl;
    total_precision += precision;
    total_recall += recall;
    total_accuracy += accuracy;
    total_precision_wo_veg += precision_wo_veg;
    total_recall_wo_veg += recall_wo_veg;
    total_accuracy_wo_veg += accuracy_wo_veg;
    total_time += ground_seg_time;
    frame_count++;

    // Compute averages
    double avg_precision = total_precision / frame_count;
    double avg_recall = total_recall / frame_count;
    double avg_accuracy = total_accuracy / frame_count;
    double avg_precision_wo_veg = total_precision_wo_veg / frame_count;
    double avg_recall_wo_veg = total_recall_wo_veg / frame_count;
    double avg_accuracy_wo_veg = total_accuracy_wo_veg / frame_count;
    double avg_time = total_time / frame_count;
    std::cout << "\033[1;32m Average Precision (without vegetation): " << avg_precision_wo_veg 
              << " | Average Recall (without vegetation): " << avg_recall_wo_veg 
              << " | Average Accuracy (without vegetation): " << avg_accuracy_wo_veg << "\033[0m" << std::endl;

    std::cout << "\033[1;34m Average Precision: " << avg_precision 
              << " | Average Recall: " << avg_recall 
              << " | Average Accuracy: " << avg_accuracy << "\033[0m" << std::endl;

    std::cout << "\033[1;35m Average Time: " << avg_time << "\033[0m" << std::endl;
    sensor_msgs::PointCloud2 ground_cloud_msg;
    pcl::toROSMsg(*ground_pc, ground_cloud_msg);
    ground_cloud_msg.header = node_header;
    pub_ground_cloud.publish(ground_cloud_msg);

    sensor_msgs::PointCloud2 nonground_cloud_msg;
    pcl::toROSMsg(*nonground_pc, nonground_cloud_msg);
    nonground_cloud_msg.header = node_header;
    pub_nonground_cloud.publish(nonground_cloud_msg);

    discern_ground_without_vegetation(*ground_pc, TP, FP);
    discern_ground_without_vegetation(*nonground_pc, FN, TN);
    
    sensor_msgs::PointCloud2 TP_cloud_msg;
    pcl::toROSMsg(TP, TP_cloud_msg);
    TP_cloud_msg.header = node_header;
    pub_TP_cloud.publish(TP_cloud_msg);

    sensor_msgs::PointCloud2 FN_cloud_msg;
    pcl::toROSMsg(FN, FN_cloud_msg);
    FN_cloud_msg.header = node_header;
    pub_FN_cloud.publish(FN_cloud_msg);

    sensor_msgs::PointCloud2 FP_cloud_msg;
    pcl::toROSMsg(FP, FP_cloud_msg);
    FP_cloud_msg.header = node_header;
    pub_FP_cloud.publish(FP_cloud_msg);

    // CloudPublisher.publish(cloud2msg(pc_curr));
    // TPPublisher.publish(cloud2msg(TP));
    // FPPublisher.publish(cloud2msg(FP));
    // FNPublisher.publish(cloud2msg(FN));



    // Apply above-ground object segmentation
    // travel_object_seg->segmentObjects(nonground_pc, labeled_pc);
    // std::cout << "\033[1;35m Above-Ground Seg: -> " << labeled_pc->size() << "\033[0m" << std::endl;
    // evalUSE<PointXYZILID>(labeled_pc);
    // // Just for 3DUIS benchmark
    // // Please refer to the site:
    // // https://codalab.lisn.upsaclay.fr/competitions/2183?secret_key=4763e3d2-1f22-45e6-803a-a862528426d2
    // if (save_labels_) {
    //     saveLabels(abs_save_dir_, node_msg.header.seq, *cloud_in, *labeled_pc);
    // }
    // sensor_msgs::PointCloud2 labeled_cloud_msg;
    // pcl::toROSMsg(*labeled_pc, labeled_cloud_msg);
    // labeled_cloud_msg.header = node_header;
    // pub_labeled_cloud.publish(labeled_cloud_msg);

    return;
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "gtravelp_graph");
    ros::NodeHandle nh;
    ROS_INFO("gTravelp: Graph Cluster");

    // Set Parameters
    std::string node_topic_;
    nh.param<string>("/node_topic"          , node_topic_, "/node");
    std::cout << "\033[1;32m" << "Node topic: " << node_topic_ << "\033[0m" << std::endl;
    nh.param<bool> ("/save_results/save_labels"  , save_labels_, false);
    nh.param<string> ("/save_results/abs_save_dir"  , abs_save_dir_, "");

    int vert_scan, horz_scan;
    float min_vert_angle, max_vert_angle;
    nh.param<float> ("/lidar/min_range"     , min_range_, 0.0);
    nh.param<float> ("/lidar/max_range"     , max_range_, 30.0);
    nh.param<int>   ("/lidar/vert_scan"     , vert_scan, 64);
    nh.param<int>   ("/lidar/horz_scan"     , horz_scan, 1800);
    nh.param<float> ("/lidar/min_vert_angle", min_vert_angle, -30.0);
    nh.param<float> ("/lidar/max_vert_angle", max_vert_angle, 50.0);

    float tgf_res, th_seeds, th_dist, th_outlier, th_normal, th_weight, th_lcc_normal_similarity, th_lcc_planar_dist, th_merging;
    int num_iter, num_lpr, num_min_pts;
    bool refine_mode, viz_mode, rgnr, tnnr, mnpf;
    double sensor_height, th_seed_disparity, adaptive_seeds_margin;
    nh.param<float>("/tgs/resolution"    , tgf_res, 5.0);
    nh.param<int>  ("/tgs/num_iter"     , num_iter, 3);
    nh.param<int>  ("/tgs/num_lpr"      , num_lpr, 20);
    nh.param<int>  ("/tgs/num_min_pts"  , num_min_pts, 10);
    nh.param<float>("/tgs/th_seeds"     , th_seeds, 0.4);
    nh.param<float>("/tgs/th_dist"      , th_dist, 0.3);
    nh.param<float>("/tgs/th_outlier"   , th_outlier, 0.3);
    nh.param<float>("/tgs/th_normal"    , th_normal, 0.707);
    nh.param<float>("/tgs/th_weight"    , th_weight, 1.5);
    nh.param<float>("/tgs/th_merging"  , th_merging, 1.5);
    nh.param<float>("/tgs/th_lcc_normal", th_lcc_normal_similarity , 1.5);
    nh.param<float>("/tgs/th_lcc_planar", th_lcc_planar_dist , 1.5);
    nh.param<bool> ("/tgs/refine_mode"  , refine_mode, true);
    nh.param<bool> ("/tgs/visualization", viz_mode, true);  
    nh.param<bool> ("/tgs/rgnr", rgnr, false);
    nh.param<bool> ("/tgs/tnnr", tnnr, false);
    nh.param<bool> ("/tgs/mnpf", mnpf, false);
    nh.param<double> ("/tgs/sensor_height", sensor_height, 1.723);
    nh.param<double> ("/tgs/th_seed_disparity", th_seed_disparity, 3);
    nh.param<double> ("/tgs/adaptive_seeds_margin", adaptive_seeds_margin, -1.2);
    float car_width, car_length, lidar_width_offset, lidar_length_offset, horz_merge_thres, vert_merge_thres;
    int downsample, vert_scan_size, horz_scan_size, horz_skip_size, horz_extension_size, min_cluster_size, max_cluster_size;
    bool debug;
    nh.param<int>  ("/aos/downsample"           , downsample, 2);
    nh.param<float>("/aos/car_width"            , car_width, 1.0);
    nh.param<float>("/aos/car_length"           , car_length, 1.0);
    nh.param<float>("/aos/lidar_width_offset"   , lidar_width_offset, 0.0);
    nh.param<float>("/aos/lidar_length_offset"  , lidar_length_offset, 0.0);
    nh.param<float>("/aos/th_horz_merg"         , horz_merge_thres, 0.3);
    nh.param<float>("/aos/th_vert_merg"         , vert_merge_thres, 1.0);
    nh.param<int>  ("/aos/vert_scan_size"       , vert_scan_size, 4);
    nh.param<int>  ("/aos/horz_scan_size"       , horz_scan_size, 4);
    nh.param<int>  ("/aos/horz_skip_size"       , horz_skip_size, 4);
    nh.param<int>  ("/aos/horz_extension_size"  , horz_extension_size, 3);
    nh.param<int>  ("/aos/min_cluster_size"     , min_cluster_size, 4);
    nh.param<int>  ("/aos/max_cluster_size"     , max_cluster_size, 100);
    // travel_ground_seg.reset(new travel::TravelGroundSeg<PointT>());
    gtravelp_ground_seg.reset(new gtravelp::gTravelpGroundSeg<PointT>(&nh));
    gtravelp_object_seg.reset(new gtravelp::ObjectCluster<PointT>());

    std::cout << "Max Range: " << max_range_ << std::endl;
    std::cout << "Min Range: " << min_range_ << std::endl;
    gtravelp_ground_seg->setParams(max_range_, min_range_, tgf_res, 
                                num_iter, num_lpr, num_min_pts, th_seeds, 
                                th_dist, th_outlier, th_normal, th_weight, 
                                th_lcc_normal_similarity, th_lcc_planar_dist, th_merging, sensor_height,
                                refine_mode, viz_mode, rgnr, th_seed_disparity, adaptive_seeds_margin, tnnr, mnpf);

    gtravelp_object_seg->setParams(vert_scan, horz_scan, min_range_, max_range_, 
                                min_vert_angle, max_vert_angle,
                                horz_merge_thres, vert_merge_thres, vert_scan_size,
                                horz_scan_size, horz_extension_size, horz_skip_size, downsample, 
                                min_cluster_size, max_cluster_size);
    
    cloud_in.reset(new pcl::PointCloud<PointT>());
    filtered_pc.reset(new pcl::PointCloud<PointT>());
    ground_pc.reset(new pcl::PointCloud<PointT>());
    nonground_pc.reset(new pcl::PointCloud<PointT>());
    labeled_pc.reset(new pcl::PointCloud<PointT>());
    // TP.reset(new pcl::PointCloud<PointT>());
    // FP.reset(new pcl::PointCloud<PointT>());
    // FN.reset(new pcl::PointCloud<PointT>());

    pub_raw_cloud = nh.advertise<sensor_msgs::PointCloud2>("gtravelp/original_pc", 1);
    pub_nonground_cloud = nh.advertise<sensor_msgs::PointCloud2>("gtravelp/nonground_pc", 1);
    pub_ground_cloud = nh.advertise<sensor_msgs::PointCloud2>("gtravelp/ground_pc", 1);
    pub_labeled_cloud = nh.advertise<sensor_msgs::PointCloud2>("gtravelp/segmented_pc", 1);
    pub_TP_cloud = nh.advertise<sensor_msgs::PointCloud2>("gtravelp/TP", 1);
    pub_FN_cloud = nh.advertise<sensor_msgs::PointCloud2>("gtravelp/FN", 1);
    pub_FP_cloud = nh.advertise<sensor_msgs::PointCloud2>("gtravelp/FP", 1);


    ros::Subscriber sub_ptCloud = nh.subscribe<gtravelp::node>(node_topic_, 4000, callbackNode, ros::TransportHints().tcpNoDelay());
    ros::spin();

    return 0;
}
