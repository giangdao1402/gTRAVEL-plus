#ifndef gTRAVELp_GSEG_H
#define gTRAVELp_GSEG_H
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <random>
#include <mutex>
#include <thread>
#include <chrono>
#include <math.h>
#include <fstream>
#include <memory>
#include <signal.h>
#include <cmath>

#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/filter.h>
#include <pcl/common/centroid.h>
#include <pcl/filters/voxel_grid.h>
#include <tf/transform_listener.h>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>

#include <jsk_recognition_msgs/PolygonArray.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <opencv2/opencv.hpp>
#include <unordered_set>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <mutex> 
#include "tbb/blocked_range2d.h"
#include "gtravelp/atan.hpp"
namespace gtravelp {

    #define PTCLOUD_SIZE 132000
    #define NODEWISE_PTCLOUDSIZE 5000

    #define UNKNOWN 1
    #define NONGROUND 2
    #define GROUND 3

    using Eigen::MatrixXf;
    using Eigen::JacobiSVD;
    using Eigen::VectorXf;

    template <typename PointType>
    bool point_z_cmp(PointType a, PointType b){
        return a.z<b.z;
    }

    struct TriGridIdx {
        int row, col, tri;
        bool operator<(const TriGridIdx& other) const {
            if (row != other.row) return row < other.row; // Compare rows
            if (col != other.col) return col < other.col; // Compare columns if rows are equal
            return tri < other.tri; // Compare triangles if rows and columns are equal
        }
    };

    struct TriGridEdge {
        std::pair<TriGridIdx, TriGridIdx> Pair;
        bool is_traversable;
    };

    template <typename PointType>
    struct TriGridNode {
        int node_type;
        bool mark;
        pcl::PointCloud<PointType> ptCloud;
        bool is_curr_data;
        // sub graph
        bool is_processed;
        // planar model
        Eigen::Vector3f normal;
        Eigen::Vector3f mean_pt;
        double d;

        Eigen::Vector3f singular_values;
        Eigen::Matrix3f eigen_vectors;
        double weight;

        double th_dist_d;
        double th_outlier_d;

        // graph_searching
        bool need_recheck;
        bool is_visited;
        bool is_rejection;
        int check_life;
        int depth;
    };

    struct TriGridCorner {
        double x, y, z;
        std::vector<double> zs;
        std::vector<double> weights;
    };

    template <typename PointType>
    using GridNode = std::vector<TriGridNode<PointType>>;

    template <typename PointType>
    using TriGridField = std::vector<std::vector<GridNode<PointType>>>;

    template <typename PointType>
    class gTravelpGroundSeg{
    private:
        ros::NodeHandle node_handle_;
        pcl::PCLHeader cloud_header_;
        std_msgs::Header msg_header_;

        ros::Publisher pub_trigrid_nodes_;
        ros::Publisher pub_trigrid_edges_;
        ros::Publisher pub_trigrid_corners_;
        ros::Publisher pub_tgseg_ground_cloud;
        ros::Publisher pub_tgseg_nonground_cloud;
        ros::Publisher pub_tgseg_outliers_cloud;
        ros::Publisher pub_initial_seeds;
        ros::Publisher pub_dominant_cloud;
        ros::Publisher pub_noise_cloud;
        jsk_recognition_msgs::PolygonArray viz_trigrid_polygons_;
        visualization_msgs::Marker viz_trigrid_edges_;
        pcl::PointCloud<pcl::PointXYZ> viz_trigrid_corners_;

        bool REFINE_MODE_;
        bool VIZ_MDOE_;
        bool RGNR;
        bool TNNR;
        bool MNPF;
        double MAX_RANGE_;
        double MIN_RANGE_;
        double TGF_RESOLUTION_;

        int NUM_ITER_;
        int NUM_LRP_;
        int NUM_MIN_POINTS_;
        
        double TH_SEEDS_;
        double TH_DIST_;
        double TH_OUTLIER_;
        
        double TH_NORMAL_;
        double TH_WEIGHT_;
        double TH_LCC_NORMAL_SIMILARITY_;
        double TH_LCC_PLANAR_MODEL_DIST_;
        double TH_MERGING_;
        double TH_SEED_DISPARITY_;


        double SENSOR_HEIGHT_, ADAPTIVE_SEEDS_MARGIN_;

        TriGridField<PointType> trigrid_field_;
        std::vector<TriGridEdge> trigrid_edges_;
        std::vector<std::vector<TriGridCorner>> trigrid_corners_;
        std::vector<std::vector<TriGridCorner>> trigrid_centers_;

        pcl::PointCloud<PointType> empty_cloud_;
        TriGridNode<PointType>  empty_trigrid_node_;
        GridNode<PointType> empty_grid_nodes_;
        TriGridCorner empty_trigrid_corner_;
        TriGridCorner empty_trigrid_center_;

        pcl::PointCloud<PointType> ptCloud_tgfwise_ground_;
        pcl::PointCloud<PointType> ptCloud_tgfwise_nonground_;
        pcl::PointCloud<PointType> ptCloud_tgfwise_outliers_;
        pcl::PointCloud<PointType> ptCloud_tgfwise_obstacle_;
        pcl::PointCloud<PointType> ptCloud_nodewise_ground_;
        pcl::PointCloud<PointType> ptCloud_nodewise_nonground_;
        pcl::PointCloud<PointType> ptCloud_nodewise_outliers_;
        pcl::PointCloud<PointType> ptCloud_nodewise_obstacle_;
        pcl::PointCloud<PointType> noise_;
        pcl::PointCloud<PointType> init_seeds_;
        std::map<TriGridIdx, double> idx_seed_map;
        std::map<TriGridIdx, bool> idx_have_noise_map;
        std::map<TriGridIdx, bool> idx_type_map;
        // pcl::PointCloud<PointType> seeds_for_mean_;
    public:
        gTravelpGroundSeg(ros::NodeHandle* nh): node_handle_(*nh) {
        // TravelGroundSeg(){
            // Init ROS related
            ROS_INFO("Initializing Traversable Ground Segmentation...");

            pub_trigrid_nodes_  = node_handle_.advertise<jsk_recognition_msgs::PolygonArray>("/gtravelp_gseg/nodes", 1);
            pub_trigrid_edges_  = node_handle_.advertise<visualization_msgs::Marker>("/gtravelp_gseg/edges", 1);
            pub_trigrid_corners_= node_handle_.advertise<pcl::PointCloud<pcl::PointXYZ>>("/gtravelp_gseg/corners", 1);
            pub_tgseg_ground_cloud    = node_handle_.advertise<sensor_msgs::PointCloud2>("/gtravelp_gseg/ground_cloud", 1);
            pub_tgseg_nonground_cloud = node_handle_.advertise<sensor_msgs::PointCloud2>("/gtravelp_gseg/nonground_cloud", 1);
            pub_tgseg_outliers_cloud   = node_handle_.advertise<sensor_msgs::PointCloud2>("/gtravelp_gseg/outlier_cloud", 1);
            pub_initial_seeds = node_handle_.advertise<sensor_msgs::PointCloud2>("/gtravelp_gseg/seeds", 1);
            pub_dominant_cloud = node_handle_.advertise<sensor_msgs::PointCloud2>("/gtravelp_gseg/dominant", 1);
            pub_noise_cloud = node_handle_.advertise<sensor_msgs::PointCloud2>("/gtravelp_gseg/noise", 1);
        }
        
        void setParams(const double max_range, const double min_range, const double resolution, 
                            const int num_iter, const int num_lpr, const int num_min_pts, const double th_seeds, 
                            const double th_dist, const double th_outlier, const double th_normal, const double th_weight, 
                            const double th_lcc_normal_similiarity, const double th_lcc_planar_model_dist, const double th_merging, const double sensor_height,
                            const bool refine_mode, const bool visualization, const bool rgnr, const double th_seed_disparity, 
                            const double adaptive_seeds_margin, const bool tnnr, bool mnpf) {
            std::cout<<""<<std::endl;
            ROS_INFO("Set gTRAVELp_GSEG Parameters");

            SENSOR_HEIGHT_ = sensor_height;
            ROS_INFO("Sensor height: %f", SENSOR_HEIGHT_);

            ADAPTIVE_SEEDS_MARGIN_ = adaptive_seeds_margin;
            ROS_INFO("Adaptive seeds margin: %f", ADAPTIVE_SEEDS_MARGIN_);

            MAX_RANGE_ = max_range;
            ROS_INFO("Max Range: %f", MAX_RANGE_);

            MIN_RANGE_ = min_range;
            ROS_INFO("Min Range: %f", MIN_RANGE_);
            
            TGF_RESOLUTION_ = resolution;
            ROS_INFO("Resolution: %f", TGF_RESOLUTION_);
            
            NUM_ITER_ = num_iter;
            ROS_INFO("Num of Iteration: %d", NUM_ITER_);
            
            NUM_LRP_ = num_lpr;
            ROS_INFO("Num of LPR: %d", NUM_LRP_);
            
            NUM_MIN_POINTS_ = num_min_pts;
            ROS_INFO("Num of min. points: %d", NUM_MIN_POINTS_);

            TH_SEEDS_ = th_seeds;
            ROS_INFO("Seeds Threshold: %f", TH_SEEDS_);
            
            TH_DIST_ = th_dist;
            ROS_INFO("Distance Threshold: %f", TH_DIST_);
            
            TH_OUTLIER_ = th_outlier;
            ROS_INFO("Outlier Threshold: %f", TH_OUTLIER_);

            TH_NORMAL_ = th_normal;
            ROS_INFO("Normal Threshold: %f", TH_NORMAL_);

            TH_WEIGHT_ = th_weight;
            ROS_INFO("Node Weight Threshold: %f", TH_WEIGHT_);

            TH_LCC_NORMAL_SIMILARITY_ = th_lcc_normal_similiarity;
            ROS_INFO("LCC Normal Similarity: %f", TH_LCC_NORMAL_SIMILARITY_);

            TH_LCC_PLANAR_MODEL_DIST_ = th_lcc_planar_model_dist;
            ROS_INFO("LCC Plane Distance   : %f", TH_LCC_PLANAR_MODEL_DIST_);

            TH_MERGING_ = th_merging;
            ROS_INFO("Merging threshold  : %f", TH_MERGING_);
            
            TH_SEED_DISPARITY_ = th_seed_disparity;
            ROS_INFO("Seed disparity threshold  : %f", TH_SEED_DISPARITY_);

            RGNR = rgnr;
            REFINE_MODE_ = refine_mode;
            VIZ_MDOE_ = visualization;
            TNNR = tnnr;
            MNPF = mnpf;

            ROS_INFO("Set TGF Parameters");
            initTriGridField(trigrid_field_);
            initTriGridCorners(trigrid_corners_, trigrid_centers_);

            ptCloud_tgfwise_ground_.clear();
            ptCloud_tgfwise_ground_.reserve(PTCLOUD_SIZE);
            ptCloud_tgfwise_nonground_.clear();
            ptCloud_tgfwise_nonground_.reserve(PTCLOUD_SIZE);
            ptCloud_tgfwise_outliers_.clear();
            ptCloud_tgfwise_outliers_.reserve(PTCLOUD_SIZE);
            ptCloud_tgfwise_obstacle_.clear();
            ptCloud_tgfwise_obstacle_.reserve(PTCLOUD_SIZE);
    
            ptCloud_nodewise_ground_.clear();
            ptCloud_nodewise_ground_.reserve(NODEWISE_PTCLOUDSIZE);
            ptCloud_nodewise_nonground_.clear();
            ptCloud_nodewise_nonground_.reserve(NODEWISE_PTCLOUDSIZE);
            ptCloud_nodewise_outliers_.clear();
            ptCloud_nodewise_outliers_.reserve(NODEWISE_PTCLOUDSIZE);
            ptCloud_nodewise_obstacle_.clear();
            ptCloud_nodewise_obstacle_.reserve(NODEWISE_PTCLOUDSIZE);

            if (VIZ_MDOE_) {
                viz_trigrid_polygons_.polygons.clear();
                viz_trigrid_polygons_.polygons.reserve(rows_ * cols_);
                viz_trigrid_polygons_.likelihood.clear();
                viz_trigrid_polygons_.likelihood.reserve(rows_ * cols_);
                

                viz_trigrid_edges_.ns = "trigrid_edges";
                viz_trigrid_edges_.action = visualization_msgs::Marker::ADD;
                viz_trigrid_edges_.type = visualization_msgs::Marker::LINE_LIST;
                viz_trigrid_edges_.pose.orientation.w = 1.0;
                viz_trigrid_edges_.scale.x = 0.5;
                viz_trigrid_edges_.id = 0;

                viz_trigrid_corners_.clear();
                viz_trigrid_corners_.reserve(rows_*cols_ + (rows_+ 1)*(cols_+1));
            }
        };

        void estimateGround(const pcl::PointCloud<PointType>& cloud_in,
                            pcl::PointCloud<PointType>& cloudGround_out,
                            pcl::PointCloud<PointType>& cloudNonground_out,
                            double& time_taken, double& time_taken_embed, double& time_taken_modelnode, double& time_taken_BFS, double& time_taken_refine,
                            double& time_taken_merging, double& time_taken_segment, double& time_taken_rgnr){
        
            // 0. Init
            static time_t start, end, start_embed, end_embed, start_modelnode, end_modelnode, start_BFS,end_BFS, start_rgnr, end_rgnr, start_refine, end_refine, start_merging, end_merging, start_segment, end_segment;
            cloud_header_ = cloud_in.header;  // is already filter
            pcl_conversions::fromPCL(cloud_header_, msg_header_);
            ROS_INFO("TriGrid Field-based Traversable Ground Segmentation...");
            start = clock();
            ptCloud_tgfwise_outliers_.clear();
            ptCloud_tgfwise_outliers_.reserve(cloud_in.size());
            ptCloud_tgfwise_nonground_.clear();
            ptCloud_tgfwise_nonground_.reserve(cloud_in.size());
            ptCloud_tgfwise_ground_.clear();
            ptCloud_tgfwise_ground_.reserve(cloud_in.size());

            // 1. Embed PointCloud to TriGridField
            clearTriGridField(trigrid_field_);
            clearTriGridCorners(trigrid_corners_, trigrid_centers_);

            start_embed = clock();
            embedCloudToTriGridField(cloud_in, trigrid_field_);
            end_embed = clock();

            // 2. Node-wise Terrain 
            start_modelnode = clock();
            modelNodeWiseTerrain(trigrid_field_);
            end_modelnode = clock();
            
            // 3. Merged Node Plane Fitting
            start_merging = clock();    
            if (MNPF)
            {
                MergedNodePlaneFitting(trigrid_field_);
            }
            end_merging = clock();

            // 4. Breadth-first Traversable Graph Search
            start_BFS = clock();
            BreadthFirstTraversableGraphSearch(trigrid_field_);
            end_BFS = clock();
            
            // 5. Rejected Ground Node Revert
            start_rgnr = clock();
            if (RGNR) RejectedGroundNodeRevert(trigrid_field_); //REJECTED GROUND NODE REVERT
            end_rgnr = clock();  

            start_refine = clock();
            setTGFCornersCenters(trigrid_field_, trigrid_corners_, trigrid_centers_);

            // 6. TGF-wise Traversable Terrain Model Fitting
            if (REFINE_MODE_){
                fitTGFWiseTraversableTerrainModel(trigrid_field_, trigrid_corners_, trigrid_centers_);
            }
            end_refine = clock();

            // 7. Ground Segmentation            
            start_segment = clock();
            segmentTGFGround(trigrid_field_, ptCloud_tgfwise_ground_, ptCloud_tgfwise_nonground_, ptCloud_tgfwise_obstacle_, ptCloud_tgfwise_outliers_);
            end_segment = clock();
            cloudGround_out = ptCloud_tgfwise_ground_;
            cloudNonground_out = ptCloud_tgfwise_nonground_;
            cloudGround_out.header = cloudNonground_out.header = cloud_header_;

            end = clock();
            time_taken = (double)(end - start) / CLOCKS_PER_SEC;
            time_taken_embed = (double)(end_embed - start_embed) / CLOCKS_PER_SEC;
            time_taken_modelnode = (double)(end_modelnode - start_modelnode) / CLOCKS_PER_SEC;
            time_taken_BFS = (double)(start_BFS - end_BFS) / CLOCKS_PER_SEC;
            time_taken_merging = (double)(start_merging - end_merging) / CLOCKS_PER_SEC;
            time_taken_rgnr = (double)(start_rgnr - end_rgnr) / CLOCKS_PER_SEC;
            time_taken_refine = (double)(start_refine - end_refine) / CLOCKS_PER_SEC;
            time_taken_segment = (double)(start_segment - end_segment) / CLOCKS_PER_SEC; 
            // 6. Publish Results and Visualization
            if (VIZ_MDOE_){
                publishTriGridFieldGraph();
                publishTriGridCorners();
                publishPointClouds();
                pub_initial_seeds.publish(convertCloudToRosMsg(init_seeds_, cloud_header_.frame_id));
                pub_noise_cloud.publish(convertCloudToRosMsg(noise_, cloud_header_.frame_id));
            }
            noise_.clear();
            init_seeds_.clear();
            return;
        };

        TriGridIdx getTriGridIdx(const float& x_in, const float& y_in){
            TriGridIdx tgf_idx;
            int r_i = (x_in - tgf_min_x)/TGF_RESOLUTION_;
            int c_i = (y_in - tgf_min_y)/TGF_RESOLUTION_;
            int t_i = 0;
            double angle = atan2(y_in-(c_i*TGF_RESOLUTION_ + TGF_RESOLUTION_/2 + tgf_min_y), x_in-(r_i*TGF_RESOLUTION_ + TGF_RESOLUTION_/2 + tgf_min_x));

            if (angle>=(M_PI/4) && angle <(3*M_PI/4)){
                t_i = 1;
            } else if (angle>=(-M_PI/4) && angle <(M_PI/4)){
                t_i = 0;
            } else if (angle>=(-3*M_PI/4) && angle <(-M_PI/4)){
                t_i = 3;
            } else{
                t_i = 2;
            }
            tgf_idx.row = r_i;
            tgf_idx.col = c_i;
            tgf_idx.tri = t_i;
            return tgf_idx;
        }

        TriGridNode<PointType> getTriGridNode(const float& x_in, const float& y_in){
            TriGridNode<PointType> node;
            TriGridIdx node_idx = getTriGridIdx(x_in, y_in);
            node = trigrid_field_[node_idx.row][node_idx.col][node_idx.tri];
            return node;
        };

        TriGridNode<PointType> getTriGridNode(const TriGridIdx& tgf_idx){
            TriGridNode<PointType> node;
            node = trigrid_field_[tgf_idx.row][tgf_idx.col][tgf_idx.tri];
            return node;
        };

        bool is_traversable(const float& x_in, const float& y_in){
            TriGridNode<PointType> node = getTriGridNode(x_in, y_in);
            if (node.node_type == GROUND){
                return true;
            } else{
                return false;
            }
        };

        pcl::PointCloud<PointType> getObstaclePC(){
            pcl::PointCloud<PointType> cloud_obstacle;
            cloud_obstacle = ptCloud_tgfwise_obstacle_;
            return cloud_obstacle;
        };

    private:
        double tgf_max_x, tgf_max_y, tgf_min_x, tgf_min_y;
        double rows_, cols_;

        void initTriGridField(TriGridField<PointType>& tgf_in){
            // ROS_INFO("Initializing TriGridField...");

            tgf_max_x = MAX_RANGE_;
            tgf_max_y = MAX_RANGE_;

            tgf_min_x = -MAX_RANGE_;
            tgf_min_y = -MAX_RANGE_;

            rows_ = (int)(tgf_max_x - tgf_min_x) / TGF_RESOLUTION_;
            cols_ = (int)(tgf_max_y - tgf_min_y) / TGF_RESOLUTION_;
            empty_cloud_.clear();
            empty_cloud_.reserve(PTCLOUD_SIZE);
            
            // Set Node structure
            empty_trigrid_node_.node_type = UNKNOWN;
            empty_trigrid_node_.ptCloud.clear();
            empty_trigrid_node_.ptCloud.reserve(NODEWISE_PTCLOUDSIZE);

            empty_trigrid_node_.is_curr_data = false;
            empty_trigrid_node_.need_recheck = false;
            empty_trigrid_node_.is_visited = false;
            empty_trigrid_node_.is_rejection = false;
            empty_trigrid_node_.mark = false;
            empty_trigrid_node_.is_processed = false;
            empty_trigrid_node_.check_life = 15;
            empty_trigrid_node_.depth = -1;

            empty_trigrid_node_.normal;
            empty_trigrid_node_.mean_pt;
            empty_trigrid_node_.d = 0;
            empty_trigrid_node_.singular_values;
            empty_trigrid_node_.eigen_vectors;
            empty_trigrid_node_.weight = 0;

            empty_trigrid_node_.th_dist_d = 0;
            empty_trigrid_node_.th_outlier_d = 0;

            // // Set idx map
            // for (auto& pair : idx_seed_map) {
            //     pair.second = 0.0;
            // }
            // for (auto& pair : idx_have_noise_map) {
            //     pair.second = false; // no noise
            // }

            // for (auto& pair : idx_type_map) {
            //     pair.second = false; // not ground
            // }
            // Set TriGridField
            tgf_in.clear();
            std::vector<GridNode<PointType>> vec_gridnode;

            for (int i = 0; i < 4 ; i ++) 
                empty_grid_nodes_.emplace_back(empty_trigrid_node_);
                
            for (int i=0; i< cols_; i++){ vec_gridnode.emplace_back(empty_grid_nodes_);}
            for (int j=0; j< rows_; j++){ tgf_in.emplace_back(vec_gridnode);}

            return;
        };

        void initTriGridCorners(std::vector<std::vector<TriGridCorner>>& trigrid_corners_in,
                                std::vector<std::vector<TriGridCorner>>& trigrid_centers_in){
            // ROS_INFO("Initializing TriGridCorners...");

            // Set TriGridCorner
            empty_trigrid_corner_.x = empty_trigrid_corner_.y = 0.0;
            empty_trigrid_corner_.zs.clear();
            empty_trigrid_corner_.zs.reserve(8);
            empty_trigrid_corner_.weights.clear();
            empty_trigrid_corner_.weights.reserve(8);

            empty_trigrid_center_.x = empty_trigrid_center_.y = 0.0;
            empty_trigrid_center_.zs.clear();
            empty_trigrid_center_.zs.reserve(4);
            empty_trigrid_center_.weights.clear();
            empty_trigrid_center_.weights.reserve(4);

            trigrid_corners_in.clear();
            trigrid_centers_in.clear();

            // Set Corners and Centers
            std::vector<TriGridCorner> col_corners;
            std::vector<TriGridCorner> col_centers;
            for (int i=0; i< cols_; i++){
                col_corners.emplace_back(empty_trigrid_corner_);
                col_centers.emplace_back(empty_trigrid_center_);
            }
            col_corners.emplace_back(empty_trigrid_corner_);

            for (int j=0; j< rows_; j++){
                trigrid_corners_in.emplace_back(col_corners);
                trigrid_centers_in.emplace_back(col_centers);
            }
            trigrid_corners_in.emplace_back(col_corners);

            return;
        };

        void clearTriGridField(TriGridField<PointType> &tgf_in){
            // ROS_INFO("Clearing TriGridField...");

            for (int r_i = 0; r_i < rows_; r_i++){
            for (int c_i = 0; c_i < cols_; c_i++){
                tgf_in[r_i][c_i] = empty_grid_nodes_;
            }
            }
            return;
        };

        void clearTriGridCorners(std::vector<std::vector<TriGridCorner>>& trigrid_corners_in,
                                std::vector<std::vector<TriGridCorner>>& trigrid_centers_in){
            // ROS_INFO("Clearing TriGridCorners...");

            TriGridCorner tmp_corner = empty_trigrid_corner_;
            TriGridCorner tmp_center = empty_trigrid_center_;
            for (int r_i = 0; r_i < rows_+1; r_i++){
            for (int c_i = 0; c_i < cols_+1; c_i++){
                tmp_corner.x = (r_i)*TGF_RESOLUTION_+tgf_min_x; tmp_corner.y = (c_i)*TGF_RESOLUTION_+tgf_min_y;
                tmp_corner.zs.clear();
                tmp_corner.weights.clear();
                trigrid_corners_in[r_i][c_i] = tmp_corner;
                if (r_i < rows_ && c_i < cols_) {
                    tmp_center.x = (r_i+0.5)*TGF_RESOLUTION_+tgf_min_x; tmp_center.y = (c_i+0.5)*TGF_RESOLUTION_+tgf_min_y;
                    tmp_center.zs.clear();
                    tmp_center.weights.clear();
                    trigrid_centers_in[r_i][c_i] = tmp_center;
                }
            }
            }
            return;
        };
        
        double xy_2Dradius(double x, double y){
            return sqrt(x*x + y*y);
        };

        bool filterPoint(const PointType &pt_in){
            double xy_range = xy_2Dradius(pt_in.x, pt_in.y);
            if (xy_range >= MAX_RANGE_ || xy_range <= MIN_RANGE_) return true;

            return false;
        }

        // void reflected_noise_removal(const pcl::PointCloud<PointType> &cloud_in, pcl::PointCloud<PointType> &cloud_no_noise)
        // {
        //     for (int i = 0; i < cloud_in.size(); i++)
        //     {
        //         double r = sqrt(cloud_in[i].x * cloud_in[i].x + cloud_in[i].y * cloud_in[i].y);
        //         double z = cloud_in[i].z;
        //         double ver_angle_in_deg = atan2(z, r) * 180 / M_PI;

        //         bool is_low = z < -SENSOR_HEIGHT_ - 0.8;
        //         bool is_angle_low = ver_angle_in_deg < -15;
        //         bool is_intensity_low = cloud_in[i].intensity < 0.2;

        //         if (!(is_low && is_angle_low & is_intensity_low))
        //         {
        //             cloud_no_noise.push_back(cloud_in[i]);
        //         }
        //         else {
        //             noise_.emplace_back(cloud_in[i]);
        //         }
        //     }

        //     // if (verbose_) cout << "[ RNR ] Num of noises : " << noise_pc_.points.size() << endl;
        // }


       void triangle_wise_reflected_noise_removal(TriGridNode<PointType>& tgf_in, double& local_height) {
            
            std::vector<int> noise_z_ver_indices;
            std::vector<bool> selected_indices(tgf_in.ptCloud.size(), false);

            // Step 1: Identify noise points
            for (int i = 0; i < tgf_in.ptCloud.size(); i++) {
                double r = sqrt(tgf_in.ptCloud[i].x * tgf_in.ptCloud[i].x + tgf_in.ptCloud[i].y * tgf_in.ptCloud[i].y);
                double ver_angle_in_deg = atan2(tgf_in.ptCloud[i].z, r) * 180 / M_PI;
                if (tgf_in.ptCloud[i].z < local_height - 1.1 
                && ver_angle_in_deg < -15
                // && tgf_in.ptCloud[i].intensity < 0.2
                ) {
                    noise_z_ver_indices.push_back(i);
                }
            }

            // Mark indices in selected_indices for filtering
            for (int idx : noise_z_ver_indices) {
                selected_indices[idx] = true;
            }

            // Separate noise and filtered points
            pcl::PointCloud<PointType> filtered_ptCloud;
            filtered_ptCloud.reserve(tgf_in.ptCloud.size());

            for (int i = 0; i < tgf_in.ptCloud.size(); ++i) {
                if (!selected_indices[i]) {
                    filtered_ptCloud.push_back(tgf_in.ptCloud[i]);
                } else {
                    // ptCloud_tgfwise_nonground_.points.emplace_back(tgf_in.ptCloud[i]);
                    noise_.emplace_back(tgf_in.ptCloud[i]);
                }
            }

            // Replace original point cloud with filtered cloud
            tgf_in.ptCloud = std::move(filtered_ptCloud);

        }


        void embedCloudToTriGridField(const pcl::PointCloud<PointType>& cloud_in, TriGridField<PointType>& tgf_out) {
            // ROS_INFO("Embedding PointCloud to TriGridField...");

            for (auto const &pt: cloud_in.points){
                if (filterPoint(pt)){
                    ptCloud_tgfwise_nonground_.points.push_back(pt);
                    continue;   
                }
                // double range = xy_2Dradius(pt.x, pt.y);
                // std::cout << "range: " << range << std::endl;
                int r_i = (pt.x - tgf_min_x)/TGF_RESOLUTION_;
                int c_i = (pt.y - tgf_min_y)/TGF_RESOLUTION_;

                if (r_i < 0 || r_i >= rows_ || c_i < 0 || c_i >= cols_) {
                    ptCloud_tgfwise_nonground_.points.push_back(pt);
                    continue;
                }

                double angle = atan2approx(pt.y-(c_i*TGF_RESOLUTION_ + TGF_RESOLUTION_/2 + tgf_min_y), pt.x-(r_i*TGF_RESOLUTION_ + TGF_RESOLUTION_/2 + tgf_min_x));
                if (angle>=(M_PI/4) && angle <(3*M_PI/4)){
                    // left side
                    tgf_out[r_i][c_i][1].ptCloud.push_back(pt);
                    if(!tgf_out[r_i][c_i][1].is_curr_data) {tgf_out[r_i][c_i][1].is_curr_data = true;}
                } else if (angle>=(-M_PI/4) && angle <(M_PI/4)){
                    // upper side
                    tgf_out[r_i][c_i][0].ptCloud.push_back(pt);
                    if (!tgf_out[r_i][c_i][0].is_curr_data){tgf_out[r_i][c_i][0].is_curr_data = true;}
                    
                } else if (angle>=(-3*M_PI/4) && angle <(-M_PI/4)){
                    // right side
                    tgf_out[r_i][c_i][3].ptCloud.push_back(pt);
                    if (!tgf_out[r_i][c_i][3].is_curr_data) {tgf_out[r_i][c_i][3].is_curr_data = true;}
                } else{
                    // lower side
                    tgf_out[r_i][c_i][2].ptCloud.push_back(pt);
                    if (!tgf_out[r_i][c_i][2].is_curr_data) {tgf_out[r_i][c_i][2].is_curr_data = true;}
                }
            }

            return;
        };

        void extractInitialSeeds(const pcl::PointCloud<PointType>& p_sorted, pcl::PointCloud<PointType>& init_seeds, TriGridIdx& cur_idx){
            //function for uniform mode
            init_seeds.points.clear();
            // LPR is the mean of Low Point Representative
            double sum = 0;
            int cnt = 0;
            int init_idx = 0;

            for (int i = 0; i < p_sorted.points.size(); i++) {
                // Calculate xy_range (2D radius from the origin or a reference point)
                double xy_range = xy_2Dradius(p_sorted.points[i].x, p_sorted.points[i].y);

                // Use the xy_range and z-value to determine if the point qualifies as part of the LRP
                if (xy_range < 13 && p_sorted.points[i].z < ADAPTIVE_SEEDS_MARGIN_ * SENSOR_HEIGHT_) {
                    // noise_.points.push_back(p_sorted.points[i]);
                    ++init_idx;  // Increment init_idx without adding to init_seeds
                } else {
                    break;  // Stop if conditions are not met
                }
            }

            for (int i = init_idx; i < p_sorted.points.size() && cnt < NUM_LRP_; i++) {
                sum += p_sorted.points[i].z;
                cnt++;
                // vertical_seeds.points.push_back(p_sorted.points[i]);
            }

            double lpr_height = cnt!=0?sum/cnt:0;

            double total_z = 0.0;
            int seed_count = 0;

            for(int i=0 ; i< (int) p_sorted.points.size() ; i++){
                if(p_sorted.points[i].z < lpr_height + TH_SEEDS_){
                    // if (p_sorted.points[i].z < lpr_height-TH_OUTLIER_) continue;
                    init_seeds.points.push_back(p_sorted.points[i]);
                    init_seeds_.points.push_back(p_sorted.points[i]);
                    total_z += p_sorted.points[i].z;
                    seed_count++;
                }
            }
            if (seed_count > 0) {
                idx_seed_map[cur_idx] = total_z / seed_count;
            }
            return;
        }

        void extractInitialSeeds(const pcl::PointCloud<PointType>& p_sorted, pcl::PointCloud<PointType>& init_seeds){
            //function for uniform mode
            init_seeds.points.clear();
            // LPR is the mean of Low Point Representative
            double sum = 0;
            int cnt = 0;
            int init_idx = 0;

            for (int i = 0; i < p_sorted.points.size(); i++) {
                // Calculate xy_range (2D radius from the origin or a reference point)
                double xy_range = xy_2Dradius(p_sorted.points[i].x, p_sorted.points[i].y);

                // Use the xy_range and z-value to determine if the point qualifies as part of the LRP
                if (xy_range < 13 && p_sorted.points[i].z < ADAPTIVE_SEEDS_MARGIN_ * SENSOR_HEIGHT_) {
                    // noise_.points.push_back(p_sorted.points[i]);
                    ++init_idx;  // Increment init_idx without adding to init_seeds
                } else {
                    break;  // Stop if conditions are not met
                }
            }

            for (int i = init_idx; i < p_sorted.points.size() && cnt < NUM_LRP_; i++) {
                sum += p_sorted.points[i].z;
                cnt++;
                // vertical_seeds.points.push_back(p_sorted.points[i]);
            }

            double lpr_height = cnt!=0?sum/cnt:0;

            for(int i=0 ; i< (int) p_sorted.points.size() ; i++){
                if(p_sorted.points[i].z < lpr_height + TH_SEEDS_){
                    // if (p_sorted.points[i].z < lpr_height-TH_OUTLIER_) continue;
                    init_seeds.points.push_back(p_sorted.points[i]);
                    init_seeds_.points.push_back(p_sorted.points[i]);
                }
            }
            return;
        }

        void estimatePlanarModel(const pcl::PointCloud<PointType>& ground_in, TriGridNode<PointType>& node_out) {

            // function for uniform mode
            Eigen::Matrix3f cov_;
            Eigen::Vector4f pc_mean_;
            pcl::computeMeanAndCovarianceMatrix(ground_in, cov_, pc_mean_);

            // Singular Value Decomposition: SVD
            Eigen::JacobiSVD<Eigen::MatrixXf> svd(cov_, Eigen::DecompositionOptions::ComputeFullU);

            // Use the least singular vector as normal
            node_out.eigen_vectors = svd.matrixU();
            if (node_out.eigen_vectors.col(2)(2,0)<0) {
                node_out.eigen_vectors.col(0)*= -1;
                // node_out.eigen_vectors.col(1)*= -1;
                node_out.eigen_vectors.col(2)*= -1;
            }
            node_out.normal = node_out.eigen_vectors.col(2);
            node_out.singular_values = svd.singularValues();

            // mean ground seeds value
            node_out.mean_pt = pc_mean_.head<3>();

            // according to normal.T*[x,y,z] = -d
            node_out.d = -(node_out.normal.transpose()*node_out.mean_pt)(0,0);

            // set distance theshold to 'th_dist - d'
            node_out.th_dist_d = TH_DIST_ - node_out.d;
            node_out.th_outlier_d = -node_out.d - TH_OUTLIER_;

            return;
        }    


        void estimatePlanarModel(const pcl::PointCloud<PointType>& ground_in, TriGridNode<PointType>& node_out, double TH_DIST_NEW) {

            // function for uniform mode
            Eigen::Matrix3f cov_;
            Eigen::Vector4f pc_mean_;
            pcl::computeMeanAndCovarianceMatrix(ground_in, cov_, pc_mean_);

            // Singular Value Decomposition: SVD
            Eigen::JacobiSVD<Eigen::MatrixXf> svd(cov_, Eigen::DecompositionOptions::ComputeFullU);

            // Use the least singular vector as normal
            node_out.eigen_vectors = svd.matrixU();
            if (node_out.eigen_vectors.col(2)(2,0)<0) {
                node_out.eigen_vectors.col(0)*= -1;
                node_out.eigen_vectors.col(1)*= -1;
                node_out.eigen_vectors.col(2)*= -1;
            }
            node_out.normal = node_out.eigen_vectors.col(2);
            node_out.singular_values = svd.singularValues();

            // mean ground seeds value
            node_out.mean_pt = pc_mean_.head<3>();

            // according to normal.T*[x,y,z] = -d
            node_out.d = -(node_out.normal.transpose()*node_out.mean_pt)(0,0);

            // set distance theshold to 'th_dist - d'
            node_out.th_dist_d = TH_DIST_NEW - node_out.d;
            node_out.th_outlier_d = -node_out.d - TH_OUTLIER_;

            return;
        }    

        void modelPCAbasedTerrain(TriGridNode<PointType>& node_in, TriGridIdx& cur_idx) { 
            // Initailization
            if (!ptCloud_nodewise_ground_.empty()) ptCloud_nodewise_ground_.clear();
            
            // Tri Grid Initialization
            // When to initialize the planar model, we don't have prior. so outlier is removed in heuristic parameter.
            pcl::PointCloud<PointType> sort_ptCloud = node_in.ptCloud;
            // sort in z-coordinate
            sort(sort_ptCloud.points.begin(), sort_ptCloud.end(), point_z_cmp<PointType>);
            // Set init seeds
            extractInitialSeeds(sort_ptCloud, ptCloud_nodewise_ground_, cur_idx);

            // std::cout << "seeds: " <<  ptCloud_nodewise_ground_.size() << std::endl;
            Eigen::MatrixXf points(sort_ptCloud.points.size(),3);
            int j = 0;
            for (auto& p:sort_ptCloud.points){
                points.row(j++)<<p.x, p.y, p.z;
            }
            // Extract Ground
            for (int i =0; i < NUM_ITER_; i++){
                estimatePlanarModel(ptCloud_nodewise_ground_, node_in);
                if(ptCloud_nodewise_ground_.size() < 3){
    
                    node_in.node_type = NONGROUND;
                    break;
                }
                ptCloud_nodewise_ground_.clear();
                // threshold filter
                Eigen::VectorXf result = points*node_in.normal;
                for (int r = 0; r<result.rows(); r++){
                    if (i < NUM_ITER_-1){
                        if (result[r]<node_in.th_dist_d){
                            ptCloud_nodewise_ground_.push_back(sort_ptCloud.points[r]);
                        }
                    } else {
                        if (node_in.normal(2) < TH_NORMAL_ 
                        ){
                            node_in.node_type = NONGROUND;
                        } else {
                            node_in.node_type = GROUND;
                            // std::cout << "GROUND PCA: " << node_in.normal[2] << std::endl;
                        }
                    }
                }
            }

            return;
        }

        void modelPCAbasedTerrain(TriGridNode<PointType>& node_in) {
            // Initailization
            if (!ptCloud_nodewise_ground_.empty()) ptCloud_nodewise_ground_.clear();
            
            // Tri Grid Initialization
            // When to initialize the planar model, we don't have prior. so outlier is removed in heuristic parameter.
            pcl::PointCloud<PointType> sort_ptCloud = node_in.ptCloud;
            // sort in z-coordinate
            sort(sort_ptCloud.points.begin(), sort_ptCloud.end(), point_z_cmp<PointType>);
            // Set init seeds
            extractInitialSeeds(sort_ptCloud, ptCloud_nodewise_ground_);

            // std::cout << "seeds: " <<  ptCloud_nodewise_ground_.size() << std::endl;
            Eigen::MatrixXf points(sort_ptCloud.points.size(),3);
            int j = 0;
            for (auto& p:sort_ptCloud.points){
                points.row(j++)<<p.x, p.y, p.z;
            }
            // Extract Ground
            for (int i =0; i < NUM_ITER_; i++){
                estimatePlanarModel(ptCloud_nodewise_ground_, node_in);
                if(ptCloud_nodewise_ground_.size() < 3){
    
                    node_in.node_type = NONGROUND;
                    break;
                }
                ptCloud_nodewise_ground_.clear();
                // threshold filter
                Eigen::VectorXf result = points*node_in.normal;
                for (int r = 0; r<result.rows(); r++){
                    if (i < NUM_ITER_-1){
                        if (result[r]<node_in.th_dist_d){
                            ptCloud_nodewise_ground_.push_back(sort_ptCloud.points[r]);
                        }
                    } else {
                        if (node_in.normal(2) < TH_NORMAL_ 
                        ){
                            node_in.node_type = NONGROUND;
                        } else {
                            node_in.node_type = GROUND;
                            // std::cout << "GROUND PCA: " << node_in.normal[2] << std::endl;
                        }
                    }
                }
            }

            return;
        }

        double calcNodeWeight(const TriGridNode<PointType>& node_in){
            double weight = 0;
            weight = (node_in.singular_values[0] + node_in.singular_values[1])*node_in.singular_values[1]/(node_in.singular_values[0]*node_in.singular_values[2]+0.001);
            return weight;
        }

        void RejectedGroundNodeRevert(TriGridField<PointType> & tgf_in) {
            for (int r_i = 0; r_i < rows_; r_i++) {
            for (int c_i = 0; c_i < cols_; c_i++) {
            for (int s_i = 0; s_i < (int) tgf_in[r_i][c_i].size(); s_i++) {
                if (tgf_in[r_i][c_i][s_i].node_type == NONGROUND && tgf_in[r_i][c_i][s_i].is_rejection) {
                    TriGridIdx current_node_idx = {r_i, c_i, s_i};
                    std::vector<TriGridIdx> neighbor_idxs;
                    searchNeighborNodes(current_node_idx, neighbor_idxs);
                    if (neighbor_idxs.size() == 0) continue;
                    double mean = 0.0, stdev = 0.0;
                    std::vector<double> centroid_height;
                    for (int i = 0; i < (int)neighbor_idxs.size(); i++) {
                        TriGridIdx n_i = neighbor_idxs[i];
                        if (tgf_in[n_i.row][n_i.col][n_i.tri].node_type == GROUND) {
                            centroid_height.push_back(tgf_in[n_i.row][n_i.col][n_i.tri].mean_pt[2]);
                            if (LocalConvecityConcavity(tgf_in, current_node_idx, n_i, TH_LCC_NORMAL_SIMILARITY_, TH_LCC_PLANAR_MODEL_DIST_)) {
                                tgf_in[r_i][c_i][s_i].is_processed = true;
                                tgf_in[r_i][c_i][s_i].node_type = GROUND;
                                tgf_in[r_i][c_i][s_i].depth +=1;
                                tgf_in[r_i][c_i][s_i].is_rejection = false;
                            }
                        }
                        else {
                            continue;
                        }
                    }
                    calc_mean_stdev(centroid_height, mean, stdev);
                    double min_bound = mean - stdev;
                    double max_bound = mean + stdev;
                    if (tgf_in[r_i][c_i][s_i].mean_pt[2] >= max_bound) {
                        tgf_in[r_i][c_i][s_i].is_processed = false;
                        tgf_in[r_i][c_i][s_i].node_type = NONGROUND;
                        tgf_in[r_i][c_i][s_i].depth = 0;
                        tgf_in[r_i][c_i][s_i].is_rejection = true;
                    }
                    else {
                        continue;
                    }
                    }
                }
                }
                }
            }
        
        void modelNodeWiseTerrain(TriGridField<PointType>& tgf_in) {
            // ROS_INFO("Node-wise Terrain Modeling...");
            for (int r_i = 0; r_i < rows_; r_i++){
            for (int c_i = 0; c_i < cols_; c_i++){
            for (int s_i = 0; s_i < 4; s_i++){
                if (tgf_in[r_i][c_i][s_i].is_curr_data){
                    if (tgf_in[r_i][c_i][s_i].ptCloud.size() < NUM_MIN_POINTS_){
                        tgf_in[r_i][c_i][s_i].node_type = UNKNOWN;
                        continue;                    
                    } 
                    else {
                        TriGridIdx cur_idx = {r_i,c_i,s_i};
                        double local_height = idx_seed_map[cur_idx];
                        // std::cout << "local height: " << local_height << std::endl;
                        // idx_seed_map.erase(cur_idx);
                        if(TNNR && local_height != 0) triangle_wise_reflected_noise_removal(tgf_in[r_i][c_i][s_i], local_height);
                        modelPCAbasedTerrain(tgf_in[r_i][c_i][s_i], cur_idx);
                        // if (tgf_in[r_i][c_i][s_i].node_type == GROUND){ 
                            tgf_in[r_i][c_i][s_i].weight = calcNodeWeight(tgf_in[r_i][c_i][s_i]);
                        // }
                    }
                }
            }
            }
            }
            return;
        };

        void MergedNodePlaneFitting(TriGridField<PointType>& tgf_in) {
            for (int r_i = 0; r_i < rows_; r_i++) {
            for (int c_i = 0; c_i < cols_; c_i++) {
            for (int s_i = 0; s_i < tgf_in[r_i][c_i].size(); s_i++) {
                if (!tgf_in[r_i][c_i][s_i].is_curr_data) {
                    continue;
                }
                if (tgf_in[r_i][c_i][s_i].node_type == NONGROUND && !tgf_in[r_i][c_i][s_i].mark
                && !tgf_in[r_i][c_i][s_i].is_rejection 
                ) {
                    TriGridNode<PointType> cell_in_unknown;
                    TriGridIdx current_node_idx = {r_i, c_i, s_i};
                    std::vector<TriGridIdx> neighbor_idxs;
                    searchNeighborNodes(current_node_idx, neighbor_idxs);
                    if (neighbor_idxs.size() == 0) continue;
                    std::vector<TriGridIdx> contributing_neighbor_triangles;  
                    cell_in_unknown.ptCloud += tgf_in[r_i][c_i][s_i].ptCloud;
                    double avr_height = 0;
                    int count = 0;
                    for (int i = 0; i < (int)neighbor_idxs.size(); i++) {
                        TriGridIdx n_i = neighbor_idxs[i];
                        if (tgf_in[n_i.row][n_i.col][n_i.tri].node_type == GROUND) {
                            avr_height += idx_seed_map[n_i];
                            count++;
                        }
                        else if (tgf_in[n_i.row][n_i.col][n_i.tri].node_type == NONGROUND && !tgf_in[n_i.row][n_i.col][n_i.tri].mark 
                        && !tgf_in[n_i.row][n_i.col][n_i.tri].is_rejection
                        ) {
                            if (abs(idx_seed_map[current_node_idx]- idx_seed_map[n_i]) <= TH_MERGING_) {
                                cell_in_unknown.ptCloud += tgf_in[n_i.row][n_i.col][n_i.tri].ptCloud;
                                contributing_neighbor_triangles.push_back(n_i);                      
                            }
                        }
                        else {
                            continue;
                        }
                    }
                    if (count == 0) {
                        continue;
                    }
                    if (contributing_neighbor_triangles.size() > 0) { 
                        modelPCAbasedTerrain(cell_in_unknown);
                    } else {
                        continue;
                    }
                    double seed_disparity = avr_height/count;
                    // std::cout << "avr_height: " << avr_height << std::endl;
                    // std::cout << "count: " << count << std::endl;
                    if (cell_in_unknown.node_type == GROUND) {
                        if (abs(seed_disparity - idx_seed_map[current_node_idx]) > TH_SEED_DISPARITY_) {
                            std::cout << "reject" << std::endl;
                            continue;
                        }
                        // std::cout << "avr height: " << seed_disparity << std::endl;
                        // std::cout << "cur height: " << idx_seed_map[current_node_idx] << std::endl;
                        tgf_in[r_i][c_i][s_i].mark = true;
                        tgf_in[r_i][c_i][s_i].depth = 0;
                        tgf_in[r_i][c_i][s_i].is_rejection = false;
                        tgf_in[r_i][c_i][s_i].normal = cell_in_unknown.normal; 
                        tgf_in[r_i][c_i][s_i].d = cell_in_unknown.d;
                        tgf_in[r_i][c_i][s_i].th_dist_d = cell_in_unknown.th_dist_d;
                        tgf_in[r_i][c_i][s_i].mean_pt = cell_in_unknown.mean_pt;
                        tgf_in[r_i][c_i][s_i].singular_values = cell_in_unknown.singular_values;
                        tgf_in[r_i][c_i][s_i].weight = calcNodeWeight(cell_in_unknown);
                        tgf_in[r_i][c_i][s_i].node_type = GROUND;
                        for (int i = 0; i < (int) contributing_neighbor_triangles.size(); i++) {
                            TriGridIdx n_i = contributing_neighbor_triangles[i];
                            tgf_in[n_i.row][n_i.col][n_i.tri].mark = true;   
                            tgf_in[n_i.row][n_i.col][n_i.tri].depth = 0;
                            tgf_in[n_i.row][n_i.col][n_i.tri].is_rejection = false;
                            tgf_in[n_i.row][n_i.col][n_i.tri].node_type = GROUND;       
                            tgf_in[n_i.row][n_i.col][n_i.tri].normal = cell_in_unknown.normal;             
                            tgf_in[n_i.row][n_i.col][n_i.tri].th_dist_d = cell_in_unknown.th_dist_d;        
                            tgf_in[n_i.row][n_i.col][n_i.tri].d = cell_in_unknown.d;            
                            tgf_in[n_i.row][n_i.col][n_i.tri].mean_pt = cell_in_unknown.mean_pt;      
                            tgf_in[n_i.row][n_i.col][n_i.tri].singular_values = cell_in_unknown.singular_values;   
                            tgf_in[n_i.row][n_i.col][n_i.tri].weight = calcNodeWeight(cell_in_unknown);   
                            }   
                        }
                    else {
                            continue;
                        }
                }
                    else {
                        continue;
                    }
                }
            }
        }
        return;
        }

        // void UnknownMerging(TriGridField<PointType>& tgf_in) {
        //     for (int r_i = 0; r_i < rows_; r_i++) {
        //     for (int c_i = 0; c_i < cols_; c_i++) {
        //     for (int s_i = 0; s_i < tgf_in[r_i][c_i].size(); s_i++) {
        //         if (!tgf_in[r_i][c_i][s_i].is_curr_data) {
        //             continue;
        //         }
        //         if (tgf_in[r_i][c_i][s_i].node_type == UNKNOWN && !tgf_in[r_i][c_i][s_i].mark
        //         ) {
        //             TriGridNode<PointType> cell_in_unknown;
        //             TriGridIdx current_node_idx = {r_i, c_i, s_i};
        //             std::vector<TriGridIdx> neighbor_idxs;
        //             searchNeighborNodes(current_node_idx, neighbor_idxs);
        //             if (neighbor_idxs.size() == 0) continue;
        //             std::vector<TriGridIdx> contributing_neighbor_triangles;  
        //             cell_in_unknown.ptCloud += tgf_in[r_i][c_i][s_i].ptCloud;
        //             for (int i = 0; i < (int)neighbor_idxs.size(); i++) {
        //                 TriGridIdx n_i = neighbor_idxs[i];
        //                 if (tgf_in[n_i.row][n_i.col][n_i.tri].node_type == UNKNOWN && !tgf_in[n_i.row][n_i.col][n_i.tri].mark
        //                 ) {
        //                     cell_in_unknown.ptCloud += tgf_in[n_i.row][n_i.col][n_i.tri].ptCloud;
        //                     contributing_neighbor_triangles.push_back(n_i);                      
        //                 }
        //                 else {
        //                     continue;
        //                 }
        //             }
        //             if (contributing_neighbor_triangles.size() > 0 && cell_in_unknown.ptCloud.size() > NUM_MIN_POINTS_) { 
        //                 modelPCAbasedTerrain(cell_in_unknown);
        //             } else {
        //                 continue;
        //             }
        //             if (cell_in_unknown.node_type == GROUND) {
        //                 tgf_in[r_i][c_i][s_i].mark = true;
        //                 tgf_in[r_i][c_i][s_i].depth = 0;
        //                 tgf_in[r_i][c_i][s_i].is_rejection = false;
        //                 tgf_in[r_i][c_i][s_i].normal = cell_in_unknown.normal; 
        //                 tgf_in[r_i][c_i][s_i].d = cell_in_unknown.d;
        //                 tgf_in[r_i][c_i][s_i].th_dist_d = cell_in_unknown.th_dist_d;
        //                 tgf_in[r_i][c_i][s_i].mean_pt = cell_in_unknown.mean_pt;
        //                 tgf_in[r_i][c_i][s_i].singular_values = cell_in_unknown.singular_values;
        //                 tgf_in[r_i][c_i][s_i].weight = calcNodeWeight(cell_in_unknown);
        //                 tgf_in[r_i][c_i][s_i].node_type = GROUND;
        //                 for (int i = 0; i < (int) contributing_neighbor_triangles.size(); i++) {
        //                     TriGridIdx n_i = contributing_neighbor_triangles[i];
        //                     tgf_in[n_i.row][n_i.col][n_i.tri].mark = true;   
        //                     tgf_in[n_i.row][n_i.col][n_i.tri].depth = 0;
        //                     tgf_in[n_i.row][n_i.col][n_i.tri].is_rejection = false;
        //                     tgf_in[n_i.row][n_i.col][n_i.tri].node_type = GROUND;       
        //                     tgf_in[n_i.row][n_i.col][n_i.tri].normal = cell_in_unknown.normal;             
        //                     tgf_in[n_i.row][n_i.col][n_i.tri].th_dist_d = cell_in_unknown.th_dist_d;        
        //                     tgf_in[n_i.row][n_i.col][n_i.tri].d = cell_in_unknown.d;            
        //                     tgf_in[n_i.row][n_i.col][n_i.tri].mean_pt = cell_in_unknown.mean_pt;      
        //                     tgf_in[n_i.row][n_i.col][n_i.tri].singular_values = cell_in_unknown.singular_values;   
        //                     tgf_in[n_i.row][n_i.col][n_i.tri].weight = calcNodeWeight(cell_in_unknown);   
        //                     }   
        //                 }
        //             else {
        //                     continue;
        //                 }
        //         }
        //             else {
        //                 continue;
        //             }
        //         }
        //     }
        // }
        // return;
        // }


        void findDominantNode(TriGridField<PointType>& tgf_in, TriGridIdx& node_idx_out) {
            // Find the dominant node
            // ROS_INFO("Find the dominant node...");
            TriGridIdx max_tri_idx;
            TriGridIdx ego_idx;
            ego_idx.row = (int)((0-tgf_min_x)/TGF_RESOLUTION_);
            ego_idx.col = (int)((0-tgf_min_y)/TGF_RESOLUTION_);
            ego_idx.tri = 0;
            
            max_tri_idx = ego_idx;

            double weight_sum = 0.0;
            double total_weight = 0.0;

            pcl::PointCloud<PointType> combined_cloud;
            for (int r_i = ego_idx.row - 2; r_i < ego_idx.row + 2; r_i++){  //3x3 grid
            for (int c_i = ego_idx.col - 2; c_i < ego_idx.col + 2; c_i++){
            for (int s_i = 0; s_i < 4; s_i++){
                if (tgf_in[r_i][c_i][s_i].is_curr_data){
                    if (tgf_in[r_i][c_i][s_i].node_type == GROUND){
                        double xy_range = xy_2Dradius(tgf_in[r_i][c_i][s_i].mean_pt[0], tgf_in[r_i][c_i][s_i].mean_pt[1]);
                        double linearity = (tgf_in[r_i][c_i][s_i].singular_values(0) - tgf_in[r_i][c_i][s_i].singular_values(1))/ tgf_in[r_i][c_i][s_i].singular_values(0);
                        // // std::cout << "xy range: " << xy_range << std::endl;
                        if (xy_range < 10 && xy_range > MIN_RANGE_ && linearity < 0.9) {
                            double weight = tgf_in[r_i][c_i][s_i].weight;
                            double mean_height = tgf_in[r_i][c_i][s_i].mean_pt[2];
                            weight_sum += mean_height * weight;
                            total_weight += weight;
                        }
                        // combined_cloud +=  tgf_in[r_i][c_i][s_i].ptCloud;
                        // combined_cloud.points.insert(combined_cloud.points.end(),
                        //      tgf_in[r_i][c_i][s_i].ptCloud.points.begin(),
                        //      tgf_in[r_i][c_i][s_i].ptCloud.points.end());  
                        // pub_dominant_cloud.publish(convertCloudToRosMsg(const_cast<pcl::PointCloud<PointType>&>(*(&tgf_in[r_i][c_i][s_i].ptCloud)), cloud_header_.frame_id));

                        if (tgf_in[r_i][c_i][s_i].weight > tgf_in[max_tri_idx.row][max_tri_idx.col][max_tri_idx.tri].weight 
                            // && SENSOR_HEIGHT_ + tgf_in[max_tri_idx.row][max_tri_idx.col][max_tri_idx.tri].mean_pt[2] < 0.3
                        ){
                            max_tri_idx.row = r_i;
                            max_tri_idx.col = c_i;
                            max_tri_idx.tri = s_i;
                        // for (const auto& point : tgf_in[max_tri_idx.row][max_tri_idx.col][max_tri_idx.tri].ptCloud.points) {
                        //                             sum_z += point.z;
                        //                         }
                        // point_count += tgf_in[max_tri_idx.row][max_tri_idx.col][max_tri_idx.tri].ptCloud.points.size();
                        // combined_cloud.points.insert(combined_cloud.points.end(),
                        //     tgf_in[max_tri_idx.row][max_tri_idx.col][max_tri_idx.tri].ptCloud.points.begin(),
                        //     tgf_in[max_tri_idx.row][max_tri_idx.col][max_tri_idx.tri].ptCloud.points.end());  
                            // tgf_in[max_tri_idx.row][max_tri_idx.col][max_tri_idx.tri].header = cloud_header_;
                            // const pcl::PointCloud<PointType>& cloud = tgf_in[max_tri_idx.row][max_tri_idx.col][max_tri_idx.tri].ptCloud;
                            // pub_dominant_cloud.publish(convertCloudToRosMsg(const_cast<pcl::PointCloud<PointType>&>(*(&tgf_in[max_tri_idx.row][max_tri_idx.col][max_tri_idx.tri].ptCloud)), cloud_header_.frame_id));
                        }
                    }
                }
            }
            }    
            }

            node_idx_out = max_tri_idx;

            // pub_dominant_cloud.publish(convertCloudToRosMsg(const_cast<pcl::PointCloud<PointType>&>(*(&tgf_in[max_tri_idx.row][max_tri_idx.col][max_tri_idx.tri].ptCloud)), cloud_header_.frame_id));
            // std::cout << "dominant height: " << tgf_in[max_tri_idx.row][max_tri_idx.col][max_tri_idx.tri].mean_pt[2] << std::endl;

            if (total_weight > 0) {
                SENSOR_HEIGHT_  = -weight_sum / total_weight;
            }
            else {
                ROS_WARN("No ground points found in tri grid for estimating height");
            }
            std::cout << "========================" << std::endl;
            std::cout << "SENSOR HEIGHT: " << SENSOR_HEIGHT_ << std::endl;
            // pub_dominant_cloud.publish(convertCloudToRosMsg(const_cast<pcl::PointCloud<PointType>&>(*(&tgf_in[max_tri_idx.row][max_tri_idx.col][max_tri_idx.tri].ptCloud)), cloud_header_.frame_id));  
            return;
        };

        void calcSensorHeight(TriGridField<PointType>& tgf_in) {
            TriGridIdx ego_idx;
            ego_idx.row = (int)((0-tgf_min_x)/TGF_RESOLUTION_);
            ego_idx.col = (int)((0-tgf_min_y)/TGF_RESOLUTION_);
            ego_idx.tri = 0;
            double weight_sum = 0.0;
            double total_weight = 0.0;
            for (int r_i = ego_idx.row - 2; r_i < ego_idx.row + 2; r_i++){  //3x3 grid
            for (int c_i = ego_idx.col - 2; c_i < ego_idx.col + 2; c_i++){
            for (int s_i = 0; s_i < 4; s_i++){
                if (tgf_in[r_i][c_i][s_i].is_curr_data){
                    if (tgf_in[r_i][c_i][s_i].node_type == GROUND){
                        double xy_range = xy_2Dradius(tgf_in[r_i][c_i][s_i].mean_pt[0], tgf_in[r_i][c_i][s_i].mean_pt[1]);
                        double linearity = (tgf_in[r_i][c_i][s_i].singular_values(0) - tgf_in[r_i][c_i][s_i].singular_values(1))/ tgf_in[r_i][c_i][s_i].singular_values(0);
                        if (xy_range < 10 && xy_range > MIN_RANGE_ && linearity < 0.9) {
                            double weight = tgf_in[r_i][c_i][s_i].weight;
                            double mean_height = tgf_in[r_i][c_i][s_i].mean_pt[2];
                            weight_sum += mean_height * weight;
                            total_weight += weight;
                        }
                    }
                }
            }
            }    
            }

            if (total_weight > 0) {
                SENSOR_HEIGHT_  = -weight_sum / total_weight;
            }
            else {
                ROS_WARN("No ground points found in tri grid for estimating height");
            }
            std::cout << "SENSOR HEIGHT: " << SENSOR_HEIGHT_ << std::endl;
            return;
        };

        void searchNeighborNodes(const TriGridIdx &cur_idx, std::vector<TriGridIdx> &neighbor_idxs) {
            neighbor_idxs.clear();
            neighbor_idxs.reserve(14);
            int r_i = cur_idx.row;
            int c_i = cur_idx.col;
            int s_i = cur_idx.tri;

            std::vector<TriGridIdx> tmp_neighbors;
            tmp_neighbors.clear();
            tmp_neighbors.reserve(14);
            
            TriGridIdx neighbor_idx;
            for (int s_i = 0; s_i < 4 ; s_i++){
                if (s_i == cur_idx.tri) continue;

                neighbor_idx = cur_idx;
                neighbor_idx.tri = s_i;
                tmp_neighbors.push_back(neighbor_idx);
            }

            switch (s_i) {
                case 0:
                    tmp_neighbors.push_back({r_i+1, c_i+1, 2});
                    tmp_neighbors.push_back({r_i+1, c_i+1, 3});
                    tmp_neighbors.push_back({r_i+1, c_i  , 1});
                    tmp_neighbors.push_back({r_i+1, c_i  , 2});
                    tmp_neighbors.push_back({r_i+1, c_i  , 3});
                    tmp_neighbors.push_back({r_i+1, c_i-1, 1});
                    tmp_neighbors.push_back({r_i+1, c_i-1, 2});
                    tmp_neighbors.push_back({r_i  , c_i+1, 0});
                    tmp_neighbors.push_back({r_i  , c_i+1, 3});
                    tmp_neighbors.push_back({r_i  , c_i-1, 0});
                    tmp_neighbors.push_back({r_i  , c_i-1, 1});
                    break;
                case 1:
                    tmp_neighbors.push_back({r_i+1, c_i+1, 2});
                    tmp_neighbors.push_back({r_i+1, c_i+1, 3});
                    tmp_neighbors.push_back({r_i+1, c_i  , 1});
                    tmp_neighbors.push_back({r_i+1, c_i  , 2});
                    tmp_neighbors.push_back({r_i  , c_i+1, 0});
                    tmp_neighbors.push_back({r_i  , c_i+1, 2});
                    tmp_neighbors.push_back({r_i  , c_i+1, 3});
                    tmp_neighbors.push_back({r_i-1, c_i+1, 0});
                    tmp_neighbors.push_back({r_i-1, c_i+1, 3});
                    tmp_neighbors.push_back({r_i-1, c_i+1, 0});
                    tmp_neighbors.push_back({r_i-1, c_i  , 1});
                    break;
                case 2:
                    tmp_neighbors.push_back({r_i  , c_i+1, 2});
                    tmp_neighbors.push_back({r_i  , c_i+1, 3});
                    tmp_neighbors.push_back({r_i  , c_i-1, 1});
                    tmp_neighbors.push_back({r_i  , c_i-1, 2});
                    tmp_neighbors.push_back({r_i-1, c_i+1, 0});
                    tmp_neighbors.push_back({r_i-1, c_i+1, 3});
                    tmp_neighbors.push_back({r_i-1, c_i  , 0});
                    tmp_neighbors.push_back({r_i-1, c_i  , 1});
                    tmp_neighbors.push_back({r_i-1, c_i  , 3});
                    tmp_neighbors.push_back({r_i-1, c_i-1, 0});
                    tmp_neighbors.push_back({r_i-1, c_i-1, 1});
                    break;
                case 3:
                    tmp_neighbors.push_back({r_i+1, c_i  , 2});
                    tmp_neighbors.push_back({r_i+1, c_i  , 3});
                    tmp_neighbors.push_back({r_i+1, c_i-1, 1});
                    tmp_neighbors.push_back({r_i+1, c_i-1, 2});
                    tmp_neighbors.push_back({r_i  , c_i-1, 0});
                    tmp_neighbors.push_back({r_i  , c_i-1, 1});
                    tmp_neighbors.push_back({r_i  , c_i-1, 2});
                    tmp_neighbors.push_back({r_i-1, c_i  , 0});
                    tmp_neighbors.push_back({r_i-1, c_i  , 3});
                    tmp_neighbors.push_back({r_i-1, c_i-1, 0});
                    tmp_neighbors.push_back({r_i-1, c_i-1, 1});
                    break;
                default:
                    break;
            }

            for (int n_i = 0 ; n_i < (int) tmp_neighbors.size() ; n_i++) {

                if (tmp_neighbors[n_i].row >= rows_ || tmp_neighbors[n_i].row < 0) {
                    continue;
                }

                if (tmp_neighbors[n_i].col >= cols_ || tmp_neighbors[n_i].col < 0) {
                    continue;
                }

                neighbor_idxs.push_back(tmp_neighbors[n_i]);
            }
        }

        void searchNeighborNodes_new(const TriGridIdx &cur_idx, std::vector<TriGridIdx> &neighbor_idxs) {
            neighbor_idxs.clear();
            neighbor_idxs.reserve(23);
            int r_i = cur_idx.row;
            int c_i = cur_idx.col;
            int s_i = cur_idx.tri;

            std::vector<TriGridIdx> tmp_neighbors;
            tmp_neighbors.clear();
            tmp_neighbors.reserve(23);
            
            TriGridIdx neighbor_idx;
            for (int s_i = 0; s_i < 4 ; s_i++){
                if (s_i == cur_idx.tri) continue;

                neighbor_idx = cur_idx;
                neighbor_idx.tri = s_i;
                tmp_neighbors.push_back(neighbor_idx);
            }

            switch (s_i) {
                case 0:
                    tmp_neighbors.push_back({r_i+1, c_i+1, 0});
                    tmp_neighbors.push_back({r_i+1, c_i+1, 1});
                    tmp_neighbors.push_back({r_i+1, c_i+1, 2});
                    tmp_neighbors.push_back({r_i+1, c_i+1, 3});
                    tmp_neighbors.push_back({r_i+1, c_i  , 0});
                    tmp_neighbors.push_back({r_i+1, c_i  , 1});
                    tmp_neighbors.push_back({r_i+1, c_i  , 2});
                    tmp_neighbors.push_back({r_i+1, c_i  , 3});
                    tmp_neighbors.push_back({r_i+1, c_i-1, 0});
                    tmp_neighbors.push_back({r_i+1, c_i-1, 1});
                    tmp_neighbors.push_back({r_i+1, c_i-1, 2});
                    tmp_neighbors.push_back({r_i+1, c_i-1, 3});
                    tmp_neighbors.push_back({r_i  , c_i+1, 0});
                    tmp_neighbors.push_back({r_i  , c_i+1, 1});
                    tmp_neighbors.push_back({r_i  , c_i+1, 2});
                    tmp_neighbors.push_back({r_i  , c_i+1, 3});
                    tmp_neighbors.push_back({r_i  , c_i-1, 0});
                    tmp_neighbors.push_back({r_i  , c_i-1, 1});
                    tmp_neighbors.push_back({r_i  , c_i-1, 2});
                    tmp_neighbors.push_back({r_i  , c_i-1, 3});
                    break;
                case 1:
                    tmp_neighbors.push_back({r_i+1, c_i+1, 0});
                    tmp_neighbors.push_back({r_i+1, c_i+1, 1});
                    tmp_neighbors.push_back({r_i+1, c_i+1, 2});
                    tmp_neighbors.push_back({r_i+1, c_i+1, 3});
                    tmp_neighbors.push_back({r_i+1, c_i  , 0});
                    tmp_neighbors.push_back({r_i+1, c_i  , 1});
                    tmp_neighbors.push_back({r_i+1, c_i  , 2});
                    tmp_neighbors.push_back({r_i+1, c_i  , 3});
                    tmp_neighbors.push_back({r_i+1, c_i-1, 0});
                    tmp_neighbors.push_back({r_i+1, c_i-1, 1});
                    tmp_neighbors.push_back({r_i+1, c_i-1, 2});
                    tmp_neighbors.push_back({r_i+1, c_i-1, 3});
                    tmp_neighbors.push_back({r_i  , c_i+1, 0});
                    tmp_neighbors.push_back({r_i  , c_i+1, 1});
                    tmp_neighbors.push_back({r_i  , c_i+1, 2});
                    tmp_neighbors.push_back({r_i  , c_i+1, 3});
                    tmp_neighbors.push_back({r_i  , c_i-1, 0});
                    tmp_neighbors.push_back({r_i  , c_i-1, 1});
                    tmp_neighbors.push_back({r_i  , c_i-1, 2});
                    tmp_neighbors.push_back({r_i  , c_i-1, 3});
                    break;
                case 2:
                    tmp_neighbors.push_back({r_i+1, c_i+1, 0});
                    tmp_neighbors.push_back({r_i+1, c_i+1, 1});
                    tmp_neighbors.push_back({r_i+1, c_i+1, 2});
                    tmp_neighbors.push_back({r_i+1, c_i+1, 3});
                    tmp_neighbors.push_back({r_i+1, c_i  , 0});
                    tmp_neighbors.push_back({r_i+1, c_i  , 1});
                    tmp_neighbors.push_back({r_i+1, c_i  , 2});
                    tmp_neighbors.push_back({r_i+1, c_i  , 3});
                    tmp_neighbors.push_back({r_i+1, c_i-1, 0});
                    tmp_neighbors.push_back({r_i+1, c_i-1, 1});
                    tmp_neighbors.push_back({r_i+1, c_i-1, 2});
                    tmp_neighbors.push_back({r_i+1, c_i-1, 3});
                    tmp_neighbors.push_back({r_i  , c_i+1, 0});
                    tmp_neighbors.push_back({r_i  , c_i+1, 1});
                    tmp_neighbors.push_back({r_i  , c_i+1, 2});
                    tmp_neighbors.push_back({r_i  , c_i+1, 3});
                    tmp_neighbors.push_back({r_i  , c_i-1, 0});
                    tmp_neighbors.push_back({r_i  , c_i-1, 1});
                    tmp_neighbors.push_back({r_i  , c_i-1, 2});
                    tmp_neighbors.push_back({r_i  , c_i-1, 3});
                    break;
                case 3:
                    tmp_neighbors.push_back({r_i+1, c_i+1, 0});
                    tmp_neighbors.push_back({r_i+1, c_i+1, 1});
                    tmp_neighbors.push_back({r_i+1, c_i+1, 2});
                    tmp_neighbors.push_back({r_i+1, c_i+1, 3});
                    tmp_neighbors.push_back({r_i+1, c_i  , 0});
                    tmp_neighbors.push_back({r_i+1, c_i  , 1});
                    tmp_neighbors.push_back({r_i+1, c_i  , 2});
                    tmp_neighbors.push_back({r_i+1, c_i  , 3});
                    tmp_neighbors.push_back({r_i+1, c_i-1, 0});
                    tmp_neighbors.push_back({r_i+1, c_i-1, 1});
                    tmp_neighbors.push_back({r_i+1, c_i-1, 2});
                    tmp_neighbors.push_back({r_i+1, c_i-1, 3});
                    tmp_neighbors.push_back({r_i  , c_i+1, 0});
                    tmp_neighbors.push_back({r_i  , c_i+1, 1});
                    tmp_neighbors.push_back({r_i  , c_i+1, 2});
                    tmp_neighbors.push_back({r_i  , c_i+1, 3});
                    tmp_neighbors.push_back({r_i  , c_i-1, 0});
                    tmp_neighbors.push_back({r_i  , c_i-1, 1});
                    tmp_neighbors.push_back({r_i  , c_i-1, 2});
                    tmp_neighbors.push_back({r_i  , c_i-1, 3});
                    break;
                default:
                    break;
            }

            for (int n_i = 0 ; n_i < (int) tmp_neighbors.size() ; n_i++) {

                if (tmp_neighbors[n_i].row >= rows_ || tmp_neighbors[n_i].row < 0) {
                    continue;
                }

                if (tmp_neighbors[n_i].col >= cols_ || tmp_neighbors[n_i].col < 0) {
                    continue;
                }

                neighbor_idxs.push_back(tmp_neighbors[n_i]);
            }
        }

        void searchNeighborNodesNear(const TriGridIdx &cur_idx, std::vector<TriGridIdx> &neighbor_idxs) {
            neighbor_idxs.clear();
            neighbor_idxs.reserve(4);
            int r_i = cur_idx.row;
            int c_i = cur_idx.col;
            int s_i = cur_idx.tri;

            std::vector<TriGridIdx> tmp_neighbors;
            tmp_neighbors.clear();
            tmp_neighbors.reserve(4);
            
            TriGridIdx neighbor_idx;
            for (int s_i = 0; s_i < 4 ; s_i++){
                if (s_i == cur_idx.tri) continue;

                neighbor_idx = cur_idx;
                neighbor_idx.tri = s_i;
                tmp_neighbors.push_back(neighbor_idx);
            }
            switch (s_i) {
                case 0:
                    tmp_neighbors.push_back({r_i+1, c_i, 2});
                    break;
                case 1:
                    tmp_neighbors.push_back({r_i, c_i+1, 3});
                    break;
                case 2:
                    tmp_neighbors.push_back({r_i-1, c_i, 0});
                    break;
                case 3:
                    tmp_neighbors.push_back({r_i, c_i-1, 1});
                    break;
                default:
                    break;
            }
            for (int n_i = 0 ; n_i < (int) tmp_neighbors.size() ; n_i++) {

                if (tmp_neighbors[n_i].row >= rows_ || tmp_neighbors[n_i].row < 0) {
                    continue;
                }

                if (tmp_neighbors[n_i].col >= cols_ || tmp_neighbors[n_i].col < 0) {
                    continue;
                }

                neighbor_idxs.push_back(tmp_neighbors[n_i]);
            }
        }

        void searchNeighborNodesMid(const TriGridIdx &cur_idx, std::vector<TriGridIdx> &neighbor_idxs) {
            neighbor_idxs.clear();
            neighbor_idxs.reserve(3);
            int r_i = cur_idx.row;
            int c_i = cur_idx.col;
            int s_i = cur_idx.tri;

            std::vector<TriGridIdx> tmp_neighbors;
            tmp_neighbors.clear();
            tmp_neighbors.reserve(3);
            
            TriGridIdx neighbor_idx;
            for (int s_i = 0; s_i < 4 ; s_i++){
                if (s_i == cur_idx.tri) continue;

                neighbor_idx = cur_idx;
                neighbor_idx.tri = s_i;
                tmp_neighbors.push_back(neighbor_idx);
            }

            for (int n_i = 0 ; n_i < (int) tmp_neighbors.size() ; n_i++) {

                if (tmp_neighbors[n_i].row >= rows_ || tmp_neighbors[n_i].row < 0) {
                    continue;
                }

                if (tmp_neighbors[n_i].col >= cols_ || tmp_neighbors[n_i].col < 0) {
                    continue;
                }

                neighbor_idxs.push_back(tmp_neighbors[n_i]);
            }
        }


        void searchAdjacentNodes(const TriGridIdx &cur_idx, std::vector<TriGridIdx> &adjacent_idxs) {
            adjacent_idxs.clear();
            adjacent_idxs.reserve(3);
            int r_i = cur_idx.row;
            int c_i = cur_idx.col;
            int s_i = cur_idx.tri;

            std::vector<TriGridIdx> tmp_neighbors;
            tmp_neighbors.clear();
            tmp_neighbors.reserve(3);
            
            TriGridIdx neighbor_idx;

            switch (s_i) {
                case 0:
                    // tmp_neighbors.push_back({r_i+1, c_i, 2});
                    tmp_neighbors.push_back({r_i  , c_i, 1});
                    tmp_neighbors.push_back({r_i  , c_i, 2});
                    tmp_neighbors.push_back({r_i  , c_i, 3});
                    break;
                case 1:
                    // tmp_neighbors.push_back({r_i, c_i+1, 3});
                    tmp_neighbors.push_back({r_i, c_i  , 0});
                    tmp_neighbors.push_back({r_i, c_i  , 2});     
                    tmp_neighbors.push_back({r_i, c_i  , 3});                  
                    break;
                case 2:
                    // tmp_neighbors.push_back({r_i-1, c_i, 0});
                    tmp_neighbors.push_back({r_i, c_i  , 0});
                    tmp_neighbors.push_back({r_i  , c_i, 1});
                    tmp_neighbors.push_back({r_i  , c_i, 3});
                    break;
                case 3:
                    // tmp_neighbors.push_back({r_i, c_i-1, 1});
                    tmp_neighbors.push_back({r_i, c_i  , 2});
                    tmp_neighbors.push_back({r_i, c_i  , 0});
                    tmp_neighbors.push_back({r_i  , c_i, 1});
                    break;
                default:
                    break;
            }

            for (int n_i = 0 ; n_i < (int) tmp_neighbors.size() ; n_i++) {

                if (tmp_neighbors[n_i].row >= rows_ || tmp_neighbors[n_i].row < 0) {
                    continue;
                }

                if (tmp_neighbors[n_i].col >= cols_ || tmp_neighbors[n_i].col < 0) {
                    continue;
                }

                adjacent_idxs.push_back(tmp_neighbors[n_i]);
            }
        }

        bool LocalConvecityConcavity(const TriGridField<PointType> &tgf, const TriGridIdx &cur_node_idx, const TriGridIdx &neighbor_idx, 
                                    double & thr_local_normal, double & thr_local_dist) {
            TriGridNode<PointType> current_node = tgf[cur_node_idx.row][cur_node_idx.col][cur_node_idx.tri];
            TriGridNode<PointType> neighbor_node = tgf[neighbor_idx.row][neighbor_idx.col][neighbor_idx.tri];

            Eigen::Vector3f normal_src = current_node.normal; // normal current
            Eigen::Vector3f normal_tgt = neighbor_node.normal;  //normal neighbor
            Eigen::Vector3f meanPt_diff_s2t = neighbor_node.mean_pt - current_node.mean_pt; //different of mean btw neighbor and current

            double diff_norm = meanPt_diff_s2t.norm();  // norm of diff of mean
            double dist_s2t = normal_src.dot(meanPt_diff_s2t); // dist from current plane to neighbor plane
            double dist_t2s = normal_tgt.dot(-meanPt_diff_s2t); // reverse

            double normal_similarity = normal_src.dot(normal_tgt); // normal similarity
            double TH_NORMAL_cos_similarity = sin(diff_norm*thr_local_normal);

            if ((normal_similarity < (1-TH_NORMAL_cos_similarity))) {
                return false;
            }

            double TH_DIST_to_planar = diff_norm*sin(thr_local_dist);
            if ( (abs(dist_s2t) > TH_DIST_to_planar || abs(dist_t2s) > TH_DIST_to_planar) ) {
                return false;
            }
            return true;
        }

        void calc_mean_stdev(std::vector<double> vec, double &mean, double &stdev)
        {
            if (vec.size() <= 1) return;

            mean = std::accumulate(vec.begin(), vec.end(), 0.0) / vec.size();

            for (int i=0; i<vec.size(); i++) { stdev += (vec.at(i)-mean)*(vec.at(i)-mean); }
            stdev /= vec.size()-1;
            stdev = sqrt(stdev);
        }

        void BreadthFirstTraversableGraphSearch(TriGridField<PointType>& tgf_in) {

            // Find the dominant node
            std::queue<TriGridIdx> searching_idx_queue;
            std::queue<TriGridIdx> searching_new_dominant;
            TriGridIdx dominant_node_idx;
            findDominantNode(tgf_in, dominant_node_idx);
            tgf_in[dominant_node_idx.row][dominant_node_idx.col][dominant_node_idx.tri].is_visited = true;
            tgf_in[dominant_node_idx.row][dominant_node_idx.col][dominant_node_idx.tri].depth = 0;
            tgf_in[dominant_node_idx.row][dominant_node_idx.col][dominant_node_idx.tri].node_type = GROUND;

            searching_idx_queue.push(dominant_node_idx);

            double max_planar_height = 0;
            trigrid_edges_.clear();
            trigrid_edges_.reserve(rows_*cols_*4);
            TriGridEdge cur_edge;
            TriGridIdx current_node_idx;
            pcl::PointCloud<PointType> new_dominant;
            while (!searching_idx_queue.empty()){
                // set current node
                current_node_idx = searching_idx_queue.front();
                searching_idx_queue.pop();

                // search the neighbor nodes
                std::vector<TriGridIdx> neighbor_idxs;
                searchNeighborNodes(current_node_idx, neighbor_idxs);

                // set the traversable edges
                for (int i = 0; i < (int) neighbor_idxs.size(); i++){
                    // if the neighbor node is traversable, add it to the queue

                    TriGridIdx n_i = neighbor_idxs[i];


                    if (tgf_in[n_i.row][n_i.col][n_i.tri].depth >=0){
                        continue;
                    }

                    if (tgf_in[n_i.row][n_i.col][n_i.tri].is_visited  && !tgf_in[n_i.row][n_i.col][n_i.tri].mark) {
                        if (!tgf_in[n_i.row][n_i.col][n_i.tri].need_recheck){
                            continue;
                        } else {
                            if (tgf_in[n_i.row][n_i.col][n_i.tri].check_life <= 0){
                                continue;
                            }
                        }
                        continue;
                    } else {
                        if (tgf_in[n_i.row][n_i.col][n_i.tri].node_type != GROUND) {
                        continue;
                        }
                    }

                    tgf_in[n_i.row][n_i.col][n_i.tri].is_visited =true;
                    if (!LocalConvecityConcavity(tgf_in, current_node_idx, n_i, TH_LCC_NORMAL_SIMILARITY_, TH_LCC_PLANAR_MODEL_DIST_)) {
                        tgf_in[n_i.row][n_i.col][n_i.tri].is_rejection = true;
                        tgf_in[n_i.row][n_i.col][n_i.tri].node_type = NONGROUND;
                        if(tgf_in[n_i.row][n_i.col][n_i.tri].check_life > 0) {
                            tgf_in[n_i.row][n_i.col][n_i.tri].check_life -=1;
                            tgf_in[n_i.row][n_i.col][n_i.tri].need_recheck = true;
                        } 
                        else {
                            tgf_in[n_i.row][n_i.col][n_i.tri].need_recheck = false;
                        }
                        continue;
                    }
                    // if (tgf_in[n_i.row][n_i.col][n_i.tri].check_life < 15) 
                    // {
                    //     std::cout << "check life: " << tgf_in[n_i.row][n_i.col][n_i.tri].check_life << " weight: " << tgf_in[n_i.row][n_i.col][n_i.tri].weight << std::endl;
                    // }
                    // if (max_planar_height < tgf_in[n_i.row][n_i.col][n_i.tri].mean_pt[2]) max_planar_height = tgf_in[n_i.row][n_i.col][n_i.tri].mean_pt[2];
                    tgf_in[n_i.row][n_i.col][n_i.tri].node_type = GROUND;
                    tgf_in[n_i.row][n_i.col][n_i.tri].is_rejection = false;
                    tgf_in[n_i.row][n_i.col][n_i.tri].depth = tgf_in[current_node_idx.row][current_node_idx.col][current_node_idx.tri].depth + 1;

                    if (VIZ_MDOE_){
                        cur_edge.Pair.first = current_node_idx;
                        cur_edge.Pair.second = n_i;
                        cur_edge.is_traversable = true;
                        trigrid_edges_.push_back(cur_edge);
                    }

                    searching_idx_queue.push(n_i);
                }
                if (searching_idx_queue.empty()){
                    // set the new dominant node
                    for (int r_i = 0; r_i < rows_; r_i++) {
                    for (int c_i = 0; c_i < cols_; c_i++) {
                    for (int s_i = 0; s_i < (int) tgf_in[r_i][c_i].size() ; s_i++){
                        if (tgf_in[r_i][c_i][s_i].is_visited) { continue; }

                        if (tgf_in[r_i][c_i][s_i].node_type != GROUND ) { continue; }

                        if (tgf_in[r_i][c_i][s_i].depth >= 0) { continue; }

                        // if (tgf_in[r_i][c_i][s_i].weight > TH_WEIGHT_){
                            tgf_in[r_i][c_i][s_i].depth = 0;
                            tgf_in[r_i][c_i][s_i].is_visited = true;
                            TriGridIdx new_dominant_idx = {r_i, c_i, s_i};
                            new_dominant +=  tgf_in[r_i][c_i][s_i].ptCloud;
                            searching_idx_queue.push(new_dominant_idx);
                            searching_new_dominant.push(new_dominant_idx);
                        // }
                    }
                    }
                    }
                }
            }
            std::cout << "Number of points in new dominant: " << new_dominant.size() << std::endl;
            pub_dominant_cloud.publish(convertCloudToRosMsg(new_dominant, cloud_header_.frame_id));
            std::cout << "Number of new dominant: " << searching_new_dominant.size() << std::endl;
            return;
        };


        double getCornerWeight(const TriGridNode<PointType>& node_in, const pcl::PointXYZ &tgt_corner){
            double xy_dist = sqrt( (node_in.mean_pt[0]-tgt_corner.x)*(node_in.mean_pt[0]-tgt_corner.x)+(node_in.mean_pt[1]-tgt_corner.y)*(node_in.mean_pt[1]-tgt_corner.y) );
            return (node_in.weight/xy_dist);
        }

        void setTGFCornersCenters(TriGridField<PointType>& tgf_in,
                                std::vector<std::vector<TriGridCorner>>& trigrid_corners_out,
                                std::vector<std::vector<TriGridCorner>>& trigrid_centers_out) {
            pcl::PointXYZ corner_TL, corner_BL, corner_BR, corner_TR, corner_C;

            for (int r_i = 0; r_i<rows_; r_i++){
            for (int c_i = 0; c_i<cols_; c_i++){
                corner_TL.x = trigrid_corners_out[r_i+1][c_i+1].x; corner_TL.y = trigrid_corners_out[r_i+1][c_i+1].y;   // LT
                corner_BL.x = trigrid_corners_out[r_i][c_i+1].x;   corner_BL.y = trigrid_corners_out[r_i][c_i+1].y;     // LL
                corner_BR.x = trigrid_corners_out[r_i][c_i].x;     corner_BR.y = trigrid_corners_out[r_i][c_i].y;       // RL
                corner_TR.x = trigrid_corners_out[r_i+1][c_i].x;   corner_TR.y = trigrid_corners_out[r_i+1][c_i].y;     // RT
                corner_C.x = trigrid_centers_out[r_i][c_i].x;    corner_C.y = trigrid_centers_out[r_i][c_i].y;       // Center

                for (int s_i = 0; s_i< (int) tgf_in[r_i][c_i].size();s_i++){
                    if (tgf_in[r_i][c_i][s_i].is_rejection) {
                        continue;
                    }
                    if (tgf_in[r_i][c_i][s_i].node_type != GROUND) { continue; }
                    if (tgf_in[r_i][c_i][s_i].depth == -1) { continue; }

                    switch(s_i){
                        case 0: // upper Tri-grid bin
                            // RT / LT / C
                            trigrid_corners_out[r_i+1][c_i].zs.push_back( (-tgf_in[r_i][c_i][s_i].normal(0,0)*corner_TR.x - tgf_in[r_i][c_i][s_i].normal(1,0)*corner_TR.y-tgf_in[r_i][c_i][s_i].d)/tgf_in[r_i][c_i][s_i].normal(2,0) );
                            trigrid_corners_out[r_i+1][c_i].weights.push_back(getCornerWeight(tgf_in[r_i][c_i][s_i],corner_TR));

                            trigrid_corners_out[r_i+1][c_i+1].zs.push_back( (-tgf_in[r_i][c_i][s_i].normal(0,0)*corner_TL.x - tgf_in[r_i][c_i][s_i].normal(1,0)*corner_TL.y-tgf_in[r_i][c_i][s_i].d)/tgf_in[r_i][c_i][s_i].normal(2,0) );
                            trigrid_corners_out[r_i+1][c_i+1].weights.push_back(getCornerWeight(tgf_in[r_i][c_i][s_i],corner_TL));

                            trigrid_centers_out[r_i][c_i].zs.push_back( (-tgf_in[r_i][c_i][s_i].normal(0,0)*corner_C.x - tgf_in[r_i][c_i][s_i].normal(1,0)*corner_C.y-tgf_in[r_i][c_i][s_i].d)/tgf_in[r_i][c_i][s_i].normal(2,0) );
                            trigrid_centers_out[r_i][c_i].weights.push_back(getCornerWeight(tgf_in[r_i][c_i][s_i],corner_C));

                            break;
                        case 1: // left Tri-grid bin
                            // LT / LL / C
                            trigrid_corners_out[r_i+1][c_i+1].zs.push_back( (-tgf_in[r_i][c_i][s_i].normal(0,0)*corner_TL.x - tgf_in[r_i][c_i][s_i].normal(1,0)*corner_TL.y-tgf_in[r_i][c_i][s_i].d)/tgf_in[r_i][c_i][s_i].normal(2,0) );
                            trigrid_corners_out[r_i+1][c_i+1].weights.push_back(getCornerWeight(tgf_in[r_i][c_i][s_i],corner_TL));

                            trigrid_corners_out[r_i][c_i+1].zs.push_back( (-tgf_in[r_i][c_i][s_i].normal(0,0)*corner_BL.x - tgf_in[r_i][c_i][s_i].normal(1,0)*corner_BL.y-tgf_in[r_i][c_i][s_i].d)/tgf_in[r_i][c_i][s_i].normal(2,0) );
                            trigrid_corners_out[r_i][c_i+1].weights.push_back(getCornerWeight(tgf_in[r_i][c_i][s_i],corner_BL));

                            trigrid_centers_out[r_i][c_i].zs.push_back( (-tgf_in[r_i][c_i][s_i].normal(0,0)*corner_C.x - tgf_in[r_i][c_i][s_i].normal(1,0)*corner_C.y-tgf_in[r_i][c_i][s_i].d)/tgf_in[r_i][c_i][s_i].normal(2,0) );
                            trigrid_centers_out[r_i][c_i].weights.push_back(getCornerWeight(tgf_in[r_i][c_i][s_i],corner_C));

                            break;
                        case 2: // lower Tri-grid bin
                            // LL / RL / C
                            trigrid_corners_out[r_i][c_i+1].zs.push_back( (-tgf_in[r_i][c_i][s_i].normal(0,0)*corner_BL.x - tgf_in[r_i][c_i][s_i].normal(1,0)*corner_BL.y-tgf_in[r_i][c_i][s_i].d)/tgf_in[r_i][c_i][s_i].normal(2,0) );
                            trigrid_corners_out[r_i][c_i+1].weights.push_back(getCornerWeight(tgf_in[r_i][c_i][s_i],corner_BL));

                            trigrid_corners_out[r_i][c_i].zs.push_back( (-tgf_in[r_i][c_i][s_i].normal(0,0)*corner_BR.x - tgf_in[r_i][c_i][s_i].normal(1,0)*corner_BR.y-tgf_in[r_i][c_i][s_i].d)/tgf_in[r_i][c_i][s_i].normal(2,0) );
                            trigrid_corners_out[r_i][c_i].weights.push_back(getCornerWeight(tgf_in[r_i][c_i][s_i],corner_BR));

                            trigrid_centers_out[r_i][c_i].zs.push_back( (-tgf_in[r_i][c_i][s_i].normal(0,0)*corner_C.x - tgf_in[r_i][c_i][s_i].normal(1,0)*corner_C.y-tgf_in[r_i][c_i][s_i].d)/tgf_in[r_i][c_i][s_i].normal(2,0) );
                            trigrid_centers_out[r_i][c_i].weights.push_back(getCornerWeight(tgf_in[r_i][c_i][s_i],corner_C));
                            
                            break;
                        case 3: // right Tri-grid bin
                            // RL / RT / C
                            trigrid_corners_out[r_i][c_i].zs.push_back( (-tgf_in[r_i][c_i][s_i].normal(0,0)*corner_BR.x - tgf_in[r_i][c_i][s_i].normal(1,0)*corner_BR.y-tgf_in[r_i][c_i][s_i].d)/tgf_in[r_i][c_i][s_i].normal(2,0) );
                            trigrid_corners_out[r_i][c_i].weights.push_back(getCornerWeight(tgf_in[r_i][c_i][s_i],corner_BR));

                            trigrid_corners_out[r_i+1][c_i].zs.push_back( (-tgf_in[r_i][c_i][s_i].normal(0,0)*corner_TR.x - tgf_in[r_i][c_i][s_i].normal(1,0)*corner_TR.y-tgf_in[r_i][c_i][s_i].d)/tgf_in[r_i][c_i][s_i].normal(2,0) );
                            trigrid_corners_out[r_i+1][c_i].weights.push_back(getCornerWeight(tgf_in[r_i][c_i][s_i],corner_TR));

                            trigrid_centers_out[r_i][c_i].zs.push_back( (-tgf_in[r_i][c_i][s_i].normal(0,0)*corner_C.x - tgf_in[r_i][c_i][s_i].normal(1,0)*corner_C.y-tgf_in[r_i][c_i][s_i].d)/tgf_in[r_i][c_i][s_i].normal(2,0) );
                            trigrid_centers_out[r_i][c_i].weights.push_back(getCornerWeight(tgf_in[r_i][c_i][s_i],corner_C));
                            
                            break;
                        default:
                            break;
                    }
                }
            }
            }
            return;
        };
        
        TriGridCorner getMeanCorner(const TriGridCorner &corners_in){
            // get the mean of the corners

            TriGridCorner corners_out;
            corners_out.x = corners_in.x;
            corners_out.y = corners_in.y;
            corners_out.zs.clear();
            corners_out.weights.clear();

            double weighted_sum_z = 0.0;
            double sum_w = 0.0;
            for (int i = 0; i < (int) corners_in.zs.size(); i++){
                weighted_sum_z += corners_in.zs[i]*corners_in.weights[i];
                sum_w += corners_in.weights[i];
            }

            corners_out.zs.push_back(weighted_sum_z/sum_w);
            corners_out.weights.push_back(sum_w);

            return corners_out;
        }

        void updateTGFCornersCenters(std::vector<std::vector<TriGridCorner>>& trigrid_corners_out,
                                    std::vector<std::vector<TriGridCorner>>& trigrid_centers_out) {

            // update corners
            TriGridCorner updated_corner = empty_trigrid_corner_;
            for (int r_i = 0; r_i < rows_ +1; r_i++) {
            for (int c_i = 0; c_i < cols_ +1; c_i++) {
                if (trigrid_corners_out[r_i][c_i].zs.size() > 0 && trigrid_corners_out[r_i][c_i].weights.size() > 0) {
                    updated_corner = getMeanCorner(trigrid_corners_out[r_i][c_i]);
                    trigrid_corners_out[r_i][c_i] = updated_corner;
                } else {
                    trigrid_corners_out[r_i][c_i].zs.clear();
                    trigrid_corners_out[r_i][c_i].weights.clear();
                }
            }
            }        

            // update centers
            TriGridCorner updated_center = empty_trigrid_center_;
            for (int r_i = 0; r_i < rows_; r_i++) {
            for (int c_i = 0; c_i < cols_; c_i++) {
                if (trigrid_centers_out[r_i][c_i].zs.size() > 0 && trigrid_centers_out[r_i][c_i].weights.size() > 0) {
                    updated_center = getMeanCorner(trigrid_centers_out[r_i][c_i]);
                    trigrid_centers_out[r_i][c_i] = updated_center;
                    // trigrid_centers_out[r_i][c_i].z = get_mean(trigrid_centers_out[r_i][c_i].zs,trigrid_centers_out[r_i][c_i].weights);
                } else {
                    trigrid_centers_out[r_i][c_i].zs.clear();
                    trigrid_centers_out[r_i][c_i].weights.clear();
                }
            }
            }

            return;
        };

        Eigen::Vector3f convertCornerToEigen(TriGridCorner &corner_in) {
            Eigen::Vector3f corner_out;
            if (corner_in.zs.size() != corner_in.weights.size()){
                ROS_WARN("ERROR in corners");
            }
            corner_out[0] = corner_in.x;
            corner_out[1] = corner_in.y;
            corner_out[2] = corner_in.zs[0];
            return corner_out;
        };

        double centroid_delta_z(double z_corner1, double z_corner2, double z_corner3) {
            // Calculate the centroid height (z-centroid)
            // double z_centroid = (z_corner1 + z_corner2 + z_corner3) / 3.0;

            // Compute the absolute differences from the centroid
            double delta_z1 = std::abs(z_corner1 - z_corner2); // 1-2
            double delta_z2 = std::abs(z_corner2 - z_corner3); // 2 - 3
            double delta_z3 = std::abs(z_corner3 - z_corner1); // 3-1

            // Find the maximum height difference
            double delta_max = std::max({delta_z1, delta_z2, delta_z3});
            double delta_min = std::min({delta_z1, delta_z2, delta_z3});
            return delta_min/delta_max;
        }

        void revertTraversableNodes(std::vector<std::vector<TriGridCorner>>& trigrid_corners_in,
                                    std::vector<std::vector<TriGridCorner>>& trigrid_centers_in, 
                                    TriGridField<PointType>& tgf_out) {
            double weight_sum = 0.0;
            double total_weight = 0.0;
            Eigen::Vector3f refined_corner_1, refined_corner_2, refined_center;
            for (int r_i = 0; r_i < rows_; r_i++) {
            for (int c_i = 0; c_i < cols_; c_i++) {
            for (int s_i = 0; s_i < (int) tgf_out[r_i][c_i].size(); s_i++) {
                // if (tgf_out[r_i][c_i][s_i].mark) {
                //     continue;
                // }
                switch (s_i)
                {
                case 0: {
                    if ( trigrid_corners_in[r_i+1][c_i].zs.size()==0 || trigrid_corners_in[r_i+1][c_i+1].zs.size()==0  || trigrid_centers_in[r_i][c_i].zs.size()==0 ){
                        if (tgf_out[r_i][c_i][s_i].node_type != NONGROUND){
                            tgf_out[r_i][c_i][s_i].node_type = UNKNOWN;
                        }
                        continue;
                    }
                    refined_corner_1 = convertCornerToEigen(trigrid_corners_in[r_i+1][c_i]);
                    refined_corner_2 = convertCornerToEigen(trigrid_corners_in[r_i+1][c_i+1]);
                    refined_center = convertCornerToEigen(trigrid_centers_in[r_i][c_i]);
                    break;
                }
                case 1: {
                    if ( trigrid_corners_in[r_i+1][c_i+1].zs.size()==0 || trigrid_corners_in[r_i][c_i+1].zs.size()==0  || trigrid_centers_in[r_i][c_i].zs.size()==0 ){
                        if (tgf_out[r_i][c_i][s_i].node_type != NONGROUND){
                            tgf_out[r_i][c_i][s_i].node_type = UNKNOWN;
                        }
                        continue;
                    }
                    refined_corner_1 = convertCornerToEigen(trigrid_corners_in[r_i+1][c_i+1]);
                    refined_corner_2 = convertCornerToEigen(trigrid_corners_in[r_i][c_i+1]);
                    refined_center = convertCornerToEigen(trigrid_centers_in[r_i][c_i]);    
                    break;
                }
                case 2: {
                    if ( trigrid_corners_in[r_i][c_i+1].zs.size()==0 || trigrid_corners_in[r_i][c_i].zs.size()==0  || trigrid_centers_in[r_i][c_i].zs.size()==0 ){
                        if (tgf_out[r_i][c_i][s_i].node_type != NONGROUND){
                            tgf_out[r_i][c_i][s_i].node_type = UNKNOWN;
                        }
                        continue;
                    }

                    refined_corner_1 = convertCornerToEigen(trigrid_corners_in[r_i][c_i+1]);
                    refined_corner_2 = convertCornerToEigen(trigrid_corners_in[r_i][c_i]);
                    refined_center = convertCornerToEigen(trigrid_centers_in[r_i][c_i]);
                    break;
                }
                case 3: {
                    if ( trigrid_corners_in[r_i][c_i].zs.size()==0 || trigrid_corners_in[r_i+1][c_i].zs.size()==0  || trigrid_centers_in[r_i][c_i].zs.size()==0 ){
                        if (tgf_out[r_i][c_i][s_i].node_type != NONGROUND){
                            tgf_out[r_i][c_i][s_i].node_type = UNKNOWN;
                        }
                        continue;
                    }

                    refined_corner_1 = convertCornerToEigen(trigrid_corners_in[r_i][c_i]);
                    refined_corner_2 = convertCornerToEigen(trigrid_corners_in[r_i+1][c_i]);
                    refined_center = convertCornerToEigen(trigrid_centers_in[r_i][c_i]);
                    break;
                }
                default:
                    ROS_ERROR("WRONG tri-grid indexing");
                    break;
                }

                // calculate the refined planar model in the node

                Eigen::Vector3f updated_normal = (refined_corner_1-refined_center).cross(refined_corner_2-refined_center);
                updated_normal /= updated_normal.norm();

                if (updated_normal(2) < TH_NORMAL_){   // non-planar
                    // Eigen::Vector3f updated_mean_pt;
                    // updated_mean_pt[0] = (refined_corner_1[0] + refined_corner_2[0] + refined_center[0])/3;
                    // updated_mean_pt[1] = (refined_corner_1[1] + refined_corner_2[1] + refined_center[1])/3;
                    // updated_mean_pt[2] = (refined_corner_1[2] + refined_corner_2[2] + refined_center[2])/3;

                    // tgf_out[r_i][c_i][s_i].mean_pt = updated_mean_pt;


                    // tgf_out[r_i][c_i][s_i].normal = updated_normal;
                    // tgf_out[r_i][c_i][s_i].d = -(updated_normal.dot(updated_mean_pt));
                    // tgf_out[r_i][c_i][s_i].th_dist_d = TH_DIST_ - tgf_out[r_i][c_i][s_i].d;
                    // tgf_out[r_i][c_i][s_i].th_outlier_d = -TH_OUTLIER_ - tgf_out[r_i][c_i][s_i].d;
                    tgf_out[r_i][c_i][s_i].normal = updated_normal;
                    tgf_out[r_i][c_i][s_i].node_type = NONGROUND;
                } else { 
                    // planar
                    Eigen::Vector3f updated_mean_pt;
                    updated_mean_pt[0] = (refined_corner_1[0] + refined_corner_2[0] + refined_center[0])/3;
                    updated_mean_pt[1] = (refined_corner_1[1] + refined_corner_2[1] + refined_center[1])/3;
                    updated_mean_pt[2] = (refined_corner_1[2] + refined_corner_2[2] + refined_center[2])/3;

                    

                    tgf_out[r_i][c_i][s_i].normal = updated_normal;
                    tgf_out[r_i][c_i][s_i].mean_pt = updated_mean_pt;
                    tgf_out[r_i][c_i][s_i].d = -(updated_normal.dot(updated_mean_pt));
                    tgf_out[r_i][c_i][s_i].th_dist_d = TH_DIST_ - tgf_out[r_i][c_i][s_i].d;
                    tgf_out[r_i][c_i][s_i].th_outlier_d = -TH_OUTLIER_ - tgf_out[r_i][c_i][s_i].d;
                    tgf_out[r_i][c_i][s_i].node_type = GROUND;
                }
            }
            }
            }
            
            // if (total_weight > 0) {
            //     SENSOR_HEIGHT_  = -weight_sum / total_weight;
            // }
            // else {
            //     ROS_WARN("No ground points found in tri grid for estimating height");
            // }
            // // std::cout << "========================" << std::endl;
            // std::cout << "SENSOR HEIGHT: " << SENSOR_HEIGHT_ << std::endl;
            return;
        };

        void fitTGFWiseTraversableTerrainModel(TriGridField<PointType>& tgf,
                                            std::vector<std::vector<TriGridCorner>>& trigrid_corners,
                                            std::vector<std::vector<TriGridCorner>>& trigrid_centers) {

            updateTGFCornersCenters(trigrid_corners, trigrid_centers);

            revertTraversableNodes(trigrid_corners, trigrid_centers, tgf);
            return;
        };

        void segmentNodeGround(TriGridNode<PointType>& node_in,
                                TriGridIdx& cur_idx,
                                pcl::PointCloud<PointType>& node_ground_out,
                                pcl::PointCloud<PointType>& node_nonground_out,
                                pcl::PointCloud<PointType>& node_obstacle_out,
                                pcl::PointCloud<PointType>& node_outlier_out) {
            node_ground_out.clear();
            node_nonground_out.clear();
            node_obstacle_out.clear();
            node_outlier_out.clear();
            // segment ground
            Eigen::MatrixXf points(node_in.ptCloud.points.size(),3);
            int j = 0; 
            for (auto& p:node_in.ptCloud.points){
                points.row(j++)<<p.x, p.y, p.z;
            }
            Eigen::VectorXf result = points*node_in.normal;
            for (int r = 0; r<result.rows(); r++){
                    if (result[r]<node_in.th_dist_d){
                        node_ground_out.push_back(node_in.ptCloud.points[r]);
                    } else {
                        node_nonground_out.push_back(node_in.ptCloud.points[r]);
                    }
                // }
            }
            idx_type_map[cur_idx] = true;
            return;
        }

        void segmentNodeGround(TriGridNode<PointType>& node_in,
                                pcl::PointCloud<PointType>& node_ground_out,
                                pcl::PointCloud<PointType>& node_nonground_out,
                                pcl::PointCloud<PointType>& node_obstacle_out,
                                pcl::PointCloud<PointType>& node_outlier_out) {
            node_ground_out.clear();
            node_nonground_out.clear();
            node_obstacle_out.clear();
            node_outlier_out.clear();
            // segment ground
            Eigen::MatrixXf points(node_in.ptCloud.points.size(),3);
            int j = 0; 
            for (auto& p:node_in.ptCloud.points){
                points.row(j++)<<p.x, p.y, p.z;
            }
            Eigen::VectorXf result = points*node_in.normal;
            for (int r = 0; r<result.rows(); r++){
                if (result[r]<node_in.th_dist_d){
                    node_ground_out.push_back(node_in.ptCloud.points[r]);
                } else {
                    node_nonground_out.push_back(node_in.ptCloud.points[r]);
                }
            }
            return;
        }

        void segmentNodeGround123(const TriGridNode<PointType>& node_in,
                                pcl::PointCloud<PointType>& node_ground_out,
                                pcl::PointCloud<PointType>& node_nonground_out,
                                pcl::PointCloud<PointType>& node_obstacle_out,
                                pcl::PointCloud<PointType>& node_outlier_out) {
            node_ground_out.clear();
            node_nonground_out.clear();
            node_obstacle_out.clear();
            node_outlier_out.clear();

            // segment ground
            Eigen::MatrixXf points(node_in.ptCloud.points.size(),3);
            int j = 0; 
            for (auto& p:node_in.ptCloud.points){
                points.row(j++)<<p.x, p.y, p.z;
            }
            pcl::PointCloud<PointType> idk;
            Eigen::VectorXf result = points*node_in.normal;
            for (int r = 0; r<result.rows(); r++){
                if (result[r]<node_in.th_dist_d){
                    // if (result[r]<node_in.th_outlier_d) {
                        // node_outlier_out.push_back(node_in.ptCloud.points[r]);
                    // } else {
                    idk.push_back(node_in.ptCloud.points[r]);
                    node_ground_out.push_back(node_in.ptCloud.points[r]);
                    // }
                } else {
                    node_nonground_out.push_back(node_in.ptCloud.points[r]);
                }
            }
            Eigen::Matrix3f cov_;
            Eigen::Vector4f pc_mean_;
            pcl::computeMeanAndCovarianceMatrix(idk, cov_, pc_mean_);

            // Singular Value Decomposition: SVD
            Eigen::JacobiSVD<Eigen::MatrixXf> svd(cov_, Eigen::DecompositionOptions::ComputeFullU);

            VectorXf singular_values_ = svd.singularValues();
            // double line_variable = singular_values_(1) != 0 ? singular_values_(0)/singular_values_(1) : std::numeric_limits<double>::max();
            double flatness = singular_values_[1]/singular_values_[2];
            double weight = (singular_values_[0] + singular_values_[1])*singular_values_[1]/(singular_values_[0]*singular_values_[2]+0.001);
            std::cout << "weight: " << weight << std::endl;
            std::cout << "flatness: " << flatness << std::endl;
            return;
        }

        bool checkElevation(double xy_range, const double& mean_pt, const std::vector<double>& RANGE_, const std::vector<double>& ELEVATION_THR_) {
            if (xy_range >= RANGE_[0] && xy_range < RANGE_[1]) {
                return mean_pt < ELEVATION_THR_[0];
            } else if (xy_range >= RANGE_[1] && xy_range < RANGE_[2]) {
                return mean_pt < ELEVATION_THR_[1];
            } else if (xy_range >= RANGE_[2] && xy_range < RANGE_[3]) {
                return mean_pt < ELEVATION_THR_[2];
            } else if (xy_range >= RANGE_[3]) {
                return mean_pt < ELEVATION_THR_[3];
            }
            return false;
        }

        bool checkFlatness(double xy_range, const double& flatness, const std::vector<double>& RANGE_, const std::vector<double>& FLATNESS_THR_) {
            if (xy_range >= RANGE_[0] && xy_range < RANGE_[1]) {
                return flatness < FLATNESS_THR_[0];
            } else if (xy_range >= RANGE_[1] && xy_range < RANGE_[2]) {
                return flatness < FLATNESS_THR_[1];
            } else if (xy_range >= RANGE_[2] && xy_range < RANGE_[3]) {
                return flatness < FLATNESS_THR_[2];
            } else if (xy_range >= RANGE_[3]) {
                return flatness < FLATNESS_THR_[3];
            }
            return false;
        }

        void segmentTGFGround(TriGridField<PointType>& tgf_in, 
                            pcl::PointCloud<PointType>& ground_cloud_out, 
                            pcl::PointCloud<PointType>& nonground_cloud_out,
                            pcl::PointCloud<PointType>& obstacle_cloud_out,
                            pcl::PointCloud<PointType>& outlier_cloud_out) {
            for (int r_i = 0; r_i < rows_; r_i++) {
                for (int c_i = 0; c_i < cols_; c_i++) {
                    for (int s_i = 0; s_i < tgf_in[r_i][c_i].size(); s_i++) {
                        if (!tgf_in[r_i][c_i][s_i].is_curr_data) {
                            continue;
                        }
                        // Clear node-wise point clouds
                        ptCloud_nodewise_ground_.clear();
                        ptCloud_nodewise_nonground_.clear();
                        ptCloud_nodewise_outliers_.clear();
                        ptCloud_nodewise_obstacle_.clear();
                        if (tgf_in[r_i][c_i][s_i].node_type == UNKNOWN) {
                            ptCloud_nodewise_nonground_ = tgf_in[r_i][c_i][s_i].ptCloud;
                        }
                        if (tgf_in[r_i][c_i][s_i].node_type == GROUND 
                        ) {
                            TriGridIdx cur_idx;
                            cur_idx.row = r_i;
                            cur_idx.col = c_i;
                            cur_idx.tri = s_i;
                            segmentNodeGround(tgf_in[r_i][c_i][s_i], cur_idx, ptCloud_nodewise_ground_, ptCloud_nodewise_nonground_, ptCloud_nodewise_obstacle_, ptCloud_nodewise_outliers_);
                        }
                        else { 
                            ptCloud_nodewise_nonground_ = tgf_in[r_i][c_i][s_i].ptCloud;
                        }
                        ground_cloud_out += ptCloud_nodewise_ground_;
                        nonground_cloud_out += ptCloud_nodewise_nonground_;
                    }
                }
            }
            return;
        }

        // functions for visualization

        geometry_msgs::PolygonStamped setPlanarModel (const TriGridNode<PointType>& node_in, const TriGridIdx& node_idx) {
            geometry_msgs::PolygonStamped polygon_out;
            polygon_out.header = msg_header_;
            geometry_msgs::Point32 corner_0, corner_1, corner_2;
            int r_i = node_idx.row;
            int c_i = node_idx.col;
            int s_i = node_idx.tri;
            if (node_in.node_type == GROUND){
                switch (s_i){
                    case 0:
                        //topx lowy & topx topy
                        corner_1.x = (r_i+1)*TGF_RESOLUTION_+tgf_min_x; corner_1.y = (c_i)*TGF_RESOLUTION_+tgf_min_y; 
                        corner_1.z = (-node_in.normal(0,0)*corner_1.x - node_in.normal(1,0)*corner_1.y-node_in.d)/node_in.normal(2,0);

                        corner_2.x = (r_i+1)*TGF_RESOLUTION_+tgf_min_x; corner_2.y = (c_i+1)*TGF_RESOLUTION_+tgf_min_y; 
                        corner_2.z = (-node_in.normal(0,0)*corner_2.x - node_in.normal(1,0)*corner_2.y-node_in.d)/node_in.normal(2,0);

                        corner_0.x = (r_i+0.5)*TGF_RESOLUTION_+tgf_min_x; corner_0.y = (c_i+0.5)*TGF_RESOLUTION_+tgf_min_y; 
                        corner_0.z = (-node_in.normal(0,0)*corner_0.x - node_in.normal(1,0)*corner_0.y-node_in.d)/node_in.normal(2,0);
                        break;
                    case 1:
                        //topx topy & lowx topy
                        corner_1.x = (r_i+1)*TGF_RESOLUTION_+tgf_min_x; corner_1.y = (c_i+1)*TGF_RESOLUTION_+tgf_min_y; 
                        corner_1.z = (-node_in.normal(0,0)*corner_1.x - node_in.normal(1,0)*corner_1.y-node_in.d)/node_in.normal(2,0);

                        corner_2.x = (r_i)*TGF_RESOLUTION_+tgf_min_x; corner_2.y = (c_i+1)*TGF_RESOLUTION_+tgf_min_y; 
                        corner_2.z = (-node_in.normal(0,0)*corner_2.x - node_in.normal(1,0)*corner_2.y-node_in.d)/node_in.normal(2,0);

                        corner_0.x = (r_i+0.5)*TGF_RESOLUTION_+tgf_min_x; corner_0.y = (c_i+0.5)*TGF_RESOLUTION_+tgf_min_y; 
                        corner_0.z = (-node_in.normal(0,0)*corner_0.x - node_in.normal(1,0)*corner_0.y-node_in.d)/node_in.normal(2,0);
                        break;
                    case 2:
                        //lowx topy & lowx lowy
                        corner_1.x = (r_i)*TGF_RESOLUTION_+tgf_min_x; corner_1.y = (c_i+1)*TGF_RESOLUTION_+tgf_min_y; 
                        corner_1.z = (-node_in.normal(0,0)*corner_1.x - node_in.normal(1,0)*corner_1.y-node_in.d)/node_in.normal(2,0);

                        corner_2.x = (r_i)*TGF_RESOLUTION_+tgf_min_x; corner_2.y = c_i*TGF_RESOLUTION_+tgf_min_y; 
                        corner_2.z = (-node_in.normal(0,0)*corner_2.x - node_in.normal(1,0)*corner_2.y-node_in.d)/node_in.normal(2,0);

                        corner_0.x = (r_i+0.5)*TGF_RESOLUTION_+tgf_min_x; corner_0.y = (c_i+0.5)*TGF_RESOLUTION_+tgf_min_y; 
                        corner_0.z = (-node_in.normal(0,0)*corner_0.x - node_in.normal(1,0)*corner_0.y-node_in.d)/node_in.normal(2,0);
                        break;
                    case 3:
                        //lowx lowy & topx lowy 
                        corner_1.x = (r_i)*TGF_RESOLUTION_+tgf_min_x; corner_1.y = (c_i)*TGF_RESOLUTION_+tgf_min_y; 
                        corner_1.z = (-node_in.normal(0,0)*corner_1.x - node_in.normal(1,0)*corner_1.y-node_in.d)/node_in.normal(2,0);

                        corner_2.x = (r_i+1)*TGF_RESOLUTION_+tgf_min_x; corner_2.y = (c_i)*TGF_RESOLUTION_+tgf_min_y; 
                        corner_2.z = (-node_in.normal(0,0)*corner_2.x - node_in.normal(1,0)*corner_2.y-node_in.d)/node_in.normal(2,0);

                        corner_0.x = (r_i+0.5)*TGF_RESOLUTION_+tgf_min_x; corner_0.y = (c_i+0.5)*TGF_RESOLUTION_+tgf_min_y; 
                        corner_0.z = (-node_in.normal(0,0)*corner_0.x - node_in.normal(1,0)*corner_0.y-node_in.d)/node_in.normal(2,0);
                        break;
                    default:
                        break;
                }        
            } else {
                switch (s_i){
                    case 0:
                        //topx lowy & topx topy
                        corner_1.x = (r_i+1)*TGF_RESOLUTION_+tgf_min_x; corner_1.y = (c_i)*TGF_RESOLUTION_+tgf_min_y; 
                        corner_1.z = -2.0;

                        corner_2.x = (r_i+1)*TGF_RESOLUTION_+tgf_min_x; corner_2.y = (c_i+1)*TGF_RESOLUTION_+tgf_min_y; 
                        corner_2.z = -2.0;

                        corner_0.x = (r_i+0.5)*TGF_RESOLUTION_+tgf_min_x; corner_0.y = (c_i+0.5)*TGF_RESOLUTION_+tgf_min_y; 
                        corner_0.z = -2.0;
                        break;
                    case 1:
                        //topx topy & lowx topy
                        corner_1.x = (r_i+1)*TGF_RESOLUTION_+tgf_min_x; corner_1.y = (c_i+1)*TGF_RESOLUTION_+tgf_min_y; 
                        corner_1.z = -2.0;

                        corner_2.x = (r_i)*TGF_RESOLUTION_+tgf_min_x; corner_2.y = (c_i+1)*TGF_RESOLUTION_+tgf_min_y; 
                        corner_2.z = -2.0;

                        corner_0.x = (r_i+0.5)*TGF_RESOLUTION_+tgf_min_x; corner_0.y = (c_i+0.5)*TGF_RESOLUTION_+tgf_min_y; 
                        corner_0.z = -2.0;
                        break;
                    case 2:
                        //lowx topy & lowx lowy
                        corner_1.x = (r_i)*TGF_RESOLUTION_+tgf_min_x; corner_1.y = (c_i+1)*TGF_RESOLUTION_+tgf_min_y; 
                        corner_1.z = -2.0;

                        corner_2.x = (r_i)*TGF_RESOLUTION_+tgf_min_x; corner_2.y = c_i*TGF_RESOLUTION_+tgf_min_y; 
                        corner_2.z = -2.0;

                        corner_0.x = (r_i+0.5)*TGF_RESOLUTION_+tgf_min_x; corner_0.y = (c_i+0.5)*TGF_RESOLUTION_+tgf_min_y; 
                        corner_0.z = -2.0;
                        break;
                    case 3:
                        //lowx lowy & topx lowy 
                        corner_1.x = (r_i)*TGF_RESOLUTION_+tgf_min_x; corner_1.y = (c_i)*TGF_RESOLUTION_+tgf_min_y; 
                        corner_1.z = -2.0;

                        corner_2.x = (r_i+1)*TGF_RESOLUTION_+tgf_min_x; corner_2.y = (c_i)*TGF_RESOLUTION_+tgf_min_y; 
                        corner_2.z = -2.0;

                        corner_0.x = (r_i+0.5)*TGF_RESOLUTION_+tgf_min_x; corner_0.y = (c_i+0.5)*TGF_RESOLUTION_+tgf_min_y; 
                        corner_0.z = -2.0;
                        break;
                    default:
                        std::cout << "the error in sub-idx" << std::endl;
                        break;
                }
            }

            polygon_out.polygon.points.reserve(3); 
            polygon_out.polygon.points.push_back(corner_0);
            polygon_out.polygon.points.push_back(corner_2);
            polygon_out.polygon.points.push_back(corner_1);
            return polygon_out;
        }

        void publishTriGridFieldGraph() {
            viz_trigrid_polygons_.header = msg_header_;
            viz_trigrid_polygons_.polygons.clear();
            viz_trigrid_polygons_.likelihood.clear();

            // visualize the graph: nodes
            for (int r_i = 0; r_i < rows_; r_i++){
            for (int c_i = 0; c_i < cols_; c_i++){
            for (int s_i = 0; s_i < (int) trigrid_field_[r_i][c_i].size(); s_i++){
                TriGridIdx curr_idx = {r_i, c_i, s_i};
                if (trigrid_field_[r_i][c_i][s_i].node_type == GROUND) {
                    viz_trigrid_polygons_.polygons.push_back(setPlanarModel(trigrid_field_[r_i][c_i][s_i], curr_idx));
                    if (trigrid_field_[r_i][c_i][s_i].mark)
                    {
                        viz_trigrid_polygons_.likelihood.push_back(0.32);
                    }
                    else if (trigrid_field_[r_i][c_i][s_i].is_processed) {
                        viz_trigrid_polygons_.likelihood.push_back(0.75);
                    }
                    else {
                        viz_trigrid_polygons_.likelihood.push_back(0.0);
                    }    
                } else if (trigrid_field_[r_i][c_i][s_i].node_type == NONGROUND) {
                    viz_trigrid_polygons_.polygons.push_back(setPlanarModel(trigrid_field_[r_i][c_i][s_i], curr_idx));
                    if (trigrid_field_[r_i][c_i][s_i].is_rejection) {
                        viz_trigrid_polygons_.likelihood.push_back(0.86);
                    }
                    else {
                        viz_trigrid_polygons_.likelihood.push_back(1.0);
                    }
                } else if (trigrid_field_[r_i][c_i][s_i].node_type == UNKNOWN) {
                    if (trigrid_field_[r_i][c_i][s_i].is_curr_data) {
                        viz_trigrid_polygons_.polygons.push_back(setPlanarModel(trigrid_field_[r_i][c_i][s_i], curr_idx));
                        viz_trigrid_polygons_.likelihood.push_back(0.5);
                    }
                } 
                else {
                    ROS_WARN("Unknown Node Type");
                }            
            }
            }
            }

            //visualize the graph: edges
            viz_trigrid_edges_.header = msg_header_;
            viz_trigrid_edges_.points.clear();
            geometry_msgs::Point src_pt;
            geometry_msgs::Point tgt_pt;
            for (int e_i = 0; e_i < (int) trigrid_edges_.size(); e_i++){
                if (trigrid_edges_[e_i].is_traversable){
                    viz_trigrid_edges_.color.a = 1.0;
                    viz_trigrid_edges_.color.r = 1.0;
                    viz_trigrid_edges_.color.g = 1.0;
                    viz_trigrid_edges_.color.b = 0.0;
                } else {
                    viz_trigrid_edges_.color.a = 0.1;
                    viz_trigrid_edges_.color.r = 1.0;
                    viz_trigrid_edges_.color.g = 1.0;
                    viz_trigrid_edges_.color.b = 1.0;
                }

                src_pt.x = trigrid_field_[trigrid_edges_[e_i].Pair.first.row][trigrid_edges_[e_i].Pair.first.col][trigrid_edges_[e_i].Pair.first.tri].mean_pt[0];
                src_pt.y = trigrid_field_[trigrid_edges_[e_i].Pair.first.row][trigrid_edges_[e_i].Pair.first.col][trigrid_edges_[e_i].Pair.first.tri].mean_pt[1];
                src_pt.z = trigrid_field_[trigrid_edges_[e_i].Pair.first.row][trigrid_edges_[e_i].Pair.first.col][trigrid_edges_[e_i].Pair.first.tri].mean_pt[2];

                tgt_pt.x = trigrid_field_[trigrid_edges_[e_i].Pair.second.row][trigrid_edges_[e_i].Pair.second.col][trigrid_edges_[e_i].Pair.second.tri].mean_pt[0];
                tgt_pt.y = trigrid_field_[trigrid_edges_[e_i].Pair.second.row][trigrid_edges_[e_i].Pair.second.col][trigrid_edges_[e_i].Pair.second.tri].mean_pt[1];
                tgt_pt.z = trigrid_field_[trigrid_edges_[e_i].Pair.second.row][trigrid_edges_[e_i].Pair.second.col][trigrid_edges_[e_i].Pair.second.tri].mean_pt[2];

                viz_trigrid_edges_.points.push_back(src_pt);
                viz_trigrid_edges_.points.push_back(tgt_pt);
            }

            pub_trigrid_nodes_.publish(viz_trigrid_polygons_);
            pub_trigrid_edges_.publish(viz_trigrid_edges_);
            return;
        };

        void publishTriGridCorners() {
            viz_trigrid_corners_.header = cloud_header_;
            viz_trigrid_corners_.points.clear();

            TriGridCorner curr_corner;
            pcl::PointXYZ corner_pt;
            // for corners
            for (int r_i = 0; r_i < (int) trigrid_corners_.size(); r_i++){
            for (int c_i = 0; c_i < (int) trigrid_corners_[0].size(); c_i++){
                curr_corner = trigrid_corners_[r_i][c_i];
                if (curr_corner.zs.size() != curr_corner.weights.size()){
                    ROS_WARN("ERROR in corners");
                }
                for (int i = 0; i < (int) curr_corner.zs.size(); i++){
                    corner_pt.x = curr_corner.x;
                    corner_pt.y = curr_corner.y;
                    corner_pt.z = curr_corner.zs[i];
                    viz_trigrid_corners_.points.push_back(corner_pt);
                }
            }
            }

            // for centers
            for (int r_i = 0; r_i < (int) trigrid_centers_.size(); r_i++){
            for (int c_i = 0; c_i < (int) trigrid_centers_[0].size(); c_i++){
                curr_corner = trigrid_centers_[r_i][c_i];
                if (curr_corner.zs.size() != curr_corner.weights.size()){
                    ROS_WARN("ERROR in corners");
                }
                for (int i = 0; i < (int) curr_corner.zs.size(); i++){
                    corner_pt.x = curr_corner.x;
                    corner_pt.y = curr_corner.y;
                    corner_pt.z = curr_corner.zs[i];
                    viz_trigrid_corners_.points.push_back(corner_pt);
                }
            }
            }

            pub_trigrid_corners_.publish(viz_trigrid_corners_);
            return;
        };

        sensor_msgs::PointCloud2 convertCloudToRosMsg(pcl::PointCloud<PointType>& cloud, std::string &frame_id) {
            sensor_msgs::PointCloud2 cloud_msg;
            pcl::toROSMsg(cloud, cloud_msg);
            cloud_msg.header.frame_id = frame_id;
            return cloud_msg;
        };

        void publishPointClouds(){
            ptCloud_tgfwise_ground_.header = cloud_header_;
            pub_tgseg_ground_cloud.publish(convertCloudToRosMsg(ptCloud_tgfwise_ground_, cloud_header_.frame_id));
            
            ptCloud_tgfwise_nonground_.header = cloud_header_;
            pub_tgseg_nonground_cloud.publish(convertCloudToRosMsg(ptCloud_tgfwise_nonground_, cloud_header_.frame_id));

            ptCloud_tgfwise_outliers_.header = cloud_header_;
            pub_tgseg_outliers_cloud.publish(convertCloudToRosMsg(ptCloud_tgfwise_outliers_, cloud_header_.frame_id));
        }
    };
}

#endif