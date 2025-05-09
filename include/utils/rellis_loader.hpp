//
// Created by Giang on 08/10/2024.
//
#include "utils/utils_rellis.hpp"

#ifndef TRAVEL_RELLIS_LOADER_HPP
#define TRAVEL_RELLIS_LOADER_HPP

class RellisLoader {
public:
    RellisLoader(const std::string &abs_path) {
        pc_path_ = abs_path + "/os1_cloud_node_kitti_bin";
        label_path_ = abs_path + "/os1_cloud_node_semantickitti_label_id";

        for (num_frames_ = 0;; num_frames_++) {
            std::string filename = (boost::format("%s/%06d.bin") % pc_path_ % num_frames_).str();
            if (!boost::filesystem::exists(filename)) {
                break;
            }
        }
        int num_labels;
        for (num_labels = 0;; num_labels++) {
            std::string filename = (boost::format("%s/%06d.label") % label_path_ % num_labels).str();
            if (!boost::filesystem::exists(filename)) {
                break;
            }
        }

        if (num_frames_ == 0) {
            std::cerr << "\033[1;31mError: No files in " << pc_path_ << "\033[0m" << std::endl;
        }
        if (num_frames_ != num_labels) {
            std::cerr << "\033[1;31mError: The # of point clouds and # of labels are not same\033[0m" << std::endl;
        }
    }

    ~RellisLoader() {}

    size_t size() const { return num_frames_; }

    pcl::PointCloud<PointXYZILID>::ConstPtr cloud(size_t i) const {
        std::string filename = (boost::format("%s/%06d.bin") % pc_path_ % i).str();
        FILE* file = fopen(filename.c_str(), "rb");
        if (!file) {
            std::cerr << "error: failed to load " << filename << std::endl;
            return nullptr;
        }

        std::vector<float> buffer(1000000); // Should be larger than 140,000 * 4
        size_t num_points = fread(reinterpret_cast<char*>(buffer.data()), sizeof(float), buffer.size(), file) / 4;
        fclose(file);

        pcl::PointCloud<PointXYZILID>::Ptr cloud_ptr(new pcl::PointCloud<PointXYZILID>());
        cloud_ptr->resize(num_points);

        std::string label_name = (boost::format("%s/%06d.label") % label_path_ % i).str();
        std::ifstream label_input(label_name, std::ios::binary);
        if (!label_input.is_open()) {
            std::cerr << "Could not open the label!" << std::endl;
            return nullptr;
        }
        label_input.seekg(0, std::ios::beg);

        std::vector<uint32_t> labels(num_points);
        label_input.read((char*)&labels[0], num_points * sizeof(uint32_t));

        for (int i = 0; i < num_points; i++) {
            auto& pt = cloud_ptr->at(i);
            pt.x = buffer[i * 4];
            pt.y = buffer[i * 4 + 1];
            pt.z = buffer[i * 4 + 2];
            pt.intensity = buffer[i * 4 + 3];
            pt.label = labels[i] & 0xFFFF;
        }

        return cloud_ptr;
    }

private:
    int num_frames_;
    std::string label_path_;
    std::string pc_path_;
};

#endif //TRAVEL_RELLIS_LOADER_HPP
