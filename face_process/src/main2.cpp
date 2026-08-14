#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <map>
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include "net.h"

using namespace std;

// 人脸检测结果结构体
struct FaceObject {
    cv::Rect_<float> rect;
    cv::Point2f pts[5];
    float prob;
};

// 计算余弦相似度
float cosine_similarity(const std::vector<float>& v1, const std::vector<float>& v2) {
    float dot_product = 0.0f, norm1 = 0.0f, norm2 = 0.0f;
    for (size_t i = 0; i < v1.size(); ++i) {
        dot_product += v1[i] * v2[i];
        norm1 += v1[i] * v1[i];
        norm2 += v2[i] * v2[i];
    }
    return dot_product / (std::sqrt(norm1) * std::sqrt(norm2));
}

// 从 db.txt 加载数据库 (每行: 名字 特征1 特征2 ... 特征128)
bool load_db(const string& filename, std::map<string, std::vector<float>>& db) {
    std::ifstream f(filename);
    if (!f.is_open()) {
        return false; // 文件不存在属于正常情况（初次运行）
    }
    
    string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        
        std::stringstream ss(line);
        string name;
        ss >> name; // 读取每一行的第一个单词作为名字
        
        std::vector<float> feature;
        float val;
        while (ss >> val) {
            feature.push_back(val);
        }
        
        // 确保特征维度是 128 维才录入
        if (feature.size() == 128) {
            db[name] = feature;
        } else {
            std::cerr << "警告: 发现格式错误的行 (姓名: " << name << ", 特征维度: " << feature.size() << ")，已跳过。" << std::endl;
        }
    }
    f.close();
    return true;
}

// 保存数据库到 db.txt (每人一行)
bool save_db(const string& filename, const std::map<string, std::vector<float>>& db) {
    std::ofstream f(filename);
    if (!f.is_open()) {
        std::cerr << "无法打开文件以写入数据: " << filename << std::endl;
        return false;
    }
    
    // 设置高精度，防止特征值因四舍五入丢失精度
    f << std::fixed << std::setprecision(6);
    
    for (const auto& pair : db) {
        f << pair.first; // 写入名字
        for (float val : pair.second) {
            f << " " << val; // 写入128个特征值，用空格隔开
        }
        f << "\n"; // 换行，一人一行
    }
    f.close();
    return true;
}

// SCRFD 人脸检测类
class SCRFD {
public:
    SCRFD(const string& param_path, const string& bin_path) {
        scrfd.opt.use_vulkan_compute = false; 
        scrfd.opt.num_threads = 4; // 树莓派3B使用4核充分利用性能
        scrfd.load_param(param_path.c_str());
        scrfd.load_model(bin_path.c_str());
    }

    int detect(const cv::Mat& bgr, std::vector<FaceObject>& faceobjects, float prob_threshold = 0.5f) {
        int width = bgr.cols;
        int height = bgr.rows;
        
        int target_size = 640;
        int w = width;
        int h = height;
        float scale = 1.f;
        if (w > h) {
            scale = (float)target_size / w;
            w = target_size;
            h = h * scale;
        } else {
            scale = (float)target_size / h;
            h = target_size;
            w = w * scale;
        }

        ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, width, height, w, h);
        
        int wpad = (w + 31) / 32 * 32 - w;
        int hpad = (h + 31) / 32 * 32 - h;
        ncnn::Mat in_pad;
        ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.f);

        const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
        const float norm_vals[3] = {1.0f/128.0f, 1.0f/128.0f, 1.0f/128.0f};
        in_pad.substract_mean_normalize(mean_vals, norm_vals);

        ncnn::Extractor ex = scrfd.create_extractor();
        ex.input("data", in_pad);

        std::vector<FaceObject> proposals;
        int strides[] = {8, 16, 32};
        
        for (int i = 0; i < 3; i++) {
            int stride = strides[i];
            char score_blob_name[32], bbox_blob_name[32], kps_blob_name[32];
            sprintf(score_blob_name, "score_%d", stride);
            sprintf(bbox_blob_name, "bbox_%d", stride);
            sprintf(kps_blob_name, "kps_%d", stride);

            ncnn::Mat score_blob, bbox_blob, kps_blob;
            ex.extract(score_blob_name, score_blob);
            ex.extract(bbox_blob_name, bbox_blob);
            ex.extract(kps_blob_name, kps_blob);

            int base_w = in_pad.w / stride;
            int base_h = in_pad.h / stride;

            for (int q = 0; q < score_blob.c; q++) {
                const float* score_ptr = score_blob.channel(q);
                const float* bbox_ptr = bbox_blob.channel(q);
                const float* kps_ptr = kps_blob.channel(q);

                for (int y = 0; y < base_h; y++) {
                    for (int x = 0; x < base_w; x++) {
                        float score = score_ptr[0];
                        if (score > prob_threshold) {
                            float cx = (x + 0.5f) * stride;
                            float cy = (y + 0.5f) * stride;

                            FaceObject obj;
                            obj.prob = score;
                            obj.rect.x = cx - bbox_ptr[0] * stride;
                            obj.rect.y = cy - bbox_ptr[1] * stride;
                            obj.rect.width = cx + bbox_ptr[2] * stride - obj.rect.x;
                            obj.rect.height = cy + bbox_ptr[3] * stride - obj.rect.y;

                            for (int k = 0; k < 5; k++) {
                                obj.pts[k].x = cx + kps_ptr[k * 2] * stride;
                                obj.pts[k].y = cy + kps_ptr[k * 2 + 1] * stride;
                            }
                            proposals.push_back(obj);
                        }
                        score_ptr++; bbox_ptr += 4; kps_ptr += 10;
                    }
                }
            }
        }

        std::sort(proposals.begin(), proposals.end(), [](const FaceObject& a, const FaceObject& b) { return a.prob > b.prob; });
        std::vector<int> picked;
        for (size_t i = 0; i < proposals.size(); i++) {
            int keep = 1;
            for (int j : picked) {
                cv::Rect_<float> inter = proposals[i].rect & proposals[j].rect;
                float inter_area = inter.area();
                float union_area = proposals[i].rect.area() + proposals[j].rect.area() - inter_area;
                if (inter_area / union_area > 0.4f) {
                    keep = 0; break;
                }
            }
            if (keep) {
                picked.push_back(i);
            }
        }

        faceobjects.resize(picked.size());
        for (size_t i = 0; i < picked.size(); i++) {
            faceobjects[i] = proposals[picked[i]];
            float x_offset = (wpad / 2.0f);
            float y_offset = (hpad / 2.0f);
            
            faceobjects[i].rect.x = (faceobjects[i].rect.x - x_offset) / scale;
            faceobjects[i].rect.y = (faceobjects[i].rect.y - y_offset) / scale;
            faceobjects[i].rect.width /= scale;
            faceobjects[i].rect.height /= scale;

            for (int k = 0; k < 5; k++) {
                faceobjects[i].pts[k].x = (faceobjects[i].pts[k].x - x_offset) / scale;
                faceobjects[i].pts[k].y = (faceobjects[i].pts[k].y - y_offset) / scale;
            }
        }
        return 0;
    }

private:
    ncnn::Net scrfd;
};

// MobileFaceNet 人脸识别类
class MobileFaceNet {
public:
    MobileFaceNet(const string& param_path, const string& bin_path) {
        net.opt.use_vulkan_compute = false;
        net.opt.num_threads = 4;
        net.load_param(param_path.c_str());
        net.load_model(bin_path.c_str());
    }

    std::vector<float> extract_feature(const cv::Mat& bgr, const cv::Point2f pts[5]) {
        // MobileFaceNet 标准 112x112 仿射变换参考点
        float dst[5][2] = {
            {38.2946f, 51.6963f}, {73.5318f, 51.5014f},
            {56.0252f, 71.7366f},
            {41.5493f, 92.3655f}, {70.7299f, 92.2041f}
        };
        
        cv::Mat src_mat(5, 2, CV_32FC1);
        cv::Mat dst_mat(5, 2, CV_32FC1, dst);
        for (int i = 0; i < 5; i++) {
            src_mat.at<float>(i, 0) = pts[i].x;
            src_mat.at<float>(i, 1) = pts[i].y;
        }

        // 仿射变换与局部裁剪
        cv::Mat M = cv::estimateAffinePartial2D(src_mat, dst_mat);
        cv::Mat aligned_face;
        cv::warpAffine(bgr, aligned_face, M, cv::Size(112, 112), cv::INTER_LINEAR);

        ncnn::Mat in = ncnn::Mat::from_pixels(aligned_face.data, ncnn::Mat::PIXEL_BGR2RGB, 112, 112);
        
        const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
        const float norm_vals[3] = {1.0f/128.0f, 1.0f/128.0f, 1.0f/128.0f};
        in.substract_mean_normalize(mean_vals, norm_vals);

        ncnn::Extractor ex = net.create_extractor();
        ex.input("data", in);
        ncnn::Mat out;
        ex.extract("fc1", out); 

        // 归一化特征
        std::vector<float> feature(out.w);
        float norm = 0.f;
        for (int i = 0; i < out.w; i++) {
            feature[i] = out[i];
            norm += feature[i] * feature[i];
        }
        norm = std::sqrt(norm);
        for (int i = 0; i < out.w; i++) {
            feature[i] /= norm;
        }

        return feature;
    }

private:
    ncnn::Net net;
};

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s [mode: register/recognize] [image_path]\n", argv[0]);
        fprintf(stderr, "Example: %s register my_face.jpg\n", argv[0]);
        return -1;
    }

    string mode = argv[1];
    string img_path = argv[2];

    // 模型和数据库路径
    string model_dir = "/root/model/";
    string scrfd_param = model_dir + "scrfd_500m_kps-opt2.param";
    string scrfd_bin   = model_dir + "scrfd_500m_kps-opt2.bin";
    string mfn_param   = model_dir + "mobilefacenets.param";
    string mfn_bin     = model_dir + "mobilefacenets.bin";
    string db_file     = "db.txt";

    // 加载人脸数据库
    std::map<string, std::vector<float>> face_db;
    if (load_db(db_file, face_db)) {
        std::cout << ">>> 成功加载文本数据库。当前库内人数: " << face_db.size() << std::endl;
    } else {
        std::cout << ">>> 未找到现有人脸数据库文本，将创建新库。" << std::endl;
    }

    // 初始化模型
    SCRFD detector(scrfd_param, scrfd_bin);
    MobileFaceNet recognizer(mfn_param, mfn_bin);

    // 读取输入图片（320x240 或 640x360 均可自动自适应）
    cv::Mat img = cv::imread(img_path, 1);
    if (img.empty()) {
        fprintf(stderr, "cv::imread %s 失败\n", img_path.c_str());
        return -1;
    }

    std::vector<FaceObject> faceobjects;
    detector.detect(img, faceobjects);

    if (faceobjects.empty()) {
        std::cout << "未在图像中检测到人脸。" << std::endl;
        return 0;
    }

    // 挑选画面中面积最大的人脸
    int max_idx = 0;
    float max_area = 0;
    for (size_t i = 0; i < faceobjects.size(); i++) {
        float area = faceobjects[i].rect.area();
        if (area > max_area) {
            max_area = area;
            max_idx = i;
        }
    }
    // 提取128维人脸特征
    std::vector<float> feature = recognizer.extract_feature(img, faceobjects[max_idx].pts);

    if (mode == "register") {
        // ---- 注册模式 ----
        string name;
        std::cout << "检测到人脸。请输入要绑定的姓名 (不要包含空格): ";
        std::cin >> name;

        if (face_db.find(name) != face_db.end()) {
            char choice;
            std::cout << "提示: '" << name << "' 已存在。是否覆盖原有特征？(y/n): ";
            std::cin >> choice;
            if (choice != 'y' && choice != 'Y') {
                std::cout << "取消注册。" << std::endl;
                return 0;
            }
        }

        face_db[name] = feature;
        
        // 保存为 txt 文本文件
        if (save_db(db_file, face_db)) {
            std::cout << ">>> 成功注册并追加写入文本。姓名: " << name << " -> 已同步到 " << db_file << std::endl;
        } else {
            std::cerr << ">>> 错误: 写入 " << db_file << " 失败！" << std::endl;
        }
    } 
    else if (mode == "recognize") {
        // ---- 识别模式 ----
        if (face_db.empty()) {
            std::cout << "数据库为空！请先使用 register 模式注册人脸。" << std::endl;
            return 0;
        }

        string best_name = "stranger";
        float best_score = 0.0f;
        float threshold = 0.45f; // 余弦相似度阈值

        for (const auto& pair : face_db) {
            float score = cosine_similarity(feature, pair.second);
            if (score > best_score) {
                best_score = score;
                best_name = pair.first;
            }
        }

        if (best_score >= threshold) {
            std::cout << "【识别结果】: " << best_name << " (相似度: " << best_score << ")" << std::endl;
        } else {
            std::cout << "【识别结果】: stranger (最高相似度: " << best_score << ")" << std::endl;
        }
    } else {
        std::cerr << "未知的运行模式: " << mode << "，请选择 register 或 recognize" << std::endl;
    }

    return 0;
}