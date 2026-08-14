#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <map>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include "net.h"

using namespace std;

struct FaceObject {
    cv::Rect_<float> rect;
    cv::Point2f pts[5];
    float prob;
};

float cosine_similarity(const std::vector<float>& v1, const std::vector<float>& v2) {
    float dot_product = 0.0f, norm1 = 0.0f, norm2 = 0.0f;
    for (size_t i = 0; i < v1.size(); ++i) {
        dot_product += v1[i] * v2[i];
        norm1 += v1[i] * v1[i];
        norm2 += v2[i] * v2[i];
    }
    return dot_product / (std::sqrt(norm1) * std::sqrt(norm2));
}

bool load_db(const string& filename, std::map<string, std::vector<float>>& db) {
    std::ifstream f(filename);
    if (!f.is_open()) return false;
    string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        string name;
        ss >> name;
        std::vector<float> feature;
        float val;
        while (ss >> val) feature.push_back(val);
        if (feature.size() == 128) db[name] = feature;
    }
    f.close();
    return true;
}

bool save_db(const string& filename, const std::map<string, std::vector<float>>& db) {
    std::ofstream f(filename);
    if (!f.is_open()) return false;
    f << std::fixed << std::setprecision(6);
    for (const auto& pair : db) {
        f << pair.first;
        for (float val : pair.second) f << " " << val;
        f << "\n";
    }
    f.close();
    return true;
}

// SCRFD 类
class SCRFD {
public:
    SCRFD() {}
    
    bool init(const string& param_path, const string& bin_path) {
        scrfd.opt.use_vulkan_compute = false; 
        scrfd.opt.num_threads = 4;
        if (scrfd.load_param(param_path.c_str()) != 0) {
            std::cerr << "【模型错误】无法加载 SCRFD param: " << param_path << std::endl;
            return false;
        }
        if (scrfd.load_model(bin_path.c_str()) != 0) {
            std::cerr << "【模型错误】无法加载 SCRFD bin: " << bin_path << std::endl;
            return false;
        }
        return true;
    }

    int detect(const cv::Mat& bgr, std::vector<FaceObject>& faceobjects, float prob_threshold = 0.5f) {
        int width = bgr.cols; int height = bgr.rows;
        int target_size = 640; int w = width; int h = height; float scale = 1.f;
        if (w > h) { scale = (float)target_size / w; w = target_size; h = h * scale; }
        else { scale = (float)target_size / h; h = target_size; w = w * scale; }

        ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, width, height, w, h);
        int wpad = (w + 31) / 32 * 32 - w; int hpad = (h + 31) / 32 * 32 - h;
        ncnn::Mat in_pad;
        ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.f);

        const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
        const float norm_vals[3] = {1.0f/128.0f, 1.0f/128.0f, 1.0f/128.0f};
        in_pad.substract_mean_normalize(mean_vals, norm_vals);

        ncnn::Extractor ex = scrfd.create_extractor();
        
        // 核心修改点：根据 NCNN 报错提示，将 "data" 改为 "input.1"
        ex.input("input.1", in_pad); 
        
        std::vector<FaceObject> proposals;
        int strides[] = {8, 16, 32};
        
        for (int i = 0; i < 3; i++) {
            int stride = strides[i];
            char score_blob_name[32], bbox_blob_name[32], kps_blob_name[32];
            sprintf(score_blob_name, "score_%d", stride);
            sprintf(bbox_blob_name, "bbox_%d", stride);
            sprintf(kps_blob_name, "kps_%d", stride);

            ncnn::Mat score_blob, bbox_blob, kps_blob;
            
            // 增加防御性代码：如果提取输出层失败，给出明确提示
            if (ex.extract(score_blob_name, score_blob) != 0 ||
                ex.extract(bbox_blob_name, bbox_blob) != 0 ||
                ex.extract(kps_blob_name, kps_blob) != 0) {
                std::cerr << "【模型层名不匹配】提取 SCRFD 输出层失败 (" << score_blob_name << ")。" << std::endl;
                std::cerr << "请检查你的 .param 文件，看看里面的输出层是不是数字（例如 814, 815 等）而不是 score_8。" << std::endl;
                return -1;
            }

            int base_w = in_pad.w / stride; int base_h = in_pad.h / stride;
            for (int q = 0; q < score_blob.c; q++) {
                const float* score_ptr = score_blob.channel(q);
                const float* bbox_ptr = bbox_blob.channel(q);
                const float* kps_ptr = kps_blob.channel(q);
                for (int y = 0; y < base_h; y++) {
                    for (int x = 0; x < base_w; x++) {
                        float score = score_ptr[0];
                        if (score > prob_threshold) {
                            float cx = (x + 0.5f) * stride; float cy = (y + 0.5f) * stride;
                            FaceObject obj; obj.prob = score;
                            obj.rect.x = cx - bbox_ptr[0] * stride; obj.rect.y = cy - bbox_ptr[1] * stride;
                            obj.rect.width = cx + bbox_ptr[2] * stride - obj.rect.x; obj.rect.height = cy + bbox_ptr[3] * stride - obj.rect.y;
                            for (int k = 0; k < 5; k++) {
                                obj.pts[k].x = cx + kps_ptr[k * 2] * stride; obj.pts[k].y = cy + kps_ptr[k * 2 + 1] * stride;
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
                if (inter.area() / (proposals[i].rect.area() + proposals[j].rect.area() - inter.area()) > 0.4f) { keep = 0; break; }
            }
            if (keep) picked.push_back(i);
        }

        faceobjects.resize(picked.size());
        for (size_t i = 0; i < picked.size(); i++) {
            faceobjects[i] = proposals[picked[i]];
            float x_offset = (wpad / 2.0f); float y_offset = (hpad / 2.0f);
            faceobjects[i].rect.x = (faceobjects[i].rect.x - x_offset) / scale; faceobjects[i].rect.y = (faceobjects[i].rect.y - y_offset) / scale;
            faceobjects[i].rect.width /= scale; faceobjects[i].rect.height /= scale;
            for (int k = 0; k < 5; k++) {
                faceobjects[i].pts[k].x = (faceobjects[i].pts[k].x - x_offset) / scale; faceobjects[i].pts[k].y = (faceobjects[i].pts[k].y - y_offset) / scale;
            }
        }
        return 0;
    }

private:
    ncnn::Net scrfd;
};
// MobileFaceNet 类
class MobileFaceNet {
public:
    MobileFaceNet() {}
    
    bool init(const string& param_path, const string& bin_path) {
        net.opt.use_vulkan_compute = false;
        net.opt.num_threads = 4;
        if (net.load_param(param_path.c_str()) != 0) {
            std::cerr << "【模型错误】无法加载 MobileFaceNet param: " << param_path << std::endl;
            return false;
        }
        if (net.load_model(bin_path.c_str()) != 0) {
            std::cerr << "【模型错误】无法加载 MobileFaceNet bin: " << bin_path << std::endl;
            return false;
        }
        return true;
    }

    std::vector<float> extract_feature(const cv::Mat& bgr, const cv::Point2f pts[5]) {
        float dst[5][2] = { {38.2946f, 51.6963f}, {73.5318f, 51.5014f}, {56.0252f, 71.7366f}, {41.5493f, 92.3655f}, {70.7299f, 92.2041f} };
        cv::Mat src_mat(5, 2, CV_32FC1); cv::Mat dst_mat(5, 2, CV_32FC1, dst);
        for (int i = 0; i < 5; i++) { src_mat.at<float>(i, 0) = pts[i].x; src_mat.at<float>(i, 1) = pts[i].y; }
        cv::Mat M = cv::estimateAffinePartial2D(src_mat, dst_mat);
        cv::Mat aligned_face;
        cv::warpAffine(bgr, aligned_face, M, cv::Size(112, 112), cv::INTER_LINEAR);

        ncnn::Mat in = ncnn::Mat::from_pixels(aligned_face.data, ncnn::Mat::PIXEL_BGR2RGB, 112, 112);
        const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
        const float norm_vals[3] = {1.0f/128.0f, 1.0f/128.0f, 1.0f/128.0f};
        in.substract_mean_normalize(mean_vals, norm_vals);

        ncnn::Extractor ex = net.create_extractor();
        
        // 提示：MobileFaceNet 也有可能输入层叫 "input.1"，如果后面这步报错，可以参照 SCRFD 进行微调
        ex.input("data", in);
        ncnn::Mat out; 
        
        if (ex.extract("fc1", out) != 0) {
            // 尝试另一种常见的输出层命名
            if (ex.extract("embedding", out) != 0) {
                std::cerr << "【警告】MobileFaceNet 提取特征层失败，默认使用了 'fc1' 和 'embedding' 均未匹配上。" << std::endl;
            }
        } 

        std::vector<float> feature(out.w);
        float norm = 0.f;
        for (int i = 0; i < out.w; i++) { feature[i] = out[i]; norm += feature[i] * feature[i]; }
        norm = std::sqrt(norm);
        for (int i = 0; i < out.w; i++) feature[i] /= norm;
        return feature;
    }

private:
    ncnn::Net net;
};

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s [mode: register/recognize] [image_path]\n", argv[0]);
        return -1;
    }

    string mode = argv[1];
    string img_path = argv[2];

    string model_dir = "/root/model/";
    string scrfd_param = model_dir + "scrfd_500m_kps-opt2.param";
    string scrfd_bin   = model_dir + "scrfd_500m_kps-opt2.bin";
    string mfn_param   = model_dir + "mobilefacenets.param";
    string mfn_bin     = model_dir + "mobilefacenets.bin";
    string db_file     = "db.txt";

    SCRFD detector;
    MobileFaceNet recognizer;
    if (!detector.init(scrfd_param, scrfd_bin) || !recognizer.init(mfn_param, mfn_bin)) {
        return -1;
    }

    std::map<string, std::vector<float>> face_db;
    if (load_db(db_file, face_db)) {
        std::cout << ">>> 成功加载文本数据库。当前库内人数: " << face_db.size() << std::endl;
    } else {
        std::cout << ">>> 未找到现有人脸数据库文本，将创建新库。" << std::endl;
    }

    cv::Mat img = cv::imread(img_path, 1);
    if (img.empty()) {
        std::cerr << "【错误】OpenCV 读取或解码图片失败: " << img_path << std::endl;
        return -1;
    }

    std::vector<FaceObject> faceobjects;
    if (detector.detect(img, faceobjects) != 0) {
        return -1; // 检测内部发生层名错误，提早退出
    }

    if (faceobjects.empty()) {
        std::cout << "未在图像中检测到人脸。" << std::endl;
        return 0;
    }

    int max_idx = 0; float max_area = 0;
    for (size_t i = 0; i < faceobjects.size(); i++) {
        float area = faceobjects[i].rect.area();
        if (area > max_area) { max_area = area; max_idx = i; }
    }

    std::vector<float> feature = recognizer.extract_feature(img, faceobjects[max_idx].pts);
    if (feature.empty() || feature.size() < 10) {
        std::cerr << "【错误】提取到的特征向量异常，无法进行保存或比对。" << std::endl;
        return -1;
    }

    if (mode == "register") {
        string name;
        std::cout << "检测到人脸。请输入要绑定的姓名 (不要包含空格): ";
        std::cin >> name;

        if (face_db.find(name) != face_db.end()) {
            char choice;
            std::cout << "提示: '" << name << "' 已存在。是否覆盖原有特征？(y/n): ";
            std::cin >> choice;
            if (choice != 'y' && choice != 'Y') { std::cout << "取消注册。" << std::endl; return 0; }
        }

        face_db[name] = feature;
        if (save_db(db_file, face_db)) {
            std::cout << ">>> 成功注册并追加写入文本。姓名: " << name << " -> 已同步到 " << db_file << std::endl;
        } else {
            std::cerr << ">>> 错误: 写入 " << db_file << " 失败！" << std::endl;
        }
    } 
    else if (mode == "recognize") {
        if (face_db.empty()) { std::cout << "数据库为空！请先使用 register 模式注册人脸。" << std::endl; return 0; }
        string best_name = "stranger"; float best_score = 0.0f; float threshold = 0.45f;

        for (const auto& pair : face_db) {
            float score = cosine_similarity(feature, pair.second);
            if (score > best_score) { best_score = score; best_name = pair.first; }
        }

        if (best_score >= threshold) {
            std::cout << "【识别结果】: " << best_name << " (相似度: " << best_score << ")" << std::endl;
        } else {
            std::cout << "【识别结果】: stranger (最高相似度: " << best_score << ")" << std::endl;
        }
    }

    return 0;
}
