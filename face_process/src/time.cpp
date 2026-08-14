#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <map>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "net.h"

using namespace std;

struct FaceObject {
    cv::Rect_<float> rect;
    cv::Point2f pts[5];
    float prob;
};

float cosine_similarity(const std::vector<float>& v1, const std::vector<float>& v2) {
    float dot = 0.0f, norm1 = 0.0f, norm2 = 0.0f;
    for (size_t i = 0; i < v1.size(); ++i) {
        dot += v1[i] * v2[i];
        norm1 += v1[i] * v1[i];
        norm2 += v2[i] * v2[i];
    }
    return dot / (std::sqrt(norm1) * std::sqrt(norm2));
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
    return true;
}

class SCRFD {
public:
    bool init(const string& param, const string& bin) {
        scrfd.opt.use_vulkan_compute = false;
        scrfd.opt.num_threads = 4;
        if (scrfd.load_param(param.c_str()) != 0) return false;
        if (scrfd.load_model(bin.c_str()) != 0) return false;
        return true;
    }

    int detect(const cv::Mat& bgr, std::vector<FaceObject>& faceobjects, float prob_threshold = 0.5f) {
        int width = bgr.cols, height = bgr.rows;
        int target_size = 640;
        int w = width, h = height;
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
        ncnn::copy_make_border(in, in_pad, hpad/2, hpad - hpad/2, wpad/2, wpad - wpad/2, ncnn::BORDER_CONSTANT, 0.f);

        const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
        const float norm_vals[3] = {1/128.0f, 1/128.0f, 1/128.0f};
        in_pad.substract_mean_normalize(mean_vals, norm_vals);

        ncnn::Extractor ex = scrfd.create_extractor();
        ex.input("input.1", in_pad);

        std::vector<FaceObject> proposals;
        int strides[] = {8, 16, 32};
        for (int i = 0; i < 3; i++) {
            int stride = strides[i];
            char score_name[32], bbox_name[32], kps_name[32];
            sprintf(score_name, "score_%d", stride);
            sprintf(bbox_name, "bbox_%d", stride);
            sprintf(kps_name, "kps_%d", stride);

            ncnn::Mat score_blob, bbox_blob, kps_blob;
            if (ex.extract(score_name, score_blob) != 0 ||
                ex.extract(bbox_name, bbox_blob) != 0 ||
                ex.extract(kps_name, kps_blob) != 0)
                return -1;

            int base_w = in_pad.w / stride, base_h = in_pad.h / stride;
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
                                obj.pts[k].x = cx + kps_ptr[k*2] * stride;
                                obj.pts[k].y = cy + kps_ptr[k*2+1] * stride;
                            }
                            proposals.push_back(obj);
                        }
                        score_ptr++; bbox_ptr += 4; kps_ptr += 10;
                    }
                }
            }
        }

        // NMS
        std::sort(proposals.begin(), proposals.end(),
                  [](const FaceObject& a, const FaceObject& b) { return a.prob > b.prob; });
        std::vector<int> picked;
        for (size_t i = 0; i < proposals.size(); i++) {
            int keep = 1;
            for (int j : picked) {
                cv::Rect_<float> inter = proposals[i].rect & proposals[j].rect;
                if (inter.area() / (proposals[i].rect.area() + proposals[j].rect.area() - inter.area()) > 0.4f) {
                    keep = 0; break;
                }
            }
            if (keep) picked.push_back(i);
        }

        faceobjects.resize(picked.size());
        for (size_t i = 0; i < picked.size(); i++) {
            faceobjects[i] = proposals[picked[i]];
            float x_off = wpad/2.0f, y_off = hpad/2.0f;
            faceobjects[i].rect.x = (faceobjects[i].rect.x - x_off) / scale;
            faceobjects[i].rect.y = (faceobjects[i].rect.y - y_off) / scale;
            faceobjects[i].rect.width /= scale;
            faceobjects[i].rect.height /= scale;
            for (int k = 0; k < 5; k++) {
                faceobjects[i].pts[k].x = (faceobjects[i].pts[k].x - x_off) / scale;
                faceobjects[i].pts[k].y = (faceobjects[i].pts[k].y - y_off) / scale;
            }
        }
        return 0;
    }

private:
    ncnn::Net scrfd;
};

class MobileFaceNet {
public:
    bool init(const string& param, const string& bin) {
        net.opt.use_vulkan_compute = false;
        net.opt.num_threads = 4;
        if (net.load_param(param.c_str()) != 0) return false;
        if (net.load_model(bin.c_str()) != 0) return false;
        return true;
    }

    std::vector<float> extract_feature(const cv::Mat& bgr, const cv::Point2f pts[5]) {
        float dst[5][2] = {
            {38.2946f, 51.6963f}, {73.5318f, 51.5014f},
            {56.0252f, 71.7366f},
            {41.5493f, 92.3655f}, {70.7299f, 92.2041f}
        };
        cv::Mat src(5, 2, CV_32FC1), dst_mat(5, 2, CV_32FC1, dst);
        for (int i = 0; i < 5; i++) {
            src.at<float>(i,0) = pts[i].x;
            src.at<float>(i,1) = pts[i].y;
        }
        cv::Mat M = cv::estimateAffinePartial2D(src, dst_mat);
        cv::Mat aligned;
        cv::warpAffine(bgr, aligned, M, cv::Size(112,112), cv::INTER_LINEAR);

        ncnn::Mat in = ncnn::Mat::from_pixels(aligned.data, ncnn::Mat::PIXEL_BGR2RGB, 112, 112);
        const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
        const float norm_vals[3] = {1/128.0f, 1/128.0f, 1/128.0f};
        in.substract_mean_normalize(mean_vals, norm_vals);

        ncnn::Extractor ex = net.create_extractor();
        ex.input("data", in);
        ncnn::Mat out;
        if (ex.extract("fc1", out) != 0) {
            if (ex.extract("embedding", out) != 0) {
                return std::vector<float>();
            }
        }

        std::vector<float> feature(out.w);
        float norm = 0.f;
        for (int i = 0; i < out.w; i++) {
            feature[i] = out[i];
            norm += feature[i] * feature[i];
        }
        norm = std::sqrt(norm);
        for (int i = 0; i < out.w; i++) feature[i] /= norm;
        return feature;
    }

private:
    ncnn::Net net;
};

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

    // ========== 计时开始 ==========
    auto start_time = std::chrono::high_resolution_clock::now();

    string img_path = argv[1];
    string model_dir = "/root/model/";
    string scrfd_param = model_dir + "scrfd_500m_kps-opt2.param";
    string scrfd_bin   = model_dir + "scrfd_500m_kps-opt2.bin";
    string mfn_param   = model_dir + "mobilefacenets.param";
    string mfn_bin     = model_dir + "mobilefacenets.bin";
    string db_file     = "db.txt";

    std::map<string, std::vector<float>> face_db;
    load_db(db_file, face_db);

    SCRFD detector;
    MobileFaceNet recognizer;
    if (!detector.init(scrfd_param, scrfd_bin) || !recognizer.init(mfn_param, mfn_bin)) {
        std::cerr << "Model init failed" << std::endl;
        return -1;
    }

    cv::Mat img = cv::imread(img_path, 1);
    if (img.empty()) {
        std::cerr << "Failed to read image" << std::endl;
        return -1;
    }

    std::vector<FaceObject> faces;
    if (detector.detect(img, faces) != 0 || faces.empty()) {
        std::cout << "stranger" << std::endl;
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        std::cerr << "Total time: " << duration << " ms" << std::endl;
        return 0;
    }

    int max_idx = 0;
    float max_area = 0;
    for (size_t i = 0; i < faces.size(); i++) {
        float area = faces[i].rect.area();
        if (area > max_area) {
            max_area = area;
            max_idx = i;
        }
    }

    std::vector<float> feat = recognizer.extract_feature(img, faces[max_idx].pts);
    if (feat.empty() || feat.size() < 10) {
        std::cerr << "Feature extraction failed" << std::endl;
        return -1;
    }

    if (face_db.empty()) {
        std::cout << "stranger" << std::endl;
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        std::cerr << "Total time: " << duration << " ms" << std::endl;
        return 0;
    }

    string best_name = "stranger";
    float best_score = 0.0f;
    float threshold = 0.45f;
    for (const auto& p : face_db) {
        float sim = cosine_similarity(feat, p.second);
        if (sim > best_score) {
            best_score = sim;
            best_name = p.first;
        }
    }

    std::cout << std::fixed << std::setprecision(3);
    if (best_score >= threshold)
        std::cout << best_name << "," << best_score << std::endl;
    else
        std::cout << "stranger" << std::endl;

    // ========== 计时结束 ==========
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cerr << "Total time: " << duration << " ms" << std::endl;

    return 0;
}
