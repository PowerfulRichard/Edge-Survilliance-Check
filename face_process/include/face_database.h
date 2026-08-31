#pragma once
#include "types.h"
#include <string>
#include <utility>
#include <vector>

class FaceDatabase {
public:
    bool load(const std::string& path);
    bool save(const std::string& path) const;
    void clear();

    void add_template(const std::string& name, const std::vector<float>& embedding);
    MatchResult match(const std::vector<float>& query, float threshold) const;
    size_t size() const { return entries_.size(); }
    size_t count_for_name(const std::string& name) const;

    static float cosine(const std::vector<float>& a, const std::vector<float>& b);

private:
    std::vector<std::pair<std::string, std::vector<float>>> entries_;
};
