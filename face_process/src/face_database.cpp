#include "face_database.h"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

void FaceDatabase::clear() { entries_.clear(); }

void FaceDatabase::add_template(const std::string& name, const std::vector<float>& embedding) {
    entries_.push_back({name, embedding});
}

size_t FaceDatabase::count_for_name(const std::string& name) const {
    size_t n = 0;
    for (const auto& e : entries_) if (e.first == name) ++n;
    return n;
}

float FaceDatabase::cosine(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.empty() || a.size() != b.size()) return -1.f;
    double dot = 0.0, aa = 0.0, bb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        aa += static_cast<double>(a[i]) * a[i];
        bb += static_cast<double>(b[i]) * b[i];
    }
    if (aa < 1e-20 || bb < 1e-20) return -1.f;
    return static_cast<float>(dot / std::sqrt(aa * bb));
}

MatchResult FaceDatabase::match(const std::vector<float>& query, float threshold) const {
    MatchResult r;
    for (const auto& e : entries_) {
        const float sim = cosine(query, e.second);
        if (sim > r.similarity) {
            r.similarity = sim;
            r.name = e.first;
        }
    }
    r.known = !entries_.empty() && r.similarity >= threshold;
    if (!r.known) r.name = "UNKNOWN";
    return r;
}

bool FaceDatabase::load(const std::string& path) {
    std::ifstream fin(path);
    if (!fin) return false;
    entries_.clear();

    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        std::string name;
        if (!line.empty() && line[0] == '"') iss >> std::quoted(name);
        else iss >> name; // backward-compatible with the old whitespace format

        std::vector<float> emb;
        float v = 0.f;
        while (iss >> v) emb.push_back(v);
        if (!name.empty() && emb.size() == 512) entries_.push_back({name, std::move(emb)});
    }
    return true;
}

bool FaceDatabase::save(const std::string& path) const {
    const std::filesystem::path p(path);
    if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());

    const std::string tmp = path + ".tmp";
    std::ofstream fout(tmp, std::ios::trunc);
    if (!fout) return false;

    fout << "# FACE_DB_V2 embedding_dim=512 templates=" << entries_.size() << '\n';
    fout << std::setprecision(9);
    for (const auto& e : entries_) {
        fout << std::quoted(e.first);
        for (float x : e.second) fout << ' ' << x;
        fout << '\n';
    }
    fout.close();
    if (!fout) return false;

    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        std::filesystem::remove(path, ec);
        ec.clear();
        std::filesystem::rename(tmp, path, ec);
    }
    return !ec;
}
