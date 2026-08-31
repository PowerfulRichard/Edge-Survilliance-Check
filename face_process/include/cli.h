#pragma once
#include "face_pipeline.h"
#include <string>
#include <unordered_map>

using ArgMap = std::unordered_map<std::string, std::string>;
ArgMap parse_args(int argc, char** argv);
std::string arg_get(const ArgMap& args, const std::string& key, const std::string& def = "");
bool build_config(const ArgMap& args, PipelineConfig& cfg, std::string& image_path, std::string& error);
void print_usage_main();
void print_usage_mini();
