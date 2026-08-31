#pragma once
#include "types.h"
#include <string>

std::string result_to_json(const PipelineResult& result,
                           bool include_timing,
                           const std::string& mode = "infer",
                           const std::string& action = "");
