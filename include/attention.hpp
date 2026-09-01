#pragma once

#include "Tensor.hpp"
#include "loader.hpp"
#include <vector>


namespace llm {

std::vector<float> attention(
    const std::vector<std::vector<float>>& x,
    const Tensor& WQ,
    const Tensor& WK,
    const Tensor& WV,
    const Tensor& WO,
    const ModelConfig& config,
    int position
);

}

