#pragma once

#include <cstddef>
#include <vector>

namespace llm {

struct Tensor {
    std::vector<size_t> shape;
    std::vector<float> data;
};

}
