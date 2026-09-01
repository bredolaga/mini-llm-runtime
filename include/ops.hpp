#pragma once

#include <vector>
#include "Tensor.hpp"

namespace llm {

std::vector<float> multiply (
    const std::vector<float>& a,
    const std::vector<float>& b
    );

std::vector<float> matvec(
    const Tensor& matrix,
    const std::vector<float>& x
);

float dot(
    const std::vector<float>& a,
    const std::vector<float>& b
);

std::vector<float> softmax(
    const std::vector<float>& x
);

std::vector<float> rmsnorm(
    const std::vector<float>& x,
    const std::vector<float>& weight,
    float eps
);

std::vector<float> add(
    const std::vector<float>& a,
    const std::vector<float>& b
);

std::vector<float> SiLU (
	const std::vector<float>& x
);

std::vector<float> rope(
    const std::vector<float>& x,
    int position,
    float base
);

}
