#pragma once

#include "Tensor.hpp"
#include <vector>

namespace llm {

std::vector<float> mlp (
	const std::vector<float>& x,
	const Tensor& wgate,
	const Tensor& wup,
	const Tensor& wdown
);

}
