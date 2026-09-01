#pragma once

#include <vector>
#include "loader.hpp"
#include "Tensor.hpp"

namespace llm {

struct TransformerBlockWeights {
    Tensor attention_norm;

    Tensor wq;
    Tensor wk;
    Tensor wv;
    Tensor wo;

    Tensor mlp_norm;

    Tensor wgate;
    Tensor wup;
    Tensor wdown;
};

TransformerBlockWeights load_block(
    const SafeTensorLoader& loader,
    int layer
);

std::vector<std::vector<float>> transformer_block (
	const std::vector<std::vector<float>>& X,
	const TransformerBlockWeights& weights,
    const ModelConfig& config
);

}
