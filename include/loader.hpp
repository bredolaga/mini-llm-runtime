#pragma once

#include <vector>
#include <string>
#include <cstddef>
#include "Tensor.hpp"
#include "safetensors.hh"

namespace llm {


struct ModelConfig {
    int dim;
    int hidden_dim;

    int num_layers;

    int num_heads;
    int num_kv_heads;

    int vocab_size;
    int max_position_embeddings;

    float rms_norm_eps;
    float rope_theta;

};

ModelConfig load_config(
	const std::string& path
);

class SafeTensorLoader {
public:
    explicit SafeTensorLoader(
        const std::string& path
    );

    void print_tensors() const;

    Tensor load_tensor(
        const std::string& name
    ) const;

private:
    safetensors::safetensors_t st_;
};

}
