#define SAFETENSORS_CPP_IMPLEMENTATION
#include <fstream>
#include <stdexcept>
#include <nlohmann/json.hpp>
#include "safetensors.hh"
#include <string>
#include <loader.hpp>
#include <bit>
#include <cstdint>
#include <cstring>

using json = nlohmann::json;

namespace llm {

ModelConfig load_config(
	const std::string& path
) {

    std::ifstream file(path);

    if (!file) {
        throw std::runtime_error(
            "Cannot open config: " + path
        );
    }

    json j = json::parse(file);

    ModelConfig config;

    config.dim =
        j.at("hidden_size").get<int>();

    config.hidden_dim =
        j.at("intermediate_size").get<int>();

    config.num_layers =
        j.at("num_hidden_layers").get<int>();

    config.num_heads =
        j.at("num_attention_heads").get<int>();

    config.num_kv_heads =
        j.at("num_key_value_heads").get<int>();

    config.vocab_size =
        j.at("vocab_size").get<int>();

    config.max_position_embeddings =
        j.at("max_position_embeddings").get<int>();

    config.rms_norm_eps =
        j.at("rms_norm_eps").get<float>();

    config.rope_theta =
        j.at("rope_theta").get<float>();

    return config;
}

SafeTensorLoader::SafeTensorLoader(
    const std::string& path
) {
    std::string warn;
    std::string err;

    bool ok =
        safetensors::load_from_file(
            path,
            &st_,
            &warn,
            &err
        );

    if (!ok) {
        throw std::runtime_error(
            "Failed to load safetensors: " + err
        );
    }

    if (!safetensors::validate_data_offsets(
            st_,
            err
        )) {
        throw std::runtime_error(
            "Invalid safetensors offsets: " + err
        );
    }
}

Tensor SafeTensorLoader::load_tensor(
    const std::string& name
) const {
    safetensors::tensor_t tensor;
    bool found = false;

    for (size_t i = 0; i < st_.tensors.size(); ++i) {
        if (st_.tensors.keys()[i] == name) {
            if (!st_.tensors.at(i, &tensor)) {
                throw std::runtime_error(
                    "Failed to read tensor metadata: " + name
                );
            }

            found = true;
            break;
        }
    }

    if (!found) {
        throw std::runtime_error(
            "Tensor not found: " + name
        );
    }

    if (tensor.dtype != safetensors::dtype::kBFLOAT16) {
        throw std::runtime_error(
            "Tensor is not BF16: " + name
        );
    }

    Tensor result;

    for (auto dim : tensor.shape) {
        result.shape.push_back(
            static_cast<std::size_t>(dim)
        );
    }

    size_t num_elements =
        safetensors::get_shape_size(tensor);

    result.data.resize(num_elements);
    const uint8_t* buffer = nullptr;

    if (st_.mmaped) {
        buffer = st_.databuffer_addr;
    } else {
        buffer = st_.storage.data();
    }

    if (buffer == nullptr) {
        throw std::runtime_error(
            "Safetensors data buffer is null"
        );
    }

    const uint8_t* tensor_data =
        buffer + tensor.data_offsets[0];

    for (size_t i = 0; i < num_elements; ++i) {
        uint16_t bf16;

        std::memcpy(
            &bf16,
            tensor_data + i * sizeof(uint16_t),
            sizeof(uint16_t)
        );

        uint32_t fp32_bits =
            static_cast<uint32_t>(bf16) << 16;

        result.data[i] =
            std::bit_cast<float>(fp32_bits);
    }

    return result;
}

}
