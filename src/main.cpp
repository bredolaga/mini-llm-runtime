#include <iostream>
#include <string>
#include <vector>

#include "ops.hpp"
#include "attention.hpp"
#include "mlp.hpp"
#include "model.hpp"
#include "loader.hpp"
#include "tokenizer.hpp"

int main() {

    // 1. Load model config

    auto config = llm::load_config(
        "../models/SmolLM2-135M/config.json"
    );


    // 2. Load model weights

    llm::SafeTensorLoader loader(
        "../models/SmolLM2-135M/model.safetensors"
    );


    // 3. Load tokenizer

    llm::Tokenizer tokenizer(
        "../models/SmolLM2-135M/vocab.json",
        "../models/SmolLM2-135M/merges.txt"
    );


    // 4. Load embeddings

    auto embeddings = loader.load_tensor(
        "model.embed_tokens.weight"
    );


    // 5. Load final RMSNorm

    auto final_norm = loader.load_tensor(
        "model.norm.weight"
    );


    // 6. Load all Transformer blocks

    std::vector<llm::TransformerBlockWeights> blocks;

    blocks.reserve(config.num_layers);

    for (int layer = 0;
         layer < config.num_layers;
         ++layer) {

        blocks.push_back(
            llm::load_block(loader, layer)
        );
    }


    // 7. Read prompt

    std::cout << "Prompt: ";

    std::string prompt;
    std::getline(std::cin, prompt);

    auto ids = tokenizer.encode(prompt);

    if (ids.empty()) {
        std::cerr << "Tokenizer returned no tokens\n";
        return 1;
    }


    std::cout << "\nGenerated:\n";
    std::cout << prompt << std::flush;


    // 8. Autoregressive generation

    const int max_new_tokens = 20;

    for (int step = 0;
         step < max_new_tokens;
         ++step) {

        if (ids.size() >=
            static_cast<std::size_t>(
                config.max_position_embeddings
            )) {

            break;
        }


        // IDs > embeddings

        std::vector<std::vector<float>> X;

        X.reserve(ids.size());

        for (int token_id : ids) {

            auto begin =
                embeddings.data.begin()
                + token_id * config.dim;

            X.emplace_back(
                begin,
                begin + config.dim
            );
        }


        // 30 Transformer blocks

        auto hidden = X;

        for (int layer = 0;
             layer < config.num_layers;
             ++layer) {

            hidden = llm::transformer_block(
                hidden,
                blocks[layer],
                config
            );
        }


        // Final RMSNorm

        auto normalized = llm::rmsnorm(
            hidden.back(),
            final_norm.data,
            config.rms_norm_eps
        );


        // Hidden > logits
        // tied embeddings

        auto logits = llm::matvec(
            embeddings,
            normalized
        );


        // Greedy decoding
        // argmax(logits)

        int next_token = 0;

        for (int i = 1;
             i < static_cast<int>(logits.size());
             ++i) {

            if (logits[i] > logits[next_token]) {
                next_token = i;
            }
        }


        // Append token to context

        ids.push_back(next_token);


        // Decode and print token

        std::cout
            << tokenizer.decode({next_token})
            << std::flush;


        // EOS

        if (next_token == 0) {
            break;
        }
    }


    std::cout << "\n";

    return 0;
}
