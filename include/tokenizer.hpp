#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <array>

namespace llm {

class Tokenizer {
public:
    Tokenizer(
        const std::string& vocab_path,
        const std::string& merges_path
    );

    std::vector<int> encode(
        const std::string& text
    ) const;

    std::string decode(
        const std::vector<int>& ids
    ) const;

private:
    std::unordered_map<std::string, int> vocab;
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int> merge_rank;

    std::array<std::string, 256> byte_encoder;

    std::unordered_map<
        std::string,
        unsigned char
    > byte_decoder;

    std::string byte_encode(
        const std::string& text
    ) const;

    std::vector<std::string> bpe(
        const std::string& piece
    ) const;
};

}
