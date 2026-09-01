#include "tokenizer.hpp"
#include <fstream>
#include <stdexcept>
#include <nlohmann/json.hpp>
#include <limits>
#include <utility>
#include <regex>

namespace llm {


static std::string utf8_from_codepoint(
    unsigned int cp
) {
    std::string result;

    if (cp <= 0x7F) {
        result.push_back(
            static_cast<char>(cp)
        );
    }
    else if (cp <= 0x7FF) {
        result.push_back(
            static_cast<char>(
                0xC0 | (cp >> 6)
            )
        );

        result.push_back(
            static_cast<char>(
                0x80 | (cp & 0x3F)
            )
        );
    }
    else {
        result.push_back(
            static_cast<char>(
                0xE0 | (cp >> 12)
            )
        );

        result.push_back(
            static_cast<char>(
                0x80 | ((cp >> 6) & 0x3F)
            )
        );

        result.push_back(
            static_cast<char>(
                0x80 | (cp & 0x3F)
            )
        );

    }
    return result;
}

static std::vector<std::string> pretokenize(
    const std::string& text
) {
    std::vector<std::string> pieces;

    std::regex pattern(
        R"('s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^ \t\r\nA-Za-z0-9]+|[ \t\r\n]+)"
    );

    auto begin =
        std::sregex_iterator(
            text.begin(),
            text.end(),
            pattern
        );

    auto end =
        std::sregex_iterator();

    for (auto it = begin; it != end; ++it) {
        pieces.push_back(
            it->str()
        );
    }

    return pieces;
}

static std::array<std::string, 256>
make_byte_encoder() {

    std::array<std::string, 256> result;

    std::vector<int> bytes;
    std::vector<int> codepoints;

    for (int b = 33; b <= 126; ++b) {
        bytes.push_back(b);
    }

    for (int b = 161; b <= 172; ++b) {
        bytes.push_back(b);
    }

    for (int b = 174; b <= 255; ++b) {
        bytes.push_back(b);
    }

    codepoints = bytes;

    bool used[256] = {};

    for (int b : bytes) {
        used[b] = true;
    }

    int extra = 0;

    for (int b = 0; b < 256; ++b) {

        if (!used[b]) {
            bytes.push_back(b);

            codepoints.push_back(
                256 + extra
            );

            ++extra;
        }
    }

    
    for (int i = 0;
         i < static_cast<int>(bytes.size());
         ++i) {

        result[bytes[i]] =
            utf8_from_codepoint(
                codepoints[i]
            );
    }

    return result;
}

static std::vector<std::string>
split_utf8_chars(const std::string& text) {

    std::vector<std::string> result;

    for (std::size_t i = 0; i < text.size();) {

        unsigned char c =
            static_cast<unsigned char>(text[i]);

        std::size_t len = 1;

        if ((c & 0x80) == 0x00) {
            len = 1;
        }
        else if ((c & 0xE0) == 0xC0) {
            len = 2;
        }
        else if ((c & 0xF0) == 0xE0) {
            len = 3;
        }
        else if ((c & 0xF8) == 0xF0) {
            len = 4;
        }

        result.push_back(
            text.substr(i, len)
        );

        i += len;
    }

    return result;
}

Tokenizer::Tokenizer(
    const std::string& vocab_path,
    const std::string& merges_path
) {
    std::ifstream file(vocab_path);

    if (!file) {
        throw std::runtime_error(
            "Cannot open vocab: " + vocab_path
        );
    }

    nlohmann::json j;
    file >> j;

    int max_id = 0;

    for (auto it = j.begin(); it != j.end(); ++it) {
        int id = it.value().get<int>();

        vocab[it.key()] = id;

        if (id > max_id) {
            max_id = id;
        }
    }

    id_to_token.resize(max_id + 1);

    for (const auto& [token, id] : vocab) {
        id_to_token[id] = token;
    }
    
    std::ifstream merges_file(merges_path);

    if (!merges_file) {
        throw std::runtime_error(
                "Cannot open merges: " + merges_path
            );
    }

    std::string line;
    int rank = 0;

    while (std::getline(merges_file, line)) {
        
        if (line.empty() || line[0] == '#') {
            continue;
        }

    merge_rank[line] = rank;
    ++rank;
}

byte_encoder = make_byte_encoder();

for (int b = 0; b < 256; ++b) {
    byte_decoder[
        byte_encoder[b]
    ] = static_cast<unsigned char>(b);
}

}


std::string Tokenizer::byte_encode(
        const std::string& text
    ) const {
        std::string result;

        for (unsigned char byte : text) {
            result += byte_encoder[byte];
        }

    return result;
}

std::vector<std::string> Tokenizer::bpe(
    const std::string& piece
) const {

    std::vector<std::string> parts =
        split_utf8_chars(piece);

    if (parts.size() <= 1) {
        return parts;
    }

    while (true) {

        int best_rank =
            std::numeric_limits<int>::max();

        std::string best_left;
        std::string best_right;

        bool found = false;

        for (std::size_t i = 0;
             i + 1 < parts.size();
             ++i) {

            std::string key =
                parts[i] + " " + parts[i + 1];

            auto it = merge_rank.find(key);

            if (it != merge_rank.end() &&
                it->second < best_rank) {

                best_rank = it->second;

                best_left = parts[i];
                best_right = parts[i + 1];

                found = true;
            }
        }

        if (!found) {
            break;
        }

        std::vector<std::string> merged;

        for (std::size_t i = 0;
             i < parts.size();) {

            if (
                i + 1 < parts.size() &&
                parts[i] == best_left &&
                parts[i + 1] == best_right
            ) {
                merged.push_back(
                    parts[i] + parts[i + 1]
                );

                i += 2;
            }
            else {
                merged.push_back(parts[i]);
                ++i;
            }
        }

        parts = std::move(merged);
    }

    return parts;
}

std::vector<int> Tokenizer::encode(
    const std::string& text
) const {
    std::vector<int> ids;

    auto pieces = pretokenize(text);

    for (const auto& piece : pieces) {

        std::string encoded =
            byte_encode(piece);

        auto tokens =
            bpe(encoded);

        for (const auto& token : tokens) {

            auto it = vocab.find(token);

            if (it == vocab.end()) {
                throw std::runtime_error(
                    "Token not found in vocab: "
                    + token
                );
            }

            ids.push_back(
                it->second
            );
        }
    }

    return ids;
}

std::string Tokenizer::decode(
    const std::vector<int>& ids
) const {
    std::string encoded;

    for (int id : ids) {

        if (id < 0 ||
            id >= static_cast<int>(id_to_token.size())) {

            throw std::runtime_error(
                "Invalid token id"
            );
        }

        encoded += id_to_token[id];
    }

    auto symbols =
        split_utf8_chars(encoded);

    std::string result;

    for (const auto& symbol : symbols) {

        auto it =
            byte_decoder.find(symbol);

        if (it == byte_decoder.end()) {
            throw std::runtime_error(
                "Unknown byte symbol"
            );
        }

        result.push_back(
            static_cast<char>(it->second)
        );
    }

    return result;
}

}
