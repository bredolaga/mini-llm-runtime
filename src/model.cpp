#include "ops.hpp"
#include "attention.hpp"
#include "mlp.hpp"
#include "model.hpp"
#include <string>

namespace llm {

TransformerBlockWeights load_block(
    const SafeTensorLoader& loader,
    int layer
) {
    TransformerBlockWeights weights;

    std::string prefix =
        "model.layers."
        + std::to_string(layer)
        + ".";

    weights.attention_norm =
        loader.load_tensor(
            prefix + "input_layernorm.weight"
        );

    weights.wq =
        loader.load_tensor(
            prefix + "self_attn.q_proj.weight"
        );

    weights.wk =
        loader.load_tensor(
            prefix + "self_attn.k_proj.weight"
        );

    weights.wv =
        loader.load_tensor(
            prefix + "self_attn.v_proj.weight"
        );

    weights.wo =
        loader.load_tensor(
            prefix + "self_attn.o_proj.weight"
        );

    weights.mlp_norm =
        loader.load_tensor(
            prefix + "post_attention_layernorm.weight"
        );

    weights.wgate =
        loader.load_tensor(
            prefix + "mlp.gate_proj.weight"
        );

    weights.wup =
        loader.load_tensor(
            prefix + "mlp.up_proj.weight"
        );

    weights.wdown =
        loader.load_tensor(
            prefix + "mlp.down_proj.weight"
        );

    return weights;
}

std::vector<std::vector<float>> transformer_block (
	const std::vector<std::vector<float>>& X,
	const TransformerBlockWeights& weights,
    const ModelConfig&  config
) {
	
    std::vector<std::vector<float>> normalized1(X.size());
    std::vector<std::vector<float>> normalized2(X.size());
    std::vector<std::vector<float>> att_out(X.size());
    std::vector<std::vector<float>> Y(X.size());
    std::vector<std::vector<float>> out_mlp(X.size());
    std::vector<std::vector<float>> result(X.size());

    for (int i = 0; i < X.size();i++) {

            normalized1[i] = rmsnorm(
                X[i],
                weights.attention_norm.data,
                config.rms_norm_eps
            );

            att_out[i] = attention(
                normalized1,
                weights.wq,
                weights.wk,
                weights.wv,
                weights.wo,
                config,
                i
            );

            Y[i] = add(X[i], att_out[i]);

            normalized2[i] = rmsnorm(
                Y[i],
                weights.mlp_norm.data,
                config.rms_norm_eps
            );

            out_mlp[i] = mlp(
                normalized2[i],
                weights.wgate,
                weights.wup,
                weights.wdown
            );

            result[i] = add(Y[i], out_mlp[i]);
        }

return result;
}

}


