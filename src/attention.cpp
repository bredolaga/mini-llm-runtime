#include "attention.hpp"
#include "ops.hpp"
#include "loader.hpp"
#include "Tensor.hpp"
#include <stdexcept>
#include <cmath> 

namespace llm {

std::vector<float> attention (
    const std::vector<std::vector<float>>& x,
    const Tensor& WQ,
    const Tensor& WK,
    const Tensor& WV,
    const Tensor& WO,
    const ModelConfig& config,
    int position
) {	
	int dim = config.dim;

	int head_dim =
    		config.dim / config.num_heads;

	int kv_dim =
    		config.num_kv_heads * head_dim;

	if (WQ.shape.size() != 2
		|| WQ.shape[0] != dim 
		|| WQ.shape[1] != dim) {

		throw std::runtime_error("Wrong WQ shape");
	}
	
	if (WK.shape.size() != 2
		|| WK.shape[0] != kv_dim 
		|| WK.shape[1] != dim) {

		throw std::runtime_error("Wrong WK shape");
	}
	
	if (WV.shape.size() != 2
		|| WV.shape[0] != kv_dim
		|| WV.shape[1] != dim) {

		throw std::runtime_error("Wrong WV shape");
	}

	if (WO.shape.size() != 2
	        || WO.shape[0] != dim
		|| WO.shape[1] != dim) {

		throw std::runtime_error("Wrong WO shape");
	}
	if (x.empty()) {
    		throw std::runtime_error("Empty sequence");
	}

	if (position < 0 ||
    		position >= static_cast<int>(x.size())) {
    		throw std::runtime_error("Invalid position");
	}

	if (config.num_heads % config.num_kv_heads != 0) {
    		throw std::runtime_error(
        	"Invalid GQA configuration"
    		);
	}

	if (x[position].size() !=
    		static_cast<size_t>(dim)) {
    		throw std::runtime_error(
        	"Wrong hidden dimension"
    		);
	}

	std::vector<std::vector<float>> K (position + 1);
	std::vector<std::vector<float>> V (position + 1);

	auto q_full =
    		matvec(WQ, x[position]);

	for (int i = 0; i <= position; ++i) {
		K[i] = matvec(WK, x[i]);
    		V[i] = matvec(WV, x[i]);
	}

	int q_per_kv =
    		config.num_heads /
    		config.num_kv_heads;

	std::vector<float> concatenated(
    		dim,
    		0.0f
	);

	for (int h = 0; h < config.num_heads; ++h) {

    		int start = h * head_dim;
    		int end   = start + head_dim;

    		std::vector<float> q_head(
        		q_full.begin() + start,
        		q_full.begin() + end
    		);


    		int kv_head =
        		h / q_per_kv;

    		int kv_start =
        		kv_head * head_dim;

    		int kv_end =
        		kv_start + head_dim;
		
		q_head = rope(
    			q_head,
    			position,
    			config.rope_theta
		);
		
 		std::vector<float> scores(position + 1);

    		for (int i = 0; i <= position; ++i) {

        		std::vector<float> k_head(
            			K[i].begin() + kv_start,
            			K[i].begin() + kv_end
        		);

			k_head = rope(
    				k_head,
    				i,
    				config.rope_theta
			);

 			scores[i] =
            			dot(q_head, k_head) /
            			std::sqrt(static_cast<float>(head_dim));

		}

	auto weights = softmax(scores);
	std::vector<float> head_result(
    		head_dim,
    		0.0f
	);

	for (int i = 0; i <= position; ++i) {

    	std::vector<float> v_head(
        	V[i].begin() + kv_start,
        	V[i].begin() + kv_end
    	);

    	for (int k = 0; k < head_dim; ++k) {
        	head_result[k] +=
            	weights[i] * v_head[k];
    	}
}

    	for (int k = 0; k < head_dim; ++k) {
        	concatenated[h * head_dim + k] =
            	head_result[k];
    	}
}

return matvec(WO, concatenated);

}

}

