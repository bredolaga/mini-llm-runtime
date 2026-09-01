#include "mlp.hpp"
#include "ops.hpp"

namespace llm {

std::vector<float> mlp (
	const std::vector<float>& x,
	const Tensor& wgate,
	const Tensor& wup,
	const Tensor& wdown
) {
	std::vector<float> gate, up, hidden;
	std::vector<float> result;

	gate = matvec(
		wgate,
		x
	);

	up = matvec(
		wup,
		x
	);
	
	gate = SiLU(gate);

	hidden = multiply(gate, up);

	result = matvec(
		wdown,
		hidden
	);

	return result;
}

}
