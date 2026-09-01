#include "ops.hpp"

#include <stdexcept>
#include <cmath>

namespace llm {

std::vector<float> multiply (
        const std::vector<float>& a,
        const std::vector<float>& b
) {
        if (a.size() != b.size()) {
                throw std::runtime_error(
                "multiply dimension mismatch"
                );
        }
        std::vector <float> result (a.size());
        for (int i = 0;i < a.size();i ++) {
            result[i] = a[i] * b[i];
        }

return result;

}

std::vector<float> matvec (
		const Tensor& matrix,
		const std::vector<float>& x
) {
    if (matrix.shape.size() != 2) {
        throw std::runtime_error(
            "matvec expects 2D tensor"
        );
    }

    int rows = 
        static_cast<int>(matrix.shape[0]);

    int cols = 
        static_cast<int>(matrix.shape[1]);

    if (x.size() != static_cast<size_t>(cols)) {
        throw std::runtime_error(
            "matvec dimension mismatch"
        );
    }

	std::vector<float> Result (rows);
	for (int i = 0;i < rows; i++) {
		float count = 0.0f;
		for(int k = 0; k < cols; k++) {
			count += x[k] * matrix.data[i * cols + k];
		}
		Result[i] = count;
	}
	return Result;
}

float dot(
    const std::vector<float>& a,
    const std::vector<float>& b
) {
    float result = 0.0f;

    for (int i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }

    return result;
}

std::vector<float> softmax(
	const std::vector<float>& x
) {

	int size = x.size();
	std::vector <float> result (size);
	float max = x[0];
	for (int i = 0;i < size;i ++) {
		if (max < x[i]) max = x[i];
	}

	float denumerator = 0.0f;
	for (int i = 0;i < size; i++) {
		denumerator += std::exp(x[i] - max);
	}

	for (int i = 0; i < size;i ++) {
		float numerator = std::exp(x[i]- max);
		result[i] = numerator / denumerator;
	}
	return result;
}

std::vector<float> rmsnorm(
    const std::vector<float>& x,
    const std::vector<float>& weight,
    float eps
) {
    int d = x.size();

    float mean_square = 0.0f;

    for (int i = 0; i < d; ++i) {
        mean_square += x[i] * x[i];
    }

    mean_square /= d;

    float scale =
        1.0f / std::sqrt(mean_square + eps);

    std::vector<float> result(d);

    for (int i = 0; i < d; ++i) {
        result[i] = x[i] * scale * weight[i];
    }

    return result;

}

std::vector<float> add(
    const std::vector<float>& a,
    const std::vector<float>& b
) {
    std::vector<float> result(a.size());

    for (int i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }

    return result;
}

std::vector<float> SiLU (
	const std::vector<float>& x
) {
	std::vector<float> result(x.size());
	
	for (int i = 0; i < x.size(); i++) {
		result[i] = 
			x[i] /
			(1 + std::exp(-x[i]));
	}

	return result;
}

std::vector<float> rope(
    const std::vector<float>& x,
    int position,
    float base
) {
    int d = x.size();

    if (d % 2 != 0) {
        throw std::runtime_error(
            "RoPE expects even dimension"
        );
    }

    int half = d / 2;

    std::vector<float> result(d);

    for (int i = 0; i < half; ++i) {

        float frequency =
            1.0f / std::pow(
                base,
                (2.0f * i) / d
            );

        float angle =
            position * frequency;

        float c = std::cos(angle);
        float s = std::sin(angle);

        float a = x[i];
        float b = x[i + half];

        result[i] =
            a * c - b * s;

        result[i + half] =
            a * s + b * c;
    }

    return result;
}
}
