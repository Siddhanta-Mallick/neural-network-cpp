#pragma once

#include <vector>
#include <format>
#include <stdexcept>

class CostFunction
{
public:
    static float MSECostQ(const std::vector<float> &predicted_Q, const std::vector<float> &expected_Q)
    {
        if (predicted_Q.size() != expected_Q.size())
            throw std::runtime_error(std::format("Predicted size : {} and Estimated : {} sizes dont match ", predicted_Q.size(), expected_Q.size()));

        if (expected_Q.empty())
            throw std::runtime_error("Expected Q array (and possibly Predicted Q array) is empty !");

        float delta_sqr = 0.0f;
        for (size_t i = 0; i < predicted_Q.size(); ++i)
        {
            float diff = expected_Q[i] - predicted_Q[i];
            delta_sqr += diff * diff;
        }

        return delta_sqr / (2 * expected_Q.size());
    }

    static std::vector<float> MSEDerivative(const std::vector<float> &predicted_Q, const std::vector<float> &expected_Q)
    {
        if (predicted_Q.size() != expected_Q.size())
            throw std::runtime_error("Dimensions mismatch in MSEDerivative");

        std::vector<float> grad(predicted_Q.size());
        float n = static_cast<float>(predicted_Q.size());
        for (size_t i = 0; i < predicted_Q.size(); ++i)
        {
            // d/dy_hat [ 1/2n * sum (y - y_hat)^2 ] = 1/n * (y_hat - y)
            grad[i] = (predicted_Q[i] - expected_Q[i]) / n;
        }
        return grad;
    }
};