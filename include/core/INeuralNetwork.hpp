#pragma once

#include <vector>

class INeuralNetwork
{
public:
    virtual ~INeuralNetwork() = default;
    virtual std::vector<float> forward(const std::vector<float> &input) = 0;
    virtual std::vector<float> backward(const std::vector<float> &grad_output) = 0;
    virtual void update(float learning_rate) = 0;
    virtual void zero_grad() = 0;
};