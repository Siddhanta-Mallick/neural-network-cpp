#pragma once

#include <vector>

template <typename T>
class INeuralNetwork
{
public:
    virtual ~INeuralNetwork() = default;
    virtual std::vector<T> forward(const std::vector<T> &input) = 0;
    virtual std::vector<T> backward(const std::vector<T> &grad_output) = 0;
    virtual void update(T learning_rate) = 0;
    virtual void zero_grad() = 0;
};
