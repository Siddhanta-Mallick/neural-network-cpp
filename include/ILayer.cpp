#pragma once

#include <vector>

template <typename T>
class ILayer
{
public:
    virtual std::vector<T> forward(std::vector<T> &input) = 0;
    virtual std::vector<T> backward(std::vector<T> grad_output) = 0;
    virtual void update(T lr) = 0;
    virtual void zero_grad() = 0;
};