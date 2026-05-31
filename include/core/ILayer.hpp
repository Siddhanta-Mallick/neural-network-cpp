#pragma once

#include <vector>

class ILayer
{
public:
    virtual std::vector<float> forward(std::vector<float> &input) = 0;
    virtual std::vector<float> backward(std::vector<float> grad_output) = 0;
    virtual void update(float lr) = 0;
    virtual void zero_grad() = 0;
};