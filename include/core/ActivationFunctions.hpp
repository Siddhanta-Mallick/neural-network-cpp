#pragma once

#include <algorithm>
#include <cmath>

class Activation
{
public:
    static float ReLU(float y)
    {
        return std::fmax(y, 0.0f);
    }
    static float Sigmoid(float y)
    {
        if (y >= 0)
        {
            float z = exp(-y);
            return 1 / (1 + z);
        }
        else
        {
            float z = exp(y);
            return z / (1 + z);
        }
    }
    static float SigmoidDerivative(float y)
    {
        float sig = Sigmoid(y);
        return sig * (1 - sig);
    }
};