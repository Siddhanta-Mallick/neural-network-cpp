#pragma once

#include "core/ILayer.hpp"

class SeqDenseLayer : public ILayer
{
private:
    std::vector<std::vector<float>> weightMatrix;
    std::vector<float> bias;
    std::vector<std::vector<float>> gradWeightMatrix;
    std::vector<float> gradBias;
    unsigned int neuronCount;
    unsigned int inputCount;
    std::vector<float> z;
    std::vector<float> input;

public:
    SeqDenseLayer(unsigned int neuronCount, unsigned int inputCount);
    std::vector<float> forward(const std::vector<float> &input) override;
    std::vector<float> backward(std::vector<float> gradOutput) override;
    void printLayer(int layerId = -1);
    void update(float lr) override;
    void zero_grad() override;
};
