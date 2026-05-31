#pragma once

#include <vector>

#include "core/INeuralNetwork.hpp"
#include "layers/SeqDenseLayer.hpp"

class SeqNN : public INeuralNetwork
{
private:
    std::vector<unsigned int> architecture;
    unsigned int inputSize;
    unsigned int outputSize;
    unsigned int layerCount;
    std::vector<SeqDenseLayer *> layers;

public:
    SeqNN(std::vector<unsigned int> arch);
    std::vector<float> forward(const std::vector<float> &input) override;
    std::vector<float> backward(const std::vector<float> &grad_output) override;
    void update(float learning_rate) override;
    void zero_grad() override;
    void printNetwork();
};