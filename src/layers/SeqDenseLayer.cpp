#include <random>
#include <vector>
#include <iostream>

#include "layers/SeqDenseLayer.hpp"
#include "core/ActivationFunctions.hpp"

using std::cout, std::endl;

SeqDenseLayer::SeqDenseLayer(unsigned int neuronCount, unsigned int inputCount)
    : neuronCount(neuronCount),
      inputCount(inputCount),
      weightMatrix(neuronCount, std::vector<float>(inputCount)),
      bias(neuronCount)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> d(0.0, 1.0);

    for (unsigned int i = 0; i < neuronCount; i++)
    {
        bias[i] = d(gen);
        for (unsigned int j = 0; j < inputCount; ++j)
            weightMatrix[i][j] = d(gen);
    }
}

void SeqDenseLayer::printLayer(int layerId)
{
    if (layerId != -1)
        cout << "Layer : " << layerId << endl;

    cout << "Weight Matrix : " << neuronCount << "x" << inputCount << endl;
    for (unsigned i = 0; i < neuronCount; i++)
    {
        for (unsigned j = 0; j < inputCount; ++j)
            cout << weightMatrix[i][j] << " ";
        cout << endl;
    }
}

std::vector<float> SeqDenseLayer::forward(std::vector<float> &input)
{
    // TODO : rightnow all layers use sigmoid, create a new class member to store which activation function is to be used.
    if (input.size() != inputCount)
        throw std::runtime_error("Dimensions of input and weight matrix do not match");

    std::vector<float> y(neuronCount, 0.0);
    for (unsigned int i = 0; i < neuronCount; ++i)
    {
        for (unsigned int j = 0; j < inputCount; ++j)
            y[i] += input[j] * weightMatrix[i][j];
        y[i] += bias[i];
        y[i] = Activation::Sigmoid(y[i]);
    }

    return y;
}
std::vector<float> SeqDenseLayer::backward(std::vector<float> gradOutput)
{
}
void SeqDenseLayer::update(float lr)
{
}
void SeqDenseLayer::zero_grad()
{
}
