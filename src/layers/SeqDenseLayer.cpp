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
      bias(neuronCount),
      gradWeightMatrix(neuronCount, std::vector<float>(inputCount, 0.0f)),
      gradBias(neuronCount, 0.0f),
      z(neuronCount, 0.0f)
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

std::vector<float> SeqDenseLayer::forward(const std::vector<float> &input)
{
    this->input = std::vector<float>(input);
    // TODO : rightnow all layers use sigmoid, create a new class member to store which activation function is to be used.
    if (input.size() != inputCount)
        throw std::runtime_error("Dimensions of input and weight matrix do not match");

    std::vector<float> y(neuronCount, 0.0);
    std::fill(z.begin(), z.end(), 0.0f);
    for (unsigned int i = 0; i < neuronCount; ++i)
    {
        for (unsigned int j = 0; j < inputCount; ++j)
            z[i] += input[j] * weightMatrix[i][j];
        z[i] += bias[i];
        y[i] = Activation::Sigmoid(z[i]);
    }

    return y;
}
std::vector<float> SeqDenseLayer::backward(std::vector<float> gradOutput_l)
{
    std::vector<float> delta_l(neuronCount);
    for (unsigned int i = 0; i < neuronCount; ++i)
    {
        // delta^(l)_i = dL/da^(l)_i * sigma'(z^(l)_i)
        delta_l[i] = gradOutput_l[i] * Activation::SigmoidDerivative(z[i]);
    }

    for (unsigned int i = 0; i < neuronCount; ++i)
    {
        for (unsigned int j = 0; j < inputCount; ++j)
        {
            // dL/dW^(l)_ij = delta^(l)_i * a^(l-1)_j
            gradWeightMatrix[i][j] += delta_l[i] * input[j];
        }
        // dL/db^(l)_i = delta^(l)_i
        gradBias[i] += delta_l[i];
    }

    std::vector<float> gradOutput_prev(inputCount, 0.0f);
    for (unsigned int j = 0; j < inputCount; ++j)
    {
        for (unsigned int i = 0; i < neuronCount; ++i)
        {
            // dL/da^(l-1)_j = sum_i ( delta^(l)_i * W^(l)_ij )
            gradOutput_prev[j] += delta_l[i] * weightMatrix[i][j];
        }
    }

    return gradOutput_prev;
}
void SeqDenseLayer::update(float lr)
{
    for (unsigned int i = 0; i < neuronCount; ++i)
    {
        for (unsigned int j = 0; j < inputCount; ++j)
        {
            weightMatrix[i][j] -= lr * gradWeightMatrix[i][j];
        }
        bias[i] -= lr * gradBias[i];
    }
}
void SeqDenseLayer::zero_grad()
{
    for (unsigned int i = 0; i < neuronCount; ++i)
    {
        std::fill(gradWeightMatrix[i].begin(), gradWeightMatrix[i].end(), 0.0f);
        gradBias[i] = 0.0f;
    }
}
