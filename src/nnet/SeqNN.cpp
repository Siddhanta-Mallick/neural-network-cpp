#include "nnet/SeqNN.hpp"
#include "layers/SeqDenseLayer.hpp"

SeqNN::SeqNN(std::vector<unsigned int> arch) : architecture(arch)
{
    inputSize = architecture[0];
    outputSize = architecture[architecture.size() - 1];
    layerCount = architecture.size();
    layers = std::vector<SeqDenseLayer *>(layerCount - 1, nullptr);

    for (unsigned int i = 1; i < layerCount; ++i)
    {
        unsigned int neurons = architecture[i];
        unsigned int weights = architecture[i - 1];

        layers[i - 1] = new SeqDenseLayer(neurons, weights);
    }
}
std::vector<float> SeqNN::forward(const std::vector<float> &input)
{
}
std::vector<float> SeqNN::backward(const std::vector<float> &grad_output)
{
}
void SeqNN::update(float learning_rate)
{
}
void SeqNN::zero_grad()
{
}
void SeqNN::printNetwork()
{
    for (auto layer : layers)
        layer->printLayer();
}
