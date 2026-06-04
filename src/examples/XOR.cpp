#include <nnet/SeqNN.hpp>
#include <core/CostFunctions.cpp>
#include <vector>
#include <iostream>
#include <iomanip>

class XOR
{
public:
    static void testXOR()
    {
        // XOR architecture: 2 inputs, 3 hidden neurons, 1 output
        std::vector<unsigned int> arch({2, 3, 3, 1});
        SeqNN nn(arch);

        // XOR data
        std::vector<std::vector<float>> inputs = {
            {0.0f, 0.0f},
            {0.0f, 1.0f},
            {1.0f, 0.0f},
            {1.0f, 1.0f}};
        std::vector<std::vector<float>> targets = {
            {0.0f},
            {1.0f},
            {1.0f},
            {0.0f}};

        float learning_rate = 0.5f;
        int epochs = 10000;

        std::cout << "Training XOR..." << std::endl;

        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            float total_loss = 0.0f;
            for (size_t i = 0; i < inputs.size(); ++i)
            {
                nn.zero_grad();

                // Forward pass
                std::vector<float> output = nn.forward(inputs[i]);

                // Calculate loss
                total_loss += CostFunction::MSECostQ(output, targets[i]);

                // Backward pass
                std::vector<float> grad = CostFunction::MSEDerivative(output, targets[i]);
                nn.backward(grad);

                // Update weights
                nn.update(learning_rate);
            }

            if (epoch % 1000 == 0)
            {
                std::cout << "Epoch " << epoch << " Loss: " << total_loss / inputs.size() << std::endl;
            }
        }

        std::cout << "\nTesting XOR Results:" << std::endl;
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            std::vector<float> output = nn.forward(inputs[i]);
            std::cout << "Input: {" << inputs[i][0] << ", " << inputs[i][1] << "} "
                      << "Target: " << targets[i][0] << " "
                      << "Predicted: " << std::fixed << std::setprecision(4) << output[0] << std::endl;
        }
    }
};