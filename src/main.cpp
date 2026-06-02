#include <nnet/SeqNN.hpp>

#include <vector>
#include <iostream>

int main()
{
    std::vector<unsigned int> arch({3, 5, 2, 3, 1});
    SeqNN nn(arch);
    std::vector<float> y_hat = nn.forward({1, 2, 3});
    std::cout << y_hat[0];
}