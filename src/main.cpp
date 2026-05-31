#include <nnet/SeqNN.hpp>

#include <vector>

int main()
{
    std::vector<unsigned int> arch({3, 5, 2, 3, 1});
    SeqNN nn(arch);
    nn.printNetwork();
}