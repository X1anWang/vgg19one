#include <torch/torch.h>
#include <iostream>


torch::nn::Conv2dOptions conv_options(int64_t in_planes, int64_t out_planes, int64_t kerner_size, int64_t stride=1, int64_t padding=0, bool with_bias=false)
{
  torch::nn::Conv2dOptions conv_options = torch::nn::Conv2dOptions(in_planes, out_planes, kerner_size);
  conv_options.stride(stride);
  conv_options.padding(padding);
  conv_options.bias(with_bias);
  return conv_options;
}

struct vgg19 : public torch::nn::Module
{
    torch::nn::Conv2d c0;
    torch::nn::Conv2d c2;
    torch::nn::Conv2d c5;
    torch::nn::Conv2d c7;
    torch::nn::Conv2d c10;
    torch::nn::Conv2d c12;
    torch::nn::Conv2d c14;
    torch::nn::Conv2d c16;
    torch::nn::Conv2d c19;
    torch::nn::Conv2d c21;
    torch::nn::Conv2d c23;
    torch::nn::Conv2d c25;
    torch::nn::Conv2d c28;
    torch::nn::Conv2d c30;
    torch::nn::Conv2d c32;
    torch::nn::Conv2d c34;
    torch::nn::Linear fc0;
    torch::nn::Linear fc3;
    torch::nn::Linear fc6;

    vgg19():
        c0(conv_options(3, 64, 3, 1, 1)),
        c2(conv_options(64, 64, 3, 1, 1)),
        c5(conv_options(64, 128, 3, 1, 1)),
        c7(conv_options(128, 128, 3, 1, 1)),
        c10(conv_options(128, 256, 3, 1, 1)),
        c12(conv_options(256, 256, 3, 1, 1)),
        c14(conv_options(256, 256, 3, 1, 1)),
        c16(conv_options(256, 256, 3, 1, 1)),
        c19(conv_options(256, 512, 3, 1, 1)),
        c21(conv_options(512, 512, 3, 1, 1)),
        c23(conv_options(512, 512, 3, 1, 1)),
        c25(conv_options(512, 512, 3, 1, 1)),
        c28(conv_options(512, 512, 3, 1, 1)),
        c30(conv_options(512, 512, 3, 1, 1)),
        c32(conv_options(512, 512, 3, 1, 1)),
        c34(conv_options(512, 512, 3, 1, 1)),
        fc0(25088, 4096),
        fc3(4096, 4096),
        fc6(4096, 1000)
        {
            register_module("c0", c0);
            register_module("c2", c2);
            register_module("c5", c5);
            register_module("c7", c7);
            register_module("c10", c10);
            register_module("c12", c12);
            register_module("c14", c14);
            register_module("c16", c16);
            register_module("c19", c19);
            register_module("c21", c21);
            register_module("c23", c23);
            register_module("c25", c25);
            register_module("c28", c28);
            register_module("c30", c30);
            register_module("c32", c32);
            register_module("c34", c34);
            register_module("fc0", fc0);
            register_module("fc3", fc3);
            register_module("fc6", fc6);
        }
    
    torch::Tensor forward(torch::Tensor x)
    {
        x = c0->forward(x);
        x = torch::relu(x);
        x = c2->forward(x);
        x = torch::max_pool2d(torch::relu(x), 2, 2, 0);
        x = c5->forward(x);
        x = torch::relu(x);
        x = c7->forward(x);
        x = torch::max_pool2d(torch::relu(x), 2, 2, 0);
        x = c10->forward(x);
        x = torch::relu(x);
        x = c12->forward(x);
        x = torch::relu(x);
        x = c14->forward(x);
        x = torch::relu(x);
        x = c16->forward(x);
        x = torch::max_pool2d(torch::relu(x), 2, 2, 0);
        x = c19->forward(x);
        x = torch::relu(x);
        x = c21->forward(x);
        x = torch::relu(x);
        x = c23->forward(x);
        x = torch::relu(x);
        x = c25->forward(x);
        x = torch::max_pool2d(torch::relu(x), 2, 2, 0);
        x = c28->forward(x);
        x = torch::relu(x);
        x = c30->forward(x);
        x = torch::relu(x);
        x = c32->forward(x);
        x = torch::relu(x);
        x = c34->forward(x);
        x = torch::max_pool2d(torch::relu(x), 2, 2, 0);
        x = torch::nn::AdaptiveAvgPool2d(x, 7);
        x = x.view({x.sizes()[0], -1});
        x = torch::nn::functional(torch::relu(fc0->forward(x)), torch::nn::functional::DropoutFuncOptions().p(0.5));
        x = torch::nn::functional(torch::relu(fc3->forward(x)), torch::nn::functional::DropoutFuncOptions().p(0.5));
        x = fc6->forward(x);
        
        return x;
    }
}

int main()
{
    vgg19 model;
    model.to('cpu');

    torch::Tensor t = torch::rand({1, 3, 224, 224}).to('cpu');
    auto y = model.forward(t);
    
    std::cout << "Completed." << std::endl;

    return 0;
}