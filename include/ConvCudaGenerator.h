#ifndef CONVCUDAGENERATOR_CONVCUDAGENERATOR_H
#define CONVCUDAGENERATOR_CONVCUDAGENERATOR_H

#include <iostream>
#include <string>
using namespace std;

string BeginningCode = R"xxx(
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <cuda_runtime.h>

//use INSTANTIATE when kernel is called from other files
#define INSTANTIATE_LAYER_GPU_FORWARD(func) \
template __global__ void func<float>(float* d_out, const float* d_in, \
                                     const float* d_kernel, const float* d_kernel_bias); \
template __global__ void func<double>(double* d_out, const double* d_in, \
                                     const double* d_kernel, const double* d_kernel_bias);

using namespace std;

)xxx";

string KernelCode=R"xxx({{#info_kernel}}
template <typename Dtype>
__global__ void {{name}}(Dtype* d_out, const Dtype* d_in, const Dtype* d_kernel, const Dtype* d_kernel_bias) { d_out[0] = 0.0; }
INSTANTIATE_LAYER_GPU_FORWARD({{name}})
{{/info_kernel}})xxx";

string ConvCode= R"xxx(
{{#info_conv}}
    if (!strcmp(name.c_str(), "{{name}}")){
        {{name}} <Dtype> <<<1,36>>> (d_out, d_in, d_kernel, d_kernel_bias);
        return 0;
    }
{{/info_conv}}
)xxx";

string ConvCodeHead=R"xxx(
template <typename Dtype>
int ConvKernel(string name, Dtype* d_out, const Dtype* d_in, const Dtype* d_kernel, const Dtype* d_kernel_bias){

    cout << "Calling Conv Kernel " << name << endl;
)xxx";

string EndingCode= R"xxx(
    cerr<<"No Matching Convolution Code"<<endl;
    return -1;
}

template int ConvKernel<float>(string name, float* d_out, const float* d_in, const float* d_kernel, const float* d_kernel_bias);
template int ConvKernel<double>(string name, double* d_out, const double* d_in, const double* d_kernel, const double* d_kernel_bias);
)xxx";


#endif //CONVCUDAGENERATOR_CONVCUDAGENERATOR_H