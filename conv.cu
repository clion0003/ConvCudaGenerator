
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


template <typename Dtype>
int ConvKernel(string name, Dtype* d_out, const Dtype* d_in, const Dtype* d_kernel, const Dtype* d_kernel_bias){

    cout << "Calling Conv Kernel " << name << endl;


    cerr<<"No Matching Convolution Code"<<endl;
    return -1;
}

template int ConvKernel<float>(string name, float* d_out, const float* d_in, const float* d_kernel, const float* d_kernel_bias);
template int ConvKernel<double>(string name, double* d_out, const double* d_in, const double* d_kernel, const double* d_kernel_bias);
