#include <iostream>

// This is the kernel function that runs on the GPU
__global__ void helloFromGPU() {
    printf("Hello World from GPU!\\n");
}

int main() {
    std::cout << "Hello World from CPU!" << std::endl;

    // Launch 1 block with 1 thread
    helloFromGPU<<<1, 1>>>();

    cudaDeviceSynchronize();   //this is used to wait for gpu to complete its execution

    return 0;
}
