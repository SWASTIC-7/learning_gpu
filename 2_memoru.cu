#include <stdio.h>
#include <cuda_runtime.h>
#define N 100
__global__ void fun(int *a){
    a[threadIdx.x] += 10;
}

int main(){
    int a[N];
    int *d_a;
    for(int i=0;i<N;i++) a[i] = i;

    cudaMalloc((void**)&d_a, N*sizeof(int));   // make memory in gpu   
    cudaMemcpy(d_a, a, N*sizeof(int), cudaMemcpyHostToDevice); // copy from cpu to gpu (host to device) 

    fun<<<1,N>>>(d_a); 

    cudaMemcpy(a, d_a, N*sizeof(int), cudaMemcpyDeviceToHost);  // copy from gpu to cpu (device to host)
    cudaFree(d_a);

    for(int i=0;i<N;i++) printf("%d ", a[i]);
    printf("\n");
    return 0;
}