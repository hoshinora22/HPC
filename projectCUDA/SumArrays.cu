//
// Created by nora1 on 2023/6/10.
//

#include <iostream>
#include <vector>

#include "../include/MyUtils.h"


/// 在cpu上执行向量相加操作
void sumArrays(const double *a, const double *b, double *res, int size)
{
    // 以步长为4进行迭代，每次迭代处理4个double类型的加法
    for (int i = 0; i < size; i += 4)
    {
        res[i] = a[i] + b[i];
        res[i + 1] = a[i + 1] + b[i + 1];
        res[i + 2] = a[i + 2] + b[i + 2];
        res[i + 3] = a[i + 3] + b[i + 3];
    }
}

/// 在gpu上执行向量向量相加操作的 核函数
__global__ void sumArraysGPU(const double *a, const double *b, double *res)
{
    // 每个线程处理一个数组元素，通过索引threadIdx.x访问对应的元素
    auto i = threadIdx.x;
    //auto id = blockIdx.x * blockDim.x + threadIdx.x;
    res[i] = a[i] + b[i];
}

int main()
{
    // 将当前设备设置为第一个设备
    int dev = 0;
    cudaSetDevice(dev);

    // 定义了向量的长度
    unsigned long long elementCount = 32;
    printf("vector size : %llu\n", elementCount);

    // 向量所需的存储空间
    auto elementByte = sizeof(double) * elementCount;

    // 在主机上分配内存
    auto *host_a = (double *) malloc(elementByte);
    auto *host_b = (double *) malloc(elementByte);
    auto *host_res = (double *) malloc(elementByte);
    auto *host_resFromGPU = (double *) malloc(elementByte);

    // 将结果向量的内容初始化为0
    memset(host_res, 0, elementByte);
    memset(host_resFromGPU, 0, elementByte);

    // 在GPU上分配内存
    double *dev_a = nullptr;
    double *dev_b = nullptr;
    double *dev_res = nullptr;
    CHECK(cudaMalloc((double **) &dev_a, elementByte));
    CHECK(cudaMalloc((double **) &dev_b, elementByte));
    CHECK(cudaMalloc((double **) &dev_res, elementByte));

    // 初始host化向量的值
    Utils::initialData(host_a, (int) elementByte);
    Utils::initialData(host_b, (int) elementByte);

    // 将host上的向量的数据从主机内存复制到GPU设备内存中
    CHECK(cudaMemcpy(dev_a, host_a, elementByte, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_b, host_b, elementByte, cudaMemcpyHostToDevice));

    // 定义了核函数的执行配置“block”和grid”
    dim3 block(elementCount);
    dim3 grid(elementCount / block.x);

    // 计时器
    double timeStart, timeElapse;
    timeStart = cpuSecond();

    // 执行核函数
    sumArraysGPU<<<grid, block>>>(dev_a, dev_b, dev_res);

    // 加一个同步函数等待核函数执行完毕
    // 如果不加这个同步函数，那么测试的时间是从调用核函数，到核函数返回给主机线程的时间段，而不是核函数的执行时间
    // 加上了同步函数后，计时是从调用核函数开始，到核函数执行完并返回给主机的时间段
    cudaDeviceSynchronize();
    timeElapse = cpuSecond() - timeStart;
    printf("Execution configuration<<<%d,%d>>> Time elapsed %f sec\n", grid.x, block.x, timeElapse);

    CHECK(cudaMemcpy(host_resFromGPU, dev_res, elementByte, cudaMemcpyDeviceToHost));
    sumArrays(host_a, host_b, host_res, (int) elementCount);

    // 数据对比
    Utils::checkResult(host_res, host_resFromGPU, (int)elementCount);

    // GPU上内存释放
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_res);

    // 主机上的内存释放
    free(host_a);
    free(host_b);
    free(host_res);
    free(host_resFromGPU);

    printf("Fin\n");
    return 0;
}