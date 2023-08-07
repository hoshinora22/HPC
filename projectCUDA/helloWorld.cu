#include <iostream>
#include <stack>
#include <queue>

#include "src/MyUtils.h"

__global__ void hello_world()
{
    printf("Hello World from GPU!\n");
}

int main()
{
    // 用__global__定义的kernel是异步的，这意味着host不会等待kernel执行完就执行下一步
    hello_world<<<1, 5>>>();

    std::cout << "Hello, World!" << std::endl;
    return 0;
}
