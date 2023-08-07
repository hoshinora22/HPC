//
// Created by nora1 on 2023/6/14.
//

#include <iostream>
#include <omp.h>

int main(int argc, char **argv)
{
    //线程数目不能超过CPU核数
    auto max = omp_get_max_threads();
    printf("max : %d\n", max);
    omp_set_num_threads(8);
#pragma omp parallel
    {
        printf("Hello World!, thread: %d\n",omp_get_thread_num());
    }

    return 0;
}