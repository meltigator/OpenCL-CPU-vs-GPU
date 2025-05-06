#!/usr/bin/env bash
#
# performs vector addition on CPU vs GPU (OpenCL),
#
# written by Andrea Giani
#

set -e

# test data
C_FILE="vector_add_avg.c"
EXE_FILE="vector_add_avg"

case "$1" in
  build)
    echo "Updating and installing MSYS2 packages..."
    pacman -Syu --noconfirm
    pacman -S --needed --noconfirm \
        base-devel \
        mingw-w64-x86_64-gcc \
        mingw-w64-x86_64-cmake \
        mingw-w64-x86_64-opencl-headers \
        mingw-w64-x86_64-opencl-icd

    if [ ! -f "$EXE_FILE" ]; then
      echo "Generating $C_FILE..."
      cat > "$C_FILE" << 'EOF'
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 10000000
#define RUNS 10

double get_time_sec() {
    return (double)clock() / CLOCKS_PER_SEC;
}

const char* programSource =
"__kernel void vec_add(__global int* A, __global int* B, __global int* C) {\n"
"    int idx = get_global_id(0);\n"
"    C[idx] = A[idx] + B[idx];\n"
"}\n";

int main() {
    int *A = (int*)malloc(sizeof(int) * N);
    int *B = (int*)malloc(sizeof(int) * N);
    int *C = (int*)malloc(sizeof(int) * N);
    int *C_cpu = (int*)malloc(sizeof(int) * N);

    for (int i = 0; i < N; i++) {
        A[i] = i;
        B[i] = i * 2;
    }

    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_int err;

    err = clGetPlatformIDs(1, &platform, NULL);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueue(context, device, 0, &err);
    program = clCreateProgramWithSource(context, 1, &programSource, NULL, &err);
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error building program\n");
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("%s\n", log);
        free(log);
        return 1;
    }
    kernel = clCreateKernel(program, "vec_add", &err);
    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * N, A, &err);
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * N, B, &err);
    cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * N, NULL, &err);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
    size_t global_size = N;

    double total_cpu = 0;
    for (int run = 0; run < RUNS; run++) {
        double start = get_time_sec();
        for (int i = 0; i < N; i++) {
            C_cpu[i] = A[i] + B[i];
        }
        double end = get_time_sec();
        total_cpu += (end - start);
    }
    printf("CPU average time over %d runs: %.6f seconds\n", RUNS, total_cpu / RUNS);

    double total_gpu = 0;
    for (int run = 0; run < RUNS; run++) {
        double start = get_time_sec();
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
        clFinish(queue);
        err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, sizeof(int) * N, C, 0, NULL, NULL);
        double end = get_time_sec();
        total_gpu += (end - start);
    }
    printf("GPU (OpenCL) average time over %d runs: %.6f seconds\n", RUNS, total_gpu / RUNS);

    int correct = 1;
    for (int i = 0; i < N; i++) {
        if (C[i] != C_cpu[i]) {
            correct = 0;
            printf("Mismatch at index %d: CPU=%d, GPU=%d\n", i, C_cpu[i], C[i]);
            break;
        }
    }
    if (correct)
        printf("Result check: PASSED\n");
    else
        printf("Result check: FAILED\n");

    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    free(A);
    free(B);
    free(C);
    free(C_cpu);

    return 0;
}
EOF

      echo "Compiling $C_FILE..."
      gcc "$C_FILE" -o "$EXE_FILE" -lOpenCL
      echo "Build completed."
    else
      echo "$EXE_FILE already exists. Skipping build."
    fi
    ;;

  run)
    if [ -f "$EXE_FILE" ]; then
      echo "Running $EXE_FILE..."
      ./"$EXE_FILE"
    else
      echo "Executable $EXE_FILE not found. Please run './opencl_vs_cpu_avg.sh build' first."
    fi
    ;;

  clean)
    echo "Cleaning up..."
    rm -f "$C_FILE" "$EXE_FILE"
    echo "Cleanup completed."
    ;;

  *)
    echo "Usage: $0 {build|run|clean}"
    exit 1
    ;;
esac
