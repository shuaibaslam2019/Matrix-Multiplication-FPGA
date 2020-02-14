#ifdef APPLE
#include <OpenCL/opencl.h>
#else
#include "CL/cl.h"
#endif
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cstring>
#include <time.h>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include "../inc/matrixMult.h"

using namespace aocl_utils;

// Size of the matrices - K, M, N (squared)
#define SIZE 4096

// localworkgroup sizes ()
//#define TILE_SIZE 32
// Opencl RUNTIME Configuration 

cl_platform_id platform = 0;
cl_device_id device = 0;
scoped_array<cl_device_id> device_num;
cl_context context = 0;
cl_command_queue queue = 0;
cl_program program = 0;
cl_event event = 0;
cl_int status = 0;
cl_kernel kernel = 0;


// fUNCTION PROTOTYPES
float rand_float();
bool init();
void cleanup();

// Matrix-multiplication using a custom OpenCL matrixMult kernel.
int main(int argc, char* argv[]) {

	if (!init())
	{
		return -1;
	}

	// Set the MATRIX sizes
	int K = SIZE;
	int M = SIZE;
	int N = SIZE;
	// Set up matrices
	int  szA = M * K;
	int  szB = K * N;
	int  szC = M * N;
	// Create the matrices and initialize them with random values
	float* A = (float*)malloc(szA * sizeof(float*));
	float* B = (float*)malloc(szB * sizeof(float*));
	float* C = (float*)malloc(szC * sizeof(float*));
	for (int i = 0; i < szA; i++)
	{

		A[i] = rand_float();
	}
	for (int i = 0; i < szB; i++) {
		B[i] = rand_float();
	}
	for (int i = 0; i < szC; i++) {
		C[i] = 0.0;
	}
		
	// Prepare OpenCL memory objects
	cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_CHANNEL_1_INTELFPGA, szA * sizeof(float), NULL, &status);
	checkError(status, "Failed to create buffer for input A");
	cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_CHANNEL_2_INTELFPGA, szB * sizeof(float), NULL, &status);
	checkError(status, "Failed to create buffer for input B");
	cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_CHANNEL_1_INTELFPGA, szC * sizeof(float), NULL, &status);
	checkError(status, "Failed to create buffer for output");

	// Copy matrices to the FPGA
	status = clEnqueueWriteBuffer(queue, bufA, CL_FALSE, 0, szA * sizeof(float), A, 0, NULL, NULL);
	checkError(status, "Failed to transfer input A");
	status = clEnqueueWriteBuffer(queue, bufB, CL_FALSE, 0, szB * sizeof(float), B, 0, NULL, NULL);
	checkError(status, "Failed to transfer input B");
	status = clEnqueueWriteBuffer(queue, bufC, CL_FALSE, 0, szC * sizeof(float), C, 0, NULL, NULL);
	checkError(status, "Failed to transfer input C");
	// Wait for all queues to finish.
	clFinish(queue);

	// Configure the matrixMult kernel and set its arguments

	const double start_time = getCurrentTimestamp();
	clSetKernelArg(kernel, 0, sizeof(M), (void*)&M);
	clSetKernelArg(kernel, 1, sizeof(N), (void*)&N);
	clSetKernelArg(kernel, 2, sizeof(K), (void*)&K);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&bufA);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&bufB);
	clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&bufC);

	printf("\nKernel initialization is complete.\n");
	
	printf("\n----------------------------------------------------------.\n");

	printf("Launching the kernel...\n\n");

	// Execute the kernel over the entire range of C matrix elements
	const size_t global_work_size[2] = { M, N };
	const size_t local_work_size[2] = { TILE_SIZE , TILE_SIZE };
	
	printf("Global size: %zu, %zu\n", global_work_size[0], global_work_size[1]);
	printf("Local size: %zu, %zu\n",  local_work_size[0], local_work_size[1]);

	status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, &event);
	checkError(status, "Failed to launch kernel");
	// Wait for calculations to be finished
	clWaitForEvents(1, &event);
	
	// End the timed loop
	const double end_time = getCurrentTimestamp();
	const double total_time = end_time - start_time;

	// Wall - clock time taken.
	printf("\nTime: %0.3f ms\n", total_time * 1e3);

	printf("\n----------------------------------------------------------.\n");

	// kernel time through event Profiling API
	cl_ulong time_ns = getStartEndTime(event);
	printf("Kernel time : %0.3f ms\n", double(time_ns) * 1e-6);


	printf("\n----------------------------------------------------------.\n");

	// Compute the throughput (GFLOPS).
	const float flops = (float)(2.0f * M * N * K / total_time);
	printf("\nThroughput: %0.2f GFLOPS\n\n", flops * 1e-9);
	
	// Copy the output matrix C back to the CPU memory
	status = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, szC * sizeof(float), C, 0, NULL, NULL);
	checkError(status, "Failed to read output matrix");
	// Free the OpenCL memory objects
	clReleaseMemObject(bufA);
	clReleaseMemObject(bufB);
	clReleaseMemObject(bufC);

	cleanup();

	// Free the host memory objects
	free(A);
	free(B);
	free(C);

	// Exit
	return 0;
}


void cleanup() {

	clReleaseCommandQueue(queue);
	clReleaseContext(context);
	clReleaseProgram(program);
	clReleaseKernel(kernel);

}


bool init()
{
	cl_int status = 0;

	if (!setCwdToExeDir())
	{
		return false;
	}

	// Get the OpenCL platform.
	platform = findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
	if (platform == NULL)
	{
		printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform.\n");
		return false;
	}

	// Query the available OpenCL devices.
	scoped_array<cl_device_id> devices;
	cl_uint num_devices;

	devices.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));

	// We'll just use the first device.
	device = devices[0];

	// Display some device information.
	// display_device_info(device);

	// Create the context.
	context = clCreateContext(NULL, 1, &device, &oclContextCallback, NULL, &status);
	checkError(status, "Failed to create context");

	// Create the command queue.
	queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
	checkError(status, "Failed to create command queue");

	// Create the program.
	std::string binary_file = getBoardBinaryFile("mmul", device);
	printf("Using AOCX: %s\n", binary_file.c_str());
	program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);

	// Build the program that was just created.
	status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
	checkError(status, "Failed to build program");

	// Create the kernel - name passed in here must match kernel name in the
	// original CL file, that was compiled into an AOCX file using the AOC tool
	const char *kernel_name = "mmul"; // Kernel name, as defined in the CL file
	kernel = clCreateKernel(program, kernel_name, &status);
	checkError(status, "Failed to create kernel");

	return true;
}

// Randomly generate a floating-point number between -10 and 10.
float rand_float() {
	return float(rand()) / float(RAND_MAX) * 20.0f - 10.0f;
}