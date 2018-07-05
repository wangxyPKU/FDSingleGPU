// GUDA runtime
#include "cuda_runtime.h"
#include "assert.h"
#include "cuda.h"

// helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>
#include <timer.h>

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <string.h>



#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

using namespace std;


__device__ float single_fai(float *fai, unsigned int i,unsigned int j,size_t pitch) {
	float *a = (float*)((char*)fai + (i - 1)*pitch);
	float *b = (float*)((char*)fai + (i + 1)*pitch);
	float *c = (float*)((char*)fai + i*pitch);
	return ((a[j] + b[j] + c[j - 1] + c[j + 1]) / 4);
}

__device__ float WestTime(float *fai, unsigned int j)
{
    for(int k=0; k<1000; k++){
        fai[j] += 1;
    }
    for(int k=0; k<1000; k++){
        fai[j] = fai[j]-1;
    }
    return fai[j];
}


__global__ void fai_iter(float *fai_n,float *fai,size_t pitch, int M, int N) {
	//unsigned int i = blockDim.y*blockIdx.y + threadIdx.y;
	//unsigned int j = blockDim.x*blockIdx.x + threadIdx.x;
	int temp;
    for (temp=0;temp<20;)
        temp += 1;
	for (int i = blockDim.y*blockIdx.y + threadIdx.y; i <M; i += blockDim.y*gridDim.y) {
		float *fai_row_n = (float*)((char*)fai_n + i*pitch);
		for (int j = blockDim.x*blockIdx.x + threadIdx.x; j < N; j += blockDim.x*gridDim.x) {
			if (i > 0 && i < M - 1 && j > 0 && j < N - 1){
				fai_row_n[j] = single_fai(fai, i, j, pitch);
				fai_row_n[j] = WestTime(fai_row_n, j);
			}
		}
	}
}



//save data
int data_save(float *fai, int M, int N) {
	char filename[100];
	strcpy(filename,"/public/home/wang_xiaoyue/data/fai_data_std.txt");
	ofstream f(filename);
	if (!f) {
		cout << "file open error!" << endl;
		return 0;
	}
	for (int i = 0; i < M*N; i++) {
		f << fai[i] << ' ';
		if ((i + 1) % N == 0)
			f << endl;
	}
	f.close();
	return 1;
}

void GetDeviceName() 
{ 
    int count= 0;
    cudaGetDeviceCount(&count);
    cudaDeviceProp prop;
    if (count== 0)
    {
        cout<< "There is no device."<< endl;
    }
    for(int i= 0;i< count;++i)
    {
    	cudaGetDeviceProperties(&prop,i) ;
    	cout << "Device " <<i<<": "<< prop.name<< endl;
    } 
}


int main(int argc, char* argv[])
{

	unsigned int i,n;
	int M,N;

	cout<<"\nPlease input number of grids(height,width): "<<endl;
	cin >> M >> N;

	//CPU malloc
	float *fai = (float*)malloc(M * N * sizeof(float));

	//GPU malloc
	float *fai_dev, *fai_dev_n,*temp;
	size_t pitch;

	struct timeval start_t,end_t;
	double timeuse;

    cout<<"\nInitializing data..."<<endl;
	gettimeofday(&start_t, NULL);
	//initialization and boundary condition
	//fai_up=100, fai_right=fai_left=fai_down=0

	for (i = 0; i < M*N; i++) {
		if (i < N)
			fai[i] = 100;  
		else
			fai[i] = 0;    
	}

	cout<<"Starting GPU calculation (include allocating GPU memory)..."<<endl;
	StartTimer();

	//GetDeviceName();

	CUDA_CALL(cudaMallocPitch((void**)&fai_dev, &pitch, N * sizeof(float), M));
	CUDA_CALL(cudaMallocPitch((void**)&fai_dev_n, &pitch, N * sizeof(float), M));
//	cout << "pitch: " << pitch << endl;
//	cout << sizeof(float)*N << endl;

	CUDA_CALL(cudaMemcpy2D(fai_dev, pitch, fai, sizeof(float)*N, sizeof(float)*N, M, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy2D(fai_dev_n, pitch, fai, sizeof(float)*N, sizeof(float)*N, M, cudaMemcpyHostToDevice));
	
	const dim3 blockDim(32, 16,1);
	const dim3 gridDim(8, 8,1);


	for (n = 0; n < 5000; n++) {
		fai_iter << <gridDim, blockDim >> > (fai_dev_n, fai_dev, pitch, M, N);
		temp = fai_dev;
		fai_dev = fai_dev_n;
		fai_dev_n = temp;

		//CUDA_CALL(cudaMemcpy2D(fai_dev, pitch, fai_dev_n, pitch, sizeof(float)*N, M, cudaMemcpyDeviceToDevice));
	}

	CUDA_CALL(cudaMemcpy2D(fai, sizeof(float)*N,fai_dev,pitch,sizeof(float)*N,M,cudaMemcpyDeviceToHost));

	cudaFree(fai_dev);
	cudaFree(fai_dev_n);
	
	cudaDeviceSynchronize();
		
        cout<<"    GPU Processing time: "<<GetTimer()/1e3<<"(s)"<<endl;
        if(argc==2){
		data_save(fai, M, N);
	}

	free(fai);

	gettimeofday(&end_t,NULL);

	timeuse=end_t.tv_sec-start_t.tv_sec + (end_t.tv_usec-start_t.tv_usec)/1e6;
	cout << "number of grids: " << M << " x " << N << endl;
	cout << "iterations: " << n << endl;
	cout << "Total time: " << timeuse << "(s)" << endl;

	return 0;

}


