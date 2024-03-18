#include <iostream>
#include <cuda.h>
#include <cmath>
#include <ctime>

using namespace std;

#define N 256
#define ITER 15000
#define PI 3.14159265358979323846

__device__ float mu(float B) {
	return 6500 * exp(-B * B / 2 / 0.16) / 0.4 / sqrt(2 * PI) + 1;
}

__global__ void Relax(float* U0, float* U1, float* Alph1,
					  float* Alph2, float* Alph3, float* Alph4, 
					  int n, float *J, float h, bool b)
{
	int ind = blockDim.x * blockIdx.x + threadIdx.x;
	int i = ind % n;
	int j = ind / n;
	if (ind % n != 0 && i <= 2 * n / 3 && j <= 2 * n / 3) {
		if (b) {
			
			if ((i == n / 2 + n / 8 && j <= n / 2 + 3) || (i <= n / 2 + n / 8 && j == n / 2 + 3)) {
				float m4 = mu(sqrt(pow(U0[ind] + U0[ind - n] - U0[ind - 1] - U0[ind - n - 1], 2) / (2 * h) + pow(U0[ind - 1] + U0[ind] - U0[ind - 1 - n] - U0[ind - n], 2) / (2 * h)));
				U1[ind] = (m4 * h * U0[ind + 1] + h * U0[ind - 1]) / (m4 * h + h);
			}
		}
		else {
			
			if (j == 0) {
				float a1 = Alph1[ind];
				float a2 = Alph2[ind];
				float a3 = Alph3[ind];
				float a4 = Alph4[ind];
				float m0 = 1.2567e-6;
				float f = -m0 * h * h * (J[ind + 1] + J[ind + n + 1] + J[ind + n] + J[ind]) / 4;
				U1[ind] = (a1 * U0[ind + 1] + a2 * U0[ind + n] + a3 * U0[ind - 1]
					 + a4* U0[ind + n] + 2 * f) / (a1 + a2 + a3 + a4);
			}
			else if (j == n / 3 - 2) {
				float a1 = Alph1[ind];
				float a2 = Alph2[ind];
				float a3 = Alph3[ind];
				float a4 = Alph4[ind];
				float m0 = 1.2567e-6;
				float f = -m0 * h * h * (J[ind + 1] + J[ind + n + 1] + J[ind + n] + J[ind]) / 4;
				U1[ind] = (a1 * U0[ind + 1] + a2 * U0[ind - n] + a3 * U0[ind - 1]
					+ a4 * U0[ind - n] + 2 * f) / (a1 + a2 + a3 + a4);
			}
			else {
				float a1 = Alph1[ind];
				float a2 = Alph2[ind];
				float a3 = Alph3[ind];
				float a4 = Alph4[ind];
				float m0 = 1.2567e-6;
				float f = -m0 * h * h * (J[ind + 1] + J[ind + n + 1] + J[ind + n] + J[ind]) / 4;
				U1[ind] = (a1 * U0[ind + 1] + a2 * U0[ind + n] + a3 * U0[ind - 1]
					+ a4 * U0[ind - n] + 2 * f) / (a1 + a2 + a3 + a4);
			}
		}
	}
	
}


__global__ void Alph(float* u, float* Alph1,
	float* Alph2, float* Alph3, float* Alph4, int n, float h)
{
	int ind = blockDim.x * blockIdx.x + threadIdx.x;
	int i = ind % n;
	int j = ind / n;
	if (ind > n - 1 && ind % n != 0 && i <= 2 * n / 3 && j <= 2 * n / 3) {
		if ((i >= n / 3 + n / 8) || j >= n / 3 + 3) {
			float m1 = mu(sqrt(pow(u[ind + 1] + u[ind - n + 1] - u[ind] - u[ind - n], 2) / (2 * h) + pow(u[ind] + u[ind + 1] - u[ind + 1 - n] - u[ind - n], 2) / (2 * h)));
			float m2 = mu(sqrt(pow(u[ind + 1 + n] + u[ind + 1] - u[ind] - u[ind + n], 2) / (2 * h) + pow(u[ind + n] + u[ind + 1 + n] - u[ind] - u[ind + 1], 2) / (2 * h)));
			float m3 = mu(sqrt(pow(u[ind + n] + u[ind] - u[ind - 1 + n] - u[ind - 1], 2) / (2 * h) + pow(u[ind - 1 + n] + u[ind + n] - u[ind - 1] - u[ind], 2) / (2 * h)));
			float m4 = mu(sqrt(pow(u[ind] + u[ind - n] - u[ind - 1] - u[ind - n - 1], 2) / (2 * h) + pow(u[ind - 1] + u[ind] - u[ind - 1 - n] - u[ind - n], 2) / (2 * h)));
			Alph1[ind] = (m1 + m2) / (m1 * m2);
			Alph2[ind] = (m3 + m2) / (m3 * m2);
			Alph3[ind] = (m4 + m3) / (m4 * m3);
			Alph4[ind] = (m1 + m4) / (m1 * m4);
		}
	}

	
} // ядро для вычисления проницаемости, если задача не с постоянной магн. проницаемостью (нужна зависимость от индукции)


int main()
{
	int numBytes = N * N * sizeof(float);
	float h = 0.004;
	float* u0 = new float[N * N];
	float* u1 = new float[N * N];
	float* alph1 = new float[N * N];
	float* alph2 = new float[N * N];
	float* alph3 = new float[N * N];
	float* alph4 = new float[N * N];
	float* J = new float[N * N];
	float m = 8000; // мю ферромагнетика
	for (int j = 0; j < N; j++) 
		for (int i = 0; i < N; i++)
		{
			int k = N * j + i;
			u0[k] = 0.0f; // начальное приближение для потенциала
			u1[k] = 0.0f;
			if ((i >= N / 3 - N / 16 && i <= N / 3 + N / 16) && j <= N / 3 - 2) { 
				J[k] = 10000.0f;
				alph1[k] = 2.0f;  
				alph2[k] = 2.0f;
				alph3[k] = 2.0f;
				alph4[k] = 2.0f;
			}
			else if ((i < N / 3 + N / 8) && j < N / 3 + 3) { //область вакуума
				J[k] = 0.0f;
				alph1[k] = 2.0f;  
				alph2[k] = 2.0f;
				alph3[k] = 2.0f;
				alph4[k] = 2.0f;
			}
			else {
				J[k] = 0.0f;
				alph1[k] = 2 / m;
				alph2[k] = 2 / m;
				alph3[k] = 2 / m;
				alph4[k] = 2 / m;
			}
		}
	
	float* du0 = NULL; // копии для device
	float* du1 = NULL;
	float* dalph1 = NULL;
	float* dalph2 = NULL;
	float* dalph3 = NULL;
	float* dalph4 = NULL;
	float* dJ = NULL;

	cudaMalloc((void**)&du0, numBytes);
	cudaMalloc((void**)&du1, numBytes);
	cudaMalloc((void**)&dalph1, numBytes);
	cudaMalloc((void**)&dalph2, numBytes);
	cudaMalloc((void**)&dalph3, numBytes);
	cudaMalloc((void**)&dalph4, numBytes);
	cudaMalloc((void**)&dJ, numBytes);

	dim3 threads(512, 1);
	dim3 blocks(N * N / threads.x, 1);

	cudaEvent_t start, stop;
	float gpuTime = 0.0f;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cudaMemcpy(du0, u0, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(du1, u1, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dalph1, alph1, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dalph2, alph2, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dalph3, alph3, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dalph4, alph4, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dJ, J, numBytes, cudaMemcpyHostToDevice);
	

	for (int x = 0; x <= ITER; x++) {

		Relax << <blocks, threads >> > (du0, du1, dalph1, dalph2, dalph3, dalph4, N, dJ, h, false);
		cudaMemcpy(du0, du1, numBytes, cudaMemcpyDeviceToDevice);
		Alph << <blocks, threads >> > (du0, dalph1, dalph2, dalph3, dalph4, N, h);


	}

	cudaMemcpy(u1, du1, numBytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(alph1, dalph1, numBytes, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpuTime, start, stop);
	printf("time spent executing by the GPU: %.2f millseconds\n", gpuTime);

	FILE* pfile;
	pfile = fopen("mf.txt", "w");

	for (int j = 0; j < N; j++)
		for (int i = 0; i < N; i++)
		{
			int k = N * j + i;
			if ((i + 1) % N == 0) {
				fprintf(pfile, "%12.12f\n", u1[k]);
			}
			else {
				fprintf(pfile, "%12.12f ", u1[k]);
			}
		}
	fclose(pfile);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(du0);
	cudaFree(du1);
	cudaFree(dalph1);
	cudaFree(dalph2);
	cudaFree(dalph3);
	cudaFree(dalph4);
	cudaFree(dJ);

	delete u0;
	delete u1;
	delete alph1;
	delete alph2;
	delete alph3;
	delete alph4;
	delete J;

	return 0;
}