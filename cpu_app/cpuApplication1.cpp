#include <iostream>

#include <cmath>
#include <ctime>


using namespace std;

#define N 128
#define ITER 6000
#define PI 3.14159265358979323846

float mu(float B) {
	return 6500 * exp(-B * B / 2 / 0.16) / 0.4 / sqrt(2 * PI) + 1;
}


void Relax(float* U0, float* U1, float* Alph1,
	float* Alph2, float* Alph3, float* Alph4,
	int n, float* J, float h, float w)
{
	float a1;
	float a2;
	float a3;
	float a4;
	float m0 = 1.2567e-6;
	float f;
	float u12;
	int ind;
	for (ind = 0; ind < n * n; ind++) {
		int i = ind % n;
		int j = ind / n;
		if (ind % n != 0 && i <= 2 * n / 3 && j <= 2 * n / 3) {
			if (j == 0) {
				a1 = Alph1[ind];
				a2 = Alph2[ind];
				a3 = Alph3[ind];
				a4 = Alph4[ind];
				f = -m0 * h * h * (J[ind + 1] + J[ind + n + 1] + J[ind + n] + J[ind]) / 4;
				u12 = (a1 * U0[ind + 1] + a2 * U0[ind + n] + a3 * U1[ind - 1]
					+ a4 * U0[ind + n] + 2 * f) / (a1 + a2 + a3 + a4);
				U1[ind] = (1 - w) * U0[ind] + w * u12;
			}
			else if (j == n / 3 - 2) {
				a1 = Alph1[ind];
				a2 = Alph2[ind];
				a3 = Alph3[ind];
				a4 = Alph4[ind];
				f = -m0 * h * h * (J[ind + 1] + J[ind + n + 1] + J[ind + n] + J[ind]) / 4;
				u12 = (a1 * U0[ind + 1] + a2 * U1[ind - n] + a3 * U1[ind - 1]
					+ a4 * U1[ind - n] + 2 * f) / (a1 + a2 + a3 + a4);
				U1[ind] = (1 - w) * U0[ind] + w * u12;
			}
			else {
				a1 = Alph1[ind];
				a2 = Alph2[ind];
				a3 = Alph3[ind];
				a4 = Alph4[ind];
				f = -m0 * h * h * (J[ind + 1] + J[ind + n + 1] + J[ind + n] + J[ind]) / 4;
				u12 = (a1 * U0[ind + 1] + a2 * U0[ind + n] + a3 * U1[ind - 1]
					+ a4 * U1[ind - n] + 2 * f) / (a1 + a2 + a3 + a4);
				U1[ind] = (1 - w) * U0[ind] + w * u12;
			}

		}
	}
}


void Alph(float* u, float* Alph1,
	float* Alph2, float* Alph3, float* Alph4, int n, float h)
{
	float m1;
	float m2;
	float m3;
	float m4;
	int ind;
	for (int j = 0; j < n; j++)
		for (int i = 0; i < n; i++)
		{
			ind = n * j + i;
			if (ind > n - 1 && ind % n != 0 && i <= 2*n/3 && j <= 2 * n / 3) {
				if ((i >= n / 3 + n / 8) || j >= n / 3 + 3) {
					m1 = mu(sqrt(pow(u[ind + 1] + u[ind - n + 1] - u[ind] - u[ind - n], 2) / (2 * h) + pow(u[ind] + u[ind + 1] - u[ind + 1 - n] - u[ind - n], 2) / (2 * h)));
					m2 = mu(sqrt(pow(u[ind + 1 + n] + u[ind + 1] - u[ind] - u[ind + n], 2) / (2 * h) + pow(u[ind + n] + u[ind + 1 + n] - u[ind] - u[ind + 1], 2) / (2 * h)));
					m3 = mu(sqrt(pow(u[ind + n] + u[ind] - u[ind - 1 + n] - u[ind - 1], 2) / (2 * h) + pow(u[ind - 1 + n] + u[ind + n] - u[ind - 1] - u[ind], 2) / (2 * h)));
					m4 = mu(sqrt(pow(u[ind] + u[ind - n] - u[ind - 1] - u[ind - n - 1], 2) / (2 * h) + pow(u[ind - 1] + u[ind] - u[ind - 1 - n] - u[ind - n], 2) / (2 * h)));
					Alph1[ind] = (m1 + m2) / (m1 * m2);
					Alph2[ind] = (m3 + m2) / (m3 * m2);
					Alph3[ind] = (m4 + m3) / (m4 * m3);
					Alph4[ind] = (m1 + m4) / (m1 * m4);
				}
			}
		}

} // ядро для вычисления проницаемости, если задача не с постоянной магн. проницаемостью (нужна зависимость от индукции)


int main()
{
	int numBytes = N * N * sizeof(float);
	float w = 1.0;
	float h = 0.004;
	float e = 1e-11;
	float* u0 = new float[N * N];
	float* u1 = new float[N * N];
	float* alph1 = new float[N * N];
	float* alph2 = new float[N * N];
	float* alph3 = new float[N * N];
	float* alph4 = new float[N * N];
	float* J = new float[N * N];
	float m = 8000; // мю ферромагнетика
	int k;
	
	float e01 = 0;
	float e11 = 1;
	float e02 = 0;
	float e12 = 1;
	float e03 = 0;
	float e13 = 1;
	float e04 = 0;
	float e14 = 1;
	int iter = 0;

	for (w = 1.0; w < 1.05; w += 0.1) {
		for (int j = 0; j < N; j++)
			for (int i = 0; i < N; i++)
			{
				k = N * j + i;
				u0[k] = 0.0f; // начальное приближение для потенциала
				u1[k] = 0.0f;
				if ((i > N / 3 - N / 16 && i < N / 3 + N / 16) && j <= N / 3 - 2) { // область обмотки (считаю за ферромагнетик)
					J[k] = 10000.0f;
					alph1[k] = 2.0f;  // т.к. магн. проницаемость вакуума 1, и h постоянна на сетке
					alph2[k] = 2.0f;
					alph3[k] = 2.0f;
					alph4[k] = 2.0f;
				}
				else if ((i < N / 3 + N / 8) && j < N / 3 + 3) { //область вакуума
					J[k] = 0.0f;
					alph1[k] = 2.0f;  // т.к. магн. проницаемость вакуума 1, и h постоянна на сетке
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

		e01 = 0;
		e11 = 1;
		e02 = 0;
		e12 = 1;
		e03 = 0;
		e13 = 1;
		e04 = 0;
		e14 = 1;
		iter = 0;
		unsigned int start_time = clock();

		for (int x = 0; x <= ITER; x++) {

			Relax(u0, u1, alph1, alph2, alph3, alph4, N, J, h, w);
			for (int i = 0; i < N * N; i++) {
				u0[i] = u1[i];
			}
			Alph(u0, alph1, alph2, alph3, alph4, N, h);

		}
		unsigned int end_time = clock();
		unsigned int search_time = end_time - start_time;
		printf("w: %f ", w);
		printf("time: %d ", search_time);
		printf("iter: %d\n", iter);
	}

	FILE* pfile;
	fopen_s(&pfile, "mf.txt", "w");

	for (int j = 0; j < N; j++)
		for (int i = 0; i < N; i++)
		{
			int k = N * j + i;
			if ((i + 1) % N == 0) {
				fprintf(pfile, "%10.10f\n", u1[k]);
			}
			else {
				fprintf(pfile, "%10.10f ", u1[k]);
			}
		}
	fclose(pfile);

	
	delete u0;
	delete u1;
	delete alph1;
	delete alph2;
	delete alph3;
	delete alph4;
	delete J;

	return 0;
}