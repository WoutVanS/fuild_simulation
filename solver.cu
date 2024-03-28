#include "cuda.h"


#define N 100
#define size (N+2)*(N+2)
#define IX(i, j) ((i) + (N+2) * (j))
#define SWAP(x0,x) {float *tmp=x0;x0=x;x=tmp;}

#define threads 1024
#define blocks (size + threads - 1) / threads

///////////////////////////////////////////////// gpu //////////////////////////////////////////////////////////////////
__global__ void add_source_kernel(float * x, float * s, float dt ){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    x[index] += dt*s[index];
}

void add_source(float * x, float * s, float dt ){
    add_source_kernel<<<blocks, threads>>>(x, s, dt);
    cudaDeviceSynchronize();
}

__global__ void lin_solve_kernel(float *x, float *x0, float a, float c) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int i = index % (N + 2); // Recover 2D i index
    int j = index / (N + 2); // Recover 2D j index

    if (i >= 1 && i <= N && j >= 1 && j <= N) {
        // Perform the relaxation step
        x[IX(i, j)] = (x0[IX(i, j)] + a * (x[IX(i - 1, j)] + x[IX(i + 1, j)] + x[IX(i, j - 1)] + x[IX(i, j + 1)])) / c;
    }
}

__global__ void set_bnd_kernel(int b, float *x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i <= N) {
        x[IX(0 ,i)] = (b==1)? -x[IX(1,i)] : x[IX(1,i)];
        x[IX(N+1,i)] = (b==1)? -x[IX(N,i)] : x[IX(N,i)];
        x[IX(i,0 )] = (b==2)? -x[IX(i,1)] : x[IX(i,1)];
        x[IX(i,N+1)] = (b==2)? -x[IX(i,N)] : x[IX(i,N)];
    }

    __syncthreads();

    if (i == 1) { // Ensuring these operations are done by only one thread
        x[IX(0 ,0 )] = 0.5f*(x[IX(1,0 )]+x[IX(0 ,1)]);
        x[IX(0 ,N+1)] = 0.5f*(x[IX(1,N+1)]+x[IX(0 ,N )]);
        x[IX(N+1,0 )] = 0.5f*(x[IX(N,0 )]+x[IX(N+1,1)]);
        x[IX(N+1,N+1)] = 0.5f*(x[IX(N,N+1)]+x[IX(N+1,N )]);
    }
}

void set_bnd (int b, float *x ) {
    int threadsNum = std::min(N, threads);
    int blocksNum = (N + threadsNum -1)/threadsNum;

    set_bnd_kernel<<<blocksNum, threadsNum>>>(b, x);
    cudaDeviceSynchronize();
}

void lin_solve(int b, float *x, float *x0, float a, float c){
    int i, j, n;
    for ( n=0 ; n<20 ; n++ ) {
        lin_solve_kernel<<<blocks, threads>>>(x, x0, a, c);
        cudaDeviceSynchronize();
        set_bnd(b, x);
    }
}

void diffuse (int b, float *x, float *x0, float diff, float dt ){
    float a=dt*diff*N*N;
    lin_solve(b, x, x0, a, (1+4*a));
}

__global__ void advect_kernel(float *d, float *d0, float *u, float *v, float dt) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int i = index % (N + 2); // Convert 1D index to 2D i
    int j = index / (N + 2); // Convert 1D index to 2D j

    if (i >= 1 && i <= N && j >= 1 && j <= N) { // Ensure we're inside the relevant grid
        float dt0 = dt * N;
        float x = i - dt0 * u[IX(i, j)];
        float y = j - dt0 * v[IX(i, j)];

        if (x < 0.5f) x = 0.5f;
        if (x > N + 0.5f) x = N + 0.5f;

        int i0 = (int)x;
        int i1 = i0 + 1;

        if (y < 0.5f) y = 0.5f;
        if (y > N + 0.5f) y = N + 0.5f;

        int j0 = (int)y;
        int j1 = j0 + 1;

        float s1 = x - i0;
        float s0 = 1 - s1;
        float t1 = y - j0;
        float t0 = 1 - t1;

        d[IX(i, j)] = s0 * (t0 * d0[IX(i0, j0)] + t1 * d0[IX(i0, j1)]) +
                s1 * (t0 * d0[IX(i1, j0)] + t1 * d0[IX(i1, j1)]);
    }
}

void advect (int b, float * d, float * d0, float * u, float * v, float dt ){
    advect_kernel<<<blocks, threads>>>(d, d0, u, v, dt );
    cudaDeviceSynchronize();
    set_bnd (b,d);
}
__global__ void project_kernel_div(float * u, float * v, float * p, float * div){
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int i = index % (N + 2); // Convert 1D index to 2D i
    int j = index / (N + 2); // Convert 1D index to 2D j

    float h = 1.0 / N;

    if (i >= 1 && i <= N && j >= 1 && j <= N) {
        div[IX(i, j)] = -0.5f * h * (u[IX(i + 1, j)] - u[IX(i - 1, j)] + v[IX(i, j + 1)] - v[IX(i, j - 1)]);
        p[IX(i, j)] = 0;
    }
}

__global__ void project_kernel_UV(float * u, float * v, float * p){
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int i = index % (N + 2); // Convert 1D index to 2D i
    int j = index / (N + 2); // Convert 1D index to 2D j

    if (i >= 1 && i <= N && j >= 1 && j <= N) {
        u[IX(i, j)] -= 0.5f * N * (p[IX(i + 1, j)] - p[IX(i - 1, j)]);
        v[IX(i, j)] -= 0.5f * N * (p[IX(i, j + 1)] - p[IX(i, j - 1)]);
    }
}

void project(float * u, float * v, float * p, float * div){

    project_kernel_div<<<blocks, threads>>>( u, v, p, div);
    cudaDeviceSynchronize();

    set_bnd (0, div );
    set_bnd ( 0, p );
    lin_solve (0, p, div, 1, 4 );

    project_kernel_UV<<<blocks, threads>>>(u, v, p);
    cudaDeviceSynchronize();

    set_bnd (1, u );
    set_bnd (2, v );
}

void dens_step (float * x, float * x0, float * u, float * v, float diff,float dt ){
    add_source(x, x0, dt );
    SWAP( x0, x );
    diffuse( 0, x, x0, diff, dt );
    SWAP( x0, x );
    advect(0, x, x0, u, v, dt );
}

void vel_step (float * u, float * v, float * u0, float * v0, float visc, float dt ){
    add_source ( u, u0, dt );
    add_source (v, v0, dt );
    SWAP ( u0, u );
    diffuse (1, u, u0, visc, dt );
    SWAP ( v0, v );
    diffuse (2, v, v0, visc, dt );
    project ( u, v, u0, v0 );
    SWAP ( u0, u );
    SWAP ( v0, v );
    advect ( 1, u, u0, u0, v0, dt );
    advect (2, v, v0, u0, v0, dt );
    project (u, v, u0, v0 );
}