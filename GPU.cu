#include <iostream>
#include "SFML/Graphics.hpp"
#include "cuda.h"
#include <chrono>
#include "fstream"

#define N 510
#define size ((N+2)*(N+2))
#define IX(i, j) ((i) + (N+2) * (j))
#define SWAP(x0,x) {float *tmp=x0;x0=x;x=tmp;}

#define threads 1024
#define blocks (size / threads)

#define SOURCE_SIZE 16
#define FORCE_SIZE 8

// Additional setup for SFML
const int WINDOW_WIDTH = 1024;
const int WINDOW_HEIGHT = 1024;

bool mouse_down[3];
int omx = -1, omy = -1;

///////////////////////////////////////// Initialization function for fluid ////////////////////////////////////////////
void initializeFluid(float * u, float * v, float * u_prev, float * v_prev,float * dens, float * dens_prev ) {
    for (int i = 0; i < size; ++i) {
        u[i] = v[i] = u_prev[i] = v_prev[i] = dens[i] = dens_prev[i] = 0.0f;
    }
}

void reset_arrays(float* arr) {
    for (int i = 0; i <size; ++i) {
        arr[i] = 0.0f;
    }
}


/////////////////////////////////////////// fluid calculations on gpu //////////////////////////////////////////////////
__global__ void add_source_kernel(float * x, float * s, float dt ){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    x[index] += dt*s[index];
}

void add_source(float * x, float * s, float dt ){
    add_source_kernel<<<blocks, threads>>>(x, s, dt);
    //cudaDeviceSynchronize();
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

    //__syncthreads();

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
    //cudaDeviceSynchronize();
}

void lin_solve(int b, float *x, float *x0, float a, float c){
    for (int n=0 ; n<20 ; n++ ) {
        lin_solve_kernel<<<blocks, threads>>>(x, x0, a, c);
        //cudaDeviceSynchronize();
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
    //cudaDeviceSynchronize();
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
    //cudaDeviceSynchronize();

    set_bnd (0, div );
    set_bnd ( 0, p );
    lin_solve (0, p, div, 1, 4 );

    project_kernel_UV<<<blocks, threads>>>(u, v, p);
    //cudaDeviceSynchronize();

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

///////////////////////////////////////// UI function for fluid simulation /////////////////////////////////////////////
void get_from_UI(sf::RenderWindow& window, float* d, float* u, float* v, float& diff, float& visc, float force,
                 float source, bool& simulating, bool& clearData) {

    reset_arrays(d);
    reset_arrays(u);
    reset_arrays(v);

    sf::Event event;
    while (window.pollEvent(event)) {
        switch (event.type) {
            case sf::Event::Closed:
                window.close();
                simulating = false;
                break;
            case sf::Event::KeyPressed:
                switch (event.key.code) {
                    case sf::Keyboard::C:
                        clearData = true;
                        break;
                    case sf::Keyboard::V:
                        //dvel = !dvel;
                        break;
                    case sf::Keyboard::A:
                        diff += 0.00001f;
                        break;
                    case sf::Keyboard::Q:
                        diff = std::max(diff - 0.00001f, 0.0f);
                        break;
                    case sf::Keyboard::Z:
                        visc += 0.00001f;
                        break;
                    case sf::Keyboard::S:
                        visc = std::max(visc - 0.000001f, 0.0f);
                        break;
                    default:
                        break;
                }
                break;
            case sf::Event::MouseButtonPressed:
                mouse_down[event.mouseButton.button] = true;
                break;
            case sf::Event::MouseButtonReleased:
                mouse_down[event.mouseButton.button] = false;
                break;
            case sf::Event::MouseMoved:
                int mx = event.mouseMove.x;
                int my = event.mouseMove.y;
                int i = int((mx / float(WINDOW_WIDTH)) * N + 1);
                int j = int((my / float(WINDOW_HEIGHT)) * N + 1);

                if (1 <= i && i <= N && 1 <= j && j <= N) {
                    if (omx >= 0 && omy >= 0 && mouse_down[sf::Mouse::Left]) {
                        for (int x = std::max(i - FORCE_SIZE, 1); x <= std::min(i + FORCE_SIZE / 4, N); ++x) {
                            for (int y = std::max(j - FORCE_SIZE / 4, 1); y <= std::min(j + FORCE_SIZE / 4, N); ++y) {
                                u[IX(x, y)] = force * (mx - omx);
                                v[IX(x, y)] = force * (my - omy);
                            }
                        }
                    }
                    if (mouse_down[sf::Mouse::Right]) {
                        for (int x = std::max(i - SOURCE_SIZE, 1); x <= std::min(i + SOURCE_SIZE, N); ++x) {
                            for (int y = std::max(j - SOURCE_SIZE, 1); y <= std::min(j + SOURCE_SIZE, N); ++y) {
                                d[IX(x, y)] = source;
                            }
                        }
                    }
                }
                omx = mx;
                omy = my;
                break;
        }
    }
}

__global__ void draw_density_kernel(float* dens, int* colors){
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int i = index % (N + 2); // Convert 1D index to 2D i
    int j = index / (N + 2); // Convert 1D index to 2D j

    if (i >= 1 && i <= N && j >= 1 && j <= N) {

        float avg_dens = (dens[IX(i, j)] + dens[IX(i, j + 1)] + dens[IX(i + 1, j)] + dens[IX(i + 1, j + 1)]) / 4;
        int color_intensity = static_cast<int>(avg_dens * 255);
        color_intensity = (color_intensity < 255)? color_intensity : 255;

        colors[IX(i, j)] = color_intensity;

    }
}

void draw_density(sf::RenderWindow& window,sf::VertexArray& quads,sf::Color& color, float* dens, int *colors,  int*colors_d) {
    //calculate density colors
    draw_density_kernel<<<blocks, threads>>>(dens, colors_d);
    cudaMemcpy(colors, colors_d, size * sizeof(int), cudaMemcpyDeviceToHost);

    //give color to the quads
    for (int i = 1; i <= N; ++i) {
        for (int j = 1; j <= N; ++j) {

            color.r = colors[IX(i,j)];
            color.b = colors[IX(i,j)];
            color.g = colors[IX(i,j)];

            // Calculate the index in the vertex array
            int quadIndex = ((i - 1) * N + (j - 1)) * 4;

            // Set the color for the vertices
            quads[quadIndex + 0].color = color;
            quads[quadIndex + 1].color = color;
            quads[quadIndex + 2].color = color;
            quads[quadIndex + 3].color = color;
        }
    }

    // Draw the entire set of quads with a single draw call
    window.clear();
    window.draw(quads);
    window.display();
}

void initQuads(sf::VertexArray& quads){
    float h_x = static_cast<float>(WINDOW_WIDTH) / (N);
    float h_y = static_cast<float>(WINDOW_HEIGHT) / (N);

    for (int i = 1; i <= N; ++i) {
        for (int j = 1; j <= N; ++j) {
            float ix_h_x = (i - 1) * h_x; // Precompute to use in positions
            float j_h_y = (j - 1) * h_y; // Precompute to use in positions

            int quadIndex = ((i - 1) * N + (j - 1)) * 4;

            // Define the four corners of the rectangle
            quads[quadIndex + 0].position = sf::Vector2f(ix_h_x, j_h_y);
            quads[quadIndex + 1].position = sf::Vector2f(i * h_x, j_h_y);
            quads[quadIndex + 2].position = sf::Vector2f(i * h_x, j * h_y);
            quads[quadIndex + 3].position = sf::Vector2f(ix_h_x, j * h_y);
        }
    }
}
//////////////////////////////////////////////////// main //////////////////////////////////////////////////////////////
int main() {

    std::ofstream myFile("fluid_simulation_GPU.csv");
    if(!myFile.is_open()){
        std::cout<< "failed to open the file." << std::endl;
        return 1;
    }

    sf::RenderWindow window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), "Fluid Simulation");
    sf::VertexArray quads(sf::Quads, 4 * N * N);
    sf::Color color(0, 0, 0);
    initQuads(quads);

    static float u[size], v[size], u_prev[size], v_prev[size];
    static float dens[size], dens_prev[size];
    int *colors = (int*) malloc(size*sizeof(int));

    float *u_d, *v_d, *u_prev_d, *v_prev_d;
    float *dens_d, *dens_prev_d;
    int *colors_d;

// Allocate memory on the device
    cudaMalloc((void **)&u_d, size * sizeof(float));
    cudaMalloc((void **)&v_d, size * sizeof(float));
    cudaMalloc((void **)&u_prev_d, size * sizeof(float));
    cudaMalloc((void **)&v_prev_d, size * sizeof(float));
    cudaMalloc((void **)&dens_d, size * sizeof(float));
    cudaMalloc((void **)&dens_prev_d, size * sizeof(float));
    cudaMalloc((void **)&colors_d, size * sizeof(int));

    float source = 1.0f;
    float force = 10.0f;

    float visc = 0.000000f;
    float diff = 0.00000f;
    float dt = 0.05f;

    bool simulating = true;
    bool clearData = true;

    int teller = 0;

    while(simulating){
        while(window.isOpen()) {

            if(clearData){
                initializeFluid(u, v, u_prev, v_prev, dens, dens_prev);
                cudaMemcpy(u_d, u, size * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(v_d, v, size * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(dens_d, dens, size * sizeof(float), cudaMemcpyHostToDevice);
                clearData = false;
            }

            auto startCPU = std::chrono::high_resolution_clock::now();

            get_from_UI(window, dens_prev, u_prev, v_prev, diff,visc,force, source, simulating, clearData);

            cudaMemcpy(u_prev_d, u_prev, size * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(v_prev_d, v_prev, size * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(dens_prev_d, dens_prev, size * sizeof(float), cudaMemcpyHostToDevice);

            vel_step(u_d, v_d, u_prev_d, v_prev_d, visc, dt);
            dens_step(dens_d, dens_prev_d, u_d, v_d, diff, dt);

            cudaMemcpy(u_prev, u_prev_d, size * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(v_prev, v_prev_d, size * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(dens_prev, dens_prev_d, size * sizeof(float), cudaMemcpyDeviceToHost);

            draw_density(window,quads, color,dens_d, colors, colors_d);

            auto stopCPU = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float, std::milli> time = stopCPU - startCPU;
            //std::cout<<" fps " << 1000/time.count() << std::endl;
            if(teller < 1000){
                myFile << time.count() << "\n";
                teller++;
            }
            if(teller == 1000) std::cout<< "file is complete";

        }
    }

    myFile.close();

    free(colors);

    cudaFree(u_d);
    cudaFree(v_d);
    cudaFree(u_prev_d);
    cudaFree(v_prev_d);
    cudaFree(dens_d);
    cudaFree(dens_prev_d);
    cudaFree(colors_d);

    return 0;
}
