// invert_colors.cu
#include "invert_colors.cuh"

__global__ void invertColorsKernel(unsigned char *img, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = (y * width + x) * 3; // Assume 3 channels (BGR)
        img[idx] = 255 - img[idx];     // Invert Blue
        img[idx+1] = 255 - img[idx+1]; // Invert Green
        img[idx+2] = 255 - img[idx+2]; // Invert Red
    }
}

void cudaInvertColors(unsigned char *img, int width, int height) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    invertColorsKernel<<<numBlocks, threadsPerBlock>>>(img, width, height);
}