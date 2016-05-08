#include "myConvKernel.hpp"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#pragma offload_attribute(push, target(mic))

#include <omp.h>

float *weights, *inputPlanes, *outputPlanes;
double *biases;
int max_nInputPlanes, max_nOutputPlanes, max_nWeights;
int nInputPlanes, nOutputPlanes, nWeights;
int wWidth, wHeight, ioWidth, ioHeight, paddedInWidth, paddedInHeight;
int wSize, paddedInSize, outputSize;

#pragma offload_attribute(pop)

#define ALLOC alloc_if(1) free_if(0)
#define REUSE alloc_if(0) free_if(0)
#define FREE  alloc_if(0) free_if(1)
#define TEMP  alloc_if(1) free_if(1)

double totalGflops, totalTimeCost;

__attribute__((target(mic)))
int BLOCK_LOW(int _id, int _p, int _n)
{
    long long id = _id;
    long long p  = _p;
    long long n  = _n;
    long long _res = id * n / p;
    int res = (int)(_res);
    return res;
}

__attribute__((target(mic)))
int BLOCK_HIGH(int id, int p, int n)
{
    return ( BLOCK_LOW((id)+1,p,n) - 1 );
}

__attribute__((target(mic)))
int BLOCK_SIZE(int id, int p, int n)
{
    return ( BLOCK_HIGH(id,p,n) - BLOCK_LOW(id,p,n) + 1 );
}

void copyFromCVMatF(const cv::Mat &src, float *dst, const int nRow, const int nCol, const int ldd)
{
    for (int i = 0; i < nRow; i++)
    {
        float *dst_base = dst + i * ldd;
        const float *src_base = src.ptr<float>(i);
        memcpy(dst_base, src_base, sizeof(float) * nCol);
    }
}

void padInputPlaneCVWith3x3Kernel(cv::Mat _inputPlane, float *inputPlane)
{
    float *inputPlaneLeftTop = inputPlane + paddedInWidth + 1;
    float *inputPlaneLeftButtom = inputPlaneLeftTop + paddedInWidth * (ioHeight - 1);
    float *ptr_IPTop = inputPlane + 1;
    float *ptr_IPButtom = inputPlane + paddedInWidth * (paddedInHeight - 1) + 1;
    
    // Copy the same matrix in the middle
    copyFromCVMatF(_inputPlane, inputPlaneLeftTop, ioHeight, ioWidth, paddedInWidth);
    
    // Fill the top and buttom 
    memcpy(ptr_IPTop, inputPlaneLeftTop, sizeof(float) * ioWidth);
    memcpy(ptr_IPButtom, inputPlaneLeftButtom, sizeof(float) * ioWidth);
    
    // Fill the left and right sides
    float *ptr_IPLeft = inputPlane + paddedInWidth;
    float *ptr_IPRight = ptr_IPLeft + ioWidth;
    for (int iRow = 0; iRow < ioHeight; iRow++)
    {
        ptr_IPLeft[0] = ptr_IPLeft[1];
        ptr_IPRight[1] = ptr_IPRight[0];
        ptr_IPLeft += paddedInWidth;
        ptr_IPRight += paddedInWidth;
    }
    
    // Copy 4 elements on the corner
    inputPlane[0] = inputPlane[paddedInWidth + 1];
    inputPlane[paddedInWidth - 1] = inputPlane[2 * paddedInWidth - 2];
    inputPlane[paddedInWidth * (paddedInHeight - 1)] = inputPlane[paddedInWidth * (paddedInHeight - 2) + 1];
    inputPlane[paddedInWidth * paddedInHeight - 1] = inputPlane[paddedInWidth * (paddedInHeight - 1) - 2];
}

__attribute__((target(mic)))
void copyFromMemMatF(float *src, float *dst, const int nRow, const int nCol, const int lds, const int ldd)
{
    for (int i = 0; i < nRow; i++)
    {
        float *src_base = src + i * lds;
        float *dst_base = dst + i * ldd;
        memcpy(dst_base, src_base, sizeof(float) * nCol);
    }
}

__attribute__((target(mic)))
void padInputPlaneWith3x3KernelFromOutput(
    float *_outputPlane, const int ioHeight, const int ioWidth,
    float *inputPlane, const int paddedInHeight, const int paddedInWidth
)
{
    float *inputPlaneLeftTop = inputPlane + paddedInWidth + 1;
    float *inputPlaneLeftButtom = inputPlaneLeftTop + paddedInWidth * (ioHeight - 1);
    float *ptr_IPTop = inputPlane + 1;
    float *ptr_IPButtom = inputPlane + paddedInWidth * (paddedInHeight - 1) + 1;
    
    // Copy the same matrix in the middle
    copyFromMemMatF(_outputPlane, inputPlaneLeftTop, ioHeight, ioWidth, ioWidth, paddedInWidth);
    
    // Fill the top and buttom 
    memcpy(ptr_IPTop, inputPlaneLeftTop, sizeof(float) * ioWidth);
    memcpy(ptr_IPButtom, inputPlaneLeftButtom, sizeof(float) * ioWidth);
    
    // Fill the left and right sides
    float *ptr_IPLeft = inputPlane + paddedInWidth;
    float *ptr_IPRight = ptr_IPLeft + ioWidth;
    for (int iRow = 0; iRow < ioHeight; iRow++)
    {
        ptr_IPLeft[0] = ptr_IPLeft[1];
        ptr_IPRight[1] = ptr_IPRight[0];
        ptr_IPLeft += paddedInWidth;
        ptr_IPRight += paddedInWidth;
    }
    
    // Copy 4 elements on the corner
    inputPlane[0] = inputPlane[paddedInWidth + 1];
    inputPlane[paddedInWidth - 1] = inputPlane[2 * paddedInWidth - 2];
    inputPlane[paddedInWidth * (paddedInHeight - 1)] = inputPlane[paddedInWidth * (paddedInHeight - 2) + 1];
    inputPlane[paddedInWidth * paddedInHeight - 1] = inputPlane[paddedInWidth * (paddedInHeight - 1) - 2];
}

void initLocalMem(
    const int _max_nInputPlanes, const int _max_nOutputPlanes,
    const int _ioWidth, const int _ioHeight, cv::Mat _1stInputPlane,
    const int _wWidth, const int _wHeight
)
{
    max_nInputPlanes  = _max_nInputPlanes;
    max_nOutputPlanes = _max_nOutputPlanes;
    max_nWeights      = max_nInputPlanes * max_nOutputPlanes;
    
    wWidth         = _wWidth;    // This should be 3
    wHeight        = _wHeight;   // This should be 3, too
    ioWidth        = _ioWidth;
    ioHeight       = _ioHeight;
    paddedInWidth  = ioWidth + wWidth - 1;
    paddedInHeight = ioHeight + wHeight - 1;
    wSize          = wWidth * wHeight;
    paddedInSize   = paddedInWidth * paddedInHeight;
    outputSize     = ioWidth * ioHeight;
    
    inputPlanes  = (float*)  _mm_malloc(sizeof(float)  * paddedInSize * max_nInputPlanes,  512);
    outputPlanes = (float*)  _mm_malloc(sizeof(float)  * outputSize   * max_nOutputPlanes, 512);
    weights      = (float*)  _mm_malloc(sizeof(float)  * wSize        * max_nWeights,      512);
    biases       = (double*) _mm_malloc(sizeof(double) * max_nWeights,                     512);
    assert(weights != NULL && inputPlanes != NULL && outputPlanes != NULL && biases != NULL);
    
    // outputPlanes are all zero matrices, though it should be reset before each time
    memset(outputPlanes, 0, sizeof(float) * outputSize * max_nOutputPlanes);
    
    // The first input plane is THE ONLY ONE input, others are from output planes
    // To make the operation same, copy this input plane to the 1st output plane,
    // it will be copied to the input plane in the 1st round
    copyFromCVMatF(_1stInputPlane, outputPlanes, ioHeight, ioWidth, ioWidth);
    
    #pragma offload_transfer target(mic) \
    nocopy(inputPlanes  : length(paddedInSize * max_nInputPlanes)  ALLOC) \
    in    (outputPlanes : length(outputSize   * max_nOutputPlanes) ALLOC) \
    nocopy(weights      : length(wSize        * max_nWeights)      ALLOC) \
    nocopy(biases       : length(max_nWeights)                     ALLOC)
}

void copyInMatrices(
    const int _nInputPlanes, const int _nOutputPlanes,
    const std::vector<cv::Mat> &_weights, const std::vector<double> _biases
)
{
    nInputPlanes   = _nInputPlanes;
    nOutputPlanes  = _nOutputPlanes;
    nWeights       = nInputPlanes * nOutputPlanes;
    
    // copy weightMatrices to local
    for (int i = 0; i < nWeights; i++)
        copyFromCVMatF(_weights[i], weights + wSize * i, wHeight, wWidth, wWidth);
    
    // copy baises to local
    for (int i = 0; i < nWeights; i++)
        biases[i] = _biases[i];
    
    #pragma offload target(mic) \
    in(ioHeight) in(ioWidth) in(paddedInHeight) in(paddedInWidth) in(outputSize) \
    nocopy(inputPlanes  : length(paddedInSize * max_nInputPlanes)  REUSE) \
    nocopy(outputPlanes : length(outputSize   * max_nOutputPlanes) REUSE) \
    in    (weights      : length(wSize        * max_nWeights)      REUSE) \
    in    (biases       : length(max_nWeights) REUSE)
    {
        // Pad for input planes to simplify the compute kernel
        // inputPlanes are the outputPlanes from the previous round
        // outputPlanes are all zero matrices
        #pragma omp parallel for
        for (int i = 0; i < nInputPlanes; i++)
        {
            padInputPlaneWith3x3KernelFromOutput(
                outputPlanes + outputSize * i, ioHeight, ioWidth,
                inputPlanes + paddedInSize * i, paddedInHeight, paddedInWidth
            );
            memset(outputPlanes + outputSize * i, 0, sizeof(float) * outputSize);
        }
    }
}

__attribute__((target(mic)))
void convolve3x3withPad(
    float *inputPlane, float *outputPlane, float *weightMatrix,
    const int ioWidth, const int ioHeight_spos, const int ioHeight_epos
)
{   
    int paddedInWidth = ioWidth + 2;
    for (int opY = ioHeight_spos; opY < ioHeight_epos; opY++)
    {
        for (int opX = 0; opX < ioWidth; opX++)
        {
            register float res = 0.0;
            res += inputPlane[(opY    ) * paddedInWidth + (opX    )] * weightMatrix[0];
            res += inputPlane[(opY    ) * paddedInWidth + (opX + 1)] * weightMatrix[1];
            res += inputPlane[(opY    ) * paddedInWidth + (opX + 2)] * weightMatrix[2];
            res += inputPlane[(opY + 1) * paddedInWidth + (opX    )] * weightMatrix[3];
            res += inputPlane[(opY + 1) * paddedInWidth + (opX + 1)] * weightMatrix[4];
            res += inputPlane[(opY + 1) * paddedInWidth + (opX + 2)] * weightMatrix[5];
            res += inputPlane[(opY + 2) * paddedInWidth + (opX    )] * weightMatrix[6];
            res += inputPlane[(opY + 2) * paddedInWidth + (opX + 1)] * weightMatrix[7];
            res += inputPlane[(opY + 2) * paddedInWidth + (opX + 2)] * weightMatrix[8];
            outputPlane[opY  * ioWidth + opX] = res;
        }
    } 
}

__attribute__((target(mic)))
void convolve3x3withPad_1line(
    float *inputPlane, float *outputPlane, float *weightMatrix,
    const int ioWidth, const int ioHeight_spos, const int ioHeight_epos
)
{   
    int paddedInWidth = ioWidth + 2;
    for (int opY = ioHeight_spos; opY < ioHeight_epos; opY++)
    {
        float *oP_base = outputPlane + opY * ioWidth;
        memset(oP_base, 0, sizeof(float) * ioWidth);
        
        for (int shiftY = 0; shiftY < 3; shiftY++)
            for (int shiftX = 0; shiftX < 3; shiftX++)
            {
                float *iP_spos = inputPlane + (opY + shiftY) * paddedInWidth + shiftX;
                float w = weightMatrix[shiftY * 3 + shiftX];
                
                #pragma simd
                for (int opX = 0; opX < ioWidth; opX++)
                    oP_base[opX] += w * iP_spos[opX];
            }
    }   
}

__attribute__((target(mic)))
void convolve3x3withPad_vec1(
    float *inputPlane, float *outputPlane, float *weightMatrix,
    const int ioWidth, const int ioHeight_spos, const int ioHeight_epos
)
{   
    int paddedInWidth = ioWidth + 2;
    float inbuf[9], outbuf[9];
    for (int ipY = 1 + ioHeight_spos; ipY < 1 + ioHeight_epos; ipY++)
    {
        for (int ipX = 1; ipX < ioWidth + 1; ipX++)
        {
            inbuf[0] = inputPlane[(ipY - 1) * paddedInWidth + (ipX - 1)];
            inbuf[1] = inputPlane[(ipY - 1) * paddedInWidth + (ipX    )];
            inbuf[2] = inputPlane[(ipY - 1) * paddedInWidth + (ipX + 1)];
            inbuf[3] = inputPlane[(ipY    ) * paddedInWidth + (ipX - 1)];
            inbuf[4] = inputPlane[(ipY    ) * paddedInWidth + (ipX    )];
            inbuf[5] = inputPlane[(ipY    ) * paddedInWidth + (ipX + 1)];
            inbuf[6] = inputPlane[(ipY + 1) * paddedInWidth + (ipX - 1)];
            inbuf[7] = inputPlane[(ipY + 1) * paddedInWidth + (ipX    )];
            inbuf[8] = inputPlane[(ipY + 1) * paddedInWidth + (ipX + 1)];
            for (int i = 0; i < 9; i++)
                outbuf[i] = inbuf[i] * weightMatrix[i];
            
            float res0 = outbuf[0] + outbuf[1];
            float res1 = outbuf[2] + outbuf[3];
            float res2 = outbuf[4] + outbuf[5];
            float res3 = outbuf[6] + outbuf[7];
            res0 += res1;
            res2 += res3;
            outputPlane[(ipY - 1) * ioWidth + (ipX - 1)] = outbuf[8] + res0 + res2;
        }
    }   
}

__attribute__((target(mic)))
void addVec(const int length, float *src, float *dst)
{
    for (int i = 0; i < length; i++) dst[i] += src[i];
}

__attribute__((target(mic)))
void addBias(const int length, const float bias, float *dst)
{
    for (int i = 0; i < length; i++) dst[i] += bias;
}

__attribute__((target(mic)))
void scaleIfLessThanX(const int length, float *dst, const float X, const float alpha)
{
    for (int i = 0; i < length; i++)
        if (dst[i] < X)
            dst[i] *= alpha;
}

void myConvKernel()
{
    float *filterOutput_buf = (float*) _mm_malloc(sizeof(float) * outputSize, 512); 
    assert(filterOutput_buf != NULL);
    
    int ipIndexStep = 4;
    
    memset(outputPlanes, 0, outputSize * nOutputPlanes);
    
    #pragma offload target(mic) \
    in(ioHeight) in(ioWidth) in(outputSize) \
    in(paddedInHeight) in(paddedInWidth) in(paddedInSize) \
    in(wSize) in(nInputPlanes) in(nOutputPlanes) \
    in(filterOutput_buf : length(outputSize) TEMP) \
    nocopy(inputPlanes  : length(paddedInSize * max_nInputPlanes)  REUSE) \
    nocopy(outputPlanes : length(outputSize   * max_nOutputPlanes) REUSE) \
    nocopy(weights      : length(wSize        * max_nWeights)      REUSE) \
    nocopy(biases       : length(max_nWeights) REUSE)
    {
        #pragma omp parallel 
        {
            int tid = omp_get_thread_num();
            int nthreads = omp_get_num_threads();
            
            int ioHeight_spos = BLOCK_LOW(tid, nthreads, ioHeight);
            int ioHeight_epos = BLOCK_LOW(tid + 1, nthreads, ioHeight);
            
            int oS_spos = ioHeight_spos * ioWidth;
            int oS_size = (ioHeight_epos - ioHeight_spos) * ioWidth;
          
            for (int ipIndex = 0; ipIndex < nInputPlanes; ipIndex += ipIndexStep)
            {
                for (int opIndex = 0; opIndex < nOutputPlanes; opIndex++)
                {
                    float *filterOutput = filterOutput_buf;                    
                    float *outputPlane = outputPlanes + opIndex * outputSize; 
                    int end_ipIndex = ipIndex + ipIndexStep;
                    if (end_ipIndex >= nInputPlanes) end_ipIndex = nInputPlanes;
                    
                    for (int ipIndex2 = ipIndex; ipIndex2 < end_ipIndex; ipIndex2++)
                    {
                        int wMatIndex = nInputPlanes * opIndex + ipIndex2;
                        float *inputPlane = inputPlanes + ipIndex2 * paddedInSize;
                        float *weightMatrix = weights + wMatIndex * wSize;
                        
                        convolve3x3withPad_1line(
                            inputPlane, filterOutput, weightMatrix,
                            ioWidth, ioHeight_spos, ioHeight_epos
                        );

                        addVec(oS_size, filterOutput + oS_spos, outputPlane + oS_spos);
                    }
                }
            }
            
            #pragma omp barrier
            
            #pragma omp for
            for (int opIndex = 0; opIndex < nOutputPlanes; opIndex++)
            {
                int wMatIndex = nInputPlanes * opIndex;
                float *outputPlane = outputPlanes + opIndex * outputSize;    
                addBias(outputSize, (float)(biases[opIndex]), outputPlane); 
                scaleIfLessThanX(outputSize, outputPlane, 0.0, 0.1);  
            }
        }
    }

    _mm_free(filterOutput_buf);
}

void copyToCVMatF(const float *src, cv::Mat &dst, const int nRow, const int nCol, const int lds)
{
    for (int i = 0; i < nRow; i++)
    {
        const float *src_base = src + i * lds;
        float *dst_base = dst.ptr<float>(i);
        memcpy(dst_base, src_base, sizeof(float) * nCol);
    }
}

void copyOutResults(std::vector<cv::Mat> &_outputPlanes)
{
    #pragma offload_transfer target(mic) \
    nocopy(inputPlanes  : length(paddedInSize * max_nInputPlanes)  FREE) \
    out   (outputPlanes : length(outputSize   * max_nOutputPlanes) FREE) \
    nocopy(weights      : length(wSize        * max_nWeights)      FREE) \
    nocopy(biases       : length(max_nWeights)                     FREE)
    
    copyToCVMatF(outputPlanes, _outputPlanes[0], ioHeight, ioWidth, ioWidth);
    
    if (weights != NULL)      _mm_free(weights);
    if (inputPlanes != NULL)  _mm_free(inputPlanes);
    if (outputPlanes != NULL) _mm_free(outputPlanes);
    if (biases != NULL)       _mm_free(biases);
}

void resetTotalGFlops()
{
    totalGflops   = 0.0;
    totalTimeCost = 0.0;
}

void addGFlops(double newGFlops, double newTimeCost)
{
    totalGflops   += newGFlops;
    totalTimeCost += newTimeCost;
}

void reportTotalGFlops()
{
    double res = totalGflops / totalTimeCost;
    printf("\n===== Total Performance Report =====\n");
    printf("Total time   = %lf (seconds)\n", totalTimeCost);
    printf("Total GFlops = %lf (single precision)\n", res);
    printf("====================================\n");
}