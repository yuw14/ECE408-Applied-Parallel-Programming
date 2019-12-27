// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void add(float *output1, float *scanBlock, int len){
//  int t =threadIdx.x;
 // int b =blockIdx.x;
  //int start=2*b*blockDim.x;
  //if(b!=0){
  //if(start+t<len){
  //output1[start+t]+=scanBlock[b-1];
  //}
  //if(start+t+b<len){
  //output1[start+b+t]+=scanBlock[b-1];  
  //}
  //}
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<len&&blockIdx.x>0){
    output1[i]+=scanBlock[blockIdx.x-1];
  }

  //int start=2*(b*BLOCK_SIZE+t);
 // if(start<len)
   // output1[start]+=scanBlock[b];
  //if(start+1<len)
   // output1[start+1]+=scanBlock[b];  
}

__global__ void scan(float *input, float *output,float *Auxiliary, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float scanResult[2*BLOCK_SIZE];
  int t=threadIdx.x;
  int start=2*blockIdx.x*blockDim.x;
  //load, definitely true
  if(start+t<len){
    scanResult[t]=input[start+t];    
  }
  else{
    scanResult[t]=0.0;    
  }
  if(start+t+blockDim.x<len){
    scanResult[blockDim.x+t]=input[start+blockDim.x+t];        
  }
  else{
    scanResult[blockDim.x+t]=0.0;            
  }
  //scan, definitely true
  int stride = 1;
  while(stride < 2*BLOCK_SIZE)
  {
      __syncthreads();
      int index = (threadIdx.x+1)*stride*2 - 1;
      if(index < 2*BLOCK_SIZE && (index-stride) >=0)
          scanResult[index] += scanResult[index-stride];
      stride = stride*2;
    }
  //true
  int stride2 = BLOCK_SIZE/2;
  while(stride2 > 0)
    {
        __syncthreads();
        int index = (threadIdx.x+1)*stride2*2 - 1;
        if((index+stride2) < 2*BLOCK_SIZE)
        {
	       scanResult[index+stride2] += scanResult[index];
        }				
        stride2 = stride2 / 2;	
     }	
 //true 
  __syncthreads();
  if(start+t<len){
  output[start+t]=scanResult[t];
  }
  if(start+t+blockDim.x<len){
  output[start+t+blockDim.x]=scanResult[t+blockDim.x];
  } 
  //??
  __syncthreads();
  Auxiliary[blockIdx.x]=scanResult[2*blockDim.x-1];
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list
  float *deviceAuxiliary;
  float *devicescanBlock;
  float* noUse;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceAuxiliary, ceil(numElements/(2.0*BLOCK_SIZE)) * sizeof(float)));
  wbCheck(cudaMalloc((void **)&devicescanBlock, ceil(numElements/(2.0*BLOCK_SIZE)) * sizeof(float)));
  wbCheck(cudaMalloc((void **)&noUse, sizeof(float)));  
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  int numBlocks=ceil(numElements/(2.0*BLOCK_SIZE));
  
  dim3 dimGrid1(ceil(numElements/(2.0*BLOCK_SIZE)),1,1);  
  dim3 dimGrid2(1,1,1);    
  dim3 dimBlock1(BLOCK_SIZE,1,1);
  dim3 dimBlock2(ceil(numBlocks/2.0),1,1);
  dim3 dimBlock3(BLOCK_SIZE*2,1,1);
  
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<dimGrid1,dimBlock1>>>(deviceInput,deviceOutput,deviceAuxiliary,numElements);
  cudaDeviceSynchronize();  
  scan<<<dimGrid2,dimBlock2>>>(deviceAuxiliary,devicescanBlock,noUse,ceil(numElements/(2.0*BLOCK_SIZE)));
  cudaDeviceSynchronize();  
  add<<<dimGrid1,dimBlock3>>>(deviceOutput,devicescanBlock,numElements);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(deviceAuxiliary);
  cudaFree(devicescanBlock);
  cudaFree(noUse);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
