// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256

//@@ insert code here
__global__ void floatToChar(float* input, unsigned char *output, int width, int height){
  int x=blockDim.x*blockIdx.x+threadIdx.x;
  int y=blockDim.y*blockIdx.y+threadIdx.y;
  if(x<width&&y<height){
  int idx=blockIdx.z*width*height+y*width+x;
  output[idx]=(unsigned char)((HISTOGRAM_LENGTH-1)*input[idx]);
  }
}

__global__ void RGBtoGrey(unsigned char *input, unsigned char *output, int width, int height)
{
  int x=blockDim.x*blockIdx.x+threadIdx.x;
  int y=blockDim.y*blockIdx.y+threadIdx.y;
  if(x<width&&y<height){
  int idx=y*width+x;
    unsigned char r=input[3*idx];
    unsigned char g=input[3*idx+1];
    unsigned char b=input[3*idx+2];
    output[idx] = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
  }
}

__global__ void histogramOfGrey(unsigned char *input,unsigned int *output, int width, int height)
{
  __shared__ unsigned int histogram[HISTOGRAM_LENGTH]  ;
  //don't know why
  int i = threadIdx.x + threadIdx.y * blockDim.x;
  //int i=blockDim.x*blockIdx.x+threadIdx.x;
  if(i<HISTOGRAM_LENGTH){
    histogram[i]=0;
  }
  __syncthreads();
  int x=blockIdx.x*blockDim.x+threadIdx.x;
  int y=blockIdx.y*blockDim.y+threadIdx.y;
  if(x<width&&y<height){
    int idx=y*width+x;
    atomicAdd(&(histogram[input[idx]]),1);
  }
  __syncthreads();
  if(i<HISTOGRAM_LENGTH){
    //why??
  atomicAdd(&(output[i]),histogram[i]);
  //output[i]=histogram[i];
  }
} 

__global__ void scan(unsigned int *input, float *output, int width, int height)
{
  __shared__  unsigned int cdf[HISTOGRAM_LENGTH];
  cdf[threadIdx.x]=input[threadIdx.x];
  int stride = 1;
  while(stride < HISTOGRAM_LENGTH)
  {
      __syncthreads();
      int index = (threadIdx.x+1)*stride*2 - 1;
      if(index < HISTOGRAM_LENGTH && (index-stride) >=0)
          cdf[index] += cdf[index-stride];
      stride = stride*2;
    }
  int stride2 = HISTOGRAM_LENGTH/4;
  while(stride2 > 0)
    {
        __syncthreads();
        int index = (threadIdx.x+1)*stride2*2 - 1;
        if((index+stride2) < HISTOGRAM_LENGTH)
        {
	       cdf[index+stride2] += cdf[index];
        }				
        stride2 = stride2 / 2;	
     }	
  __syncthreads();
  output[threadIdx.x]=cdf[threadIdx.x]/((float)(width*height));
  
}

__global__ void correct_color(unsigned char *image, float *cdf, int width, int height)
{
  int x=blockDim.x*blockIdx.x+threadIdx.x;
  int y=blockDim.y*blockIdx.y+threadIdx.y;
  if(x<width&&y<height){
    int index=blockIdx.z*width*height+y*width+x;
    unsigned char val=image[index];
    float temp=255*(cdf[val]-cdf[0])/(1.0-cdf[0]);
    float clamp=min(max(temp,0.0),255.0);
    image[index]=(unsigned char) (clamp);
  }
}

__global__ void backToFloat(unsigned char *input, float *output, int width, int height)
{
  int x=blockDim.x*blockIdx.x+threadIdx.x;
  int y=blockDim.y*blockIdx.y+threadIdx.y;    
   if(x<width&&y<height){
     int idx=blockIdx.z*width*height+y*width+x;
     output[idx]=(float)(input[idx]/255.0);
   }
}
  
int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float *deviceImageFloat;
  unsigned char *deviceImageUchar;
  unsigned char *greyScale;
  unsigned int *Histogram;
  float *CDF;
  

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData=wbImage_getData(inputImage);
  hostOutputImageData=wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  cudaMalloc((void**) &deviceImageFloat, imageWidth*imageHeight*imageChannels*sizeof(float));
  cudaMalloc((void**) &deviceImageUchar, imageWidth*imageHeight*imageChannels*sizeof(unsigned char));
  cudaMalloc((void**) &greyScale, imageWidth*imageHeight*sizeof(unsigned char));
  cudaMalloc((void**) &Histogram, HISTOGRAM_LENGTH*sizeof(unsigned int));
  cudaMalloc((void**) &CDF, HISTOGRAM_LENGTH*sizeof(float) );
  
  cudaMemcpy(deviceImageFloat,hostInputImageData, imageWidth*imageHeight*imageChannels*sizeof(float),cudaMemcpyHostToDevice);
  
  dim3 dimGrid1(ceil(imageWidth/32.0),ceil(imageHeight/32.0),imageChannels);
  dim3 dimBlock1(32,32,1);
  floatToChar<<<dimGrid1,dimBlock1>>>(deviceImageFloat,deviceImageUchar,imageWidth,imageHeight);
  cudaDeviceSynchronize();
  
  dim3 dimGrid2(ceil(imageWidth/32.0),ceil(imageHeight/32.0),1);
  dim3 dimBlock2(32,32,1);
  RGBtoGrey<<<dimGrid2,dimBlock2>>>(deviceImageUchar,greyScale,imageWidth,imageHeight);
  cudaDeviceSynchronize();
  
  dim3 dimGrid3(ceil(imageWidth/32.0),ceil(imageHeight/32.0),1);
  dim3 dimBlock3(32,32,1);
  histogramOfGrey<<<dimGrid3,dimBlock3>>>(greyScale,Histogram,imageWidth,imageHeight);
  cudaDeviceSynchronize();
  
  dim3 dimGrid4(1,1,1);
  dim3 dimBlock4(HISTOGRAM_LENGTH,1,1);
  scan<<<dimGrid4,dimBlock4>>>(Histogram,CDF,imageWidth,imageHeight);
  cudaDeviceSynchronize();  
  
  dim3 dimGrid5(ceil(imageWidth/32.0),ceil(imageHeight/32.0),imageChannels);
  dim3 dimBlock5(32,32,1);
  correct_color<<<dimGrid5,dimBlock5>>>(deviceImageUchar,CDF,imageWidth,imageHeight);
  cudaDeviceSynchronize();
  
  dim3 dimGrid6(ceil(imageWidth/32.0),ceil(imageHeight/32.0),imageChannels);
  dim3 dimBlock6(32,32,1);
  backToFloat<<<dimGrid6,dimBlock6>>>(deviceImageUchar,deviceImageFloat,imageWidth,imageHeight);
  cudaDeviceSynchronize();  
  
  cudaMemcpy(hostOutputImageData,deviceImageFloat,imageWidth*imageHeight*imageChannels*sizeof(float),cudaMemcpyDeviceToHost);
  
  cudaFree(deviceImageFloat);
  cudaFree(deviceImageUchar);
  cudaFree(greyScale);
  cudaFree(Histogram);
  cudaFree(CDF);
  
  wbSolution(args, outputImage);

  //@@ insert code here

  return 0;
}
