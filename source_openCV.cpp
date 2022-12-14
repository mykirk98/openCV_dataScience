#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>

#include <opencv2/opencv.hpp>


using namespace cv;


typedef struct {
	int r, g, b;
}int_rgb;

int** IntAlloc2(int height, int width)
{
	int** tmp;
	tmp = (int**)calloc(height, sizeof(int*));
	for (int i = 0; i < height; i++)
		tmp[i] = (int*)calloc(width, sizeof(int));
	return(tmp);
}

void IntFree2(int** image, int height, int width)
{
	for (int i = 0; i < height; i++)
		free(image[i]);

	free(image);
}

float** FloatAlloc2(int height, int width)
{
	float** tmp;
	tmp = (float**)calloc(height, sizeof(float*));
	for (int i = 0; i < height; i++)
		tmp[i] = (float*)calloc(width, sizeof(float));
	return(tmp);
}

void FloatFree2(float** image, int height, int width)
{
	for (int i = 0; i < height; i++)
		free(image[i]);

	free(image);
}

int_rgb** IntColorAlloc2(int height, int width)
{
	int_rgb** tmp;
	tmp = (int_rgb**)calloc(height, sizeof(int_rgb*));
	for (int i = 0; i < height; i++)
		tmp[i] = (int_rgb*)calloc(width, sizeof(int_rgb));
	return(tmp);
}

void IntColorFree2(int_rgb** image, int height, int width)
{
	for (int i = 0; i < height; i++)
		free(image[i]);

	free(image);
}

int** ReadImage(char* name, int* height, int* width)
{
	Mat img = imread(name, IMREAD_GRAYSCALE);
	int** image = (int**)IntAlloc2(img.rows, img.cols);

	*width = img.cols;
	*height = img.rows;

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			image[i][j] = img.at<unsigned char>(i, j);

	return(image);
}

void WriteImage(char* name, int** image, int height, int width)
{
	Mat img(height, width, CV_8UC1);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			img.at<unsigned char>(i, j) = (unsigned char)image[i][j];

	imwrite(name, img);
}


void ImageShow(char* winname, int** image, int height, int width)
{
	Mat img(height, width, CV_8UC1);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			img.at<unsigned char>(i, j) = (unsigned char)image[i][j];
	imshow(winname, img);
	waitKey(0);
}



int_rgb** ReadColorImage(char* name, int* height, int* width)
{
	Mat img = imread(name, IMREAD_COLOR);
	int_rgb** image = (int_rgb**)IntColorAlloc2(img.rows, img.cols);

	*width = img.cols;
	*height = img.rows;

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++) {
			image[i][j].b = img.at<Vec3b>(i, j)[0];
			image[i][j].g = img.at<Vec3b>(i, j)[1];
			image[i][j].r = img.at<Vec3b>(i, j)[2];
		}

	return(image);
}

void WriteColorImage(char* name, int_rgb** image, int height, int width)
{
	Mat img(height, width, CV_8UC3);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
			img.at<Vec3b>(i, j)[0] = (unsigned char)image[i][j].b;
			img.at<Vec3b>(i, j)[1] = (unsigned char)image[i][j].g;
			img.at<Vec3b>(i, j)[2] = (unsigned char)image[i][j].r;
		}

	imwrite(name, img);
}

void ColorImageShow(char* winname, int_rgb** image, int height, int width)
{
	Mat img(height, width, CV_8UC3);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
			img.at<Vec3b>(i, j)[0] = (unsigned char)image[i][j].b;
			img.at<Vec3b>(i, j)[1] = (unsigned char)image[i][j].g;
			img.at<Vec3b>(i, j)[2] = (unsigned char)image[i][j].r;
		}
	imshow(winname, img);

}

template <typename _TP>
void ConnectedComponentLabeling(_TP** seg, int height, int width, int** label, int* no_label)
{

	//Mat bw = threshval < 128 ? (img < threshval) : (img > threshval);
	Mat bw(height, width, CV_8U);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++)
			bw.at<unsigned char>(i, j) = (unsigned char)seg[i][j];
	}
	Mat labelImage(bw.size(), CV_32S);
	*no_label = connectedComponents(bw, labelImage, 8); // 0???? ?????? ??????

	(*no_label)--;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++)
			label[i][j] = labelImage.at<int>(i, j);
	}
}

#define imax(x, y) ((x)>(y) ? x : y)
#define imin(x, y) ((x)<(y) ? x : y)

int BilinearInterpolation(int** image, int width, int height, double x, double y)
{
	int x_int = (int)x;
	int y_int = (int)y;

	int A = image[imin(imax(y_int, 0), height - 1)][imin(imax(x_int, 0), width - 1)];
	int B = image[imin(imax(y_int, 0), height - 1)][imin(imax(x_int + 1, 0), width - 1)];
	int C = image[imin(imax(y_int + 1, 0), height - 1)][imin(imax(x_int, 0), width - 1)];
	int D = image[imin(imax(y_int + 1, 0), height - 1)][imin(imax(x_int + 1, 0), width - 1)];

	double dx = x - x_int;
	double dy = y - y_int;

	double value
		= (1.0 - dx) * (1.0 - dy) * A + dx * (1.0 - dy) * B
		+ (1.0 - dx) * dy * C + dx * dy * D;

	return((int)(value + 0.5));
}


void DrawHistogram(char* comments, int* Hist)
{
	int histSize = 256; /// Establish the number of bins
						// Draw the histograms for B, G and R
	int hist_w = 512; int hist_h = 512;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(255, 255, 255));
	Mat r_hist(histSize, 1, CV_32FC1);
	for (int i = 0; i < histSize; i++)
		r_hist.at<float>(i, 0) = Hist[i];
	/// Normalize the result to [ 0, histImage.rows ]
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	/// Draw for each channel
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
	}

	/// Display
	namedWindow(comments, WINDOW_AUTOSIZE);
	imshow(comments, histImage);

	waitKey(0);

}

int EX0916_1()
{
	int height, width;
	int** img = (int**)ReadImage((char*)"barbara.png", &height, &width);
	
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (img[y][x] >= 200)
			{
				img[y][x] = 0;
			}
			else
			{
				img[y][x] = 255;
			}
			//img[y][x] = 255;
			//printf("(%d, %d) ", y, x);
			//printf("%d ", img[y][x]);
		}
	}
	//printf("\n height = %d, width = %d", height, width);
	
	ImageShow((char*)"TEST", img, height, width);
	
	return 0;
}

int EX0916_2(int** img, int height, int width)					//int height, width XXXXX int height, int width ???? ???????? ?????? ???? ???? ???? ?????? ??????????
{
//	int height, width;
//	int** img = (int**)ReadImage((char*)"barbara.png", &height, &width);

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (img[y][x] >= 200)
			{
				img[y][x] = 0;
			}
			else
			{
				img[y][x] = 255;
			}
			//img[y][x] = 255;
			//printf("(%d, %d) ", y, x);
			//printf("%d ", img[y][x]);
		}
	}
	//printf("\n height = %d, width = %d", height, width);

	ImageShow((char*)"TEST", img, height, width);

	return 0;
}

int EX0916_3()
{
	int height, width;
	int** img0 = (int**)ReadImage((char*)"barbara.png", &height, &width);
	int** img1 = (int**)ReadImage((char*)"lena.png", &height, &width);

	EX0916_2(img1, height, width);

	return 0;
}

int circle()
{
	int height, width;
	int** img = (int**)ReadImage((char*)"barbara.png", &height, &width);
	
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (((x - 256) * (x - 256) + (y - 256) * (y - 256)) <= 10000)
			{
				img[y][x] = 0;
			}
		}
	}
	ImageShow((char*)"TEST", img, height, width);
	
	return 0;
}

int circle2()
{
	int height, width;
	int** img = (int**)ReadImage((char*)"barbara.png", &height, &width);
	
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			img[y][x] = 0;
			if (((x - 200) * (x - 200) + (y - 256) * (y - 256)) <= 20000)
			{
				img[y][x] = 250;
			}
			if(((x - 300) * (x - 300) + (y - 256) * (y - 256)) <= 10000)
			{
				img[y][x] = 100;
			}
			if((((x - 200) * (x - 200) + (y - 256) * (y - 256)) <= 20000) && (((x - 300) * (x - 300) + (y - 256) * (y - 256)) <= 10000))
			{
				img[y][x] = 180;
			}
		}
	}
	
	ImageShow((char*)"TEST", img, height, width);

	return 0;
}

#define GetMax(x,y)	((x > y) ? x : y)
#define GetMin(x,y)	((x < y) ? x : y)
#define Clipping(x) (GetMax(GetMin(x,255),0))

void addValue2Image(int add_number,		//???? ??
	int** img_input,		//???? ??????
	int height,		//???? ????
	int width,		//???? ??
	int** img_output		//???? ??????
)
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			img_output[y][x] = img_input[y][x] + add_number;
		}
	}
}
void Image_Clipping(int** img_input, int height, int width, int** img_output)
{

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{

			img_output[y][x] = Clipping(img_input[y][x]);
			/////////////////////if???? ?????? ?????? 2/////////////
			//img_output[y][x] = GetMax(GetMin(img_input[y][x], 255), 0);
			
			/////////////////////if???? ?????? ?????? 1/////////////
			/*if (img_output[y][x] < 0)
			{
				img_output[y][x] = 0;
			}
			else if (img_output[y][x] > 255)
			{
				img_output[y][x] = 255;
			}
			else
			{
				img_output[y][x] = img_output[y][x];
			}*/
		}
	}
}
void EX0923_1()
{
	int height, width;
	int** img = (int**)ReadImage((char*)"barbara.png", &height, &width);
	int** img2 = (int**)ReadImage((char*)"barbara.png", &height, &width);
	
	int** img_out1 = (int**)IntAlloc2(height, width);
	int** img_out2 = (int**)IntAlloc2(height, width);

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			img_out1[y][x] = img[y][x] + 50;
			img_out2[y][x] = img[y][x] - 50;
		}
	}

	addValue2Image(50, img, height, width, img_out1);
	addValue2Image(-50, img, height, width, img_out1);

//	for (int y = 0; y < height; y++)
//	{
//		for (int x = 0; x < width; x++)
//		{
//			img_out1[y][x] = img[y][x] + 50;
////			img_out2[y][x] = img[y][x] - 50;
//
//			if (img_out1[y][x] < 0)
//			{
//				img_out1[y][x] = 0;
//			}
//			else if (img_out1[y][x] > 255)
//			{
//				img_out1[y][x] = 255;
//			}
//			else
//			{
//				img_out1[y][x] = img_out1[y][x];
//			}
//		}
//	}

	ImageShow((char*)"????1????????", img, height, width);
	ImageShow((char*)"????2????????", img, height, width);

}

void EX0923_2()
{
	int height, width;
	int** img = (int**)ReadImage((char*)"barbara.png", &height, &width);

	int** img_out1 = (int**)IntAlloc2(height, width);
	int** img_out2 = (int**)IntAlloc2(height, width);


	addValue2Image(50, img, height, width, img_out1);
	addValue2Image(-50, img, height, width, img_out2);

	ImageShow((char*)"????1 ???? + 50 ?? ????????", img_out1, height, width);
	ImageShow((char*)"????2 ???? - 50 ?? ????????", img_out2, height, width);
	
	Image_Clipping(img_out1, height, width, img_out1);
	Image_Clipping(img_out2, height, width, img_out2);

	ImageShow((char*)"????????????", img, height, width);
	ImageShow((char*)"????1 ???? + 50 ?? ?????? ?? ????????", img_out1, height, width);
	ImageShow((char*)"????2 ???? - 50 ?? ?????? ?? ????????", img_out2, height, width);
}

void EX0923_3()
{
	int height, width;
	int** img = (int**)ReadImage((char*)"barbara.png", &height, &width);

	int** img_out1 = (int**)IntAlloc2(height, width);
	int** img_out2 = (int**)IntAlloc2(height, width);

	int maxvalue = GetMax(3, 2);
	int minvalue = GetMin(3, 2);

	int a = 300;
	int b = -10;
	int c = 200;

	a =GetMax(GetMin(a, 255), 0);
	b = GetMax(GetMin(b, 255), 0);
	c = GetMax(GetMin(c, 255), 0);

	printf("%d %d %d\n", a, b, c);
}

void ImageMixing(float alpha, int** img_input1, int** img_input2, int height, int width, int** img_output)
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			img_output[y][x] = alpha * img_input1[y][x] + (1.0 - alpha) * img_input2[y][x];
		}
	}
}

void EX0923_4(char* window_name1, char* window_name2, char* window_name3)
{
	int height, width;
	int** img1 = (int**)ReadImage((char*)"barbara.png", &height, &width);
	int** img2 = (int**)ReadImage((char*)"lena.png", &height, &width);

	int** img_out = (int**)IntAlloc2(height, width);

	float alpha = 0.5;

	ImageMixing(alpha, img1, img2, height, width, img_out);
	//for (int y = 0; y < height; y++)
	//{
	//	for (int x = 0; x < width; x++)
	//	{
	//		img_out[y][x] = alpha * img1[y][x] + (1.0 - alpha) * img2[y][x];
	//	}
	//}

	ImageShow(window_name1, img1, height, width);
	ImageShow(window_name2, img2, height, width);
	ImageShow(window_name3, img_out, height, width);
}

#define RoundUp(x) ((int)(x+0.5))

void Stretching_1(int height, int width, int** img_input, int** img_output, int a, int b, int c, int d)
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (0 < img_input[y][x] && img_input[y][x] <= a)
			{
				img_output[y][x] = RoundUp((float)c / a * img_input[y][x]);
			}
			else if(a < img_input[y][x] && img_input[y][x] <= b)
			{
				img_output[y][x] = RoundUp(((float)d - c) / (b - a) * (img_input[y][x] - a) + c);
			}
			else
			{
				img_output[y][x] = RoundUp((float)(255 - d) / (255 - b) * (img_input[y][x] - b) + d);
			}
		}
	}
}

void EX0930_1()
{
	int height, width;
	int** img = (int**)ReadImage((char*)"barbara.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	int a = 100, b = 150, c = 50, d = 200;

	Stretching_1(height, width, img, img_out, a, b, c, d);
	
	ImageShow((char*)"????????????", img, height, width);
	ImageShow((char*)"????????????", img_out, height, width);

}

void GetHistogram_1(int height, int width, int** img, int* Hist)
{
	for (int brightness = 0; brightness < 256; brightness++)
	{
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				if (img[y][x] == brightness)
				{
					Hist[brightness]++;
				}
			}
		}
	}
}

void GetHistogram_2(int height, int width, int** img, int* Hist)
{
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				Hist[img[y][x]]++;
			}
		}
}
void EX0930_2()
{
	int height, width;
	int** img = (int**)ReadImage((char*)"lena.png", &height, &width);
	int count = 0;

	int Hist[256] = { 0 };

	GetHistogram_1(height, width, img, Hist);

	ImageShow((char*)"????????????", img, height, width);
	DrawHistogram((char*)"??????????" , Hist);		// histogram?? ???????? ????
}

void C_Histogram(int** img, int height, int width, int* C_Hist)
{
	int Hist[256] = { 0 };
	GetHistogram_1(height, width, img, Hist);
	C_Hist[0] = Hist[0];

	for (int n = 1; n < 256; n++)
	{
		C_Hist[n] = Hist[n] + C_Hist[n - 1];
	}
}

void norm_C_Histogram(int** img, int height, int width, int* NC_Hist)
{
	
	int C_Hist[256] = { 0 };
	C_Histogram(img, height, width, C_Hist);


	for (int n = 0; n < 256; n++)
	{
		NC_Hist[n] = C_Hist[n] * 255 / (width * height);
		//NC_Hist[n] = (float)C_Hist[n] * (width * height) / 255;
	}

}
	
void EX1014_1()
{
	int height, width;
	int** img = (int**)ReadImage((char*)"lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	int C_Hist[256] = {0};
	C_Histogram(img, height, width, C_Hist);
	
	int Hist[256] = { 0 };
	
	ImageShow((char*)"????????????", img, height, width);
	DrawHistogram((char*)"??????????", C_Hist);		// histogram?? ???????? ????

}

void EX1014_2()
{
	int height, width;
	int** img = (int**)ReadImage((char*)"lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	int NC_Hist[256] = { 0 };
	norm_C_Histogram(img, height, width, NC_Hist);
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			img_out[y][x] = NC_Hist[img[y][x]];
		}
	}

	ImageShow((char*)"????????????", img, height, width);
	DrawHistogram((char*)"??????????????", NC_Hist);		// norm_C_Histogram?? ???????? ????
}

void HistogramEqualization(int** img, int height, int width, int** img_out)
{
	int NC_Hist[256] = { 0 };

	norm_C_Histogram(img, height, width, NC_Hist);

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			img_out[y][x] = NC_Hist[img[y][x]];
		}
	}
}

void EX1014_3()
{
	int height, width;
	int** img = (int**)ReadImage((char*)"lenax0.5.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);
	
	int NC_Hist[256] = { 0 };
	HistogramEqualization(img, height, width, img_out);
	norm_C_Histogram(img, height, width, NC_Hist);

	ImageShow((char*)"????????????", img, height, width);
	ImageShow((char*)"????????????", img_out, height, width);
	DrawHistogram((char*)"??????????????", NC_Hist);		// norm_C_Histogram?? ???????? ????
}

void AVERAGE_FILTER()
{
	int height, width;
	int** img = (int**)ReadImage((char*)"lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	for (int y = 1; y < height - 1; y++)
	{
		for (int x = 1; x < width - 1; x++)
		{
			img_out[y][x] = (img[y - 1][x - 1] + img[y - 1][x] + img[y - 1][x + 1] + img[y][x - 1] + img[y][x] + img[y][x + 1] + img[y + 1][x - 1] + img[y + 1][x] + img[y + 1][x + 1]) / 9;
		}
	}
	ImageShow((char*)"????????????", img, height, width);
	ImageShow((char*)"????????????", img_out, height, width);
}

void Avg3x3(int** img, int height, int width, int** img_out)
{
	
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (x == 0 || (x == width - 1) || y == 0 || (y == height - 1))
			{
				img_out[y][x] = img[y][x];
			}
			else
			{
				int sum = 0;
				for (int i = -1; i <= 1; i++)
				{
					for (int j = -1; j <= 1; j++)
					{
						sum += img[y + i][x + j];
					}
				}
				img_out[y][x] = sum / 9.0 + 0.5;
			}
		}
	}
}

void AvgNxN(int N, int** img, int height, int width, int** img_out)
{
	int delta = (N - 1) / 2;
	
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			
			
			if (x <= delta || x >= width - delta || y <= delta || y >= height - delta)
			{
				img_out[y][x] = img[y][x];
			}
			else
			{
				int sum = 0;
				for (int i = -delta; i <= delta; i++)
				{
					for (int j = -delta; j <= delta; j++)
					{
						sum += img[y + i][x + j];
					}
				}
				img_out[y][x] = (float)sum / (N*N) + 0.5;
			}
		}
	}
}

void AvgNxN_two(int N, int** img, int height, int width, int** img_out)
{
	int delta = (N - 1) / 2;
	
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			
			int sum = 0;
			for (int i = -delta; i <= delta; i++)
			{
				for (int j = -delta; j <= delta; j++)
				{
					sum += img[GetMin(GetMax(y + i,0), height-1)][GetMin(GetMax(x + j,0),width-1)];
				}
			}
			img_out[y][x] = (float)sum / (N * N) + 0.5;
		}
	}
}

void EX1021_4(int** img, int height, int width, int** img_out)
{
	float mask[3][3] = { {1 / 9.0, 1 / 9.0, 1 / 9.0}
						, {1 / 9.0, 1 / 9.0, 1 / 9.0}
						, {1 / 9.0, 1 / 9.0, 1 / 9.0} };
	
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (x == 0 || (x == width - 1) || y == 0 || (y == height - 1))
			{
				img_out[y][x] = img[y][x];
			}
			else
			{
				int sum = 0;
				float avg = 0.0;
				for (int i = -1; i <= 1; i++)
				{
					for (int j = -1; j <= 1; j++)
					{
						avg += mask[i+1][j+1] * img[y + i][x + j];
					}
				}
				img_out[y][x] = avg+ 0.5;
			}
		}
	}
}

void AVG_3X3_MASK_2nd(float** mask, int** img, int height, int width, int** img_out)
{

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (x == 0 || (x == width - 1) || y == 0 || (y == height - 1))
			{
				img_out[y][x] = img[y][x];
			}
			else
			{
				int sum = 0;
				float avg = 0.0;
				for (int i = -1; i <= 1; i++)
				{
					for (int j = -1; j <= 1; j++)
					{
						avg += mask[i + 1][j + 1] * img[y + i][x + j];
					}
				}
				img_out[y][x] =Clipping(avg + 0.5);
			}
		}
	}
}

void EX1021_5(float mask[3][3], int** img, int height, int width, int** img_out)
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (x == 0 || (x == width - 1) || y == 0 || (y == height - 1))
			{
				img_out[y][x] = img[y][x];
			}
			else
			{
				int sum = 0;
				float avg = 0.0;
				for (int i = -1; i <= 1; i++)
				{
					for (int j = -1; j <= 1; j++)
					{
						avg += mask[i + 1][j + 1] * img[y + i][x + j];
					}
				}
				img_out[y][x] = avg + 0.5;
			}
		}
	}
}

void main_1021()
{
	int height, width;
	int** img = (int**)ReadImage((char*)"lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);
	int** img_out333 = (int**)IntAlloc2(height, width);
//	int** img_out555 = (int**)IntAlloc2(height, width);

	float** mask = (float**)FloatAlloc2(3, 3);
	
	mask[0][0] = 0;		mask[0][1] = -1 / 4.0;		mask[0][2] = 0;
	mask[1][0] = -1 / 4.0;		mask[1][1] = 2.0;		mask[1][2] = -1 / 4.0;
	mask[2][0] = 0;		mask[2][1] = -1 / 4.0;		mask[2][2] = 0;
	int N = 15;

	
//	Avg3x3(img, height, width, img_out333);
//	AvgNxN_two(N, img, height, width, img_out);

	AVG_3X3_MASK_2nd(mask, img, height, width, img_out);

	ImageShow((char*)"????????????", img, height, width);
//	ImageShow((char*)"3X3 ????????????", img_out333, height, width);
//	ImageShow((char*)"9X9????????????", img_out555, height, width);
	ImageShow((char*)"3X3 ????????????", img_out, height, width);
}

void MagGradient_X(int** img, int height, int width, int** img_out)
{
	for (int i = 0; i < height - 1; i++)
	{
		for (int j = 0; j < width; j++)
		{
			img_out[i][j] = abs(img[i + 1][j] - img[i][j]);
		}
	}
}

void MagGradient_Y(int** img, int height, int width, int** img_out)
{
	for(int i = 0; i < height; i++)
	{
		for (int j = 0; j < width - 1; j++)
		{
			img_out[i][j] = abs(img[i][j + 1] - img[i][j]);
		}
	}
}

void MagGradient(int** img, int height, int width, int** img_out)
{
	for (int i = 0; i < height - 1; i++)
	{
		for (int j = 0; j < width - 1; j++)
		{
			img_out[i][j] = abs(img[i + 1][j] - img[i][j]) + abs(img[i][j + 1] - img[i][j]);
		}
	}
}

int FindMaxValue(int** img, int height, int width)
{
//	img[0][0] = 0;
	int max_value = img[0][0];

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			max_value = GetMax(max_value, img[i][j]);
		}
	}

	return (max_value);		//?????? ???? ?????? ???? ??????, ?????????? ??????????
}

void NormalizeByMax(int** img, int height, int width, int** img_out)
{
	int max_value = FindMaxValue(img, height, width);
	
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			img_out[i][j] = (float)img[i][j] / max_value * 255;
		}
	}
}


void main()
{
	int height, width;
	int** img = (int**)ReadImage((char*)"lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);
	int** img_out_X = (int**)IntAlloc2(height, width);
	int** img_out_Y = (int**)IntAlloc2(height, width);
	int** img_out_XY = (int**)IntAlloc2(height, width);
	int** img_out_NBM = (int**)IntAlloc2(height, width);
	
	MagGradient_X(img, height, width, img_out_X);
	MagGradient_Y(img, height, width, img_out_Y);
	MagGradient(img, height, width, img_out_XY);

	int max_value_main = FindMaxValue(img_out_XY, height, width);
	
	NormalizeByMax(img_out_XY, height, width, img_out_NBM);




	ImageShow((char*)"????????????", img, height, width);
	ImageShow((char*)"????????????_MagGradient_X", img_out_X, height, width);
	ImageShow((char*)"????????????_MagGradient_Y", img_out_Y, height, width);
	ImageShow((char*)"????????????_MagGradient_XY", img_out_XY, height, width);
	ImageShow((char*)"????????????_MagGradient_NBM", img_out_NBM, height, width);

	printf("\n\n\n\n\n\n\n max = %d\n\n\n\n\n\n\n", max_value_main);
}