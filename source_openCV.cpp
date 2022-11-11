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
	*no_label = connectedComponents(bw, labelImage, 8); // 0까지 포함된 갯수임

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

void Thresholding(int threshold, int** img, int height, int width, int** img_out)
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (img[y][x] >= threshold)
			{
				img_out[y][x] = 255;
			}
			else
			{
				img_out[y][x] = 0;
			}
		}
	}
}

void Threshold_MAIN()
{
	int height, width;
	int** img = (int**)ReadImage((char*)"barbara.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	int threshold = 128;

	Thresholding(threshold, img, height, width, img_out);

	ImageShow((char*)"입력영상보기", img, height, width);
	ImageShow((char*)"출력영상보기", img_out, height, width);
}

#define GetMax(x,y) ((x>y)?x:y)
#define GetMin(x,y) ((x<y)?x:y)
#define Clipping(x) (GetMax(GetMin(x,255),0))

void AddValue2Image(int value, int** img_in, int height, int width, int** img_out)
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			img_out[y][x] = img_in[y][x] + value;
		}
	}
}

void ImageClpping(int** img_in, int height, int width, int** img_out)
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (img_in[y][x] > 255)
			{
				img_out[y][x] = 255;
			}
			else if (img_in[y][x] < 0)
			{
				img_out[y][x] = 0;
			}
			else
				img_out[y][x] = img_in[y][x];
		}
	}
}

void Clipping_MAIN()
{
	int height, width;
	int** img = (int**)ReadImage((char*)"barbara.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);
	int** img_out1 = (int**)IntAlloc2(height, width);
	int** img_out2 = (int**)IntAlloc2(height, width);
	
	AddValue2Image(50, img, height, width, img_out1);
	AddValue2Image(-50, img, height, width, img_out2);

	ImageShow((char*)"입력영상보기", img, height, width);
	ImageShow((char*)"픽셀값+50출력영상1보기", img_out1, height, width);
	ImageShow((char*)"픽셀값-50출력영상2보기", img_out2, height, width);

	ImageClpping(img_out1, height, width, img_out1);
	ImageClpping(img_out2, height, width, img_out2);

	
	ImageShow((char*)"CLIPPING출력영상1보기", img_out1, height, width);
	ImageShow((char*)"CLIPPING출력영상2보기", img_out2, height, width);
}

void ImageMixing(float alpha,int** img_in1, int** img_in2, int height, int width, int** img_out)
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			img_out[y][x] = alpha * img_in1[y][x] + (1 - alpha) * img_in2[y][x];
		}
	}
}
void ImageMixing_MAIN()
{
	int height, width;

	int** img1 = (int**)ReadImage((char*)"barbara.png", &height, &width);
	int** img2 = (int**)ReadImage((char*)"lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);
	
	float alpha = 0.5;

	ImageMixing(alpha, img1, img2, height, width, img_out);

	ImageShow((char*)"입력영상보기", img1, height, width);
	ImageShow((char*)"출력영상보기", img_out, height, width);
}

#define RoundUp(x) ((int)(x+0.5))
void ImageStretch(int a, int b, int c, int d, int** img_in, int height, int width, int** img_out)
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (0 <= img_in[y][x] && img_in[y][x] < a)
			{
				img_out[y][x] = RoundUp((float)c / a * img_in[y][x]);
			}
			if (a <= img_in[y][x] && img_in[y][x] < b)
			{
				img_out[y][x] = RoundUp((float)(d - c) / (b - a) * (img_in[y][x] - a) + c);
			}
			if (b <= img_in[y][x] && img_in[y][x] < 255)
			{
				img_out[y][x] = RoundUp((float) (255 - d) / (255 - b) * (img_in[y][x] - b) + d);
			}
		}
	}
}

void ImageStretch_MAIN()
{
	int height, width;

	int** img = (int**)ReadImage((char*)"barbara.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	int a = 100, b = 200, c = 100, d = 150;
	ImageStretch(a, b, c, d, img, height, width, img_out);

	ImageShow((char*)"입력영상보기", img, height, width);
	ImageShow((char*)"출력영상보기", img_out, height, width);
}

void Histogram(int** img_in, int height, int width, int* Hist)
{
	for (int brightness = 0; brightness < 256; brightness++)
	{
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				if (img_in[y][x] == brightness)
				{
					Hist[brightness]++;
				}
			}
		}
	}
}

void Histogram2(int** img_in, int height, int width, int* Hist)
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			Hist[img_in[y][x]]++;
		}
	}
}
void C_Histogram(int** img_in, int height, int width, int* C_Hist)
{
	int Hist[256] = { 0 };
	C_Hist[0] = Hist[0];
	Histogram(img_in, height, width, Hist);

	for (int cum = 0; cum < 256; cum++)
	{
		C_Hist[cum] = C_Hist[cum - 1] + Hist[cum];
	}
}

void Norm_C_Histogram(int** img_in, int height, int width, int* NC_Hist)
{
	int C_Hist[256] = { 0 };
	C_Histogram(img_in, height, width, C_Hist);

	for (int I = 0; I < 256; I++)
	{
		NC_Hist[I] = C_Hist[I] * (float)255 / (height * width);
	}
}
void Histogram_MAIN()
{
	int height, width;

	int** img = (int**)ReadImage((char*)"barbara.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	int Hist[256] = { 0 };
	Histogram(img, height, width, Hist);

	int C_Hist[256] = { 0 };
	C_Histogram(img, height, width, C_Hist);
	
	int NC_Hist[256] = { 0 };
	Norm_C_Histogram(img, height, width, NC_Hist);

	ImageShow((char*)"입력영상보기", img, height, width);
	DrawHistogram((char*)"히스토그램", Hist);
	DrawHistogram((char*)"누적히스토그램", C_Hist);
	DrawHistogram((char*)"정규화 누적히스토그램", NC_Hist);
}

void AVG_3X3(int** img_in, int height, int width, int** img_out)
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (y == 0 || y == height - 1 || x == 0 || x == width - 1)
			{
				img_out[y][x] = img_in[y][x];
			}
			else
			{
				for (int i = -1; i <= 1; i++)
				{
					for (int j = -1; j <= 1; j++)
					{
						img_out[y][x] += img_in[y + i][x + j] / 9.0;
					}
				}
				img_out[y][x] += 0.5;
			}
		}
	}
}

void AVG_NXN(int N, int** img_in, int height, int width, int** img_out)
{
	int delta = (N - 1) / 2;

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (y <= delta || y >= height - delta || x <= delta || x >= width - delta)
			{
				img_out[y][x] = img_in[y][x];
			}
			else
			{
				for (int i = -delta; i <= delta; i++)
				{
					for (int j = -delta; j <= delta; j++)
					{
						img_out[y][x] += img_in[y + i][x + j] / (float)(N * N);
					}
				}
				img_out[y][x] += 0.5;
			}	
		}
	}
}

void AVG_NXN_TWO(int N, int** img_in, int height, int width, int** img_out)
{
	int delta = (N - 1) / 2;

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			for (int i = -delta; i <= delta; i++)
			{
				for (int j = -delta; j <= delta; j++)
				{
					img_out[y][x] += img_in[GetMin(GetMax(y + i, 0), height - 1)][GetMin(GetMax(x + j, 0), width - 1)] / (float)(N * N);
				}
			}
			img_out[y][x] += 0.5;
		}
	}
}

void AVG_3X3_MASK(int** img_in, int height, int width, int** img_out)
{
	float mask[3][3] = { {1 / 9.0, 1 / 9.0, 1 / 9.0},
						{1 / 9.0, 1 / 9.0, 1 / 9.0},
						{1 / 9.0, 1 / 9.0, 1 / 9.0} };

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if ((y < (3 - 1) / 2) || (y >= height - (3 - 1) / 2) || (x < (3 - 1) / 2) || (x >= width - (3 - 1) / 2))
			{
				img_out[y][x] = img_in[y][x];
			}
			else
			{
				for (int i = -(3 - 1) / 2; i <= (3 - 1) / 2; i++)
				{
					for (int j = -(3 - 1) / 2; j <= (3 - 1) / 2; j++)
					{
						img_out[y][x] += img_in[y + i][x + j] * mask[i + 1][j + 1];
					}
				}
				img_out[y][x] += 0.5;
			}
		}
	}
}

void AVG_3X3_with_MASK_input(float** mask, int** img_in, int height, int width, int** img_out)
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if ((y < (3 - 1) / 2) || (y >= height - (3 - 1) / 2) || (x < (3 - 1) / 2) || (x >= width - (3 - 1) / 2))
			{
				img_out[y][x] = img_in[y][x];
			}
			else
			{
				for (int i = -(3 - 1) / 2; i <= (3 - 1) / 2; i++)
				{
					for (int j = -(3 - 1) / 2; j <= (3 - 1) / 2; j++)
					{
						img_out[y][x] += img_in[y + i][x + j] * mask[i + 1][j + 1];
					}
				}
				img_out[y][x] += 0.5;
			}
		}
	}
}

void Filtering_MAIN()
{
	int height, width;
	int** img = (int**)ReadImage((char*)"barbara.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);
//	int** img_out_AVG_3X3 = (int**)IntAlloc2(height, width);
//	int** img_out_AVG_NXN = (int**)IntAlloc2(height, width);
	int** img_out_AVG_3X3_MASK = (int**)IntAlloc2(height, width);

//	AVG_3X3(img, height, width, img_out1);
//	AVG_NXN_TWO(9, img, height, width, img_out2);
	AVG_3X3_MASK(img, height, width, img_out_AVG_3X3_MASK);

	//float** mask = (float**)FloatAlloc2(3, 3);

	//mask[0][0] = 0;
	//mask[0][-1] = -1/4.0;
	//mask[0][2] = 0;
	//mask[1][0] = -1/4.0;
	//mask[1][1] = 2.0;
	//mask[1][2] = -1 / 4.0;
	//mask[2][0] = 0;
	//mask[2][1] = -1 / 4.0;
	//mask[2][2] = 0;

//	AVG_3X3_with_MASK_input(mask, img, height, width, img_out_AVG_3X3_MASK);
	
	ImageShow((char*)"입력영상보기", img, height, width);
//	ImageShow((char*)"3X3출력영상보기", img_out_AVG_3X3, height, width);
//	ImageShow((char*)"NXN출력영상보기", img_out_AVG_NXN, height, width);
	ImageShow((char*)"NXN출력영상보기", img_out_AVG_3X3_MASK, height, width);
}

void MagGradient_X(int** img_in, int height, int width, int** img_out)
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width - 1; x++)
		{
			img_out[y][x] = abs(img_in[y][x + 1] - img_in[y][x]);
		}
	}
}

void MagGradient_Y(int** img_in, int height, int width, int** img_out)
{
	for (int y = 0; y < height - 1; y++)
	{
		for (int x = 0; x < width; x++)
		{
			img_out[y][x] = abs(img_in[y + 1][x] - img_in[y][x]);
		}
	}
}

void MagGradient_XY(int** img_in, int height, int width, int** img_out)
{
	for (int y = 0; y < height - 1; y++)
	{
		for (int x = 0; x < width - 1; x++)
		{
			img_out[y][x] = abs(img_in[y][x + 1] - img_in[y][x]) + abs(img_in[y + 1][x] - img_in[y][x]);
		}
	}
}

int FindMaxValue(int** img_in, int height, int width)
{
	int max_value = img_in[0][0];

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			max_value = GetMax(max_value, img_in[y][x]);
		}
	}

	return max_value;
}

void NormalizeByMax(int** img_in, int height, int width, int** img_out)
{
	int max_value = FindMaxValue(img_in, height, width);

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			img_out[y][x] = (float)img_in[y][x] / max_value * 255;
		}
	}
}

void main()
{
	int height, width;
	int** img = (int**)ReadImage((char*)"lena.png", &height, &width);
//	int** img_out_MagGradient_X = (int**)IntAlloc2(height, width);
//	int** img_out_MagGradient_Y = (int**)IntAlloc2(height, width);
	int** img_out_MagGradient_XY = (int**)IntAlloc2(height, width);
	int** img_out_NormalizeByMax = (int**)IntAlloc2(height, width);

//	MagGradient_X(img, height, width, img_out_MagGradient_X);
//	MagGradient_Y(img, height, width, img_out_MagGradient_Y);
	MagGradient_XY(img, height, width, img_out_MagGradient_XY);
	NormalizeByMax(img_out_MagGradient_XY, height, width, img_out_NormalizeByMax);

	ImageShow((char*)"입력영상보기", img, height, width);
//	ImageShow((char*)"MagGradinet_X출력영상보기", img_out_MagGradient_X, height, width);
//	ImageShow((char*)"MagGradinet_Y출력영상보기", img_out_MagGradient_Y, height, width);
	ImageShow((char*)"MagGradinet_XY출력영상보기", img_out_MagGradient_XY, height, width);
	ImageShow((char*)"MagGradinet_NBM출력영상보기", img_out_NormalizeByMax, height, width);
}