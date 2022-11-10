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
	*no_label = connectedComponents(bw, labelImage, 8); // 0���� ���Ե� ������

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

	ImageShow((char*)"�Է¿��󺸱�", img, height, width);
	ImageShow((char*)"��¿��󺸱�", img_out, height, width);
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

	ImageShow((char*)"�Է¿��󺸱�", img, height, width);
	ImageShow((char*)"�ȼ���+50��¿���1����", img_out1, height, width);
	ImageShow((char*)"�ȼ���-50��¿���2����", img_out2, height, width);

	ImageClpping(img_out1, height, width, img_out1);
	ImageClpping(img_out2, height, width, img_out2);

	
	ImageShow((char*)"CLIPPING��¿���1����", img_out1, height, width);
	ImageShow((char*)"CLIPPING��¿���2����", img_out2, height, width);
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

	ImageShow((char*)"�Է¿��󺸱�", img1, height, width);
	ImageShow((char*)"��¿��󺸱�", img_out, height, width);
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

	ImageShow((char*)"�Է¿��󺸱�", img, height, width);
	ImageShow((char*)"��¿��󺸱�", img_out, height, width);
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
void Histogram_MAIN()
{
	int height, width;

	int** img = (int**)ReadImage((char*)"barbara.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	int Hist[256] = { 0 };
	Histogram(img, height, width, Hist);

	int C_Hist[256] = { 0 };
	C_Histogram(img, height, width, C_Hist);

	ImageShow((char*)"�Է¿��󺸱�", img, height, width);
	DrawHistogram((char*)"������׷�", Hist);
	DrawHistogram((char*)"����������׷�", C_Hist);
}