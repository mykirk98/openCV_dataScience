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

int EX0916_2(int** img, int height, int width)					//int height, width XXXXX int height, int width 함수 매개변수 선언은 각각 타입 이름 순으로 적어줘야함
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

void addValue2Image(int add_number,		//더할 값
	int** img_input,		//입력 이미지
	int height,		//영상 높이
	int width,		//영상 폭
	int** img_output		//출력 이미지
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
			/////////////////////if문을 이요한 클리핑 2/////////////
			//img_output[y][x] = GetMax(GetMin(img_input[y][x], 255), 0);
			
			/////////////////////if문을 이용한 클리핑 1/////////////
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

	ImageShow((char*)"출력1영상보기", img, height, width);
	ImageShow((char*)"출력2영상보기", img, height, width);

}

void EX0923_2()
{
	int height, width;
	int** img = (int**)ReadImage((char*)"barbara.png", &height, &width);

	int** img_out1 = (int**)IntAlloc2(height, width);
	int** img_out2 = (int**)IntAlloc2(height, width);


	addValue2Image(50, img, height, width, img_out1);
	addValue2Image(-50, img, height, width, img_out2);

	ImageShow((char*)"출력1 밝기 + 50 후 영상보기", img_out1, height, width);
	ImageShow((char*)"출력2 밝기 - 50 후 영상보기", img_out2, height, width);
	
	Image_Clipping(img_out1, height, width, img_out1);
	Image_Clipping(img_out2, height, width, img_out2);

	ImageShow((char*)"입력영상보기", img, height, width);
	ImageShow((char*)"출력1 밝기 + 50 후 클리핑 후 영상보기", img_out1, height, width);
	ImageShow((char*)"출력2 밝기 - 50 후 클리핑 후 영상보기", img_out2, height, width);
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

void main()
{
	EX0923_4((char*)"지능형", (char*)"영상", (char*)"처리");
}