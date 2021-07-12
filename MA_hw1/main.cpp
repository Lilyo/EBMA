#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <core/core.hpp>  
#include <iostream>
#include <stdlib.h> 
#include <stdio.h> 
#include <windows.h>

using namespace std;
using namespace cv;

int Checkopen(Mat img);
int CheckImgSize(Mat img1,Mat img2); 
void drawredbox(int x,int y,int length,const Scalar& color,int thickness,Mat &dst);
int count_file_num_in_a_folder( char* target_folder );// calculate the files number of specific folder
int uMAD(Mat f1,Mat f2);
Mat bilinear_interpolation(Mat matSrc,int custom_scale_x,int custom_scale_y);
Mat extract_half_pixel(Mat f1,int block_h,int block_w);
void MSE_PSNR(int rows,int cols,Mat image,Mat shift_image);
char folder_adderss[128]="BlowingBubbles_416x240_50";
char filename[128]="";

int main(){

	int check_current_img=0;
	int check_target_img=0;
	int number_of_file=count_file_num_in_a_folder(folder_adderss);
	printf("Num.:%d \n",number_of_file);

	double *Minus_Pearson_Correlation_arr= new double[number_of_file];
	Minus_Pearson_Correlation_arr[0]=0;

	char szDir1[128],szDir2[128];
	sprintf(szDir1, "%s/%s%d.bmp",folder_adderss,filename,10);
	sprintf(szDir2, "%s/%s%d.bmp",folder_adderss,filename,1);
	//printf("%s\n%s\n",szDir1,szDir2);

	Mat current_img = imread(szDir1,CV_LOAD_IMAGE_UNCHANGED); //current path
	Mat target_img = imread(szDir2,CV_LOAD_IMAGE_UNCHANGED);


	if(!Checkopen(current_img) && !Checkopen(target_img) ){
		if(CheckImgSize(current_img,target_img)!=-1){
			CvScalar color;
			color.val[0]= 255; //Blue
			color.val[1]= 255; //Green
			color.val[2]= 255; //Red

			Mat grey_current;
			Mat grey_target;
			Mat mirr1,mirr2,concat1,concat2,concat3;
			int img_h=current_img.rows;
			int img_w=current_img.cols;
			int blocksize = 16;
			int serach_region = 16;
			int imagesize = img_w*img_h;
			//imshow( "current_img", current_img);
			//imshow( "target_img", target_img);


			flip(target_img,mirr1,1); 
			hconcat(mirr1,target_img,concat1);
			hconcat(concat1,mirr1,concat1);
			flip(concat1,concat2,0);  
			vconcat(concat2,concat1,concat3);
			vconcat(concat3,concat2,concat3);
			Mat concat3cut=Mat(concat3,Range(img_h-serach_region,img_h+img_h+serach_region),Range(img_w-serach_region,img_w+img_w+serach_region));
			//imshow( "concat3", concat3);

			int concat_h=concat3.rows;
			int concat_w=concat3.cols;
			int concat_size = concat_h*concat_w;
			cout << "image =(" << img_h<<","<< img_w <<")"<< endl;
			cout << "imagesize = " << imagesize << endl;
			cout << "blocksize = " << blocksize << endl;
			cout << "concat image =(" << concat_h<<","<< concat_w <<")"<< endl;
			cout << "concat imagesize = " << concat_size << endl;

			Mat integer_result(img_h, img_w, CV_8UC3, Scalar(255,255,255));
			Mat half_result(img_h, img_w, CV_8UC3, Scalar(255,255,255));
			int MAD_min;
			int dx=0;
			int dy=0;
			int temp_mad=0;
			for (int y = 0; y < img_h; y += blocksize){
				for (int x = 0; x < img_w; x += blocksize){
					MAD_min=256*blocksize*blocksize;
					dx=0;
					dy=0;
					temp_mad=0;
					Mat block1,block2,MAD_block;

					/*Integer Scan*/
					for (int ry = -serach_region; ry <= serach_region; ry++){
						for (int rx = -serach_region; rx <= serach_region; rx++){

							Mat temp;
							Mat curr = current_img.clone();
							temp=Mat(curr,Range(y,y+blocksize),Range(x,x+blocksize));

							temp.copyTo(block1);
							drawredbox(x,y,blocksize+1,Scalar(0,0,255),1,curr);
							imshow("Scan curr", curr);
							curr.release();

							Mat mask = concat3.clone();
							temp=Mat(mask,Range(img_h+y+ry,img_h+y+ry+blocksize),Range(img_w+x+rx,img_w+x+rx+blocksize));

							temp.copyTo(block2);
							drawredbox(img_w+x-serach_region-1,img_h+y-serach_region-1,serach_region*3+1,Scalar(0,0,255),1,mask);
							drawredbox(img_w+x+rx-1,img_h+y+ry-1,blocksize+1,Scalar(0,0,255),1,mask);
							imshow("Scan target", mask);
							mask.release();

							temp_mad=uMAD(block1,block2);
							if(MAD_min>temp_mad){
								MAD_min=temp_mad;
								dx=rx;
								dy=ry;
								block2.copyTo(MAD_block);
							}
							Mat block;
							hconcat(block1,block2,block);
							hconcat(block,MAD_block,block);
							imshow("Integer Scan Compare", block);
							
							waitKey(1);
						}
					}
					int locx=img_w+x+dx;
					int locy=img_h+y+dy;
					printf("MAD_block(%d,%d) = %d\n",img_w+x+dx,img_h+y+dy,MAD_min);
					MAD_block.copyTo(integer_result(cv::Rect(x,y,MAD_block.cols, MAD_block.rows)));
					imshow("Integer result", integer_result);
					waitKey(1);

					/*Candidate Block*/
					Mat candidate_block;
					Mat mask = concat3.clone();
					Mat temp=Mat(mask,Range(img_h+y+dy-1,img_h+y+dy-1+blocksize+2),Range(img_w+x+dx-1,img_w+x+dx-1+blocksize+2));
					temp.copyTo(candidate_block);
					drawredbox(img_w+x-serach_region-1,img_h+y-serach_region-1,serach_region*3+1,Scalar(0,0,255),1,mask);
					drawredbox(img_w+x+dx-1,img_h+y+dy-1,blocksize+1,Scalar(0,255,0),1,mask);
					imshow("Scan target", mask);
					mask.release();

					//imshow("candidate_block", candidate_block);
					Mat bi_candidate_block= bilinear_interpolation(candidate_block,2,2);
					Mat bi_block1= bilinear_interpolation(block1,2,2);//using bilinear interpolation,but only calculate interlace pixel
					Mat MAD_half_block,ex_block;

					int init_pox=2;
					int init_poy=2;
					dx=0;
					dy=0;
					temp_mad=0;
					MAD_min=256*(blocksize*2)*(blocksize*2);

					/*Half Scan*/
					for (int ry = -1; ry <= 1; ry++){
						for (int rx = -1; rx <= 1; rx++){

							Mat temp;
							bi_candidate_block.copyTo(temp);
							drawredbox(init_pox-2,init_poy-2,4,Scalar(255,0,0),1,temp);
							drawredbox(init_pox+rx,init_poy+ry,blocksize*2,Scalar(255,0,0),1,temp);
							imshow("Scan bi_candidate_block", temp);

							Mat bi_block2=Mat(bi_candidate_block,Range(init_poy+ry,init_poy+ry+blocksize*2),Range(init_pox+rx,init_pox+rx+blocksize*2));
							temp_mad=uMAD(bi_block1,bi_block2);
							if(MAD_min>temp_mad){
								MAD_min=temp_mad;
								dx=rx;
								dy=ry;
								bi_block2.copyTo(MAD_half_block);
								ex_block = extract_half_pixel(MAD_half_block,blocksize,blocksize);
								//imshow("ex_block", ex_block);
							}

							Mat half_block;
							Mat org_candidate_block = extract_half_pixel(bi_block2,blocksize,blocksize);
							hconcat(block1,org_candidate_block,half_block);
							hconcat(half_block,ex_block,half_block);
							imshow("Half Scan Compare", half_block);

							waitKey(1);
						}
					}

					printf("half_block(%d,%d) = %d\n",locx+dx,locy+dy,MAD_min);
					ex_block.copyTo(half_result(cv::Rect(x,y,ex_block.cols, ex_block.rows)));
					imshow("Half result", half_result);
					waitKey(1);
				}
			}
			MSE_PSNR(img_h,img_w,current_img,integer_result);
			MSE_PSNR(img_h,img_w,current_img,half_result);
		}
	}
	
	waitKey(0);

	system("pause");
	return 0;
}

int Checkopen(Mat img){
	if(img.empty()){
		cout<<"Error load img！"<<endl;
		system("pause");
		return -1;
	}
	return 0;
}
int CheckImgSize(Mat img1,Mat img2){
	int w1=img1.cols;
	int h1=img1.rows;
	int w2=img2.cols;
	int h2=img2.rows;

	if(w1==w2 && h1==h2){
		return 0;
	}else{
		printf("Image size error!");
	}
	return -1;
}
int count_file_num_in_a_folder( char* target_folder ){
    int count=-1;                //檔案的counter
    char szDir[256];           //要讀取的資料夾的位址。 
    WIN32_FIND_DATA FileData;    //指著目前讀取到的File的指標。
    HANDLE hList;                //指著要讀取的資料夾的指標。
    sprintf(szDir, "%s/*",target_folder );
    if ( (hList = FindFirstFile(szDir, &FileData))==INVALID_HANDLE_VALUE )
        cout<<"No directories be found."<<endl<<endl;
    else {
        while (1) {
            if (!FindNextFile(hList, &FileData)) {
                if (GetLastError() == ERROR_NO_MORE_FILES)
                    break;
            }
            count++;
        }
    }
    FindClose(hList);
    return count;
}
void drawredbox(int x,int y,int length,const Scalar& color,int thickness,Mat &dst){
	line(dst,Point(x,y),Point(x+length,y),color,thickness);
	line(dst,Point(x,y+length),Point(x+length,y+length),color,thickness);
	line(dst,Point(x,y),Point(x,y+length),color,thickness);
	line(dst,Point(x+length,y),Point(x+length,y+length),color,thickness);
}
int uMAD(Mat f1,Mat f2){
	int mad=0;
	Mat gray_f1;
	Mat gray_f2;
	cvtColor(f1, gray_f1, CV_BGR2GRAY);
	cvtColor(f2, gray_f2, CV_BGR2GRAY);
	for(int i=0;i<f1.rows;i++){
		for(int j=0;j<f1.cols;j++){
			mad=mad+abs((int)(gray_f1.at<uchar>(i, j))-(int)(gray_f2.at<uchar>(i, j)));
			//printf("%f %f\n",(float)(gray_f1.at<uchar>(i, j)),(float)(gray_f2.at<uchar>(i, j)));
		}
	}
	return mad;
}
Mat bilinear_interpolation(Mat matSrc,int custom_scale_x,int custom_scale_y){
	Mat matDst1;  
	matDst1 = Mat(Size(matSrc.cols*custom_scale_x,matSrc.rows*custom_scale_y), matSrc.type(), Scalar::all(0));  
	double scale_x = (double)matSrc.cols / matDst1.cols;  
	double scale_y = (double)matSrc.rows / matDst1.rows;  
	uchar* dataDst = matDst1.data;  

	int stepDst = matDst1.step;  
	uchar* dataSrc = matSrc.data;  
	int stepSrc = matSrc.step;  
	int iWidthSrc = matSrc.cols;  
	int iHiehgtSrc = matSrc.rows;  
  
	for (int j = 0; j < matDst1.rows; ++j)  
	{  
		float fy = (float)((j + 0.5) * scale_y - 0.5);  
		int sy = cvFloor(fy);  
		fy -= sy;  
		sy = min(sy, iHiehgtSrc - 2);  
		sy = max(0, sy);  
  
		short cbufy[2];  
		cbufy[0] = cv::saturate_cast<short>((1.f - fy) * 2048);  
		cbufy[1] = 2048 - cbufy[0];  
  
		for (int i = 0; i < matDst1.cols; ++i)  
		{  
			float fx = (float)((i + 0.5) * scale_x - 0.5);  
			int sx = cvFloor(fx);  
			fx -= sx;  
  
			if (sx < 0) {  
				fx = 0, sx = 0;  
			}  
			if (sx >= iWidthSrc - 1) {  
				fx = 0, sx = iWidthSrc - 2;  
			}  
  
			short cbufx[2];  
			cbufx[0] = cv::saturate_cast<short>((1.f - fx) * 2048);  
			cbufx[1] = 2048 - cbufx[0];  
  
			for (int k = 0; k < matSrc.channels(); ++k)  
			{  
				*(dataDst+ j*stepDst + 3*i + k) = (*(dataSrc + sy*stepSrc + 3*sx + k) * cbufx[0] * cbufy[0] +   
					*(dataSrc + (sy+1)*stepSrc + 3*sx + k) * cbufx[0] * cbufy[1] +   
					*(dataSrc + sy*stepSrc + 3*(sx+1) + k) * cbufx[1] * cbufy[0] +   
					*(dataSrc + (sy+1)*stepSrc + 3*(sx+1) + k) * cbufx[1] * cbufy[1]) >> 22;  
			}  
		}  
	}  
	return matDst1;
}
Mat extract_half_pixel(Mat f1,int block_h,int block_w){
	Mat extract_f1(block_h, block_w, CV_8UC3, Scalar(255,255,255));
	for(int i=0;i<f1.rows/2;i++){
		for(int j=0;j<f1.cols/2;j++){
			extract_f1.at<Vec3b>(j,i)[0]= f1.at<Vec3b>(j*2,i*2)[0];
			extract_f1.at<Vec3b>(j,i)[1]= f1.at<Vec3b>(j*2,i*2)[1];
			extract_f1.at<Vec3b>(j,i)[2]= f1.at<Vec3b>(j*2,i*2)[2];
		}
	}
	return extract_f1;
}
void MSE_PSNR(int rows,int cols,Mat image,Mat shift_image){//(高度,寬度,原圖,右移圖)

	double difference=0;//原圖與右移圖位置相減之pixel值
	double sum=0;
	double mse;
	double psnr;
	//計算座標(i,j)之值
	for(int i=0;i<rows;i++){
		for(int j=0;j<cols;j++){
			difference=(double)image.at<unsigned char>(i, j)-(double)shift_image.at<unsigned char>(i, j);//原圖與右移圖位置相減之pixel值
			sum=sum+difference*difference;//將相減之pixel值平方後加總起來
		}
	}
	mse=sum/(rows*cols);//加總起來之值除以高度*寬度，即得到MSE值
	printf("sum=%f\n""MSE=%f\n",sum,mse);//印出加總值和MSE值
	psnr=10*log10(255*255/mse);//PSNR運算
	printf("PSNR=%f\n",psnr);

 //以下為int轉double之寫法，但在mse=total_sum/(rows*cols)之total_sum需是double型態，得到MSE之值才會正確。
 //即int/int=int表示若有小數，則小數位自動刪除
 //但double/int=double表示若有小數，則小數位保留
 
/*int difference=0;//原圖與右移圖位置相減之pixel值
 int square=0;
 int sum=0;
 double total_sum;
 double mse;
 double psnr;
 for(i=0;i<rows;i++)
 {
 for(j=0;j<cols;j++)
 {
 difference=(int)image.at<unsigned char>(i, j)-(int)shift_image.at<unsigned char>(i, j);//原圖與右移圖位置相減之pixel值
 square=difference*difference;
 sum=sum+square;//將相減之pixel值平方後加總起來
 }
 }
 total_sum=(double)sum;
 mse=total_sum/(rows*cols);//加總起來之值除以高度*寬度，即得到MSE值
 printf("sum=%f\n""MSE=%f\n",total_sum,mse);//印出加總值和MSE值
 psnr=10*log10(255*255/mse);//PSNR運算
 printf("PSNR=%f\n",psnr);
 */
}