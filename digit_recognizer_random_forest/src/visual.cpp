#include <bits/stdc++.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/video.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/superres.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
/*
int main(int argc, char **argv){
	//CV_8UC3 means that matrix accepts unsigned char type, scalar is init each pixel as, and first two param are for matrix size
	Mat img(28,28, CV_8UC3, Scalar(255,255,255));
	//print matrix
    cout << "img = " << endl << " " << img << endl << endl;
    vector<int> compression_params;
    //we are going to compress png
    compression_params.push_back(IMWRITE_PNG_COMPRESSION);
    //compression value, from 0 to 9
    compression_params.push_back(3);
    try{
		imwrite("test1.png", img, compression_params);
    } catch (runtime_error &ex){				//ex is a reference
    	cout<<ex.what()<<endl;
    	return 1;						//something went wrong
    }
	return 0;
}*/

int main(int argc, char **argv){
	//collect and visualize data
	freopen("../test/test.csv","r",stdin);
	std::string ignore_line;
	cin>>ignore_line;
	int img_id = 0;
	while(true){
		char *in = (char*)malloc(sizeof(char)*3135);
		try{
			scanf("%s",in);
		} catch (...){
			//end of collecting images
			return 1;
		}
		string tokenize_str(in);
		vector<string> tokens;
		stringstream ss(in);
		string string_builder = "";
		char curr;
		while(ss >> curr){
			string_builder += curr;
			if(ss.peek() == ','){
				//deliminator
				tokens.push_back(string_builder);
				string_builder = "";
				ss.ignore();
			}
		}
		if(string_builder.compare("") != 0)		//get last token
			tokens.push_back(string_builder);
		Mat img(14, 14, CV_8UC3, Scalar(0,0,0));
		for(int i = 0; i < 28; i+=2){
			for(int j = 0; j < 28; j+=2){
				//read from one over because the first token is class id
				int grayscale_val = stoi(tokens[28*i+j]);
				grayscale_val += stoi(tokens[28*(i+1)+j]);
				grayscale_val += stoi(tokens[28*i+(j+1)]);
				grayscale_val += stoi(tokens[28*(i+1)+(j+1)]);
				Vec3b color(grayscale_val/4,grayscale_val/4,grayscale_val/4);
				img.at<Vec3b>(Point(j/2,i/2)) = color;
			}
		}
		vector <int> compression_params;
		compression_params.push_back(IMWRITE_PNG_COMPRESSION);
		compression_params.push_back(3);
		try{
			imwrite("../MNIST-TEST/digit_test"+to_string(img_id)+".png", img, compression_params);
	    } catch (runtime_error &ex){			//ex is a reference
	    	cout<<ex.what()<<endl;
	    	return 1;							//something went wrong
	    }
		free(in);
		img_id++;
	}
}