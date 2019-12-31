#include "Sequence.hpp"
#include "tools.hpp"
#include "Constants.hpp"
#include "Calibration.hpp"
#include "Sequence.hpp"
#include "Tracking.hpp"
#include "Triangulation.hpp"
#include <string>
#include <iostream>



using namespace CVLab;
using namespace cv;
using namespace std;



Sequence::Sequence(const string &folder, const Calibration &c) : calib(c) {
	// read both videos

	readVideo(folder + Constants::sequence1File, images[0], calib.getCamera1(), calib.getDistortion1());
	readVideo(folder + Constants::sequence2File, images[1], calib.getCamera2(), calib.getDistortion2());

	// check if both videos have the same amount of frames
	if (images[0].size() != images[1].size()) {
		throw "both videos have different number of frames";
	}

	// load marker positions for both videos
	readMarkers(folder + Constants::markers1File, markers[0], images[0][0]);
	readMarkers(folder + Constants::markers2File, markers[1], images[1][0]);

	// check if both videos have the same amount of markers
	if (markers[0].size() != markers[1].size()) {
		throw "both videos have different number of markers";
	}

	// sort the markers so that they have the same ordering for both videos
	sortMarkers();
}

Sequence::Sequence(const Sequence &other) : calib(other.calib) {
	// loop over all cameras
	for (unsigned int camera = 0; camera < 2; ++camera) {
		// copy images
		images[camera].resize(other.images[camera].size());
		for (unsigned int frame = 0; frame < images[camera].size(); ++frame) {
			images[camera][frame] = other.images[camera][frame].clone();
		}

		// copy marker positions
		markers[camera] = other.markers[camera];
	}
}

size_t Sequence::getNumberOfFrames() const {
	return images[0].size();
}

const vector<Mat> & Sequence::operator[](unsigned int camera) const {
	// check camera index
	if (camera > 1) {
		throw "there are only two cameras";
	}

	// return sequence of images
	return images[camera];
}

vector<Point2f> Sequence::getMarkers(unsigned int camera) const {
	// check camera index
	if (camera > 1) {
		throw "there are only two cameras";
	}

	// return marker positions
	return markers[camera];
}

void Sequence::readVideo(const string &file, vector<Mat> &data, const Mat &K, const Mat &distortion) {
	// open video file
	VideoCapture vid(file);
	if (!vid.isOpened()) {
		throw "could not open video file " + file;
	}

	// get number of frames from the video file
	const unsigned int numberOfFrames = static_cast<unsigned int>(vid.get(CAP_PROP_FRAME_COUNT));
	
	// resize vector to number of frames
	data.clear();
	data.resize(numberOfFrames);

	// load images from video
	for (unsigned int i = 0; i < numberOfFrames; ++i) {
		Mat img, gray, undistorted;

		// load next frame
		vid >> img;

		// convert frame to grayscale
		cvtColor(img, gray, COLOR_BGR2GRAY);

		// undistort the image
		undistort(gray, undistorted, K, distortion);

		// save the undistorted image in the vector
		undistorted.copyTo(data[i]);
	}
}

void Sequence::readMarkers(const string &file, vector<Point2f> &data, const Mat &firstImage) {
	// read raw data from file
	Mat markerData = readMatrix(file);

	// check matrix dimension for validity
	checkMatrixDimensions(markerData, -1, 2, "marker positions");

	// resize vector to take marker positions
	data.clear();
	data.resize(markerData.rows);

	// save marker positions in the vector
	for (int i = 0; i < markerData.rows; ++i) {
		data[i].x = markerData.at<float>(i, 0);
		data[i].y = markerData.at<float>(i, 1);
	}

	// and refine the marker positions
	cornerSubPix(firstImage, data, Constants::markerRefinementWindowSize, Constants::markerRefinementZeroZone, Constants::markerRefinementCriteria);
}

void Sequence::sortMarkers() {
	//read data from calibration folder and sequence 0 to check the images with the markers coordinates
	string calibFold="D:/TUHH/3D_Computer_Vision/Project_WS19/Project/20/Calibration_data/";
	//string seqFold ="D:/TUHH/3D_Computer_Vision/Project_WS19/Project/20/Sequence4/";
	//creating new object from class Calibration
	Calibration calibration(calibFold);
	//creating new vector to put the second marker coordinates in
	vector <float> camera2_points;

	cerr << "in sort sequence" << endl;
	// we first create a matrix to put the values of the markers coordinates in
	//float to_compare_C1[2][2]={0};
	//float to_compare_C2[2][2]={0};
	// now we place values inside the matrix created
	// this is for the first camera with x and y coordinates
	//to_compare_C1[0][0]=markers[0][0].x; //711
	//to_compare_C1[1][0]=markers[0][0].y; //92
	//to_compare_C1[0][1]=markers[0][1].x; //729
	//to_compare_C1[1][1]=markers[0][1].y; //210

	// this is for the second camera with x and y coordinates
	//to_compare_C2[0][0]=markers[1][0].x; //798
	//to_compare_C2[1][0]=markers[1][0].y; //312
	//to_compare_C2[0][1]=markers[1][1].x; //922
	//to_compare_C2[1][1]=markers[1][1].y; //227

	//points from second camera but transposed
	Mat camera2_points_transpose=(Mat_<float>(1,3)<<markers[1][0].x,markers[1][0].y,1,CV_32FC1);
	//points from first camera
	Mat camera1=(Mat_<float>(3,1)<<markers[0][0].x,markers[0][0].y ,1,CV_32FC1);
	//fundemental matrix in a new variable
	Mat My_fundemental_matrix=calibration.getFundamentalMat();
	//x'.transpose * fundemental matrix * x = 0
	Mat check_result= camera2_points_transpose * My_fundemental_matrix * camera1;
	//reading the coordinates to take values from the origional file
	vector<Point2f> input_Markers;
	input_Markers= this->getMarkers(0);
	input_Markers= this->getMarkers(1);

	Mat output_Marker_coordinates;
	//check if the matrix multiplication is less than 0.45
	if(abs(check_result.at<float>(0,0))>0.45){
				//if so, then replace the first row with the second row and write in the new matrix
						markers[1][0].x=input_Markers[1].x;
						markers[1][0].y=input_Markers[1].y;
						// we simulate a swaping function here
						markers[1][1].x=input_Markers[0].x;
						markers[1][1].y=input_Markers[0].y;

		}
	//show the 2 images with the new matrix values
	//showImageMarkers(images[0][0],markers[0],"first camera view");
	//showImageMarkers(images[1][0],markers[1],"second camera view");








}
