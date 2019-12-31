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

int frame_limit{1};

Tracking::Tracking(const Calibration &c) : calib(c) {


}

Tracking::Tracking(const Tracking &other) : calib(other.calib) {
}


vector<vector<Point2f>> Tracking::operator()(const vector<Mat> &images, const vector<Point2f> &initMarkers) const {

	//defining vectors to be used for the function
	vector<uchar> status;
	vector<float> err;
	vector<vector<Point2f>> markers_pos;
	//initializing the markers positions vector as it has to be initialized in order to work with the function
	for (size_t i=0; i<images.size();i++)
		{
			markers_pos.push_back(initMarkers);
		}

//now we iterate over the frames (consecutive ones) and add the vectors needed for the function
	for(size_t i=0;i<(images.size()-frame_limit);i++){

		calcOpticalFlowPyrLK(images[i], images[i+1], markers_pos[i], markers_pos[i+1], status, err);

	}



	//showSequenceMarkers(images,markers_pos,"title");
	return markers_pos;
}
