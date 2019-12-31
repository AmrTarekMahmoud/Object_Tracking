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

Triangulation::Triangulation(const Calibration &c) : calib(c) {
}

Triangulation::Triangulation(const Triangulation &other) : calib(other.calib) {
}

vector<Point3f> Triangulation::operator()(const vector<Point2f> &markers1, const vector<Point2f> &markers2) const {

	//setting argument path to create an object of class Calibration
	string calibFold="D:/TUHH/3D_Computer_Vision/Project_WS19/Project/20/Calibration_data/";
	//creating an object from Calibration
	Calibration calibration(calibFold);
	//find P matrix for camera 1
	Mat K_camera1=calibration.getCamera1();
	//find P matrix for camera 2
	Mat K_camera2=calibration.getCamera2();
	//since camera 1 is at origin, the using the rotation around x axis and substituting alpha with 0 we get identity matrix for rotation and 0 vector for translation
	Mat Rotation_origin = (Mat_<float>(3,4)<<1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0);
	//since matrix getTransCamera1World is 3x2, the multiplication of vector to world coordinates will loose a dimension, so we will add another row at the bottom to be
	//4x2 instead of 3x2
	Mat addiotinal =(Mat_<float>(1,4)<<0,0,0,1);
	//get the fundamental matrix
	Mat fundamental_mat=calibration.getFundamentalMat();
	//get the transformation matrix between camera 1 and camera 2
	Mat C1_C2_Transformation=calibration.getTransCamera1Camera2();
	//calculate P1
	Mat P1=K_camera1 * Rotation_origin;
	//cerr<<P1<<endl;
	//calculate P2
	Mat P2=K_camera2*C1_C2_Transformation;
	//cerr<<P2<<endl;
	//creating new vector to put the optimized ideal point coordinates from the observed coordinates
	vector<Point2f> optimized_points_1;
	vector<Point2f> optimized_points_2;
	//get the distortion matrix for the first camera
	Mat D_camera1=calibration.getDistortion1();
	//get the distortion matrix for the second camera
	Mat D_camera2=calibration.getDistortion2();
	//The next function implements the Optimal Triangulation Method. For each given point correspondence points1[i] <-> points2[i],
	//and a fundamental matrix F, it computes the corrected correspondences newPoints1[i] <-> newPoints2[i] that minimize the
	//geometric error d(points1[i], newPoints1[i])^2 + d(points2[i],newPoints2[i])^2 (where d(a,b) is the geometric distance between points a and b )
	//subject to the epipolar constraint newPoints2^T * F * newPoints1 = 0
	correctMatches(fundamental_mat,markers1,markers2,optimized_points_1,optimized_points_2);
	//Computes the ideal point coordinates from the observed point coordinates
	//applying the computation for camera 1
	//undistortPoints(optimized_points_1,optimized_points_1,K_camera1,D_camera1,noArray(),P1);
	//applying the computation for camera 2
	//cerr<<optimized_points_1;
	//undistortPoints(optimized_points_2,optimized_points_2,K_camera2,D_camera2,noArray(),P2);
	//create a matrix to hold the values for coordinates points
	Mat homogeneous_coordinates;
	//now we compute the triangulated points
	triangulatePoints(P1,P2,optimized_points_1,optimized_points_2,homogeneous_coordinates);
	//cerr<<homogeneous_coordinates<<endl;
	//next, we transform the points from homogeneous coordinates to world coordinates
	//first we get a transformation for camera 2 to the world coordinates
	//now we create a matrix to hold the values in world coordinates
	Mat Trans_camera1_toworld=calibration.getTransCamera1World();
	//now adding the 4th row that was mentioned before
	Trans_camera1_toworld.push_back(addiotinal);
	//set the world coordinates
	Mat World_coordinates=Trans_camera1_toworld*homogeneous_coordinates;
	//Transformation matrix
	Mat Transformed_world_coordinates;
	//cerr<<World_coordinates;
	convertPointsFromHomogeneous(World_coordinates.t(),Transformed_world_coordinates);
	//cerr<<Transformed_world_coordinates;
	//create an output vector for the tiangulation
	vector<Point3f> output_Triangulation;
	output_Triangulation.push_back(Point3f(Transformed_world_coordinates.at<Point3f>(0, 0)));
	output_Triangulation.push_back(Point3f(Transformed_world_coordinates.at<Point3f>(1, 0)));
	//cerr<<output;
	return output_Triangulation;

}

vector<vector<Point3f> > Triangulation::operator()(const vector<vector<Point2f> > &markers1, const vector<vector<Point2f> > &markers2) const {
	// do nothing if there is no data
	if (markers1.empty()) {
		return vector<vector<Point3f>>();
	}

	// check for same number of frames
	if (markers1.size() != markers2.size()) {
		throw "different number of frames";
	}

	// create result vector
	vector<vector<Point3f>> result(markers1.size());

	// trinagulate each frame for itself and store result
	for (unsigned int i = 0; i < markers1.size(); ++i) {
		result[i] = (*this)(markers1[i], markers2[i]);
	}

	// and return result
	return result;
}

vector<vector<Point3f>> Triangulation::calculateMotion(const vector<vector<Point3f> > &data) {
	//create an int to use in for loop to avoid magic numbers
	int vector_limit=2;
	//create a vector to output data to and initialize it with values to avoid termination for no memory access
	vector<vector<Point3f>> output_motion_vector=data;
	for(size_t i=0;i<data.size();i++){
		for(size_t j=0;j<vector_limit;j++){
			output_motion_vector[i][j]=data[i][j]-data[0][j];
		}
	}

	return output_motion_vector;


}
