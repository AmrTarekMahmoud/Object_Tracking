#include <opencv2/opencv.hpp>

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


int main(int argc, char **argv) {
	try {
		// get calibration folder, sequence folder and output file from command line
		string calibFolder, sequenceFolder, outputFile;
		if (argc == 4) {
			calibFolder = string(argv[1]) + "/";
			sequenceFolder = string(argv[2]) + "/";
			outputFile = string(argv[3]);
		} else {
			cerr << "Please specify folder with calibration data, folder with sequence and output file" << endl;
			return EXIT_FAILURE;
		}

		// load calibration data
		logMessage("load calibration data from " + calibFolder);
		Calibration calib(calibFolder);
		logMessage("loaded calibration data");

		// load sequence
		logMessage("load sequence from " + sequenceFolder);
		//string SeqFolder="D:/TUHH/3D_Computer_Vision/Project_WS19/Project/20/Sequence4/";
		Sequence sequence(sequenceFolder,calib);
		logMessage("finished loading sequence with " + to_string(sequence.getNumberOfFrames()) + " frames");

		// track the markers in the sequence
		logMessage("start tracking of markers");
		Tracking tracking(calib);
		vector<vector<Point2f>>Camera1_line_track=tracking(sequence.operator[](0),sequence.getMarkers(0));
		vector<vector<Point2f>>Camera2_line_track=tracking(sequence.operator[](1),sequence.getMarkers(1));

		logMessage("finished tracking of markers");

		// triangulate the marker positions
		logMessage("start triangulation");
		Triangulation triangulation(calib);

		vector<vector<Point3f>> Markers_triangulation;
		Markers_triangulation=triangulation.operator ()(Camera1_line_track,Camera2_line_track);

		//showTriangulation(Markers_triangulation,"show Triangulation");
		logMessage("finished triangulation");
		// calculate the motion of the markers
		logMessage("calculate motion of markers");

		vector<vector<Point3f>>Markers_motion=triangulation.calculateMotion(Markers_triangulation);
		showTriangulation(Markers_motion,"show Triangulation");
		logMessage("finished calculation of motion of markers");
		// write the result to the output file
		logMessage("write results to " + outputFile);

		writeResult(outputFile,Markers_motion);
		logMessage("finished writing results");



		// and exit program with code for success
		return EXIT_SUCCESS;
	} catch (const string &err) {
		// print error message and exit program with code for failure
		cerr << err << endl;
		return EXIT_FAILURE;
	}
}
