/////////////////////////////////////////////////////////////////////
// File:        sceneTextOCR.cpp
// Description: scene text detection and recognition
// Author:      Yan Ge
// Created:     Aug 16 2017
/////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
/////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////
 /* (C) Copyright 2006, Google Inc.
 ** Licensed under the Apache License, Version 2.0 (the "License");
 ** you may not use this file except in compliance with the License.
 ** You may obtain a copy of the License at
 ** http://www.apache.org/licenses/LICENSE-2.0
 ** Unless required by applicable law or agreed to in writing, software
 ** distributed under the License is distributed on an "AS IS" BASIS,
 ** WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 ** See the License for the specific language governing permissions and
 ** limitations under the License.
 */
///////////////////////////////////////////////////////////////////////////////
// CPP
#include <iostream>
#include <utility>
#include <list>
#include <unordered_map>
#include <queue>
#include <iterator>
#include <algorithm>
#include <ctime>
#include <sys/time.h>

// OpenCV2
#include <opencv2/opencv.hpp>
#include "opencv2/core/utility.hpp"
#include "opencv2/text.hpp"

// Tesseract & Leptonica
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

//@@@ IMPORTANT @@@ Please replace CAMERA_DEVICE_ID with your cam ID
#define CAMERA_DEVICE_ID 0

using namespace std;
using namespace cv;
using namespace cv::text;

/************************************************************************/
// This function is from OPENCV source 
void er_draw(vector<Mat> &channels, vector<vector<ERStat> > &regions, vector<Vec2i> group, Mat& segmentation)
{
  for (int r=0; r<(int)group.size(); r++)
    {
      ERStat er = regions[group[r][0]][group[r][1]];
      if (er.parent != NULL) // deprecate the root region
        {
	  int newMaskVal = 255;
	  int flags = 4 + (newMaskVal << 8) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;

	  floodFill(channels[group[r][0]], segmentation,
		    Point(er.pixel%channels[group[r][0]].cols,
			  er.pixel/channels[group[r][0]].cols),
		    Scalar(255), 0, Scalar(er.level), Scalar(0), flags);
        }
    }
}

/************************************************************************/
// This function is from OPENCV source
void groups_draw(Mat &src, vector<Rect> &groups)
{
  for (int i=(int)groups.size()-1; i>=0; i--)
    {
      if (src.type() == CV_8UC3)
	rectangle(src,groups.at(i).tl(),groups.at(i).br(),Scalar( 0, 255, 255 ), 3, 8 );
      else
	rectangle(src,groups.at(i).tl(),groups.at(i).br(),Scalar( 255 ), 3, 8 );
    }
}


/************************************************************************/
// This function is from OPENCV source
void er_show(vector<Mat> &channels, vector<vector<ERStat> > &regions)
{
  for (int c=0; c<(int)channels.size(); c++)
    {
      Mat dst = Mat::zeros(channels[0].rows+2,channels[0].cols+2,CV_8UC1);

      for (int r=0; r<(int)regions[c].size(); r++)
        {
	  ERStat er = regions[c][r];
	  if (er.parent != NULL) // deprecate the root region
            {
	      int newMaskVal = 255;
	      int flags = 4 + (newMaskVal << 8) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;
	      floodFill(channels[c],dst,Point(er.pixel%channels[c].cols,er.pixel/channels[c].cols),
			Scalar(255),0,Scalar(er.level),Scalar(0),flags);
            }
        }
      char buff[10]; char *buff_ptr = buff;
      sprintf(buff, "channel %d", c);
      imshow(buff_ptr, dst);
    }
  waitKey(-1);
}


inline bool compare_rect (const Rect& first, const Rect& second)
{
  return first.x * first.y < second.x * second.y;
}



/************************************************************************/
// Main
int main(int args, char **argv)
{
  clock_t start, stop, start_all, stop_all;
  
  // Init cam
  VideoCapture cam(CAMERA_DEVICE_ID);
  
  // Check cam
  if (!cam.isOpened())
    {
      cerr << "camera failed!";
      exit(1);
    }
  cout << ">>>>>>>>> \tCamera initialization is done! \n";
  
  Ptr<ERFilter> er_filter1 = createERFilterNM1(loadClassifierNM1("trained_classifierNM1.xml"), 
					       8, //  threshold step when extracting the component tree
					       0.00025f, // min area (% of image size) allowed for ER
					       0.13f, // the max area allowed for ER
					       0.4f, // default = 0.4; min probability P(er | character) allowed for ER
					       true, // nonMaxSuppression
					       0.1f); // minProbabilityDiff
  
  Ptr<ERFilter> er_filter2 = createERFilterNM2(loadClassifierNM2("trained_classifierNM2.xml"),
					       0.5); // the min probability P(er | character) allowed for ER

  if (er_filter1 != nullptr && er_filter2 != nullptr)
    cout << ">>>>>>>>> \tText detecting filer initialization is done! \n";

  Mat src;
  namedWindow("capture", 1);

  // if key 'ESC' pressed, break loop
  while (1)
    {
      cam >> src;
      imshow("webcam", src);
      
      char key = (char)waitKey(30);
      
      /* ESC == 27 */
      if ( key == 27 )
	{
	  // close camera
	  cam.release();
	  cout << ">>>>>>>>> \t\"ESC\" entered, releasing all resource! \n";
	  exit(0);
	}
      else if ( key == ' ' ) // if entered space key  
	{
	  //////////////////////////////////////////////////////
	  /*            Step 1: capture camera frame          */
	  //////////////////////////////////////////////////////

	  start = clock();
	  start_all = start;

	  cout << ">>>>>>>>> \tCapturing frame \t";
	  Mat frame;
	  // frame = imread(argv[1]);
	  // cvtColor(src, frame, CV_BGR2GRAY);
	  src.copyTo(frame);
	  imshow("capture", frame);
	   
	  stop = clock();
	  cout << "time: " << (double) (stop - start) / CLOCKS_PER_SEC * 1000 << "ms\n";
	
	  //////////////////////////////////////////////////////
	  //             Step 2: text detection               
	  //////////////////////////////////////////////////////
	  
	  start = clock();

	  cout << ">>>>>>>>> \tDetecting Text regions\t";
	  vector<Mat> channels;
	  computeNMChannels(frame, channels);
   
	  // Determine channel size
	  size_t n_channels = channels.size();	  
	  vector< vector<ERStat> > regions(n_channels);
	  	  
	  for (int i = 0; i < n_channels; i++)
	    {
	      er_filter1->run(channels[i], regions[i]);
	      er_filter2->run(channels[i], regions[i]);
	    }

	  stop = clock();
	  cout << "time: " << (double) (stop - start) / CLOCKS_PER_SEC * 1000 << "ms\n";
       
	  /////////////////////////////////////////////////////
	  //             Step 3: group text region            /
	  /////////////////////////////////////////////////////

	  start = clock();
	  
	  cout << ">>>>>>>> \tGrouping text regions \t";
	  vector< vector<Vec2i> > region_groups;
	  vector<Rect> boxes;
	  // Use in horizontal direction only!
	  erGrouping(frame, channels, regions, region_groups, boxes, ERGROUPING_ORIENTATION_HORIZ);	  
	  // Any direction.
	  //  erGrouping(frame, channels, regions, region_groups, boxes, ERGROUPING_ORIENTATION_ANY, "./trained_classifier_erGrouping.xml", 0.5);

	  // groups_draw(frame, boxes);
	  // imshow("grouping",frame);

	  stop = clock();
	  cout << "time: " << (double) (stop - start) / CLOCKS_PER_SEC * 1000 << "ms\n";
       

	  /////////////////////////////////////////////////////
	  //             Step 4: sort text region            /
	  /////////////////////////////////////////////////////
	  start = clock();
	  
	  cout << ">>>>>>>>> \tSorting text regions \t";
	  list<Rect> boxes_list;
	  for (auto i = 0; i < boxes.size(); ++i)
	    boxes_list.push_back( boxes[i] );
	  
	  boxes_list.sort(compare_rect);
	
	  stop = clock();
	  cout << "time: " << (double) (stop - start) / CLOCKS_PER_SEC * 1000 << "ms\n";
       	  
	  /////////////////////////////////////////////////////
	  //             Step 5: regroup  text region         /
	  /////////////////////////////////////////////////////
	  
	  start = clock();
	  
	  cout << ">>>>>>>>> \tMerge text regions \t";

	  // now group boxes_list
	  auto it_prev = boxes_list.begin();

	  for ( auto it = std::next(boxes_list.begin(), 1); it != boxes_list.end(); it++ )
	    { 
	      // if overlap with next box, combine
	      if ( (*it_prev & *it).area() )
		{
		  //  cout << "Warnning: prev rect " << *it_prev << " and \ncurrent rect " << *it << " overlap!\n";	  
		  // now combine them 
		  // get left-up most point, i.e. get smalest x, y
		  int left_up_x = std::min( it_prev->x, it->x );
		  int left_up_y = std::min( it_prev->y, it->y );

		  // get right-down most point.
		  int right_down_x = std::max( it_prev->x + it_prev->width,
					       it->x + it->width);
		  int right_down_y = std::max( it_prev->y + it_prev->height,
					       it->y + it->height);
		  it->x = left_up_x;
		  it->y = left_up_y;
		  it->width = right_down_x - left_up_x;
		  it->height = right_down_y - left_up_y;

		  boxes_list.erase(it_prev);
		}
	      it_prev = it;
	    }
	  stop = clock();
	  cout << "time: " << (double) (stop - start) / CLOCKS_PER_SEC * 1000 << "ms\n";
       	  


	  /////////////////////////////////////////////////////
	  //             Step 5: segment  text region         /
	  /////////////////////////////////////////////////////
	  
	  start = clock();
	  
	  cout << ">>>>>>>>> \tSegment text regions \t";

	  vector<Rect> merged_boxes;
	  vector<Mat> textBoxes;
	  for ( auto it : boxes_list )
	    {
	      merged_boxes.push_back(it);

	      if (it.x + it.width > frame.cols ||
		  it.y + it.height > frame.rows)
		continue;
	      Mat tmp;
	      frame(it).copyTo(tmp);    
	      // imshow( "merged textBox_" + to_string(i++), tmp );
	      textBoxes.push_back(move(tmp));
	    }
	  stop = clock();
	  cout << "time: " << (double) (stop - start) / CLOCKS_PER_SEC * 1000 << "ms\n";
       	  
	  /////////////////////////////////////////////////////
	  //             Step 6: Draw  text region            /
	  /////////////////////////////////////////////////////
	  start = clock();
	  cout << ">>>>>>>>> \tDraw text boxes \t";

	  groups_draw(frame, merged_boxes);
	  imshow("capture", frame);
	  
	  stop = clock();
	  cout << "time: " << (double) (stop - start) / CLOCKS_PER_SEC * 1000 << "ms\n";
       	  


	  /////////////////////////////////////////////////////
	  //             Step 7: OCR                          /
	  /////////////////////////////////////////////////////
	  start = clock();
	  cout << ">>>>>>>>> \ttesserect ocr start\n";

	  
	  // Init Tesseract API
	  tesseract::TessBaseAPI *tess_api = new tesseract::TessBaseAPI();
	  
	  tess_api->Init("/usr/local/share/tessdata/", "eng", tesseract::OEM_TESSERACT_ONLY); 
	  
	  string char_whitelist = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
	  
	  if ( !tess_api->SetVariable("tessedit_char_whitelist", char_whitelist.c_str()) )
	    cerr << "Tesseract:\tsetVariable \"tessedit_char_whitelist\" failed!\n";
	  
	  // do things parallel. cilk for.
	  for ( const auto& i : textBoxes ) 
	    {
	      tess_api->SetImage((uchar*)i.data,
				 i.size().width,
				 i.size().height,
				 i.channels(),
				 i.step1());

	      tess_api->Recognize(0);
	      tesseract::ResultIterator* ri_word = tess_api->GetIterator();
	      tesseract::PageIteratorLevel level_word = tesseract::RIL_WORD;

	      if (ri_word != 0) {
		do
		  {
		    const char* word = ri_word->GetUTF8Text(level_word);
		    float conf = ri_word->Confidence(level_word);
		    printf("word: '%s';   \tconf: %.2f;\n", word, conf);
		    delete[] word;
		  }
		while (ri_word->Next(level_word));
	      }
	    }
	  tess_api->End();	  
	  stop = clock();
	  cout << ">>>>>>>>> \tOCR done! \t\ttime: " << (double) (stop - start) / CLOCKS_PER_SEC * 1000 << "ms\n";
       	  
	  


	  cout << "\n>>>>>>>>> \tALL DONE!  \t";
	  stop_all = clock();
	  cout << "  total time: " << (double) (stop_all - start_all) / CLOCKS_PER_SEC * 1000 << "ms\n";
       	  
	  
	  cout << "__________________________________________________\n";
	  
	  /**/

	  
	} // else if space key end.
      
      
	  
    } // end of while (1)
  
  
  
  
  return 0;
}


