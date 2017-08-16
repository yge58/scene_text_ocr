
#include <omp.h>
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
#define NUM_THREADS 4

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
  int num_threads = 0;

  omp_set_dynamic(0);
  //omp_set_num_threads(NUM_THREADS);
#pragma omp parallel
  {
#pragma omp single
    num_threads = omp_get_num_threads();
  }

  cout <<"Requested " <<  num_threads << " threads from system. " << endl;


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

  
  /*
  start = clock();
  Ptr<ERFilter> er_filter1 = createERFilterNM1(loadClassifierNM1("trained_classifierNM1.xml"), 
					       8, //  threshold step when extracting the component tree
					       0.00025f, // min area (% of image size) allowed for ER
					       0.13f, // the max area allowed for ER
					       0.4f, // default = 0.4; min probability P(er | character) allowed for ER
					       true, // nonMaxSuppression
					       0.1f); // minProbabilityDiff
  
  Ptr<ERFilter> er_filter2 = createERFilterNM2(loadClassifierNM2("trained_classifierNM2.xml"),
					       0.5); // the min probability P(er | character) allowed for ER
  stop = clock();
  cout << "er_filters init time: " << (double) (stop - start) / CLOCKS_PER_SEC * 1000 << "ms\n";
  
  */
 

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
	  vector<Mat> NMchannels;
	  computeNMChannels(frame, NMchannels);
   
	  // Determine channel size
	  size_t n_channels = NMchannels.size();	  
	  vector<vector<ERStat> > regions;
	  cout << "total channel:\t " << n_channels << endl;

	  vector<Mat> channels;




	  cout << ">>>> parallel extracting Extremal Regions" << endl;
#pragma omp parallel for shared (NMchannels)  schedule (dynamic)
	  for (int i = 0; i < n_channels; i++ )
	   {
	     printf("thread id: %d \n",  omp_get_thread_num());

	     
	     Ptr<ERFilter> er_filter1 = createERFilterNM1(loadClassifierNM1("trained_classifierNM1.xml"), 8, 0.00025f, 0.13f, 0.4f, true, 0.1f); 
	  
	     Ptr<ERFilter> er_filter2 = createERFilterNM2(loadClassifierNM2("trained_classifierNM2.xml"), 0.5 );
	  
	     vector<ERStat>  localER;
	     
	     er_filter1->run(NMchannels[i],  localER);
	     er_filter2->run(NMchannels[i],  localER);

#pragma omp critical
	     {
	       // printf("thread id %d enter critical region\n", id);
	       channels.push_back( move(NMchannels[i]) );
	       regions.push_back( move(localER) );
	     }
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
	  tess_api->SetVariable("tessedit_char_whitelist", char_whitelist.c_str() );


	  //#pragma omp parallel for shared(textBoxes) schedule(dynamic) 
	  for (int i = 0; i < textBoxes.size(); ++i)
	    {
	      //printf("thread : %d\n", omp_get_thread_num() );
	      
	      tess_api->SetImage((uchar*)textBoxes[i].data,
				 textBoxes[i].size().width,
				 textBoxes[i].size().height,
				 textBoxes[i].channels(),
				 textBoxes[i].step1());

	      tess_api->Recognize(0);
	      tesseract::ResultIterator* ri_word = tess_api->GetIterator();
	      tesseract::PageIteratorLevel level_word = tesseract::RIL_WORD;

	      if (ri_word != 0)
		{
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


