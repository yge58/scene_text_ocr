// CPP
#include <iostream>
#include <utility>
#include <list>
#include <unordered_map>
#include <queue>

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
	  floodFill(channels[group[r][0]],
		    segmentation,
		    Point(er.pixel%channels[group[r][0]].cols, er.pixel/channels[group[r][0]].cols),
		    Scalar(255),
		    0,
		    Scalar(er.level),
		    Scalar(0),
		    flags);
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



// comparison, not case sensitive.
inline bool compare_rect (const Rect* first, const Rect* second)
{
  return first->x * first->y < second->x * second->y;
}



/************************************************************************/
// Main
int main(int args, char **argv)
{
  // Init cam
  VideoCapture cam(CAMERA_DEVICE_ID);
  
  // Check cam
  if (!cam.isOpened())
    {
      cerr << "camera failed!";
      exit(1);
    }
  cout << ">>>>>>>>> Camera initialization is done! \n";

  /* 
     Initialize text detection extremal region filter, as described in 
     (http://docs.opencv.org/3.0-beta/modules/text/doc/erfilter.html)
  */
  Ptr<ERFilter> er_filter1 = createERFilterNM1(loadClassifierNM1
					       ("trained_classifierNM1.xml"),8,0.00015f,0.13f,0.2f,true,0.1f);

  Ptr<ERFilter> er_filter2 = createERFilterNM2(loadClassifierNM2
					       ("trained_classifierNM2.xml"),0.5);

  if (er_filter1 != nullptr && er_filter2 != nullptr)
    cout << ">>>>>>>>> Text detecting filer initialization is done! \n";

  
  // Init Tesseract API
   tesseract::TessBaseAPI *tess_api = new tesseract::TessBaseAPI();
  
   
  // Init with english, can I init with chinese. french, multiple langs at once ?????
   tess_api->Init(NULL, "eng", tesseract::OEM_LSTM_ONLY);
    

  //tess_api->SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);

   string char_whitelist = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

  if ( !tess_api->SetVariable("tessedit_char_whitelist", char_whitelist.c_str()) )
    {
      cerr << "setVariable tessedit_char_whitelist failed!\n";
    }


   
  /**/
  
  cout << ">>>>>>>>> Tesseract initialization is done! \n";

  Mat frame; 
  namedWindow("Text", 1);

  
  // loop till key 'ESC' is pressed  
  while (1)
    {
      char key = (char)waitKey(-1);

      /* ESC == 27 */
      if ( key == 27 )
	{
	  cout << ">>>>>>>>>  ESC entered, program closing \n";
	  exit(0);
	}
      else if ( key == ' ' ) // if entered space key  
	{
	  //////////////////////////////////////////////////////
	  /*            Step 1: capture camera frame          */
	  //////////////////////////////////////////////////////
	  cout << ">>>>>>>>>  Capturing frame \n";
	  // open camera
	  cam.open(CAMERA_DEVICE_ID);
	  // check camera
	  if (!cam.isOpened())
	    {
	      cerr << ">>>>>>>>> Err:  Camera failed to open! \n";
	      exit(1);
	    }
	  // get a new frame from camera
	  // cam >> frame;
	  // close camera
	  cam.release();


	  // read a image from argv[1]
	  frame = imread(argv[1]);


	  // convert to grey-scale
	  // cvtColor(frame, grey_frame, CV_BGR2GRAY);
	  // imshow("grey_frame", grey_frame);

	
	  //////////////////////////////////////////////////////
	  /*             Step 2: text detection               */
	  //////////////////////////////////////////////////////
	  cout << ">>>>>>>>> Detecting text \n";

	  vector<Mat> channels;
	  computeNMChannels(frame, channels);
	  
	  // Determine channel size
	  size_t n_channels = channels.size();
	  
	  vector< vector<ERStat> > regions(n_channels);
	  
	  // OpenMP??? multithread parallel.
	  for (int i = 0; i < n_channels; i++)
	    {
	      er_filter1->run(channels[i], regions[i]);
	      er_filter2->run(channels[i], regions[i]);
	    }
	  
	  cout << ">>>>>>>>> Text detection done! \n ";
	  

	  /////////////////////////////////////////////////////
	  /*             Step 3: group text region           */
	  /////////////////////////////////////////////////////
	  cout << ">>>>>>>>> Text region grouping start \n";

	  vector< vector<Vec2i> > region_groups;
	  vector<Rect> boxes;

	  // Use in horizontal direction only!
	  // erGrouping(frame, channels, regions, region_groups, boxes, ERGROUPING_ORIENTATION_HORIZ);

	  // Any direction. what is 0.5?
	  erGrouping(frame, channels, regions, region_groups, boxes,
		     ERGROUPING_ORIENTATION_ANY, "./trained_classifier_erGrouping.xml", 0.5);


	  
	  
	  










	  cout << ">>>>>>>>> Text segmentation start ... \n";
	  
	  vector<Mat> segments;
	  
	  list<Rect*> list_ptrRect; 
	  unordered_map<Rect*, Mat> hash_mat;
	  
	  for ( auto i = 0; i < boxes.size(); ++i )
	    {
	      Mat seg = Mat::zeros(frame.rows+2, frame.cols+2, CV_8UC1);
	      
	      er_draw(channels, regions, region_groups[i], seg);

	      
	      // push to list]
	      list_ptrRect.push_back( &boxes[i] );
	      // hash map insert 
	      hash_mat.insert( { &boxes[i], move( seg(boxes[i]) ) } );	     
	    }
	  
	  
	  
	  

	  
	  cout << ">>>>>>>>> sorting text regions start ... \n";
	  
	  list_ptrRect.sort(compare_rect);
	  
	  
	  cout << ">>>>>>>>> grouping text regions start ... \n";
	  
	  int oldX = 0;
	  int newGroupIndex = -1;

	  vector<vector<Mat> > text_region_groups;

	  for ( auto it : list_ptrRect )
	    {
	      int currentX= it->x;
	      
	      if ( std::abs(currentX - oldX) > 10 )
		{
		  newGroupIndex++;
		  oldX = currentX;
		  vector<Mat> group = { move(hash_mat[it]) };
		  text_region_groups.push_back(group);
		}
	      else
		text_region_groups[newGroupIndex].push_back( move(hash_mat[it]) );
	    }



















	  
	  
	  cout << ">>>>>>>>> OCR grouped text regions start ... \n";

	  
	  for (int i = 0; i < text_region_groups.size(); ++i)
	    {
	      cout << "\nGroup " << i << " >>>>>>>>>>>>>>>>>>>>>" << endl;
	      
	      vector<list<pair<string, int> > > sentence_lists;
	        
	      int n_words = 0;

	      // for each group
	      for (int j = 0; j < text_region_groups[i].size(); ++j)
		{
		  // show grouped text region
		  //string s  = "Group " + to_string(i)  +  ("-" + to_string(j));
		  //imshow(s, text_region_groups[i][j]);
		  
		  

		  
		  tess_api->SetImage((uchar*)text_region_groups[i][j].data,
				     text_region_groups[i][j].size().width,
				     text_region_groups[i][j].size().height,
				     text_region_groups[i][j].channels(),
				     text_region_groups[i][j].step1());
	      
		  tess_api->Recognize(0);

		  char *ptrText= tess_api->GetUTF8Text();
		  string text = string(ptrText);
        
		  vector<string> words;
       
		  while( *ptrText )
		    {
		      string s;
		      while (*ptrText == ' ' || *ptrText == '\n')
			ptrText++;	        
		      
		      while(*ptrText != ' ' && *ptrText != '\n' && *ptrText)
			s.push_back(*ptrText++);

		      if (!s.empty())
			words.push_back(move(s));
		    }
		  
		  // get confidences of each word
		  int* confs = tess_api->AllWordConfidences();

		  list<pair<string, int> > sentence;
		  
		  for (int i = 0; i < words.size(); ++i)
		      sentence.push_back( make_pair(move(words[i]), move(confs[i])) );
		    
		  sentence_lists.push_back(move(sentence));
		}

	      cout << "print sentence_lists...\n";
	      for(int i = 0; i < sentence_lists.size(); ++i)
		{
		  for (auto it : sentence_lists[i])
		    cout << it.first << "(" << it.second << ") ";

		  cout << endl;

		}

	      cout << "Group " << i << " >> ready for post processing \n" ; 
	      
	    
	      //show grouped words and conf
	      cout << "num of sentences: " << sentence_lists.size() << endl; 

	  
	      
	      vector<string> candidate_sentence;
	
	      




	      {
	      // for ()
	      
	      /*
	      while( !sentence_lists.empty() )
		{
		  string candidate_word;
		  int max_conf = 0;
		  // choose the longest word as candidate word
		  int len = 0;
		  for (int i = 0; i < sentence_lists.size(); ++i)
		    {
		      if (sentence_lists[i].front().first.size() > len)
			{
			  len = sentence_lists[i].front().first.size();
			  candidate_word = sentence_lists[i].front().first;
			  max_conf = sentence_lists[i].front().second;
			}
		    }

		  cout <<"candidate word: "<< candidate_word << endl;

		  
		  
		  
		  for (int i = 0; i <  sentence_lists.size(); ++i)
		    {
		      // if list empty, continue;
		      if (sentence_lists[i].empty())
			{
			  sentence_lists.erase(sentence_lists.begin() + i);
			  continue;
			}

		      // if same word, skip;
		      if (sentence_lists[i].front().first.compare(candidate_word) == 0)
			{
			  sentence_lists[i].pop_front();
			  continue;
			}

		      // if lengh is small
		      // for example, candidate word is PROHIBITED. but recognized as PROHIB TED
		      if (sentence_lists[i].front().first.size() < len )
			{
			  sentence_lists[i].pop_front();
			  int len_remain = len - sentence_lists[i].front().first.size();
			 
			  if (sentence_lists[i].front().first.size() < len_remain)
			    sentence_lists[i].pop_front(); 

			  continue;
			}



		      sentence_lists[i].pop_front();
		      
		       cout << "candidate word found: " << candidate_word << endl;


		    }
		  
		  /**/
		}

		/****************** BUG          */
	      
	      cout << "Group " << i << " finished post procesisng >>>>>>>>>>>>>>>>>>>>>" << endl;

	    }
	  
	  cout << "done";
      
	  
	} // else if space key end.
      
      
	  
    } // end of while (1)
  
  
  
  
  return 0;
}


