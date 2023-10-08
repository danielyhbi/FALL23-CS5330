/*
  Bruce A. Maxwell
  S21

  Sample code to identify image fils in a directory
*/
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <vector>
using namespace std;

/*
  Given a directory on the command line, scans through the directory for image files.

  Prints out the full path name for each file.  This can be used as an argument to fopen or to cv::imread.
 */
vector<string> readFiles(const std::string &argv)
{
  char dirname[256];
  char buffer[256];
  FILE *fp;
  DIR *dirp;
  struct dirent *dp;
  int i;

  vector<string> result;

  // // check for sufficient arguments
  // if( argc < 2) {
  //   printf("usage: %s <directory path>\n", argv[0]);
  //   exit(-1);
  // }

  // get the directory path
  strcpy(dirname, argv.c_str());
  printf("Processing directory %s\n", dirname);

  // open the directory
  dirp = opendir(dirname);
  if (dirp == NULL)
  {
    printf("Cannot open directory %s\n", dirname);
    exit(-1);
  }

  // loop over all the files in the image file listing
  while ((dp = readdir(dirp)) != NULL)
  {

    // check if the file is an image
    if (strstr(dp->d_name, ".jpg") ||
        strstr(dp->d_name, ".png") ||
        strstr(dp->d_name, ".ppm") ||
        strstr(dp->d_name, ".tif"))
    {

      //printf("processing image file: %s\n", dp->d_name);

      // build the overall filename
      string output = argv + "/" + dp->d_name;

      // strcpy(buffer, dirname);
      // strcat(buffer, "/");
      // strcat(buffer, dp->d_name);

      result.push_back(output);

      //printf("full path name: %s\n", buffer);
      //printf("full path name: %s\n", output.c_str());
    }
  }

  printf("Terminating\n");

  return result;
}