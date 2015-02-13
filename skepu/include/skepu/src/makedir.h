#ifndef MAKE_DIR
#define MAKE_DIR

#ifndef _WIN32

//#include "jlss.h"
//#include "emalloc.h"

#include <stdlib.h>
#include <errno.h>
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif /* HAVE_UNISTD_H */

#include <string.h>
#include <sys/stat.h>
#include <cassert>
//#include "sysstat.h"    /* Fix up for Windows - inc mode_t */

#include <pwd.h>

typedef struct stat Stat;

#ifndef lint
/* Prevent over-aggressive optimizers from eliminating ID string */
const char jlss_id_mkpath_c[] = "@(#)$Id: mkpath.c,v 1.12 2008/05/19 00:43:33 jleffler Exp $";
#endif /* lint */

int do_mkdir(const char *path, mode_t mode)
{
   Stat            st;
   int             status = 0;

   if (stat(path, &st) != 0)
   {
      /* Directory does not exist */
      if (mkdir(path, mode) != 0)
         status = -1;
   }
   else if (!S_ISDIR(st.st_mode))
   {
      errno = ENOTDIR;
      status = -1;
   }

   return(status);
}

/**
** mkpath - ensure all directories in path exist
** Algorithm takes the pessimistic view and works top-down to ensure
** each directory in path exists, rather than optimistically creating
** the last element and working backwards.
*/
int mkpath(const char *path, mode_t mode)
{
   char           *pp;
   char           *sp;
   int             status;
   char           *copypath = strdup(path);

   status = 0;
   pp = copypath;
   while (status == 0 && (sp = strchr(pp, '/')) != 0)
   {
      if (sp != pp)
      {
         /* Neither root nor double slash in path */
         *sp = '\0';
         status = do_mkdir(copypath, mode);
         *sp = '/';
      }
      pp = sp + 1;
   }
   if (status == 0)
      status = do_mkdir(path, mode);
   free(copypath);
   return (status);
}



bool fileExists(const std::string& filename)
{
   struct stat buf;
   if (stat(filename.c_str(), &buf) != -1)
   {
      return true;
   }
   return false;
}


std::string convertIntToString(int val)
{
   std::ostringstream convert;
   convert << val;
   return convert.str();
}


void createPath(std::string &path)
{
   if (mkpath(path.c_str(), 0777) != 0)
   {
      assert(false);
   }
}



std::string getUserHomeDirectory()
{
   std::string homedir;

   // first check for HOME environment variable...
   char *var = getenv("HOME");
   if(!var)
   {
      struct passwd *pw = getpwuid(getuid());
      homedir = pw->pw_dir;
   }
   else
      homedir = var;

   return homedir;
}


std::string getPMDirectory()
{
   return getUserHomeDirectory() + "/.skepu/";
}



std::string::value_type up_char(std::string::value_type ch)
{
   return std::use_facet< std::ctype< std::string::value_type > >( std::locale() ).toupper( ch );
}

std::string::value_type lower_char(std::string::value_type ch)
{
   return std::use_facet< std::ctype< std::string::value_type > >( std::locale() ).tolower( ch );
}

/** @brief To Capitalize (a-z to A-Z) the string */
std::string capitalizeString(const std::string &src)
{
   std::string result;
   std::transform( src.begin(), src.end(), std::back_inserter( result ), up_char );
   return result;
}


/** @brief To Capitalize (a-z to A-Z) the string */
std::string unCapitalizeString(const std::string &src)
{
   std::string result;
   std::transform( src.begin(), src.end(), std::back_inserter( result ), lower_char );
   return result;
}


#endif

#endif

