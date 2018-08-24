/*
 system-function for Fortran
 (C) Marat Khairoutdinov */

#if ( defined SUNOS ) || ( defined IRIX64 ) || ( defined OSF1 ) || ( defined LINUX ) 
#define FORTRANUNDERSCORE
#elif ( defined CRAY ) || ( defined T3D )
#define FORTRANCAPS
#elif ( defined AIX || defined MACOSX)
#endif

#if ( defined FORTRANCAPS )

#define systemf SYSTEMF
#define gammafff GAMMAFFF
#define erfff ERFFF

#elif ( defined FORTRANUNDERSCORE )

#define systemf systemf_
#define gammafff gammafff_
#define erfff erfff_

#elif ( defined FORTRANDOUBLEUNDERSCORE )

#define systemf systemf__
#define gammafff gammafff__
#define erfff erfff__

#endif

#include <math.h>
#include <stdio.h>


#ifdef __cplusplus 
extern "C" {
#endif

void systemf (const char *string) {system(string);}
float gammafff (float *x) {return (float)exp(lgamma(*x));}
  float erfff (float *x) {return (float)erf(*x);}


#ifdef __cplusplus
}
#endif
