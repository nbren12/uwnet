#include <stdlib.h>
#include <stdio.h>
#include "plugin.h"

int main(int argc, char *argv[])
{
  // Call hello world
  hello_world();

  // pass array to python and add 1
  int n = 100;
  double * array = calloc(n, sizeof(double));
  add_one(array, n);

  printf("Printing array after call to python. It should just be a sequence from 0 to 99\n.");
  int i;
  for (i=0; i < n ; i++) printf("%.0f ", array[i]);



  free(array);
  return 0;
}
