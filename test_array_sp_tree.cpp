#include <cstring>
#include <cstdlib>
#include<cmath>
#include<iostream>
// #include "sptree.h"
#include "array_sptree.h"

using namespace std;

int main(int argc, char **argv) {

  cout << min(1., 2.) << endl; 
  cout << "Hello World!" << endl;

  unsigned int no_dims = 2;
  // unsigned int N_data = 10;
  int square_dim = 8;
  int N_data = square_dim*square_dim; 
  unsigned int T_time_steps = 2;
  unsigned int pts_per_ts = N_data/T_time_steps;

  
  double *test_data = (double *) calloc(N_data*no_dims, sizeof(double));
  int *assignments = (int*) calloc(N_data, sizeof(int)); 
  /*
  for(int i=0; i<N_data; i++) {
    test_data[i*no_dims] = (double) i;
    test_data[i*no_dims + 1] = (double) i;
    assignments[i] = (int) i/pts_per_ts;
    // cout << "(" << i << ", " << i << ")" << assignments[i] << endl;
  }
  */

  // Let's use an example that we know what the COM's should be?

  double x;
  double y; 
  for(int i=0; i<square_dim; i++) {
    for(int j=0; j<square_dim; j++) {
      x = (double)i-((double)(square_dim/2) - .5); 
      y = (double)j-((double)(square_dim/2) - .5);

      test_data[(i*square_dim + j)*no_dims] = x;
      test_data[(i*square_dim + j)*no_dims + 1] = y;

      assignments[i*square_dim + j] = (i) % T_time_steps;

    }
  }

  /*
  for(int n=0; n<N_data; n++) {
    cout << "(" << test_data[n*no_dims] << "," << test_data[n*no_dims+1] << "); " << assignments[n]
	 << endl;
  }
  */

  // cout << endl;
  
  //SPTree sptree_test; 
  ArraySPTree *array_sptree_test = new ArraySPTree(no_dims, test_data, N_data, T_time_steps,
						   assignments);
  // SPTree *sptree_test = new SPTree(no_dims, test_data, N_data);
  // array_sptree_test->print();
  // sptree_test->print();

  // Our next test is to test computeNonEdgeForces

  
}
