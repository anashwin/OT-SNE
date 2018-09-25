#include <cstring>
#include <cstdlib>
#include<cmath>
#include<iostream>
// #include "sptree.h"
#include "array_sptree.h"
#include "sptree.h"

using namespace std;

int main(int argc, char **argv) {
  /*
  cout << min(1., 2.) << endl; 
  cout << "Hello World!" << endl;
  */
  
  unsigned int no_dims = 2;
  // unsigned int N_data = 10;
  int square_dim = 4;
  int N_data = square_dim*square_dim + 2; 
  unsigned int T_time_steps = 4;
  // unsigned int pts_per_ts = N_data/T_time_steps;

  
  double *test_data = (double *) calloc(N_data*no_dims, sizeof(double));

  double *sp_test_data = (double *) calloc(N_data*no_dims/2+1, sizeof(double));
  
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
  int sp_iter = 0; 
  for(int i=0; i<square_dim; i++) {
    for(int j=0; j<square_dim; j++) {
      x = (double)i-((double)(square_dim/2) - .5); 
      y = (double)j-((double)(square_dim/2) - .5);

      test_data[(i*square_dim + j)*no_dims] = x;
      test_data[(i*square_dim + j)*no_dims + 1] = y;
      
      assignments[i*square_dim + j] = (i+j) % 2;

      if (assignments[i*square_dim + j] == 0) {
	sp_test_data[sp_iter] = x;
	sp_test_data[sp_iter+1] = y;
	sp_iter += 2;
      } 
      // assignments[i*square_dim + j] = 0; 

    }
  }

  // Add two data points and different time-steps that should not be considered
  test_data[(square_dim*square_dim)*no_dims] = .3;
  test_data[(square_dim*square_dim)*no_dims+1] = -.1;
  sp_test_data[sp_iter] = .3;
  sp_test_data[sp_iter + 1] = -.1;
  assignments[square_dim*square_dim] = 0;
  sp_iter += 2; 
  
  test_data[(square_dim*square_dim+1)*no_dims] = -1.3;
  test_data[(square_dim*square_dim+1)*no_dims+1] = -1.1;
  sp_test_data[sp_iter] = -1.3;
  sp_test_data[sp_iter + 1] = -1.1;

  assignments[square_dim*square_dim+1] = 0;
  for (int i=0; i<(square_dim*square_dim)/2 + 2; i++) {
    cout << "~~~~" << endl;
    cout << sp_test_data[i*no_dims] << ", " << sp_test_data[i*no_dims+1] << endl;
    /*
    if (i%2==0) {
      cout << sp_test_data[i*no_dims/2] << ", " << sp_test_data[i*no_dims/2 +1] << endl;									         
    } 
    */
  }
  
  double theta = .5;
  
  double neg_f_tmp[3] = {0};
  double cur_qsum = .0;
  int T_offset = 0;
  
  //SPTree sptree_test; 
  ArraySPTree *array_sptree_test = new ArraySPTree(no_dims, test_data, N_data, T_time_steps,
						   assignments);

  array_sptree_test->print();
  
  int targ_ind = 0; 
  cout << "TARGET: (" << test_data[targ_ind*no_dims] << ", " << test_data[targ_ind*no_dims+1] << ")  t="
       << assignments[targ_ind] << endl;
  cout << "DEF TARGET: (" << sp_test_data[targ_ind*no_dims] << ", " << sp_test_data[targ_ind*no_dims+1] << ")  t="
       << assignments[targ_ind] << endl;
  array_sptree_test->computeNonEdgeForces(targ_ind, theta, neg_f_tmp, &cur_qsum, T_offset, assignments[targ_ind]);
  cout << "OUT_GRAD: "; 
  for(int d=0; d<3; d++) cout << neg_f_tmp[d] << ", ";
  cout << endl;
  cout << cur_qsum << endl;
  
  double neg_f_tmp2[3] = {0};
  cur_qsum = .0;
  targ_ind = 0; 
  
  SPTree *sptree_test = new SPTree(no_dims, sp_test_data, (N_data-2)/2+2);

  sptree_test->computeNonEdgeForces(targ_ind, theta, neg_f_tmp2, &cur_qsum);
  for(int d=0; d<3; d++) cout << neg_f_tmp2[d] << ", ";
  cout << endl;

  cout << cur_qsum << endl; 
  
  // array_sptree_test->print();
  // sptree_test->print();

  // Our next test is to test computeNonEdgeForces

  // 1. two close-by nodes, one faraway one

  /*
  T_time_steps = 1;
  N_data = 3;

  test_data = (double*) realloc((double*)test_data, N_data*no_dims*sizeof(double));
  assignments = (int*) realloc((int*)assignments, N_data*sizeof(int));

  test_data[0] = .5;
  test_data[1] = .5;

  test_data[2] = -.45;
  test_data[3] = -.4;

  test_data[4] = -.4;
  test_data[5] = -.45; 

  assignments[0] = 0;
  assignments[1] = 0;
  assignments[2] = 0;

  
  ArraySPTree *sp2 = new ArraySPTree(no_dims, test_data, N_data, T_time_steps, assignments);
  sp2->print();
  sp2->computeNonEdgeForces(1, theta, neg_f_tmp, &cur_qsum, T_offset);

  for(int d=0; d<3; d++) cout << neg_f_tmp[d] << ", ";
  cout << endl;
  */
}

