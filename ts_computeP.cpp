#include <iostream>
#include <vector>
#include <boost/program_options.hpp>
#include "vptree.h"

using namespace std;
namespace po = boost::program_options;

// TODO (FOR OT-SNE): Loop over this function for each time step to comput within-dataset
// Gaussian distances (using the nearest neighbor approximation given)

// Incorporate the across time components (need to decide whether to add these separately
// or include them in the nearest neighbor calculation ... Depends on what the data look like


/* Functions taken from Laurens van der Maaten's original implementation of t-SNE */
/* Source: https://github.com/lvdmaaten/bhtsne                                    */
static void computeGaussianPerplexity(double* X, int N, int D, unsigned int** _row_P, unsigned int** _col_P, double** _val_P, double perplexity, int K);
static void symmetrizeMatrix(unsigned int** _row_P, unsigned int** _col_P, double** _val_P, int N);
static void computeSquaredEuclideanDistance(double* X, int N, int D, double* DD);
static void zeroMean(double* X, int N, int D);
/**********************************************************************************/

static bool load_data(string infile, double **data, int &num_instances, int &num_features) {

  FILE *fp = fopen(infile.c_str(), "rb");
	if (fp == NULL) {
		cout << "Error: could not open data file " << infile << endl;
		return false;
	}

  uint64_t ret;
	ret = fread(&num_instances, sizeof(int), 1, fp);
	ret = fread(&num_features, sizeof(int), 1, fp);

	*data = (double *)malloc(num_instances * num_features * sizeof(double));
  if (*data == NULL) {
    cout << "Error: memory allocation of " << num_instances << " by " 
         << num_features << " matrix failed" << endl;
    return false;
  }

  uint64_t nelem = (uint64_t)num_instances * num_features;

  size_t batch_size = 1e8;
  double *ptr = *data;
  ret = 0;
  for (uint64_t remaining = nelem; remaining > 0; remaining -= batch_size) {
    if (remaining < batch_size) {
      batch_size = remaining;
    }
    ret += fread(ptr, sizeof(double), batch_size, fp);
    ptr += batch_size;
  }
  
  if (ret != nelem) {
    cout << "Error: reading input returned incorrect number of elements (" << ret
         << ", expected " << nelem << ")" << endl;
    return false;
  }
  
	fclose(fp);

	return true;
}


static bool load_OT(string infile, int &N, unsigned int** row_P, unsigned int** col_P, double** val_P) {
  FILE *h;
	if((h = fopen(infile.c_str(), "rb")) == NULL) {
    return false;
	}

  size_t ret = 0;
  ret += fread(&N, sizeof(int), 1, h);
  *row_P = (unsigned int*)malloc((N+1) * sizeof(unsigned int));
  if (*row_P == NULL) {
    printf("Memory allocation error\n");
    exit(1);
  }
  ret += fread(*row_P, sizeof(unsigned int), N+1, h);
  *col_P = (unsigned int*)malloc((*row_P)[N] * sizeof(unsigned int));
  *val_P = (double*)malloc((*row_P)[N] * sizeof(double));
  if (*col_P == NULL || *val_P == NULL) {
    printf("Memory allocation error\n");
    exit(1);
  }
  ret += fread(*col_P, sizeof(unsigned int), (*row_P)[N], h);
  ret += fread(*val_P, sizeof(double), (*row_P)[N], h);
  fclose(h);

  printf("P successfully loaded\n");
  return true;
}

static void truncate_data(double *X, int num_instances, int num_features, int target_dims) {
  size_t i_old = 0;
  size_t i_new = 0;
  for (int r = 0; r < num_instances; r++) {
    for (int c = 0; c < num_features; c++) {
      if (c < target_dims) {
        X[i_new++] = X[i_old];
      }
      i_old++;
    }
  }
}

static bool run(double *X, int num_instances, int num_features, double perplexity,
		unsigned int **temp_row_P, unsigned int **temp_col_P, double **temp_val_P) {

  int K = (int) 3*perplexity; 
  //*temp_row_P = (unsigned int*) malloc((num_instances+1)*sizeof(unsigned int));
  //*temp_col_P = (unsigned int*) calloc(num_instances*K, sizeof(unsigned int));
  //*temp_val_P = (double *) calloc(num_instances*K, sizeof(double));
  /*
  unsigned int* row_P = *temp_row_P;
  unsigned int* col_P = *temp_col_P;
  double* val_P = *temp_val_P;
  */
  
  // Apply lower bound on perplexity from original t-SNE implementation
  if (num_instances - 1 < 3 * perplexity) {
    cout << "Error: target perplexity (" << perplexity << ") is too large "
         << "for the number of data points (" << num_instances << ")" << endl;
    return false;
  }

  printf("Processing %d data points, %d features with target perplexity %f\n",
         num_instances, num_features, perplexity);

  // Normalize input data (to prevent numerical problems)
  zeroMean(X, num_instances, num_features);
  cout << "Normalizing the features" << endl;
  double max_X = 0;
  for (size_t i = 0; i < num_instances * num_features; i++) {
    if (fabs(X[i]) > max_X) {
      max_X = fabs(X[i]);
    }
  }

  for (size_t i = 0; i < num_instances * num_features; i++) {
    X[i] /= max_X;
  }

  // Compute input similarities for exact t-SNE
  double* P; unsigned int* row_P; unsigned int* col_P; double* val_P;

  // Compute asymmetric pairwise input similarities
  cout << "Computing conditional distributions" << endl;
  computeGaussianPerplexity(X, num_instances, num_features,
    &row_P, &col_P, &val_P, perplexity, (int) (3 * perplexity));

  // Symmetrize input similarities
  cout << "Symmetrizing matrix" << endl;
  symmetrizeMatrix(&row_P, &col_P, &val_P, num_instances);
  double sum_P = .0;
  for (int i = 0; i < row_P[num_instances]; i++) {
    sum_P += val_P[i];
  }
  for (int i = 0; i < row_P[num_instances]; i++) {
    val_P[i] /= sum_P;
  }

  
  /*
  cout << "Saving to " << outfile << endl;
  FILE *fp = fopen(outfile.c_str(), "wb");
  if (fp == NULL) {
    cout << "Error: could not open output file " << outfile << endl;
    return false;
  }

  fwrite(&num_instances, sizeof(int), 1, fp);
  fwrite(row_P, sizeof(unsigned int), num_instances + 1, fp);
  fwrite(col_P, sizeof(unsigned int), row_P[num_instances], fp);
  fwrite(val_P, sizeof(double), row_P[num_instances], fp);
  */
  /*
  for (int r=0; r<=num_instances; r++) cout << row_P[r] << ", ";

  cout << endl;
  */
  
  *temp_row_P = row_P;
  *temp_col_P = col_P;
  *temp_val_P = val_P; 

  // for (int r=row_P[num_instances]-100; r<row_P[num_instances]; r++) cout << (*temp_col_P)[r] << ", " << (*temp_val_P)[r] << "; ";

  cout << endl;
  /*
  free(row_P);
  free(col_P);
  free(val_P);
  */
  
  return true;
}

bool save_sparse_mat(string outfile, unsigned int* row_P, unsigned int* col_P,
		     double* val_P, int num_instances) {

  /*
  cout << "Printing pattern" << endl;

  int cursor;
  int col_start;
  int col_end;
  string output = ""; 

  // Create blank string
  for(int r=0; r<num_instances; r++) {
    for(int c=0; c<num_instances; c++) {
      output += " "; 
    }
    output += "\n"; 
  } 

  for(int r=0; r<num_instances; r++) {
    col_start = row_P[r];
    col_end = row_P[r+1];
    for (int c=col_start; c<col_end; c++) {
      output.replace(r*(num_instances+1) + col_P[c], 1, "x"); 
    } 
  } 

  cout << output; 

  */
  
  /*
  for(int r=0; r< num_instances; r++) {
    // cout << r << "; ";
    cursor=0;
    col_start = row_P[r];
    col_end = row_P[r+1]; 

    for(int c=0; c< num_instances; c++) {
      if(c==col_P[col_start+cursor] && cursor<col_end) {
	cout << "x";
	cursor++; 
      }
      else cout << " ";	
    } 
    cout << endl;
  }
  */

  cout << "Saving to " << outfile << endl;
  FILE *fp = fopen(outfile.c_str(), "wb");
  if (fp == NULL) {
    cout << "Error: could not open output file " << outfile << endl;
    return false;
  }

  fwrite(&num_instances, sizeof(int), 1, fp);
  fwrite(row_P, sizeof(unsigned int), num_instances + 1, fp);
  fwrite(col_P, sizeof(unsigned int), row_P[num_instances], fp);
  fwrite(val_P, sizeof(double), row_P[num_instances], fp);

  // free(row_P); free(col_P); free(val_P); 
  
  return true; 
} 

int main(int argc, char **argv) {
  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "produce help message")
    ("input-file", po::value<string>()->value_name("FILE")->default_value("data"), "name of binary input file (see prepare_input.m)")
    ("output-file", po::value<string>()->value_name("FILE")->default_value("P.dat"), "name of output file to be created")
    ("perp", po::value<double>()->value_name("NUM")->default_value(30, "30"), "set target perplexity for conditional distributions")
    ("num-dims", po::value<int>()->value_name("NUM"), "if provided, only the first NUM features in the input will be used")
    ("time-steps", po::value<int>()->value_name("NUM")->default_value(1,"1"), "set number of time steps in the data")
    ("time-offset", po::value<int>()->value_name("NUM")->default_value(0,"0"), "# of time-steps to look fro OT couplings")
  ;

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).
                options(desc).run(), vm);
  po::notify(vm);    
  
  if (vm.count("help")) {
    cout << "Usage: ComputeP [options]" << endl;
    cout << desc << "\n";
    return 1;
  }

  double perplexity = vm["perp"].as<double>();
  string infile = vm["input-file"].as<string>();
  string outfile = vm["output-file"].as<string>();

  // int time_steps = vm["time-steps"].as<int>(); // when we get this working
  int time_steps = 2;

  // int time_offset = vm["time-offset"].as<int>();
  int time_offset = 1; 
  
  double *data;
  int* num_instances = (int*) calloc(time_steps, sizeof(int));
  int num_features;
  string infile_t;
  int time_start;
  int time_end;
  
  /*
  unsigned int *full_row_P = NULL;
  unsigned int *full_col_P = NULL;
  double *full_val_P = NULL; 
  */

  vector<unsigned int> full_row_P;
  vector<unsigned int> full_col_P;
  vector<double> full_val_P;
  
  unsigned int *temp_row_P = NULL;
  unsigned int *temp_col_P = NULL;
  double *temp_val_P = NULL;
  
  unsigned int current_row = 0;
  unsigned int old_row = 0; 
  unsigned col_offset = 0;
  unsigned int K = (int)(3*perplexity); // # of nearest neighbors

  unsigned int offs_ind; 
  
  full_row_P.push_back(0); 
  
  // unsigned int last
  for(int t=0; t < time_steps; t++) {
    time_start = (t - time_offset > 0) ? (t-time_offset):0;
    time_end = (t + time_offset > time_steps - 1) ? (time_steps - 1):(t+time_offset);
    

    infile_t = infile + "_"+ to_string(t) + ".dat";
    
    if (!load_data(infile_t, &data, num_instances[t], num_features)) {
      return 1;
    }

    cout << "# instances: " << num_instances[t] << endl;
    
    old_row = current_row; 
    current_row += num_instances[t];
    
    // CHANGE TO VECTORS?
    
    cout << "Current row: " << current_row << endl;

    // Update "full" vectors to hold new data
    try {
    full_row_P.reserve(current_row);
    full_col_P.reserve(current_row*K);
    full_val_P.reserve(current_row*K);

    cout << "loop size? " << full_col_P.capacity() << endl;
						  
    } catch(...) {
      cout << "Some sort of error has happened" << endl;
      return 1;
    }
    cout << infile_t << " successfully loaded" << endl;
    
    if (vm.count("num-dims")) {
      int num_dims = vm["num-dims"].as<int>();
      cout << "Using only the first " << num_dims << " dimensions" << endl;
      truncate_data(data, num_instances[t], num_features, num_dims);
      num_features = num_dims;
    }
    
    if (!run(data, num_instances[t], num_features, perplexity, &temp_row_P,
	     &temp_col_P, &temp_val_P)) {
      return 1;
    }

    // full_row_P[
    cout << "Old Row: " << old_row << endl;
    
    for(int r=1; r<= num_instances[t]; r++) {
      // offs_ind = full_row_P[old_row] + r - 1; 
      // temp_row_P[r] + full_row_P[old_row]; 
      full_row_P.push_back(temp_row_P[r] + full_row_P[old_row]);
      
      // cout << full_row_P[old_row + r] << "," << temp_row_P[r] << "; ";
			   
      for(int c=temp_row_P[r-1]; c<temp_row_P[r]; c++) {
	full_col_P.push_back(temp_col_P[c] + old_row);
	full_val_P.push_back(temp_val_P[c]);
      }
	
    } 

    cout << endl;
    /*  
    for(int c=0; c< num_instances[t]*K; c++) {
      full_col_P.push_back(temp_col_P[c] + old_row);
      full_val_P.push_back(temp_val_P[c]);
    } 
    */
      
    free(data);
    /*
    free(temp_row_P);
    free(temp_col_P);
    free(temp_val_P);
    */
    for(int c=full_col_P.size()-100; c<full_col_P.size(); c++) {
      cout << full_col_P[c] << "," << full_val_P[c] << "; "; 
    } 
    cout << endl;
    cout << "Done with t : " << t << endl;
  }

  
  cout << "size? " <<  full_row_P.size() << " " << full_col_P.size() << " " <<  full_val_P.size() << endl;
  /*
  for(int r=0; r<full_row_P.size(); r++) {
    cout << full_row_P[r] << ", "; 
  } 
  */
  cout << endl;
  
  save_sparse_mat(outfile, &full_row_P[0], &full_col_P[0], &full_val_P[0], current_row);

  // for(int r=0; r<full_row_P.size(); r++) {
  //   cout << full_row_P[r] << ", ";
  // }
  //     cout << endl;
  
  /*
  delete *full_row_P;
  delete *full_col_P;
  delete *full_val_P;
  */
  
  return 0;
}

// Compute input similarities with a fixed perplexity using ball trees (this function allocates memory another function should free)
// Source: https://github.com/lvdmaaten/bhtsne
static void computeGaussianPerplexity(double* X, int N, int D, unsigned int** _row_P, unsigned int** _col_P, double** _val_P, double perplexity, int K) {

    if(perplexity > K) printf("Perplexity should be lower than K!\n");

    // Allocate the memory we need
    *_row_P = (unsigned int*)    malloc((N + 1) * sizeof(unsigned int));
    *_col_P = (unsigned int*)    calloc(N * K, sizeof(unsigned int));
    *_val_P = (double*) calloc(N * K, sizeof(double));
    if(*_row_P == NULL || *_col_P == NULL || *_val_P == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    unsigned int* row_P = *_row_P;
    unsigned int* col_P = *_col_P;
    double* val_P = *_val_P;
    double* cur_P = (double*) malloc((N - 1) * sizeof(double));
    if(cur_P == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    row_P[0] = 0;
    for(int n = 0; n < N; n++) row_P[n + 1] = row_P[n] + (unsigned int) K;

    // Build ball tree on data set
    VpTree<DataPoint, euclidean_distance>* tree = new VpTree<DataPoint, euclidean_distance>();
    vector<DataPoint> obj_X(N, DataPoint(D, -1, X));
    for(int n = 0; n < N; n++) obj_X[n] = DataPoint(D, n, X + n * D);
    tree->create(obj_X);

    // Loop over all points to find nearest neighbors
    printf("Building tree...\n");
    vector<DataPoint> indices;
    vector<double> distances;
    for(int n = 0; n < N; n++) {

        if(n % 10000 == 0) printf(" - point %d of %d\n", n, N);

        // Find nearest neighbors
        indices.clear();
        distances.clear();
        tree->search(obj_X[n], K + 1, &indices, &distances);

        // Initialize some variables for binary search
        bool found = false;
        double beta = 1.0;
        double min_beta = -DBL_MAX;
        double max_beta =  DBL_MAX;
        double tol = 1e-5;

        // Iterate until we found a good perplexity
        int iter = 0; double sum_P;
        while(!found && iter < 200) {

            // Compute Gaussian kernel row
            for(int m = 0; m < K; m++) cur_P[m] = exp(-beta * distances[m + 1] * distances[m + 1]);

            // Compute entropy of current row
            sum_P = DBL_MIN;
            for(int m = 0; m < K; m++) sum_P += cur_P[m];
            double H = .0;
            for(int m = 0; m < K; m++) H += beta * (distances[m + 1] * distances[m + 1] * cur_P[m]);
            H = (H / sum_P) + log(sum_P);

            // Evaluate whether the entropy is within the tolerance level
            double Hdiff = H - log(perplexity);
            if(Hdiff < tol && -Hdiff < tol) {
                found = true;
            }
            else {
                if(Hdiff > 0) {
                    min_beta = beta;
                    if(max_beta == DBL_MAX || max_beta == -DBL_MAX)
                        beta *= 2.0;
                    else
                        beta = (beta + max_beta) / 2.0;
                }
                else {
                    max_beta = beta;
                    if(min_beta == -DBL_MAX || min_beta == DBL_MAX)
                        beta /= 2.0;
                    else
                        beta = (beta + min_beta) / 2.0;
                }
            }

            // Update iteration counter
            iter++;
        }

        // Row-normalize current row of P and store in matrix
        for(unsigned int m = 0; m < K; m++) cur_P[m] /= sum_P;
        for(unsigned int m = 0; m < K; m++) {
            col_P[row_P[n] + m] = (unsigned int) indices[m + 1].index();
            val_P[row_P[n] + m] = cur_P[m];
        }
    }

    // Clean up memory
    obj_X.clear();
    free(cur_P);
    delete tree;
}

// Symmetrizes a sparse matrix
// Source: https://github.com/lvdmaaten/bhtsne
static void symmetrizeMatrix(unsigned int** _row_P, unsigned int** _col_P, double** _val_P, int N) {

    // Get sparse matrix
    unsigned int* row_P = *_row_P;
    unsigned int* col_P = *_col_P;
    double* val_P = *_val_P;

    // Count number of elements and row counts of symmetric matrix
    int* row_counts = (int*) calloc(N, sizeof(int));
    if(row_counts == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    for(int n = 0; n < N; n++) {
        for(int i = row_P[n]; i < row_P[n + 1]; i++) {

            // Check whether element (col_P[i], n) is present
            bool present = false;
            for(int m = row_P[col_P[i]]; m < row_P[col_P[i] + 1]; m++) {
                if(col_P[m] == n) present = true;
            }
            if(present) row_counts[n]++;
            else {
                row_counts[n]++;
                row_counts[col_P[i]]++;
            }
        }
    }
    int no_elem = 0;
    for(int n = 0; n < N; n++) no_elem += row_counts[n];

    // Allocate memory for symmetrized matrix
    unsigned int* sym_row_P = (unsigned int*) malloc((N + 1) * sizeof(unsigned int));
    unsigned int* sym_col_P = (unsigned int*) malloc(no_elem * sizeof(unsigned int));
    double* sym_val_P = (double*) malloc(no_elem * sizeof(double));
    if(sym_row_P == NULL || sym_col_P == NULL || sym_val_P == NULL) { printf("Memory allocation failed!\n"); exit(1); }

    // Construct new row indices for symmetric matrix
    sym_row_P[0] = 0;
    for(int n = 0; n < N; n++) sym_row_P[n + 1] = sym_row_P[n] + (unsigned int) row_counts[n];

    // Fill the result matrix
    int* offset = (int*) calloc(N, sizeof(int));
    if(offset == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    for(int n = 0; n < N; n++) {
        for(unsigned int i = row_P[n]; i < row_P[n + 1]; i++) {                                  // considering element(n, col_P[i])

            // Check whether element (col_P[i], n) is present
            bool present = false;
            for(unsigned int m = row_P[col_P[i]]; m < row_P[col_P[i] + 1]; m++) {
                if(col_P[m] == n) {
                    present = true;
                    if(n <= col_P[i]) {                                                 // make sure we do not add elements twice
                        sym_col_P[sym_row_P[n]        + offset[n]]        = col_P[i];
                        sym_col_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = n;
                        sym_val_P[sym_row_P[n]        + offset[n]]        = val_P[i] + val_P[m];
                        sym_val_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = val_P[i] + val_P[m];
                    }
                }
            }

            // If (col_P[i], n) is not present, there is no addition involved
            if(!present) {
                sym_col_P[sym_row_P[n]        + offset[n]]        = col_P[i];
                sym_col_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = n;
                sym_val_P[sym_row_P[n]        + offset[n]]        = val_P[i];
                sym_val_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = val_P[i];
            }

            // Update offsets
            if(!present || (present && n <= col_P[i])) {
                offset[n]++;
                if(col_P[i] != n) offset[col_P[i]]++;
            }
        }
    }

    // Divide the result by two
    for(int i = 0; i < no_elem; i++) sym_val_P[i] /= 2.0;

    // Return symmetrized matrices
    free(*_row_P); *_row_P = sym_row_P;
    free(*_col_P); *_col_P = sym_col_P;
    free(*_val_P); *_val_P = sym_val_P;

    // Free up some memery
    free(offset); offset = NULL;
    free(row_counts); row_counts  = NULL;
}

// Compute squared Euclidean distance matrix
// Source: https://github.com/lvdmaaten/bhtsne
static void computeSquaredEuclideanDistance(double* X, int N, int D, double* DD) {
    const double* XnD = X;
    for(int n = 0; n < N; ++n, XnD += D) {
        const double* XmD = XnD + D;
        double* curr_elem = &DD[n*N + n];
        *curr_elem = 0.0;
        double* curr_elem_sym = curr_elem + N;
        for(int m = n + 1; m < N; ++m, XmD+=D, curr_elem_sym+=N) {
            *(++curr_elem) = 0.0;
            for(int d = 0; d < D; ++d) {
                *curr_elem += (XnD[d] - XmD[d]) * (XnD[d] - XmD[d]);
            }
            *curr_elem_sym = *curr_elem;
        }
    }
}

// Makes data zero-mean
// Source: https://github.com/lvdmaaten/bhtsne
static void zeroMean(double* X, int N, int D) {

	// Compute data mean
	double* mean = (double*) calloc(D, sizeof(double));
    if(mean == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    int nD = 0;
	for(int n = 0; n < N; n++) {
		for(int d = 0; d < D; d++) {
			mean[d] += X[nD + d];
		}
        nD += D;
	}
	for(int d = 0; d < D; d++) {
		mean[d] /= (double) N;
	}

	// Subtract data mean
    nD = 0;
	for(int n = 0; n < N; n++) {
		for(int d = 0; d < D; d++) {
			X[nD + d] -= mean[d];
		}
        nD += D;
	}
    free(mean); mean = NULL;
}
