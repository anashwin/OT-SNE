#include <iostream>
#include <fstream>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include "ts_sne.h"
#include <armadillo>

using namespace std;
namespace po = boost::program_options;
namespace fsys = boost::filesystem;

bool load_assignments(string infile_asgn, int* assignments, int time_steps) {
  ifstream infile(infile_asgn);

  int asgn_num;
  int ctr = 0;

  while(infile >> asgn_num) {
    assignments[ctr] = asgn_num;
    ctr ++; 

    if(ctr > time_steps) {
      cout << "Too many time points in the file!" << endl;
      return 1; 
    } 
  }

  if(ctr < time_steps) {
    cout << "Too few time points in the file!" << endl;
    return 1; 
  }

  return 0; 
} 

bool load_data(string infile_X, mat &X, int &num_instances, int &num_features) {

  FILE *fp = fopen(infile_X.c_str(), "rb");
	if (fp == NULL) {
		cout << "Error: could not open data file " << infile_X << endl;
		return false;
	}

  uint64_t ret;
	ret = fread(&num_instances, sizeof(int), 1, fp);
	ret = fread(&num_features, sizeof(int), 1, fp);

  X.set_size(num_features, num_instances);

  uint64_t nelem = (uint64_t)num_instances * num_features;

  size_t batch_size = 1e8;
  double *ptr = X.memptr();
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



int main(int argc, char **argv) {
  // Declare the supported options
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "produce help message")
    ("input-X", po::value<string>()->value_name("FILE")->default_value("data.dat"), "name of binary input file containing data feature matrix (see prepare_input.m)")
    ("input-asgn", po::value<string>()->value_name("FILE")->default_value("assignments.txt"), "name of txt file containing time assignments for each cell in X")
    ("time-steps", po::value<int>()->value_name("NUM")->default_value(1), "# of time points in the data")
    ("input-P", po::value<string>()->value_name("FILE")->default_value("P.dat"), "name of binary input file containing P matrix (see ComputeP)")
    ("input-Y", po::value<string>()->value_name("FILE"), "if this option is provided, net-SNE will train to match the provided embedding instead of using the P matrix")
    ("out-dir", po::value<string>()->value_name("DIR")->default_value("out"), "where to create output files; directory will be created if it does not exist")
    ("out-dim", po::value<int>()->value_name("NUM")->default_value(2), "number of output dimensions")
    ("max-iter", po::value<int>()->value_name("NUM")->default_value(1000), "maximum number of iterations")
    ("rand-seed", po::value<int>()->value_name("NUM")->default_value(-1), "seed for random number generator; to use current time as seed set it to -1")
    ("theta", po::value<double>()->value_name("NUM")->default_value(0.5, "0.5"), "a value between 0 and 1 that controls the accuracy-efficiency tradeoff in SPTree for gradient computation; 0 means exact")
    ("learn-rate", po::value<double>()->value_name("NUM")->default_value(0.02, "0.02"), "learning rate for gradient steps")
    ("mom-init", po::value<double>()->value_name("NUM")->default_value(0.5, "0.5"), "initial momentum between 0 and 1")
    ("mom-final", po::value<double>()->value_name("NUM")->default_value(0.8, "0.8"), "final momentum between 0 and 1 (switch point controlled by --mom-switch-iter)")
    ("mom-switch-iter", po::value<int>()->value_name("NUM")->default_value(250), "duration (number of iterations) of initial momentum")
    ("early-exag-iter", po::value<int>()->value_name("NUM")->default_value(250), "duration (number of iterations) of early exaggeration")
    ("num-local-sample", po::value<int>()->value_name("NUM")->default_value(20), "number of local samples for each data point in the mini-batch")
    ("batch-frac", po::value<double>()->value_name("NUM")->default_value(0.1, "0.1"), "fraction of data to sample for mini-batch")
    ("min-sample-Z", po::value<double>()->value_name("NUM")->default_value(0.1, "0.1"), "minimum fraction of data to use for approximating the normalization factor Z in the gradient")
    ("l2-reg", po::value<double>()->value_name("NUM")->default_value(0, "0"), "L2 regularization parameter")
    ("init-model-prefix", po::value<string>()->value_name("STR"), "prefix of model files for initialization")
    ("step-method", po::value<string>()->value_name("STR")->default_value("adam"), "gradient step schedule; 'adam', 'mom' (momentum), 'mom_gain' (momentum with gains), 'fixed'")
    ("num-input-feat", po::value<int>()->value_name("NUM"), "if set, use only the first NUM features for the embedding function")
    ("init-map", po::bool_switch()->default_value(false), "output initial mapping for the entire data")
    ("num-layers", po::value<int>()->value_name("NUM")->default_value(2), "number of layers in the neural network")
    ("num-units", po::value<int>()->value_name("NUM")->default_value(50), "number of units for each layer in the neural network")
    ("act-fn", po::value<string>()->value_name("STR")->default_value("relu"), "activation function of the neural network; 'sigmoid' or 'relu'")
    ("test-model", po::bool_switch()->default_value(false), "if set, use the model provided with --init-model-prefix and visualize the entire data set then terminate without training")
    ("no-target", po::bool_switch()->default_value(false), "if this option is provided (ignored if --test-model is not set), then only the new embedding is printed, without the objective value")
    ("perm-iter", po::value<int>()->value_name("NUM")->default_value(INT_MAX, "INT_MAX"), "After every NUM iterations, permute the ordering of data points for fast mini-batching")
    ("cache-iter", po::value<int>()->value_name("NUM")->default_value(INT_MAX, "INT_MAX"), "After every NUM iterations, write intermediary embeddings and parameters to disk. Final embedding is always reported.")
    ("no-sgd", po::bool_switch()->default_value(false), "if set, do not use SGD acceleration; equivalent to t-SNE with an additional backpropagation step to train a neural network. Effective for small datasets")
    //("batch-norm", po::bool_switch()->default_value(false), "turn on batch normalization")
    //("monte-carlo-pos", po::bool_switch()->default_value(false), "use monte-carlo integration for positive gradient term")
    //("match-pos-neg", po::bool_switch()->default_value(false), "compute negative forces for points sampled for positive force")

  ;

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).
                options(desc).run(), vm);
  po::notify(vm);    

  if (vm.count("help")) {
    cout << "Usage: RunBhtsne [options]" << endl;
    cout << desc << "\n";
    return 1;
  }

  int time_steps = vm["time-steps"].as<int>(); 
  
  string infile_P = vm["input-P"].as<string>();
  string infile_X = vm["input-X"].as<string>();
  string outdir = vm["out-dir"].as<string>();
  string infile_asgn = vm["input-asgn"].as<string>(); 
  
  fsys::path dir(outdir);
  if (fsys::is_directory(dir)) {
    cout << "Error: Output directory (" << outdir << ") already exists" << endl;
    return 1;
  }
  if (fsys::create_directory(dir)) {
    cout << "Output directory created: " << outdir << endl; 
  }

  fsys::path paramfile = dir;
  paramfile /= "param.txt";
  ofstream ofs(paramfile.string().c_str());

  TS_SNE* ts_sne = new TS_SNE();

  bool use_known_Y = vm.count("input-Y");
  string infile_Y;

  ts_sne->TEST_RUN = vm["test-model"].as<bool>();
  ts_sne->NO_TARGET = ts_sne->TEST_RUN && vm["no-target"].as<bool>();

  if (!ts_sne->NO_TARGET) {
    if (use_known_Y) {
      infile_Y = vm["input-Y"].as<string>();
      ofs << "input-Y: " << infile_Y << endl;
      cout << "Learning to match the provided embedding: " << infile_Y << endl;
    } else {
      ofs << "input-P: " << infile_P << endl;
    }
  }
  ofs << "input-X: " << infile_X << endl;
  ofs << "out-dir: " << fsys::canonical(dir).string() << endl;

  ts_sne->BATCH_FRAC = vm["batch-frac"].as<double>(); ofs << "batch-frac: " << ts_sne->BATCH_FRAC << endl;
  ts_sne->N_SAMPLE_LOCAL = vm["num-local-sample"].as<int>(); ofs << "num-local-sample: " << ts_sne->N_SAMPLE_LOCAL << endl;
  ts_sne->MIN_SAMPLE_Z = vm["min-sample-Z"].as<double>(); ofs << "min-sample-Z: " << ts_sne->MIN_SAMPLE_Z << endl;
  ts_sne->STOP_LYING = vm["early-exag-iter"].as<int>(); ofs << "early-exag-iter: " << ts_sne->STOP_LYING << endl;
  ts_sne->STEP_METHOD = vm["step-method"].as<string>(); ofs << "step-method: " << ts_sne->STEP_METHOD << endl;
  ts_sne->LEARN_RATE = vm["learn-rate"].as<double>(); ofs << "learn-rate: " <<  ts_sne->LEARN_RATE << endl;
  ts_sne->L2_REG = vm["l2-reg"].as<double>(); ofs << "l2-reg: " <<  ts_sne->L2_REG << endl;
  ts_sne->SGD_FLAG = !vm["no-sgd"].as<bool>(); ofs << "sgd: " << ts_sne->SGD_FLAG << endl;
  ts_sne->PERM_ITER = vm["perm-iter"].as<int>(); ofs << "perm-iter: " << ts_sne->PERM_ITER << endl;
  ts_sne->CACHE_ITER = vm["cache-iter"].as<int>(); ofs << "cache-iter: " << ts_sne->CACHE_ITER << endl;
  //ts_sne->BATCH_NORM = vm["batch-norm"].as<bool>(); ofs << "batch-norm: " << ts_sne->BATCH_NORM << endl;
  //ts_sne->MONTE_CARLO_POS = vm["monte-carlo-pos"].as<bool>(); ofs << "monte-carlo-pos: " << ts_sne->MONTE_CARLO_POS << endl;
  //ts_sne->MATCH_POS_NEG = vm["match-pos-neg"].as<bool>(); ofs << "match-pos-neg: " << ts_sne->MATCH_POS_NEG << endl;

  ts_sne->MODEL_PREFIX_FLAG = vm.count("init-model-prefix");
  if (ts_sne->MODEL_PREFIX_FLAG) {
    ts_sne->MODEL_PREFIX = vm["init-model-prefix"].as<string>();
  }
  ofs << "init-model-prefix: " << ts_sne->MODEL_PREFIX << endl;

  if (ts_sne->TEST_RUN) {
    ts_sne->COMPUTE_INIT = true;

    if (!ts_sne->MODEL_PREFIX_FLAG) {
      cout << "Error: if --test-model is set, then --init-model-prefix must be provided; see --help" << endl;
      return 1;
    }
  } else {
    ts_sne->COMPUTE_INIT = vm["init-map"].as<bool>();
  }
  ofs << "init-map: " << ts_sne->COMPUTE_INIT << endl;

  if (!ts_sne->MODEL_PREFIX_FLAG) {
    ts_sne->NUM_LAYERS = vm["num-layers"].as<int>(); ofs << "num-layers: " << ts_sne->NUM_LAYERS << endl;
    ts_sne->NUM_UNITS = vm["num-units"].as<int>(); ofs << "num-units: " << ts_sne->NUM_UNITS << endl;
    ts_sne->ACT_FN = vm["act-fn"].as<string>(); ofs << "act-fn: " << ts_sne->ACT_FN << endl;
  }

  if (ts_sne->STEP_METHOD == "mom" || ts_sne->STEP_METHOD == "mom_gain") {
    ts_sne->MOM_SWITCH_ITER = vm["mom-switch-iter"].as<int>(); ofs << "mom-switch-iter: " << ts_sne->MOM_SWITCH_ITER << endl;
    ts_sne->MOM_INIT = vm["mom-init"].as<double>(); ofs << "mom-init: " << ts_sne->MOM_INIT << endl;
    ts_sne->MOM_FINAL = vm["mom-final"].as<double>(); ofs << "mom-final: " << ts_sne->MOM_FINAL << endl;
  }

  int no_dims = vm["out-dim"].as<int>(); ofs << "out-dim: " << no_dims << endl;
  double theta = vm["theta"].as<double>(); ofs << "theta: " << theta << endl;
  int rand_seed = vm["rand-seed"].as<int>(); ofs << "rand-seed: " << rand_seed << endl;
  int max_iter = vm["max-iter"].as<int>(); ofs << "max-iter: " << max_iter << endl;

  ofs.close();

  if (vm.count("help")) {
    cout << "Usage: RunBhtsne [options]" << endl;
    cout << desc << "\n";
    return 1;
  }

  if (ts_sne->STEP_METHOD != "adam" && ts_sne->STEP_METHOD != "mom" && ts_sne->STEP_METHOD != "mom_gain"
      && ts_sne->STEP_METHOD != "fixed") {
    cout << "Error: Unrecognized --step-method argument " << ts_sne->STEP_METHOD << "; see --help" << endl;
    return 1;
  } 

  if (ts_sne->ACT_FN != "sigmoid" && ts_sne->ACT_FN != "relu") {
    cout << "Error: Unrecognized --act-fn argument " << ts_sne->ACT_FN << "; see --help" << endl;
    return 1;
  } 

  int num_input_feat;
  if (vm.count("num-input-feat")) {
    num_input_feat = vm["num-input-feat"].as<int>();
  } else {
    num_input_feat = INT_MAX;
  }

  cout << "Loading input features ... ";
  mat X;
  int num_instances;
  int num_features;
  if (!load_data(infile_X, X, num_instances, num_features)) {
    return 1;
  }
  cout << endl;
  cout << "Loading assignments ... ";
  int* assignments = (int*)malloc(sizeof(int)*time_steps); 
  
  if (!load_assignments(infile_asgn, assignments, time_steps)) return 1; 
  
  if (X.n_rows > num_input_feat) {
    cout << "Truncating to top " << num_input_feat << " features" << endl;
    X = X.head_rows(num_input_feat);
  }

  cout << "Data feature matrix is " << X.n_rows << " by " << X.n_cols << endl;

  int N;
  unsigned int *row_P = NULL;
  unsigned int *col_P = NULL;
  double *val_P = NULL;
  mat target_Y;

  if (ts_sne->NO_TARGET) {

    cout << "No target provided" << endl;
    N = X.n_cols;

  } else if (use_known_Y) {

    cout << "Loading target Y ... ";
    target_Y.load(infile_Y, arma_ascii);
    target_Y = target_Y.t();
    cout << "done" << endl;

    N = target_Y.n_cols;

    if (N != X.n_cols) {
      cout << "Error: Y matrix dimensions (" << N << ") do not match with X matrix (" << X.n_cols << ")" << endl;
      return 1;
    }

  } else {

    cout << "Loading input similarities ... ";
    if (!ts_sne->load_P(infile_P, N, &row_P, &col_P, &val_P)) {
      cout << "Error: failed to load P from " << infile_P << endl;
      return 1;
    }
    cout << "done" << endl;

    if (N != X.n_cols) {
      cout << "Error: P matrix dimensions (" << N << ") do not match with X matrix (" << X.n_cols << ")" << endl;
      return 1;
    }

  }

  mat Y(no_dims, N);

  if (!ts_sne->run(N, row_P, col_P, val_P, target_Y, X, Y, no_dims, theta, rand_seed,
		   max_iter, dir, time_steps, assignments)) {
    return 1;
  }

  free(row_P);
  free(col_P);
  free(val_P);

  delete(ts_sne);

  cout << "Done" << endl;
}
