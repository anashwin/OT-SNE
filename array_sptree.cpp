/*
 *
 * Copyright (c) 2014, Laurens van der Maaten (Delft University of Technology)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *    This product includes software developed by the Delft University of Technology.
 * 4. Neither the name of the Delft University of Technology nor the names of
 *    its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY LAURENS VAN DER MAATEN ''AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL LAURENS VAN DER MAATEN BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGE.
 *
 */

#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include "array_sptree.h"

#include <armadillo>
using namespace arma;

// Constructs cell
Cell::Cell(unsigned int inp_dimension) {
    dimension = inp_dimension;
    corner = (double*) malloc(dimension * sizeof(double));
    width  = (double*) malloc(dimension * sizeof(double));
}

Cell::Cell(unsigned int inp_dimension, double* inp_corner, double* inp_width) {
    dimension = inp_dimension;
    corner = (double*) malloc(dimension * sizeof(double));
    width  = (double*) malloc(dimension * sizeof(double));
    for(int d = 0; d < dimension; d++) setCorner(d, inp_corner[d]);
    for(int d = 0; d < dimension; d++) setWidth( d,  inp_width[d]);
}

// Destructs cell
Cell::~Cell() {
    free(corner);
    free(width);
}

double Cell::getCorner(unsigned int d) {
    return corner[d];
}

double Cell::getWidth(unsigned int d) {
    return width[d];
}

void Cell::setCorner(unsigned int d, double val) {
    corner[d] = val;
}

void Cell::setWidth(unsigned int d, double val) {
    width[d] = val;
}

// Checks whether a point lies in a cell
bool Cell::containsPoint(double point[])
{
    for(int d = 0; d < dimension; d++) {
        if(corner[d] - width[d] > point[d]) return false;
        if(corner[d] + width[d] < point[d]) return false;
    }
    return true;
}


// Default constructor for ArraySPTree -- build tree, too!
ArraySPTree::ArraySPTree(unsigned int D, double* inp_data, unsigned int N, unsigned int T,
	       int* ts_assignments)
{

    int* pts_per_ts = (int*) calloc(T, sizeof(int));
    // Compute mean, width, and height of current map (boundaries of ArraySPTree)
    int nD = 0;
    double* mean_Y = (double*) calloc(D,  sizeof(double));
    double*  min_Y = (double*) malloc(D * sizeof(double)); for(unsigned int d = 0; d < D; d++)  min_Y[d] =  DBL_MAX;
    double*  max_Y = (double*) malloc(D * sizeof(double)); for(unsigned int d = 0; d < D; d++)  max_Y[d] = -DBL_MAX;
    int t; 
    for(unsigned int n = 0; n < N; n++) {
        t = inp_data[n];
	pts_per_ts[t]++; 
        for(unsigned int d = 0; d < D; d++) {
            mean_Y[d] += inp_data[n * D + d];
            if(inp_data[nD + d] < min_Y[d]) min_Y[d] = inp_data[nD + d];
            if(inp_data[nD + d] > max_Y[d]) max_Y[d] = inp_data[nD + d];
        }
        nD += D;
    }

    Nt = pts_per_ts; 
    
    for(int d = 0; d < D; d++) mean_Y[d] /= (double) N;
    
    // Construct ArraySPTree
    double* width = (double*) malloc(D * sizeof(double));
    for(int d = 0; d < D; d++) width[d] = fmax(max_Y[d] - mean_Y[d], mean_Y[d] - min_Y[d]) + 1e-5;
    init(NULL, D, inp_data, mean_Y, width, T, ts_assignments);

    direct_ptr = (ArraySPTree **) calloc(N, sizeof(ArraySPTree*));
    
    fill(N);
    
    // Clean up memory
    free(mean_Y);
    free(max_Y);
    free(min_Y);
    free(width);
}


// Constructor for ArraySPTree with particular size and parent -- build the tree, too!
ArraySPTree::ArraySPTree(unsigned int D, double* inp_data, unsigned int N, double* inp_corner, double* inp_width, unsigned int T, int* ts_assignments, int* input_Nt)
{
    Nt = input_Nt;
    init(NULL, D, inp_data, inp_corner, inp_width, T, ts_assignments);
    fill(N);
}


// Constructor for ArraySPTree with particular size (do not fill the tree)
ArraySPTree::ArraySPTree(unsigned int D, double* inp_data, double* inp_corner, double* inp_width, unsigned int T, int* ts_assignments, int* input_Nt)
{
  Nt = input_Nt;
    init(NULL, D, inp_data, inp_corner, inp_width, T, ts_assignments);
}


// Constructor for ArraySPTree with particular size and parent (do not fill tree)
ArraySPTree::ArraySPTree(ArraySPTree* inp_parent, unsigned int D, double* inp_data, double* inp_corner, double* inp_width, unsigned int T, int* ts_assignments, int* input_Nt) {
  Nt = input_Nt;
    init(inp_parent, D, inp_data, inp_corner, inp_width, T, ts_assignments);
}


// Constructor for ArraySPTree with particular size and parent -- build the tree, too!
ArraySPTree::ArraySPTree(ArraySPTree* inp_parent, unsigned int D, double* inp_data, unsigned int N, double* inp_corner, double* inp_width, unsigned int T, int* ts_assignments, int* input_Nt)
{
  Nt = input_Nt;
    init(inp_parent, D, inp_data, inp_corner, inp_width, T, ts_assignments);
    fill(N);
}


// Main initialization function
void ArraySPTree::init(ArraySPTree* inp_parent, unsigned int D, double* inp_data, double* inp_corner,
		  double* inp_width, unsigned int T, int* ts_assignments)
{
    time_steps = T;
    time_assignments = ts_assignments;

    parent = inp_parent;
    dimension = D;    
    no_children = 2;
    for(unsigned int d = 1; d < D; d++) no_children *= 2;
    data = inp_data;
    is_leaf = true;
    size = 0;
    cum_size = (unsigned int *) calloc(time_steps, sizeof(int));
    // cum_size is the size in each box (not just at leaves)

    if (parent != NULL) {
      direct_ptr = parent->getDirectPtrArray();
    } else {
      direct_ptr = NULL;
    }
    
    boundary = new Cell(dimension);
    for(unsigned int d = 0; d < D; d++) boundary->setCorner(d, inp_corner[d]);
    for(unsigned int d = 0; d < D; d++) boundary->setWidth( d, inp_width[d]);
    
    children = (ArraySPTree**) malloc(no_children * sizeof(ArraySPTree*));
    for(unsigned int i = 0; i < no_children; i++) children[i] = NULL;

    center_of_mass = (double*) malloc(time_steps*D * sizeof(double));
    for(int t = 0; t < time_steps; t++) { 
      for(unsigned int d = 0; d < D; d++) center_of_mass[t*D+d] = .0;
    }
    
    buff = (double*) malloc(D * sizeof(double));
}


// Destructor for ArraySPTree
ArraySPTree::~ArraySPTree()
{
    for(unsigned int i = 0; i < no_children; i++) {
        if(children[i] != NULL) delete children[i];
    }
    free(children);
    free(center_of_mass);
    free(buff);
    if (parent == NULL && direct_ptr != NULL) free(direct_ptr);
    delete boundary;
}


// Update the data underlying this tree
void ArraySPTree::setData(double* inp_data)
{
    data = inp_data;
}


// Get the parent of the current tree
ArraySPTree* ArraySPTree::getParent()
{
    return parent;
}

ArraySPTree** ArraySPTree::getDirectPtrArray()
{
    return direct_ptr;
}


// TODO: FINISH UPDATING CoM FOR EACH TIME STEP

// Insert a point into the ArraySPTree
ArraySPTree* ArraySPTree::insert(unsigned int new_index)
{
    int pt_t = time_assignments[new_index]; 
    // Ignore objects which do not belong in this quad tree
    double* point = data + new_index * dimension;
    
    if(!boundary->containsPoint(point))
        return NULL;

    int new_ts = time_assignments[new_index];
    
    // Online update of cumulative size (for given time step) and center-of-mass
    cum_size[pt_t]++;

    double mult1 = (double) (cum_size[pt_t] - 1) / (double) cum_size[pt_t];
    double mult2 = 1.0 / (double) cum_size[pt_t];
    
    for(unsigned int d = 0; d < dimension; d++) center_of_mass[pt_t*dimension+d] *= mult1;
    for(unsigned int d = 0; d < dimension; d++) center_of_mass[pt_t*dimension+d] += mult2 * point[d];
    
    // If there is space in this quad tree and it is a leaf, add the object here
    if(is_leaf && size < QT_NODE_CAPACITY) {
        index[size] = new_index;
        size++;
        return this;
    }
    
    // Don't add duplicates for now (this is not very nice)
    bool any_duplicate = false;
    for(unsigned int n = 0; n < size; n++) {
        bool duplicate = true;
        for(unsigned int d = 0; d < dimension; d++) {
            if(point[d] != data[index[n] * dimension + d]) { duplicate = false; break; }
        }
        any_duplicate = any_duplicate | duplicate;
    }
    if(any_duplicate) {
      cout << "DUPLICATE?" << endl;
      return this;
    }
    
    // Otherwise, we need to subdivide the current cell
    if(is_leaf) subdivide();
    
    // Find out where the point can be inserted
    for(unsigned int i = 0; i < no_children; i++) {
      ArraySPTree* ret = children[i]->insert(new_index);
      if (ret != NULL) return ret;
    }
    
    // Otherwise, the point cannot be inserted (this should never happen)
    return NULL;
}

    
// Create four children which fully divide this cell into four quads of equal area
void ArraySPTree::subdivide() {
    
    // Create new children
    double* new_corner = (double*) malloc(dimension * sizeof(double));
    double* new_width  = (double*) malloc(dimension * sizeof(double));
    for(unsigned int i = 0; i < no_children; i++) {
        unsigned int div = 1;
        for(unsigned int d = 0; d < dimension; d++) {
            new_width[d] = .5 * boundary->getWidth(d);
            if((i / div) % 2 == 1) new_corner[d] = boundary->getCorner(d) - .5 * boundary->getWidth(d);
            else                   new_corner[d] = boundary->getCorner(d) + .5 * boundary->getWidth(d);
            div *= 2;
        }
        children[i] = new ArraySPTree(this, dimension, data, new_corner, new_width, time_steps, time_assignments, Nt);
    }
    free(new_corner);
    free(new_width);
    
    // Move existing points to correct children
    for(unsigned int i = 0; i < size; i++) {
        bool success = false;
        for(unsigned int j = 0; j < no_children; j++) {
          ArraySPTree* ret = children[j]->insert(index[i]);
          if (ret != NULL) {
            direct_ptr[index[i]] = ret;
            break;
          }
        }
        index[i] = -1;
    }
    
    // Empty parent node
    size = 0;
    // cout << "is leaf is false!";
    is_leaf = false;
}

void ArraySPTree::modifyWeight(int delta) {
  cum_size += delta;
}

void ArraySPTree::modifyWeight(unsigned int index, int delta) {
  ArraySPTree* node = direct_ptr[index];
  while (node != NULL) {
    node->modifyWeight(delta);
    node = node->getParent();
  }
}

// Build ArraySPTree on dataset
void ArraySPTree::fill(unsigned int N)
{
    for(unsigned int i = 0; i < N; i++) {
      direct_ptr[i] = insert(i);
    }
}

// Checks whether the specified tree is correct
bool ArraySPTree::isCorrect()
{
    for(unsigned int n = 0; n < size; n++) {
        double* point = data + index[n] * dimension;
        if(!boundary->containsPoint(point)) return false;
    }
    if(!is_leaf) {
        bool correct = true;
        for(int i = 0; i < no_children; i++) correct = correct && children[i]->isCorrect();
        return correct;
    }
    else return true;
}



// Build a list of all indices in ArraySPTree
void ArraySPTree::getAllIndices(unsigned int* indices)
{
    getAllIndices(indices, 0);
}


// Build a list of all indices in ArraySPTree
unsigned int ArraySPTree::getAllIndices(unsigned int* indices, unsigned int loc)
{
    
    // Gather indices in current quadrant
    for(unsigned int i = 0; i < size; i++) indices[loc + i] = index[i];
    loc += size;
    
    // Gather indices in children
    if(!is_leaf) {
        for(int i = 0; i < no_children; i++) loc = children[i]->getAllIndices(indices, loc);
    }
    return loc;
}


unsigned int ArraySPTree::getDepth() {
    if(is_leaf) return 1;
    int depth = 0;
    for(unsigned int i = 0; i < no_children; i++) depth = fmax(depth, children[i]->getDepth());
    return 1 + depth;
}


// Compute non-edge forces using Barnes-Hut algorithm
void ArraySPTree::computeNonEdgeForces(unsigned int point_index, double theta, double neg_f[],
				       double* sum_Q, int T_offset, int time_step)
{
    // Record the time step of the point in question
    // int time_step = time_assignments[point_index]; 
    //int time_start = min((double)(time_step-T_offset), 0.0);
    //int time_finish = max((double)(time_step+T_offset), (double)(time_steps-1)); 
    int time_start = (time_step - T_offset > 0) ? (time_step - T_offset) : 0;
    int time_finish = (time_step + T_offset > time_steps-1) ?
	(time_steps - 1) : (time_step + T_offset);
    
    // Make sure that we spend no time on empty nodes or self-interactions
    int total_size = 0;
    for(int t=time_start; t<=time_finish; t++) {
	total_size += cum_size[t];
    }
    if(total_size == 0 || (is_leaf && size == 1 && index[0] == point_index)) return;

    // compute Bounding Box to see if this is a good summary point (in the forthcoming loop)
    double D;
    
    unsigned int ind = point_index*dimension;
    double max_width = 0.0;
    double cur_width;
    for(unsigned int d = 0; d < dimension; d++) {
	cur_width = boundary->getWidth(d);
	max_width = (max_width > cur_width) ? max_width : cur_width;
    }
    
    // Compute distance between point and center-of-masses
    for(int t=time_start; t<=time_finish; t++) {
      // cout << "Time = " << t << ";" << endl; 
	    // if there are no points from this time step skip to next time step:
	    if(cum_size[t] == 0) {
		continue; 
	    }

	    D = .0;
	    unsigned int ind = point_index * dimension;
	    for(unsigned int d = 0; d < dimension; d++) {
		buff[d] = data[ind + d] - center_of_mass[t*dimension+d];
	    }
	    
	    for(unsigned int d = 0; d < dimension; d++) D += buff[d] * buff[d];

	    if((cum_size[t] == 1) && (D < EPS)) {
	      cout << "This box has only the target point!" << endl; 
	      continue; // The point in this quadrant is our pt
	    } 

	    /*
	    double max_width = 0.0;
	    double cur_width;
	    for(unsigned int d = 0; d < dimension; d++) {
		cur_width = boundary->getWidth(d);
		max_width = (max_width > cur_width) ? max_width : cur_width;
	    }
	    */
	    
	    // Check whether we can use this node as a "summary"
	    if(is_leaf || cum_size[t] == 1 || max_width / sqrt(D) < theta) {

	      // PRINT DEBUGGING!
	      cout << (is_leaf ? "Leaf":"CoM") << ", (" << center_of_mass[t*dimension] << ", "
		   << center_of_mass[t*dimension+1] << "), t: " << t << endl;
	      // DONE PRINTING
	      
		// Compute and add t-SNE force between point and current node
		D = 1.0 / (1.0 + D);
		double mult = cum_size[t] * D;
		*sum_Q += mult;
		mult *= D;
		for(unsigned int d = 0; d < dimension; d++) neg_f[d] += mult * buff[d];
	    }
	    else {
	      // DEBUGGING!
	      cout << "Cannot be summary: (" << center_of_mass[t*dimension] << ", "
		   << center_of_mass[t*dimension+1] << "); " << (max_width / sqrt(D))
		   << "; t: " << t << endl;
	      
		// Recursively apply Barnes-Hut to children
		for(unsigned int i = 0; i < no_children; i++) {
		  children[i]->computeNonEdgeForces(point_index,theta, neg_f, sum_Q, 0, t);
		}
	    }
	}
}

// Computes edge forces
void ArraySPTree::computeEdgeForces(unsigned int* row_P, unsigned int* col_P, double* val_P, int N, double* pos_f)
{
    
    // Loop over all edges in the graph
    unsigned int ind1 = 0;
    unsigned int ind2 = 0;
    double D;
    ivec rand_indices;
    for(unsigned int n = 0; n < N; n++) {
        
      
      /*
        int nsample = 10;
        unsigned int nnei = row_P[n+1] - row_P[n];
        if (nnei < nsample) nsample = nnei;

        rand_indices.set_size(nnei); 
        for (int i = 0; i < nnei; i++) rand_indices(i) = i + row_P[n];
        rand_indices = shuffle(rand_indices);

        //printf("Selecting %llu out of %llu\n", rand_indices.n_elem, row_P[n+1]-row_P[n]);
        for (int i = 0; i < nsample; i++) {
            D = 1.0;
            unsigned int i1 = n * dimension;
            unsigned int i2 = col_P[rand_indices(i)] * dimension;
            for(unsigned int d = 0; d < dimension; d++) buff[d] = data[i1 + d] - data[i2 + d];
            for(unsigned int d = 0; d < dimension; d++) D += buff[d] * buff[d];
            D = val_P[rand_indices(i)] / D;
            
            double correction = nnei / ((double) nsample);
            // Sum positive force
            for(unsigned int d = 0; d < dimension; d++) pos_f[i1 + d] += D * buff[d] * correction;
        }

        printf("Correction %f* ", nnei/((double) nsample));
        for(unsigned int d = 0; d < dimension; d++) {
          printf("%f ", pos_f[ind1 + d]);
          pos_f[ind1 + d] = 0;
        }
        printf("\n");
        */
        
        for(unsigned int i = row_P[n]; i < row_P[n + 1]; i++) {
            // Compute pairwise distance and Q-value
            D = 1.0;
            ind2 = col_P[i] * dimension;
            for(unsigned int d = 0; d < dimension; d++) buff[d] = data[ind1 + d] - data[ind2 + d];
            for(unsigned int d = 0; d < dimension; d++) D += buff[d] * buff[d];
            D = val_P[i] / D;
            
            // Sum positive force
            for(unsigned int d = 0; d < dimension; d++) pos_f[ind1 + d] += D * buff[d];
        }
        //for(unsigned int d = 0; d < dimension; d++) printf("%f ", pos_f[ind1 + d]);
        //printf("\n");
        ind1 += dimension;
        
    }
}

void ArraySPTree::print() {
  print(0); 
} 

// Print out tree
void ArraySPTree::print(int spaces) 
{
  for(int i=0; i<spaces; i++) {
    printf(" "); 
  } 
  int total_size = 0;
  for (int t=0; t < time_steps; t++) {
    total_size += cum_size[t]; 
  }
  if(total_size == 0) {
    printf("Empty node\n");
    return;
  }

    if(is_leaf) {
      cout << "Leaf node; size = " << size << " ; "; 
        printf("Leaf node; data = [");
        for(int i = 0; i < size; i++) {
            double* point = data + index[i] * dimension;
            for(int d = 0; d < dimension; d++) printf("%f, ", point[d]);
            printf(" (index = %d)", index[i]);
	    printf(" (t = %d)", time_assignments[index[i]]);
            if(i < size - 1) printf("\n");
            else printf("]\n");
        }        
    }
    else {
        printf("Intersection node with centers-of-mass = [");
	for(int t=0; t < time_steps; t++) {
	  printf("("); 
	  for(int d = 0; d < dimension; d++) printf("%f, ", center_of_mass[t*dimension + d]);
	  printf(")"); 
	}
        printf("]; children are:\n");
        for(int i = 0; i < no_children; i++) children[i]->print(spaces+4);
    }
}

