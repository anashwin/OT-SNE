import numpy as np
import numpy.random as random
import sys

def linear_update(mean, covariance, t, state):
    if t == 0:
        state = random.normal(0.,1.,size=mean.shape)
        # state = np.array([[.5,.5],[-.5,-.5]])
        
    mean = mean + state
    return mean, state

def format_cell_frame_mixed(full_data_frame, G, D, time_steps):
    output = 'id'
    for d in range(D):
        output += '\tgene{}'.format(d)
    output += '\n'

    timestamps_out = 'id\tday\n'

    date_pairs_out = ''
    
    for t in range(time_steps):
        data_frame = full_data_frame[t]
        a_d_f = np.vstack(data_frame)
        random.shuffle(a_d_f)
        # print(t)
        # print(appended_data_frame)

        if t < time_steps-1:
            date_pairs_out += '{}\t{}\n'.format(t,t+1)
        
        for idx in range(a_d_f.shape[0]):
            cell_id = 'c_t{}_g{}'.format(int(a_d_f[idx,0]), int(a_d_f[idx,1]))
            output += cell_id
            timestamps_out += cell_id
            timestamps_out += '\t{}'.format(int(a_d_f[idx,0]))
            for d in range(D):
                output+= '\t{:.4f}'.format(a_d_f[idx,2+d])
            output += '\n'
            timestamps_out += '\n'
            
    output = output[:-1]
    timestamps_out = timestamps_out[:-1]
    date_pairs_out = date_pairs_out[:-1]
    
    print(output)
    print(timestamps_out)
    print(date_pairs_out)
    return output, timestamps_out, date_pairs_out

def simulate_gaussians(mean, covariance, N_pts, time_steps, trajectory=None):

    full_data_frame = []

    state = None
    
    for t in range(time_steps):
        data_frame = []
        for g in range(mean.shape[0]):
            N = N_pts + int(random.normal(0, .1*N_pts))# (Plus some Gaussian noise?)
            g_frame = random.multivariate_normal(mean[g,:], covariance[g,:,:],
                                                 size=N)
            full_g_frame = np.hstack((t*np.ones((N,1)), g*np.ones((N,1)), g_frame))
            
            data_frame.append(full_g_frame)
            
        full_data_frame.append(data_frame)

        # update t!
        
        mean, state = trajectory(mean, covariance, t, state)
        
        # print(mean)

    return full_data_frame

G = 4
D = 10

# mean = np.array([[.5, .5], [-.5,-.5]])
mean = random.randn(G,D)
cov = np.array([np.eye(D,D) for _ in range(G)])
print(cov)

N_pts = 10
time_steps = 3
trajectory = linear_update

gene_matrix_outfile = 'matrix.txt'
cell_timestamps_outfile = 'timestamps.txt'
date_pairs_outfile = 'date_pairs.txt'

# if len(sys.argv) > 1:

full_data_frame = simulate_gaussians(mean, cov, N_pts, time_steps, trajectory)

# for t in range(time_steps):
#     for g in range(2):
#         print(t,g)
#         print(full_data_frame[t][g])

gene_matrix, cell_timestamps, date_pairs = format_cell_frame_mixed(full_data_frame, G, D, time_steps)

with open(gene_matrix_outfile, 'w') as f:
    f.write(gene_matrix)

with open(cell_timestamps_outfile, 'w') as f:
    f.write(cell_timestamps)

with open(date_pairs_outfile, 'w') as f:
    f.write(date_pairs)
