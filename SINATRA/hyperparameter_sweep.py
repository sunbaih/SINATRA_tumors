from directions import *
from euler import *
from gp import *
from reconstruction import *
from mesh import *
from mesh_processing import *
import argparse, os


parser = argparse.ArgumentParser(description='SINATRA Pro')
args = parser.parse_args()

directory = "/users/bsun14/data/bsun14"

def get_first_file_path(folder_path):
    files = os.listdir(folder_path)
    for file in files:
        if os.path.isfile(os.path.join(folder_path, file)):
            return os.path.join(folder_path, file)
    return None

def sinatra(c, d, theta, l):

    n_sample = 19
    sm_radius = 1
    n_cone =  c
    n_direction_per_cone = d
    cap_radius = theta
    n_filtration = l
    ec_type = "DECT"
    verbose = False
    func = "linear"
    parallel = False
    label_type = "continuous"
    input_folder = "%s/BRATS_nifti_test"%(directory)
    out_directory = "%s/BRATS_nifti_out/c%d_d%d_theta%d_l%d"%(directory, c, d, theta, l)
    labels = np.loadtxt("expimap_latent.txt")[:, 9]
    first_file = get_first_file_path(input_folder)

    if not os.path.exists(out_directory):
        os.mkdir(out_directory)

    directions = generate_equidistributed_cones(n_cone=n_cone, n_direction_per_cone=n_direction_per_cone,
                                            cap_radius= cap_radius, hemisphere=False)  
    
    np.savetxt("%s/directions.txt"%(out_directory),directions)

    X, y, not_vacuum = compute_ec_curve_folder(input_folder, labels, out_directory,
                                            directions = directions,
                                            n_sample=n_sample, ec_type="SECT",
                                            n_filtration=n_filtration, sm_radius=sm_radius,
                                            parallel=parallel, n_core=-1, verbose=verbose)
    
    np.savetxt("%s/ECT.txt"%(out_directory),X)
    np.savetxt("%s/notvacuum.txt"%(out_directory),not_vacuum)
    np.savetxt('%s/labels.txt'%(out_directory),y)

    kld, rates, delta, eff_samp_size, samples = calc_rate(X,y, func= func, bandwidth= 0.01, n_mcmc= 100000,low_rank=False, parallel=parallel,
                                             n_core=-1, verbose=verbose)
    
    np.savetxt("%s/rates.txt"%(out_directory), rates)

    rates = project_rate_on_nonvacuum(rates, not_vacuum)

    np.savetxt("%s/all_rates.txt"%(out_directory),rates)

    reconstructed_verts = compute_selected_vertices_cones(directions, first_file, rates, n_filtration, threshold=1e-10, cone_size=n_direction_per_cone, ball=True, ball_radius=1.0, radius=1)
    np.savetxt("%s/reconstructed_verts.txt"%(out_directory), reconstructed_verts)

    return samples, len(reconstructed_verts)

def tune_param(defaults_list, param, param_list):
    n_cone = defaults_list[0]
    n_direction_per_cone = defaults_list[1]
    cap_radius = defaults_list[2]
    n_filtration = defaults_list[3]

    num_verts_array = np.zeros(len(param_list))
    avg_likelihood = np.zeros(len(param_list))

    if param=="c": 
        for i in range(len(param_list)):
            c = param_list[i] 
            samples, len_reconstructed_verts = sinatra(c, n_direction_per_cone, cap_radius, n_filtration)
            avg_likelihood[i] += np.mean(samples)
            num_verts_array[i] += len_reconstructed_verts
    if param=="d": 
        for i in range(len(param_list)):
            d = param_list[i] 
            samples, len_reconstructed_verts = sinatra(n_cone, d, cap_radius, n_filtration)
            avg_likelihood[i] += np.mean(samples)
            num_verts_array[i] += len_reconstructed_verts
    if param=="theta": 
        for i in range(len(param_list)):
            theta = param_list[i] 
            samples, len_reconstructed_verts = sinatra(n_cone, n_direction_per_cone, theta, n_filtration)
            avg_likelihood[i] += np.mean(samples)
            num_verts_array[i] += len_reconstructed_verts
    if param=="l": 
        for i in range(len(param_list)):
            l = param_list[i] 
            samples, len_reconstructed_verts = sinatra(n_cone, n_direction_per_cone, cap_radius, l)
            avg_likelihood[i] += np.mean(samples)
            num_verts_array[i] += len_reconstructed_verts

    print(num_verts_array)
    
    print(avg_likelihood)
    best_param = param_list[np.argmax(avg_likelihood)]
    print("The %s with the highest average likelihood is:"%(param), best_param)


def tune_params():
    default_list = [25, 5, 0.5, 100]
    c_list = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    d_list = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    theta_list = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    l_list = np.array([20, 40, 60, 80, 100, 120, 140, 160, 180, 200])

    tune_param(default_list, "c", c_list)
    tune_param(default_list, "d", d_list)
    tune_param(default_list, "theta", theta_list)
    tune_param(default_list, "l", l_list)

def main():
    tune_params()
    print("main")

if __name__ == "__main__":
    main()
    print("done")