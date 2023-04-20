from directions import *
from euler import *
from gp import *
from reconstruction import *
from mesh import *
from mesh_processing import *
import argparse, os


parser = argparse.ArgumentParser(description='SINATRA Pro')
args = parser.parse_args()

directory = "/Users/baihesun/cancer_data"
directory = "/users/bsun14"


n_sample = 19
sm_radius = 1 #r
n_cone =  20 #c
n_direction_per_cone = 5 #d
cap_radius = 0.8 #theta
n_filtration = 100 #l
ec_type = "DECT"
verbose = True
func = "linear"
parallel = False
label_type = "continuous"
input_folder = "%s/BRATS_nifti_test"%(directory)
out_directory = "%s/BRATS_nifti_out"%(directory)
labels_data = "%s/data_labels.txt"%(directory)

def get_first_file_path(folder_path):
    files = os.listdir(folder_path)
    for file in files:
        if os.path.isfile(os.path.join(folder_path, file)):
            return os.path.join(folder_path, file)
    return None

first_file = get_first_file_path(input_folder)

if not os.path.exists(out_directory):
    os.mkdir(out_directory)

"""
directions = generate_equidistributed_cones(n_cone=n_cone, n_direction_per_cone=n_direction_per_cone,
                                            cap_radius= cap_radius, hemisphere=False)

np.savetxt("%s/directions.txt"%(out_directory),directions)
"""

"""
X, y, not_vacuum = compute_ec_curve_folder(input_folder, labels_data, out_directory,
                                           directions = directions,
                                           n_sample=n_sample, ec_type="SECT",
                                           n_filtration=n_filtration, sm_radius=sm_radius,
                                           parallel=parallel, n_core=-1, verbose=verbose)

np.savetxt("%s/ECT.txt"%(out_directory),X)
np.savetxt("%s/notvacuum.txt"%(out_directory),not_vacuum)
np.savetxt('%s/labels.txt'%(out_directory),y)
"""


directions = np.loadtxt("%s/directions.txt"%(out_directory))
y = np.loadtxt(labels_data)
X = np.loadtxt("%s/ECT.txt"%(out_directory))
not_vacuum = np.loadtxt("%s/notvacuum.txt"%(out_directory))
rates = np.loadtxt("%s/rates.txt"%(out_directory))

kld, rates, delta, eff_samp_size = calc_rate(X,y, func= func, bandwidth= 0.01, n_mcmc= 100000,low_rank=False, parallel=parallel,
                                             n_core=-1, verbose=verbose)


np.savetxt("%s/rates.txt"%(out_directory),rates)


"""
directions = np.loadtxt("/Users/baihesun/trial_19/directions_20_8_0.80.txt")
y = np.loadtxt(labels_data)
X = np.loadtxt("%s/ECT.txt"%(out_directory))
not_vacuum = np.loadtxt("/Users/baihesun/trial_19/notvacuum_DECT_perturbed_xyz_None_1.0_20_8_0.80_120_norm_all.txt")
rates = np.loadtxt("/Users/baihesun/trial_19/rate_DECT_perturbed_xyz_None_1.0_20_8_0.80_120.txt")
first_file = "/Users/baihesun/trial_19/msh/perturbed_xyz_1.0/perturbed_xyz_frame0.msh"

rates = project_rate_on_nonvacuum(rates, not_vacuum)
np.savetxt("%s/all_rates.txt"%(out_directory),rates)
"""

reconstructed_faces = compute_selected_faces_cones(directions, first_file, rates, n_filtration, threshold=-1, cone_size=n_direction_per_cone, ball=True, ball_radius=1.0, radius=1)

    
np.savetxt("%s/reconstructed_faces.txt"%(out_directory),reconstructed_faces)
print(reconstructed_faces)

highlight_selected_faces(first_file, reconstructed_faces, out_directory)

print("SINATRA Pro calculation completed.")

def main():
    print("main")

if __name__ == "__main__":
    main()
    print("done")
