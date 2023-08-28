from directions import *
from euler import *
from gp import *
from reconstruction import *
from mesh import *
from mesh_processing import *
import argparse, os


parser = argparse.ArgumentParser(description='SINATRA Pro')
args = parser.parse_args()

#directory = "/Users/baihesun/cancer_data"
directory = "/users/bsun14/data/bsun14"


n_sample = 19
sm_radius = 1 #r
n_cone =  20 #c
n_direction_per_cone = 5 #d
cap_radius = 0.8 #theta
n_filtration = 100 #l
ec_type = "DECT"
verbose = False
func = "linear"
parallel = False
label_type = "continuous"
input_folder = "%s/BRATS_nifti_test"%(directory)
out_directory = "%s/BRATS_nifti_out"%(directory)
labels = np.zeros(19)

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


X, y, not_vacuum = compute_ec_curve_folder(input_folder, labels_data, out_directory,
                                           directions = directions,
                                           n_sample=n_sample, ec_type="SECT",
                                           n_filtration=n_filtration, sm_radius=sm_radius,
                                           parallel=parallel, n_core=-1, verbose=verbose)

np.savetxt("%s/ECT.txt"%(out_directory),X)
np.savetxt("%s/notvacuum.txt"%(out_directory),not_vacuum)
np.savetxt('%s/labels.txt'%(out_directory),y)


directions = np.loadtxt("%s/directions.txt"%(out_directory))
y = np.loadtxt(labels_data)
X = np.loadtxt("%s/ECT.txt"%(out_directory))
not_vacuum = np.loadtxt("%s/notvacuum.txt"%(out_directory))
rates = np.loadtxt("%s/rates.txt"%(out_directory))


kld, rates, delta, eff_samp_size = calc_rate(X,y, func= func, bandwidth= 0.01, n_mcmc= 100000,low_rank=False, parallel=parallel,
                                             n_core=-1, verbose=verbose)


np.savetxt("%s/rates.txt"%(out_directory),rates)
"""
"""
directions = generate_equidistributed_cones(n_cone=n_cone, n_direction_per_cone=n_direction_per_cone,
                                            cap_radius= cap_radius, hemisphere=False)

#np.savetxt("%s/directions.txt"%(out_directory),directions)
#directions = np.loadtxt("%s/directions.txt"%(out_directory))
X, y, not_vacuum = compute_ec_curve_folder(input_folder, labels, out_directory,
                                           directions = directions,
                                           n_sample=n_sample, ec_type="SECT",
                                           n_filtration=n_filtration, sm_radius=sm_radius,
                                           parallel=parallel, n_core=-1, verbose=verbose)
                                           """

def run_sinatra_given_y_gp(y, gp):
    directions = np.loadtxt("%s/directions.txt"%(out_directory))
    X = np.loadtxt("%s/ECT.txt"%(out_directory))
    not_vacuum = np.loadtxt("%s/notvacuum.txt"%(out_directory))
    kld, rates, delta, eff_samp_size = calc_rate(X,y, func= func, bandwidth= 0.01, n_mcmc= 100000,low_rank=False, parallel=parallel,
                                             n_core=-1, verbose=verbose)
    np.savetxt("%s/rates_%s.txt"%(out_directory, gp), rates)
    rates = project_rate_on_nonvacuum(rates, not_vacuum)
    np.savetxt("%s/all_rates_%s.txt"%(out_directory, gp),rates)
    reconstructed_verts = compute_selected_vertices_cones(directions, first_file, rates, n_filtration, threshold=1e-10, cone_size=n_direction_per_cone, ball=True, ball_radius=1.0, radius=1)
    return reconstructed_verts


def sinatra_gps():
    gps = np.genfromtxt("/users/bsun14/data/bsun14/gene_programs.txt", dtype = "str")
    ys = np.genfromtxt("expimap_latent.txt")
    significant_gps = []
    significant_vertices = []
    for i in range(len(gps)):
        gp = gps[i]
        print(gp)
        y = ys[:, i]
        reconstructed_verts = run_sinatra_given_y_gp(y, gp)
        if len(reconstructed_verts)>0:
            print("signficant:", gp)
            significant_gps.append(gps)
            significant_vertices.append(reconstructed_verts)
    verts_array = np.array([np.array(row) for row in reconstructed_verts])
    gps_array = np.array([np.array(row) for row in significant_gps], dtype='object')
    np.savetxt("%s/significant_gps.txt"%(out_directory), gps_array, fmt='%s')
    np.savetxt("%s/significant_verts.txt"%(out_directory), verts_array)



print("SINATRA Pro calculation completed.")

def main():
    print("main")

if __name__ == "__main__":
    main()
    sinatra_gps()
    print("done")
