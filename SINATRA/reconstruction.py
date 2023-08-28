
import numpy as np
from stl import mesh as msh
from euler import extract_mesh_data
from functools import reduce
from mesh import *
import trimesh 
import SimpleITK as sitk
from skimage import measure

"""
Most methods translated to Python from original SINATRA. 
"""

def project_rate_on_nonvacuum(rates,not_vacuum):
    rates_new = np.zeros(not_vacuum.size,dtype=float)
    j = 0
    for i in range(not_vacuum.size):
        if not_vacuum[i]:
            rates_new[i] = rates[j]
            j += 1
    return rates_new

def process_stl_file(stl_file_path):
    stl_mesh = mesh.Mesh.from_file(stl_file_path)
    vertices = stl_mesh.vectors.reshape(-1, 3)
    faces = np.arange(len(vertices)).reshape(-1, 3)
    edges = stl_mesh.edges.reshape(-1, 2)
    mesh_complex = {'Vertices': vertices, 'Edges': edges, 'Faces': faces}
    return mesh_complex

def compute_selected_vertices_cones(directions, mesh_complex, rate_vals, n_filtration=25, threshold=-1, cone_size=1, ball=True, ball_radius=1.0, radius=1):
   
    if threshold == -1:
        threshold = 1/len(rate_vals)
    if (directions.shape[0] % cone_size) != 0:
        print('Number of Cones not a multiple of directions')
        return 0
    coned_vertices = []
    for j in range(1, (directions.shape[0] // cone_size) + 1):
        cone_dirs = directions[((j-1)*(cone_size)): (j*cone_size), ]
        cone_rate_vals = rate_vals[(j-1)*(cone_size*n_filtration): (j*cone_size*n_filtration)]
        coned_vertices.append(summarize_vertices(cone_dirs, mesh_complex, cone_rate_vals, n_filtration, reduction_operation=np.intersect1d, threshold=threshold, cone_size=cone_size, ball=ball, ball_radius=ball_radius, radius=radius))

    total_selected_vertices = set().union(*coned_vertices)
    total_selected_vertices = np.array(list(total_selected_vertices))
    return total_selected_vertices

def highlight_selected_faces(nifti_path, reconstructed_faces, out_directory):
    print(reconstructed_faces.shape)
    vertices, edges, faces = extract_mesh_data(nifti_path)
    colors = np.zeros([len(faces), 3])
    for i in range(len(reconstructed_faces)):
        colors[reconstructed_faces[i]]=[1, 0, 0]
    mymesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mymesh.visual.face_colors = colors
    mymesh.show()
    mymesh.save('colored_model.stl')


def compute_selected_faces_cones(directions, mesh_complex, rate_vals, n_filtration=25, threshold=-1, cone_size=1, ball=True, ball_radius=None, radius=0):
    vertices, edges, faces = extract_mesh_data(mesh_complex)
    if threshold == -1:
        threshold = 1/len(rate_vals)
    if directions.shape[0] % cone_size != 0:
        print('Number of Cones not a multiple of directions')
        return 0
    coned_vertices = []
    for j in range(1, directions.shape[0] // cone_size + 1):
        cone_dirs = directions[(j-1)*cone_size:j*cone_size, :]
        cone_rate_vals = rate_vals[(j-1)*cone_size*n_filtration:j*cone_size*n_filtration]
        coned_vertices.append(summarize_vertices(cone_dirs, mesh_complex, cone_rate_vals, n_filtration, reduction_operation=np.intersect1d, threshold=threshold, cone_size=cone_size, ball=ball, ball_radius=ball_radius, radius=radius))
    total_selected_vertices = reduce(np.union1d, coned_vertices)
    reconstructed_faces = np.apply_along_axis(lambda x: np.any(np.isin(x, total_selected_vertices)), axis=1, arr=faces)
    reconstructed_faces = np.where(reconstructed_faces == True)[0]
    return reconstructed_faces


def summarize_vertices(directions, mesh_complex, rate_vals, n_filtration, reduction_operation=np.intersect1d, threshold=None, cone_size=None, ball=True, ball_radius=1, radius=0):
    vertices, edges, faces = extract_mesh_data(mesh_complex)
    """
    meshTumor= mesh()
    meshTumor.read_mesh_file(mesh_complex)
    vertices = meshTumor.vertices
    """
    picked_indices = np.where(rate_vals>=threshold)[0]
    indices = np.array([])
    
    for j in range(-radius, radius+1):
        indices = np.concatenate((indices, picked_indices+j))
    
    selected_vertices = []
    
    for i in range(directions.shape[0]):
        vtx_projection = np.dot(vertices[:,:3], directions[i,:])
        
        if ball:
            buckets = np.linspace(-ball_radius, ball_radius, num=n_filtration+1)
        else:
            buckets = np.linspace(vtx_projection.min(), vtx_projection.max(), num=n_filtration+1)
        
        projection_bucket = np.digitize(vtx_projection, buckets) 
        
        # update index to reflect rate values
        projection_bucket = projection_bucket + (i - 1)*n_filtration
                
        selected_vertices.append(np.where(np.isin(projection_bucket, indices))[0])
    
    final_selected_vertices = reduce(reduction_operation, selected_vertices)
    print(final_selected_vertices)
    return final_selected_vertices


"""
Haven't tested this one yet.
"""

def reconstruct_vertices_on_shape(directions, mesh_complex, rate_vals, n_filtration, cuts=50, cone_size=None, ball_radius=None, ball=True, radius=0):
    vertices, edges, faces = extract_mesh_data(mesh_complex)
    vert_matrix = np.zeros((vertices.shape[0], 2))
    cut = cuts
    reconstructed_vertices = np.array([], dtype=int)
    #for threshold in np.quantile(rate_vals, np.linspace(1, 0, cuts)):
    for threshold in np.linspace(0.000001, 0, cuts):
        print(threshold)
        if threshold > np.max(rate_vals):
            print(threshold, "threshold too high")
            continue
        else:
            selected_vertices = compute_selected_vertices_cones(directions=directions, mesh_complex=mesh_complex, rate_vals=rate_vals, n_filtration=n_filtration, threshold=threshold,
                                                          cone_size=cone_size, ball_radius=ball_radius, ball=ball, radius=radius)
            selected_vertices = np.setdiff1d(selected_vertices, reconstructed_vertices)
            print(selected_vertices)
            if len(selected_vertices)==0:
                print("no selected vertices")
                pass
            else: 
                vert_matrix[selected_vertices, 0] = cut
                vert_matrix[selected_vertices, 1] = threshold
            cut = cut - 1
            reconstructed_vertices = np.concatenate((reconstructed_vertices, selected_vertices))
            print(threshold, len(reconstructed_vertices))
        if len(reconstructed_vertices) == vertices.shape[0]:
            print(cut, threshold)
            break
    return vert_matrix