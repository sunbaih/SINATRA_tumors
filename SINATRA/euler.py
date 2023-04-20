#!/bin/python3

import os, sys
#from sinatra_pro.mesh import *
from mesh import *
from fast_histogram import histogram1d
import multiprocessing
from joblib import Parallel, delayed
import SimpleITK as sitk
import numpy as np
from skimage import measure
from stl import mesh
import pandas as pd 
from collections import defaultdict
from scipy.interpolate import interp1d
from itertools import combinations



def fibi_vecs(num_vecs = 100, r = 1):
    g_ratio = (np.sqrt(5) + 1)/2
    phi_i = np.array([2*np.pi*(i/g_ratio) for i in range(num_vecs)])
    z_i = np.arccos([1-(2*i/num_vecs) for i in range(num_vecs)])
    x = r*np.sin(z_i)*np.cos(phi_i)
    y = r*np.sin(phi_i)*np.sin(z_i)
    z = r*np.cos(z_i)
    return np.vstack((x,y,z)).T

def accumulate(iterator):
    total = 0
    for item in iterator:
        total += item
        yield total

def create_filtration(point_cloud, vector_set):
    to_return = np.empty((np.shape(vector_set)[0],np.shape(point_cloud)[0]))
    for i,vector in enumerate(vector_set):
        #project points onto vector
        projections = [np.dot(pc, vector)/np.linalg.norm(vector) for pc in point_cloud]
        to_return[i,:] = np.argsort(projections)[::-1].astype(int)
    return to_return

def extract_edges(tris):
    # Get all the edges in a triangulation
    edge_set = set()
    for tri in tris:
        edges = combinations(tri, 2)
        for edge in edges:
            if edge not in edge_set and edge[::-1] not in edge_set:
                edge_set.add(edge)
    edge_set = np.array(list(edge_set))
    return edge_set

def SECT(filename, d = 1, res = 10000):

    
    verts, faces, normals, values = measure.marching_cubes(sitk.GetArrayViewFromImage(sitk.ReadImage(filename)), 0)
    
    edges = extract_edges(faces)
    
    node_to_edges = defaultdict(list)
    for edge in edges:
        for node in edge:
            node_to_edges[node].append(edge)

    rand_vecs = fibi_vecs(d + 2)[1:-1]

    filtration = create_filtration(verts, rand_vecs).astype(int)

    xnew = np.linspace(0, 1, res)
    
    to_return = np.empty(d*res)

    node_to_tri = defaultdict(list)
    for tri in faces:
        for node in tri:
            node_to_tri[node].append(tri)
            
    
    for k, filt in enumerate(filtration):
        

        projected = (np.dot(verts[filt,:], rand_vecs[k]) * np.full((len(filt),3), rand_vecs[k]).T).T

        unnormed_height = np.array(list(accumulate([0, 0] + [np.linalg.norm(projected[i+1] - projected[i]) for i in range(projected.shape[0] - 1)])))

        height = unnormed_height / unnormed_height[-1]
        
        EC = np.empty((2, len(filt) + 1))
        
        EC[1, 0] = 0
        
        EC[0, :] = height

        visited_edges = set()
        visited_tri = set()
        visited_nodes = set()

        for i, node in enumerate(filt):
            ECT = EC[1, i] + 1

            visited_nodes.add(node)

            #Edges
            connected_edges = [edge for edge in filter(lambda x: all([j in visited_nodes for j in x]), node_to_edges[node])]
            ECT -= len(connected_edges)

            #Triangles
            assoc_tri = [tri for tri in filter(lambda x: all([j in visited_nodes for j in x]), node_to_tri[node])]
            ECT += len(assoc_tri)
            
            #To Return
            EC[1, i+1] = ECT
        

            # Take mean and Prep for Integration
        ys = EC[1, 1:] - np.mean(EC[1, 1:])
        smoothed = np.empty(len(ys))
        smoothed[0] = ys[0]
        # Integrate the Curve 
        for i in range(len(filt) - 1):
            smoothed[i+1] = (EC[0, i+2] - EC[0, i+1]) * ys[i + 1] + smoothed[i]
        # Interpolations for easy comparison
        f = interp1d(EC[0, 1:], smoothed, kind = 'linear')
        # Add these results to the output
        to_add = f(xnew)
        to_return[k*res:k*res + res] = to_add  - np.min(to_add)
    
    print(to_return)
    return to_return


def SECT_folder(folder_path):
    file_names = os.listdir(folder_path)
    paths = [os.path.join(folder_path, file_name) for file_name in file_names]
    paths = sorted(paths)
    RES = 10000
    VEC_NUM = 50
    data = np.empty((len(paths), RES*VEC_NUM))
    pat_ids = []
    for i, path in enumerate(paths):
        print(path)
        curve = SECT(path, VEC_NUM, RES)
        data[i,:] = curve
    df = pd.DataFrame(data = data)
    SECTs = df.values
    return SECTs


####################### SINATRA PRO VERSION BELOW #############################

def nifti_to_stl(filepath, outfile):
    nifti = sitk.ReadImage(filepath)
    data = sitk.GetArrayFromImage(nifti)
    
    # Generate the mesh using marching cubes algorithm
    vertices, faces, _, _ = measure.marching_cubes(data, level=0.0, spacing= [6.5, 1, 1])
    
    # Create an STL mesh object
    stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            stl_mesh.vectors[i][j] = vertices[f[j], :]
    
    edges = set()
    for face in faces:
        for i in range(3):
            edge = tuple(sorted((face[i], face[(i+1)%3])))
            edges.add(edge)
    edge_list = list(edges)
    
    # Save the mesh to a file
    filename = os.path.basename(filepath)
    mesh_file = os.path.splitext(filename)[0]
    mesh_path = os.path.join(outfile, filename)
    stl_mesh.save(mesh_path)
    print('Mesh saved to', mesh_path)
    return vertices, edge_list, faces

def extract_mesh_datax(mesh):
    # Extract vertices from the mesh
    vertices = []
    for point in mesh.points:
        vertices.append([point[0], point[1], point[2]])

    # Extract edges from the mesh
    edges = []
    for i, triangle in enumerate(mesh.vectors):
        edges.append([i*3, i*3+1])
        edges.append([i*3+1, i*3+2])
        edges.append([i*3+2, i*3])

    # Extract faces from the mesh
    faces = []
    for triangle in mesh.vectors:
        faces.append([triangle[0], triangle[1], triangle[2]])
    return vertices, edges, faces

def extract_mesh_data(filename):
    verts, faces, normals, values = measure.marching_cubes(sitk.GetArrayViewFromImage(sitk.ReadImage(filename)), 0)
    edges = extract_edges(faces)
    
    return verts, edges, faces 



def compute_ec_curve_single(vertices, edges, faces, direction, ball_radius, n_filtration=25, ec_type="ECT", include_faces=True):
    """
    Computes the Euler Characteristics (EC) curves in a given direction in discrete filtraion steps for a given mesh

    `mesh` is the `mesh` class containing vertices, edges, and faces of the mesh.

    `direction` is the direction for EC curve to be calculated on.

    `ball_radius` is the radius of the bounding ball.

    `n_filtration` is the number of sub-level sets for which to compute the EC curve on in a given direction.

    `ec_type` is the type of EC transform (ECT), available options: DECT / ECT / SECT.
    DECT (differential ECT) is the default method used for protein.
    ECT is the standard ECT and SECT is the smoothe ECT.

    If `included_faces` is set to False, it ignore faces from the EC calculations.
    """
    
    eulers = np.zeros(n_filtration, dtype=float)
    vertex_function = np.dot(vertices, direction)
    radius = np.linspace(-ball_radius, ball_radius, n_filtration)

    # filtrating vertices
    V = histogram1d(vertex_function, range=[-ball_radius, ball_radius], bins=(n_filtration - 1))

    # filtrating edges
    if len(edges) > 0:
        edge_function = np.amax(vertex_function[edges], axis=1)
        E = histogram1d(edge_function, range=[-ball_radius, ball_radius], bins=(n_filtration - 1))
    else:
        E = 0

    if include_faces and len(faces) > 0:
        # filtrating faces
        face_function = np.amax(vertex_function[faces], axis=1)
        F = histogram1d(face_function, range=[-ball_radius, ball_radius], bins=(n_filtration - 1))
    else:
        F = 0

    eulers[1:] = V - E + F

    if ec_type == "ECT":
        eulers = np.cumsum(eulers)
        return eulers
    elif ec_type == "DECT":
        eulers[1:] = eulers[1:] / (radius[1:] - radius[:-1])
        return eulers
    elif ec_type == "SECT":
        eulers = np.cumsum(eulers)
        eulers -= np.mean(eulers)
        eulers = np.cumsum(eulers) * ((radius[-1] - radius[0]) / n_filtration)
        return eulers
    else:
        return None


def compute_ec_curve(vertices, edges, faces, directions, n_filtration=25, ball_radius=1.0, ec_type="ECT", first_column_index=False,
                     include_faces=True):
    """Computes the Euler Characteristics (EC) curves in a given direction with single CPU core"""
    eulers = np.zeros((directions.shape[0], n_filtration), dtype=float)
    for i in range(directions.shape[0]):
        eulers[i] = compute_ec_curve_single(vertices, edges, faces, directions[i], ball_radius, n_filtration, ec_type, include_faces)
    radius = np.linspace(-ball_radius, ball_radius, n_filtration)
    return radius, eulers


def compute_ec_curve_parallel(vertices, edges, faces, directions, n_filtration=25, ball_radius=1.0, ec_type="ECT", include_faces=True,
                              n_core=-1):
    """Computes the Euler Characteristics (EC) curves in a given direction with single multiple core"""
    parameter = (ball_radius, n_filtration, ec_type, include_faces)
    if n_core == -1:
        n_core = multiprocessing.cpu_count()
    processed_list = Parallel(n_jobs=n_core)(
        delayed(compute_ec_curve_single)(vertices, edges, faces, direction, *parameter) for direction in directions)
    processed_list = np.array(processed_list)
    radius = np.linspace(-ball_radius, ball_radius, n_filtration)
    return radius, processed_list

def read_cont_labels(directory_data_A, directory_data_B, n_sample):
    label = []
    if directory_data_A == None:
        directory = [directory_data_B]
    elif directory_data_B == None:
        directory = [directory_data_A]
    else:
        directory = [directory_data_A, directory_data_B]
    for directory_data in directory:
        for filename in os.listdir(directory_data):
            if filename.endswith('.csv'):
                with open(directory_data + '/' + filename) as docking_scores:
                    data = np.loadtxt(docking_scores)
                    prot_scores = []
                    for i in range(n_sample):
                        prot_scores.append(data[i])
                label.append(prot_scores)
        print("cont_label", len(label))
    label = np.array(label)
    label = np.ndarray.flatten(label)
    return(label)

def compute_ec_curve_folder(directory_data, labels_data, outfolder,
                            directions=None, n_sample= 101, ec_type="ECT", n_filtration=25,
                            ball_radius=1.0, include_faces=True, 
                            sm_radius=4.0, hemisphere=False, parallel=False, n_core=-1, verbose=True):
    """
    'label_type' denotes whether the binomial (classification) or continuous (linear).

    Computes the Euler Characteristics (EC) curves for a set of directions for the data set.

    'data_type' denotes the form of the label: binomial(categorical) or continuous

    `ec_type` is the type of EC transform (ECT), available options: DECT / ECT / SECT.
    DECT (differential ECT) is the default method used for protein.
    ECT is the standard ECT and SECT is the smoothe ECT.

    `directory_mesh_A` and `directory_mesh_B` are the folders that contain the .msh files for meshes for class A and B respectively.
    If `directory_mesh_A` and `directory_mesh_B` are provided, the function calculates EC curves of all meshes in the two folders.

    `directions` is the list of vectors containing all directions for EC curves to be calculated on.

    `n_filtration` is the number of sub-level sets for which to compute the EC curve on in a given direction.

    `ball_radius` is the radius for EC calculation, default to be 1.0 for unit sphere, where EC curves are calculated from radius = -1.0 to +1.0.

    If `included_faces` is set to False, it ignore faces from the EC calculations.

    If `parallel` is set to True, the program runs on multiple cores for EC calculations,
    then `n_core` will be the number of cores used (the program uses all detected cores if `n_core` is not provided`).

    If `verbose` is set to True, the program prints progress in command prompt.
    """
    print("compute EC curve folder working")
    
    filenames = sorted(os.listdir(directory_data))
    ecs = []
    for filename in filenames:
        if filename.endswith(".nii.gz"):
            if verbose:
                sys.stdout.write('Calculating EC')
                sys.stdout.flush()
            filepath = os.path.join(directory_data, filename)
            vertices, edges, faces = extract_mesh_data(filepath)
            if parallel:
                t, ec = compute_ec_curve_parallel(vertices, edges, faces, directions, n_filtration=n_filtration,
                                                  ball_radius=ball_radius, ec_type=ec_type,
                                                  include_faces=include_faces, n_core=n_core)
            else:
                t, ec = compute_ec_curve(vertices, edges, faces, directions, n_filtration=n_filtration,
                                         ball_radius=ball_radius, ec_type=ec_type, include_faces=include_faces)
                
            ecs.append(ec.flatten())
    ecs = np.array(ecs)
    ecss = ecs
    

    data_A = ecss
    print(data_A.shape)
    
    vacuum = np.ones(data_A.shape[1], dtype=bool)
    for a in data_A:
        vacuum = np.logical_and(vacuum, a == 0)

    not_vacuum = np.logical_not(vacuum)
    data = data_A[:, not_vacuum]

    mean = np.average(data, axis=0)
    std = np.std(data, axis=0)

    data = np.subtract(data, mean)
    data = np.divide(data, std)
    data[np.isnan(data)] = 0

    label = []
    label = np.loadtxt(labels_data)
        
    #returns ECSS that hasn't had the vacuum spots removed. This is so that reconstruction can be simpler. 
    return data, label, not_vacuum 
