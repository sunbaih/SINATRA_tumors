import os, sys
#from sinatra_pro.mesh import *
from fast_histogram import histogram1d
import multiprocessing
from joblib import Parallel, delayed
import SimpleITK as sitk
import numpy as np
from skimage import measure
import stl
import meshio
import trimesh


# ended up not really using this, simply get the vector of verts, edges, faces in euler.py instead 

def convert_nifti_to_mesh(nifti_directory, out_directory): #Converts an input folder of nifti images to an output file of .msh 
    for filename in os.listdir(nifti_directory):
        path = os.path.join(nifti_directory, filename)
        nifti = sitk.ReadImage(path)
        print(nifti.GetSpacing())
        data = sitk.GetArrayFromImage(nifti)
        
        # Generate the mesh using marching cubes algorithm
        print("Data range:", np.min(data), np.max(data))
        vertices, faces, _, _ = measure.marching_cubes(data, level=0.0, spacing= [6.5, 1, 1])
        
        # Create an STL mesh object
        stl_mesh = stl.mesh.Mesh(np.zeros(faces.shape[0], dtype=stl.mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                stl_mesh.vectors[i][j] = vertices[f[j], :]
        
        # Save the mesh to a .stl file for visualization
        stl_out_directory = "%s/stl/"%(out_directory)
        if not os.path.exists(stl_out_directory):
            os.mkdir(stl_out_directory)
        stl_file = '%s/%s.stl'%(stl_out_directory, filename)
        stl_mesh.save(stl_file)
        
        #save the file to a .msh file for data processing 
        mesh = meshio.read(stl_file)
        # Export mesh to MSH file
        msh_out_directory = "%s/msh/"%(out_directory)
        if not os.path.exists(msh_out_directory):
            os.mkdir(msh_out_directory)
        mesh_file = '%s/%s.msh'%(msh_out_directory, filename)
        meshio.write(mesh_file, mesh)
    

def clean_files(input_dir, output_dir):
    files = os.listdir(input_dir)
    for filename in files:
        input_path = os.path.join(input_dir, filename)
        mesh = meshio.read(input_path)
        area = mesh.area
        scaled_mesh = mesh.copy()
        scaled_mesh.points /= np.sqrt(area)
        scaled_mesh.points -= scaled_mesh.points.mean(axis=0)
        output_path = os.path.join(output_dir, filename)
        meshio.write(output_path, scaled_mesh)

def scale_mesh(vertices, scale_factor):
    return vertices * scale_factor

def translate_mesh(vertices, translation):
    return vertices - translation
