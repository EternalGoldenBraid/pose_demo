import os
import json
from pathlib import Path
from numba import njit
import numpy as np
from scipy.spatial import Delaunay
import torch
from pytorch3d.io import load_ply
from numpy.random import default_rng
from ipdb import iex

from lib import rendering
#from lib.Sim3DR import RenderPipeline
#from lib.Sim3DR.Sim3DR import rasterize, get_normal

class Dataset():
    def __init__(self, data_dir, cfg, 
            cam_K, cam_height, cam_width, n_points):
        self.model_dir = Path(data_dir) / 'models_eval'
        self.cam_file = Path(data_dir) / 'camera.json'
        self.cam_K = cam_K
        self.cam_K_np = cam_K.cpu().numpy()
        self.cam_height = cam_height
        self.cam_width = cam_width
        self.model_info = None
        self.obj_model_file = dict()
        self.obj_diameter = dict()
        self.point_cloud = dict()
        self.faces = dict()
        self.object_renderer = None
        self.cfg = cfg
        self.n_points: int = n_points # Number of points

        self.model_info_file = self.model_dir / 'models_info.json'
        #self.model_info_file = os.path.abspath(os.path.join(
            #os.path.dirname(__file__), '..', self.model_dir, 'models_info.json'))
        with open(self.model_info_file, 'r') as model_f:
            self.model_info = json.load(model_f)
        
        rng = default_rng()

        ### TODO: Combine faces and verts into same container.
        for model_file in sorted(self.model_dir.iterdir()):
            #breakpoint()
            if str(model_file).endswith('.ply'):
                obj_id = int(model_file.name.split('_')[-1].split('.')[0])
                self.obj_model_file[obj_id] = model_file
                self.obj_diameter[obj_id] = self.model_info[str(obj_id)]['diameter']
                self.point_cloud[obj_id], self.faces[obj_id] = load_ply(model_file)
                self.point_cloud[obj_id] = self.point_cloud[obj_id].numpy()
                self.faces[obj_id] = self.faces[obj_id].numpy()

                #import pdb; pdb.set_trace() 
                if self.point_cloud[obj_id].shape[0] > self.n_points:
                    idxs = rng.integers(low=0, 
                            high=self.point_cloud[obj_id].shape[0], size=self.n_points)
                    self.point_cloud[obj_id] = self.point_cloud[obj_id][idxs]
                    self.faces[obj_id] = self.faces[obj_id][idxs]



        if self.cam_K == None:
            print("Warning camera intrinsics not set.")

    #@iex
    #@njit(parallel=True)
    def render_cloud(self, obj_id, R, t, image):

        ### FLIP Y, and Z coords
        #R = R@np.array([
        #                [1, 0, 0],
        #                [0, -1, 0],
        #                [0, 0, -1]], dtype=np.float32)
        R = R.dot(np.array([
                        [1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]], dtype=np.float32))
        
        P = self.cam_K_np.dot(R.dot(self.point_cloud[obj_id].T) + t.T)
        #P = self.cam_K_np.dot(t.T)

        #print(np.linalg.norm((R.dot(self.point_cloud[obj_id].T) + t.T).mean(axis=1)))
        #print(np.linalg.norm((t.T).mean(axis=1)))
        
        P = P // P[-1,:]

        if P[1].max() >= self.cam_height or P[0].max() >= self.cam_width:
            return image, False

        #view_depth *= view_cam_info['depth_scale']
        #view_depth *= cfg.MODEL_SCALING # convert to meter scale from millimeter scale
        #view_depth/=view_depth.max()
        
        P = P.astype(int)

        image[P[1], P[0], :] = 255
        return image, True

    def render_mesh(self, obj_id, R, t, image):
        ### FLIP Y, and Z coords
        R = R.dot(np.array([
                        [1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]], dtype=np.float32))
        
        P = self.cam_K_np.dot(R.dot(self.point_cloud[obj_id].T) + t.T)
        F = self.cam_K_np.dot(R.dot(self.faces[obj_id].T) + t.T)

        P = P // P[-1,:]
        F = F // F[-1,:]

        if P[1].max() >= self.cam_height or P[0].max() >= self.cam_width:
            return image, False
        
        P = P.astype(np.float32)
        F = F.astype(np.int32)

        #image = rasterize(vertices=P, triangles=F,
        #        #colors=np.zeros_like(P, dtype=np.uint8),
        #        colors=np.array([1,1,1], dtype=np.float32)[...,None],
        #        bg=image.astype(np.uint8),
        #        #height=image.shape[0], width=image.shape[1], channel=3
        #        )
        #return image, True

    def cloud_show(self, p):
        
        ### Register a point cloud
        # `my_points` is a Nx3 numpy array
        ps.register_point_cloud("my points", p)
        
        #### Register a mesh
        ## `verts` is a Nx3 numpy array of vertex positions
        ## `faces` is a Fx3 array of indices, or a nested list
        #ps.register_surface_mesh("my mesh", verts, faces, smooth_shade=True)
        
        ## Add a scalar function and a vector function defined on the mesh
        ## vertex_scalar is a length V numpy array of values
        ## face_vectors is an Fx3 array of vectors per face
        #ps.get_surface_mesh("my mesh").add_scalar_quantity("my_scalar", 
        #        vertex_scalar, defined_on='vertices', cmap='blues')
        #ps.get_surface_mesh("my mesh").add_vector_quantity("my_vector", 
        #        face_vectors, defined_on='faces', color=(0.2, 0.5, 0.5))
        
        # View the point cloud and mesh we just registered in the 3D UI
        ps.show()

    def render_depth(self, obj_id, R, t, mesh):
        self.obj_renderer = rendering.Renderer(width=self.cfg.RENDER_WIDTH, height=self.cfg.RENDER_HEIGHT)
        obj_context = rendering.SceneContext(obj=mesh, intrinsic=self.cam_K.cpu()) # define a scene
        obj_context.set_pose(rotation=R, translation=t)
        
        est_depth, est_mask = self.obj_renderer.render(obj_context)[1:]

        return est_depth
