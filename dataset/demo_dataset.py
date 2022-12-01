import os
import json
from pathlib import Path
from numba import njit
import cv2
import numpy as np
from scipy.spatial import Delaunay
import torch
from pytorch3d.io import load_ply
import open3d as o3d
from numpy.random import default_rng
from ipdb import iex

from lib import rendering
#from lib.Sim3DR import RenderPipeline
#from lib.Sim3DR.Sim3DR import rasterize, get_normal

class Dataset():
    def __init__(self, data_dir, cfg, 
            cam_K, cam_height, cam_width, n_triangles, object_renderer=None):
        self.model_dir = Path(data_dir) / 'models_eval'
        self.cam_file = Path(data_dir) / 'camera.json'
        self.cam_K = cam_K
        self.cam_K_np = cam_K.cpu().numpy()
        self.cam_height = cam_height
        self.cam_width = cam_width
        self.model_info = None
        self.obj_model_file: dict = dict()
        self.obj_diameter = dict()
        self.point_cloud = dict()
        self.faces = dict()
        self.object_renderer = object_renderer
        self.cfg = cfg
        self.n_triangles: int = n_triangles # Number of points
        self.rendering_meshes = {}

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

                mesh = o3d.io.read_triangle_mesh(filename=str(model_file))
                mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=n_triangles)
                self.point_cloud[obj_id] = np.asanyarray(mesh.vertices).astype(float)
                self.faces[obj_id] = np.asanyarray(mesh.triangles).astype(int)

                ### FOR OLD RENDERER in def render_depth
                obj_mesh, _ = rendering.load_object(model_file, resize=False, recenter=False)
                self.rendering_meshes[obj_id] = obj_mesh

        if self.cam_K == None:
            print("Warning camera intrinsics not set.")

        else:
            self.obj_renderer = rendering.Renderer(width=self.cfg.RENDER_WIDTH, height=self.cfg.RENDER_HEIGHT)
            #self.vis = o3d.visualization.Visualizer()
            #self.vis.create_window(visible=True)

    @iex
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
        
        #P = self.cam_K_np.dot(R.dot(self.point_cloud[obj_id].T) + t.T)
        P = self.cam_K_np.dot(R.dot(self.point_cloud[obj_id].T) + t)

        P = P // P[-1,:]

        if P[1].max() >= self.cam_height or P[0].max() >= self.cam_width:
            return image, False

        #view_depth *= view_cam_info['depth_scale']
        #view_depth *= cfg.MODEL_SCALING # convert to meter scale from millimeter scale
        #view_depth/=view_depth.max()
        
        P = P.astype(int)

        image[P[1], P[0], :] = 255
        return image, True

    def render_mesh(self, obj_id, R, t, image, color=(0,0,255), alpha=0.5):
        ### FLIP Y, and Z coords
        R = R.dot(np.array([
                        [1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]], dtype=np.float32))
        
        P = self.cam_K_np.dot(R.dot(self.point_cloud[obj_id].T) + t).squeeze()
        P = P // P[-1,:]

        if P[1].max() >= self.cam_height or P[0].max() >= self.cam_width:
            print("LOG: Projection out of bounds")
            return image, False

        P = P.astype(np.int32)
        #Does not fill the triangels since all are passed as a single batch.
        #Iterating over earch triangle and rendering one by one results in fill.
        image_filled = cv2.fillPoly(image.copy(), 
                                    #lineType=cv2.FILLED,
                                    #lineType=-1,
                                    pts=np.array(P.T[self.faces[obj_id]][:,:,:2]), color=color)

        image = cv2.addWeighted(image, alpha, image_filled, 1-alpha, 0.0)

        #image = cv2.drawContours(image=image, contourIdx=-1, color=color, thickness=-1, 
                #contours=np.array(P.T[self.faces[obj_id]][:,:,:2]))

        return image, True

    def render_mesh_slow(self, obj_id, R, t, image, color=(0,0,255), alpha=0.5):
        ### FLIP Y, and Z coords
        R = R.dot(np.array([
                        [1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]], dtype=np.float32))
        
        P = self.cam_K_np.dot(R.dot(self.point_cloud[obj_id].T) + t).squeeze()
        P = P // P[-1,:]

        if P[1].max() >= self.cam_height or P[0].max() >= self.cam_width:
            print("LOG: Projection out of bounds")
            return image, False

        P = P.astype(np.int32)
        #Does not fill the triangels since all are passed as a single batch.
        #Iterating over earch triangle and rendering one by one results in fill.

        #for face_idx = 
        pts = np.array(P.T[self.faces[obj_id]][:,:,:2])
        image_filled = image.copy()
        for polygon in pts:
            #breakpoint()
            image_filled = cv2.fillPoly(image_filled,
                                        #lineType=cv2.FILLED,
                                        #lineType=-1,
                                        pts=[polygon], color=color)

        image = cv2.addWeighted(image, alpha, image_filled, 1-alpha, 0.0)

        #image = cv2.drawContours(image=image, contourIdx=-1, color=color, thickness=-1, 
                #contours=np.array(P.T[self.faces[obj_id]][:,:,:2]))

        return image, True

    def render_depth(self, obj_id, R, t, image, alpha=0.5):
        #obj_context = rendering.SceneContext(obj=self.point_cloud[obj_id], intrinsic=self.cam_K.cpu()) # define a scene
        #obj_context.set_pose(rotation=R, translation=t)
        #
        #depth, mask = self.obj_renderer.render(obj_context)[1:]
        #breakpoint()
        ##image = cv2.addWeighted(image, alpha, depth, 1-alpha, 0.0)

        #self.vis.add_geometry(self.point_cloud[obj_id])
        #self.vis.update_geometry(self.point_cloud[obj_id])
        #self.vis.poll_events()
        #self.vis.update_renderer()
        #vis.destroy_window()

        R_ = torch.from_numpy(R)
        #T_[:,:3] = torch.from_numpy(t[None,...])
        T_ = torch.from_numpy(t.squeeze())

        depth, mask = rendering.rendering_views(obj_mesh=self.rendering_meshes[obj_id],
                                                intrinsic=self.cam_K.cpu(),
                                                R=R_,
                                                T=T_,
                                                width=self.cam_width,
                                                height=self.cam_height)

        rend_depth = depth.squeeze().numpy()[None,...].repeat(repeats=3, axis=2)

        #breakpoint()
        cv2.imshow('wot', rend_depth[0])
        #image = cv2.addWeighted(image, alpha, rend_depth, 1-alpha, 0.0)
        return image, True

    def __del__(self):
        pass
        #self.vis.destroy_window()

