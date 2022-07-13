import json
import torch
from pathlib import Path
from pytorch3d.io import load_ply
from lib import (rendering, network,
        agnostic_segmentation)



class Dataset():
    def __init__(self, data_dir, cfg):
        self.model_dir = Path(data_dir) / 'models_eval'
        self.cam_file = Path(data_dir) / 'camera.json'
        self.cam_K = None
        self.cam_height = None
        self.cam_width = None
        self.model_info = None
        self.obj_model_file = dict()
        self.obj_diameter = dict()
        self.point_cloud = dict()
        self.object_renderer = None
        self.cfg = cfg

        with open(self.cam_file, 'r') as cam_f:
            self.cam_info = json.load(cam_f)

        self.cam_K = torch.tensor([
            [self.cam_info['fx'], 0, self.cam_info['cx']],
            [0.0, self.cam_info['fy'], self.cam_info['cy']],
            [0.0, 0.0, 1.0]
        ], dtype=torch.float32)
        
        self.cam_height = self.cam_info['height']
        self.cam_width = self.cam_info['width']
        
        self.model_info_file = self.model_dir / 'models_info.json'
        with open(self.model_info_file, 'r') as model_f:
            self.model_info = json.load(model_f)
            
        
        for model_file in sorted(self.model_dir.iterdir()):
            if str(model_file).endswith('.ply'):
                obj_id = int(model_file.name.split('_')[-1].split('.')[0])
                self.obj_model_file[obj_id] = model_file
                self.obj_diameter[obj_id] = self.model_info[str(obj_id)]['diameter']
                self.point_cloud[obj_id], _ = load_ply(model_file)

    def render_cloud(self, obj_id, R, t, image):

        R = torch.ones([3,3], dtype=torch.float32)
        t = torch.ones(3, dtype=torch.float32)
        ### FLIP Y, and Z coords
        R = R@torch.tensor([
                        [1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]], device='cpu', dtype=torch.float32)
        
        t = t.expand(self.point_cloud[obj_id].shape[0],-1)
        
        P = (self.cam_K@(R@self.point_cloud[obj_id].T + t.T))
        #import pdb; pdb.set_trace()
        P = P / P[-1,:]
        P = P.int()
        #import pdb; pdb.set_trace()
        #view_depth *= view_cam_info['depth_scale']
        #view_depth *= cfg.MODEL_SCALING # convert to meter scale from millimeter scale
        #view_depth/=view_depth.max()
        P = P.cpu().numpy()
        
        #image[P[1], P[0], :] = P[2].expand(-1, 3)
        #import pdb; pdb.set_trace()
        #image[P[1], P[0], :] = 1
        image[P[0], P[1], :] = 1
        #return view_depth

    def render_depth(self, obj_id, R, t, mesh):
        self.obj_renderer = rendering.Renderer(width=self.cfg.RENDER_WIDTH, height=self.cfg.RENDER_HEIGHT)
        obj_context = rendering.SceneContext(obj=mhjjesh, intrinsic=self.cam_K.cpu()) # define a scene
        obj_context.set_pose(rotation=R, translation=t)
        
        est_depth, est_mask = self.obj_renderer.render(obj_context)[1:]

        return est_depth
