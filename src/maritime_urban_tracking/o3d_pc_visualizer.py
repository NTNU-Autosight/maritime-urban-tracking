import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R

from maritime_urban_tracking.o3d_line_mesh import LineMesh

class MyCoordFrame():
    def __init__(self) -> None:
        self.coord_inner = o3d.geometry.TriangleMesh.create_coordinate_frame()
        self.ori_quat = np.array([0, 0, 0, 1])
    
    @property
    def geometry(self):
        return self.coord_inner
    
    def set_pos(self, pos):
        self.coord_inner.translate(pos, relative=False)
    
    def set_ori(self, ori_quat):
        # Finding the rotation matrix that rotates the old frame to the new frame,
        # but it means transforming the points in the old frame to the points in the new frame
        ori_before = R.from_quat(self.ori_quat)
        ori = R.from_quat(ori_quat)

        rot_diff = ori_before.inv() * ori

        r_matrix = rot_diff.as_matrix()
        self.coord_inner.rotate(r_matrix)

        self.ori_quat = ori_quat

class MyPoint():
    def __init__(self, color=(0, 1.0, 0)) -> None:
        self.inner = o3d.geometry.TriangleMesh.create_sphere(radius=0.5, resolution=4)
        self.inner.paint_uniform_color(color)

    @property
    def geometry(self):
        return self.inner
    
    def set_pos(self, pos):
        self.inner.translate(pos, relative=False)

class MyArrow():
    def __init__(self, pt_from, pt_to) -> None:
        self.vec_field = MyVectorField(np.array([pt_from]), np.array([pt_to]))
    
    @property
    def geometry(self):
        assert len(self.vec_field.geometries) == 1
        return self.vec_field.geometries[0]

class MyVectorField():
    def __init__(self, pts0, pts1) -> None:
        vecs0 = np.zeros_like(pts0)
        vecs0[:,2] = 1
        vecs1 = pts1 - pts0
        Rs = get_rotations(vecs0, vecs1)
        ts = pts0
        ls = np.linalg.norm(pts1-pts0, axis=1)

        self.geometries = []
        for i in range(len(ls)):
            l = ls[i]
            if l <= 0:
                continue

            R = Rs[i,:,:]
            t = ts[i,:]

            # Along z-axis by default
            scale = 2.0
            geom = o3d.geometry.TriangleMesh.create_arrow(
                cone_height= 0.2 * l, 
                cone_radius= 0.06 * scale, 
                cylinder_height= 0.8 * l,
                cylinder_radius=  0.04 * scale,
                resolution=6
            )
            geom.paint_uniform_color((1.0, 0, 0)) # red
            geom.rotate(R, center=(0, 0, 0))
            geom.translate(t, relative=True)
            self.geometries.append(geom)

def get_rotations(vts0, vts1):
    # Vectors are horizontal
    # Rotate vts0 vectors to vts1
    # See here: 
    # https://stackoverflow.com/questions/59026581/create-arrows-in-open3d
    # and here: 
    # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    v0_norm = vts0 / np.linalg.norm(vts0, axis=1)[:, np.newaxis]
    v1_norm = vts1 / np.linalg.norm(vts1, axis=1)[:, np.newaxis]
    v = np.cross(v0_norm, v1_norm)
    # c = v0_norm.dot(v1_norm.T)
    c = np.einsum("ij,ij->i", v0_norm, v1_norm)[:,np.newaxis]
    v_skew = skew(v)
    R = np.eye(3) + v_skew + np.matmul( v_skew, v_skew ) / ((1 + c))[:,:,np.newaxis]
    return R

def skew(p):
    zeros = np.zeros_like(p[:,0])
    p_skew = np.array([
        [zeros, -p[:,2], p[:,1]],
        [p[:,2], zeros, -p[:,0]],
        [-p[:,1], p[:,0], zeros]
    ])
    p_skew_transposed = p_skew.transpose(2, 0, 1) # Move the last dim to the first one. 
    return p_skew_transposed


class O3DPointCloudVisualizer:
    def __init__(self, visualization_parameter_path = None, should_start_with_conf=True) -> None:
        self.visualization_parameter_path = visualization_parameter_path
        self.should_start_with_conf = should_start_with_conf

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="Pointcloud", visible=True)
        
        coor_frame_geometry = o3d.geometry.TriangleMesh.create_coordinate_frame()

        self.vis_initializing = True
        self.geometries = []
        self.geometries.append(coor_frame_geometry)
    
    def create_point_cloud(self):
        pcd = o3d.geometry.PointCloud()
        self.geometries.append(pcd)
        return pcd
    
    def update_point_cloud(self, pcd, pc_xyz, pc_rgb = None):
        if pc_rgb is None:
            pc_rgb = np.tile(np.array([255, 0, 0], dtype=np.uint8)/255, (pc_xyz.shape[0], 1))
        pcd.points = o3d.utility.Vector3dVector(pc_xyz)
        pcd.colors = o3d.utility.Vector3dVector(pc_rgb)
    
    def create_coord_frame(self):
        coord_frame = MyCoordFrame()
        self.geometries.append(coord_frame.geometry)
        return coord_frame
    
    def update_coord_frame(self, coord_frame: MyCoordFrame, pos, ori_quat):
        coord_frame.set_pos(pos)
        coord_frame.set_ori(ori_quat)
    
    def create_point(self, color=(0, 1.0, 0)):
        point = MyPoint(color=color)
        self.geometries.append(point.geometry)
        return point
    
    def update_point(self, point: MyPoint, pos):
        point.set_pos(pos)
    
    def add_arrow(self, pt_from, pt_to):
        arrow = MyArrow(pt_from, pt_to)
        self.add_geometries([arrow.geometry])
        return arrow
    
    def remove_arrow(self, arrow):
        self.vis.remove_geometry(arrow.geometry)
    
    def add_vector_field(self, pc0, pc1):
        vector_field = MyVectorField(pc0, pc1)
        self.add_geometries(vector_field.geometries)
        return vector_field
    
    def remove_vector_field(self, vector_field):
        for geom in vector_field.geometries:
            self.vis.remove_geometry(geom, reset_bounding_box=False)
    
    def add_geometries(self, geometries):
        for geom in geometries:
            if self.vis_initializing:
                self.vis.add_geometry(geom, reset_bounding_box=True)
            else:
                self.vis.add_geometry(geom, reset_bounding_box=False)
    
    def render(self):
        if self.vis_initializing:
            self.add_geometries(self.geometries)
            
            # Load viewpoint
            if self.visualization_parameter_path is not None and self.should_start_with_conf:
                param = o3d.io.read_pinhole_camera_parameters(self.visualization_parameter_path)
                view_ctl = self.vis.get_view_control()
                view_ctl.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)

            self.vis.run()

            # Save viewpoint
            if self.visualization_parameter_path is not None:
                param = self.vis.get_view_control().convert_to_pinhole_camera_parameters()
                o3d.io.write_pinhole_camera_parameters(self.visualization_parameter_path, param)

            self.vis_initializing = False
        
        for geometry in self.geometries:
            self.vis.update_geometry(geometry)

        self.vis.poll_events()
        self.vis.update_renderer()
    
    def draw_line(self, points):
        line_mesh = LineMesh(points, colors=[1,0,0], radius=0.1)
        line_mesh.add_line(self.vis)

def color_pc(pc_xyz, color = (1, 0, 0)):
    pc_rgb = np.tile(np.array(color, dtype=np.float32), (pc_xyz.shape[0], 1))
    return pc_rgb
