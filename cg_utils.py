import numpy as np
# Uncomment the lines below if off-screen rendering for remote server is needed
# import pyglet
# import os
# pyglet.options['shadow_window'] = False
# os.environ['MESHRENDER_EGL_OFFSCREEN'] = 't'

import trimesh
from autolab_core import RigidTransform
from perception import CameraIntrinsics
from meshrender import Scene, MaterialProperties, AmbientLight, PointLight, SceneObject, VirtualCamera


# Load 3D mesh model as a PyOpenGL Scene object
def load_3d_model(model_path):
    # Start with an empty scene
    scene = Scene()
    # Add objects to the scene
    # Begin by loading meshes
    pawn_mesh = trimesh.load_mesh(model_path)
    # Set up object's pose in the world
    pawn_pose = RigidTransform(
        rotation=np.eye(3),
        translation=np.array([0.0, 0.0, 0.0]),
        from_frame='obj',
        to_frame='world'
    )
    # Set up each object's material properties
    pawn_material = MaterialProperties(
        color=np.array([1.0, 1.0, 1.0]),
        k_a=1.0,
        k_d=1.0,
        k_s=0.0,
        alpha=1.0,
        smooth=False,
        wireframe=False
    )
    # Create SceneObjects for each object
    pawn_obj = SceneObject(pawn_mesh, pawn_pose, pawn_material)
    # Add the SceneObjects to the scene
    scene.add_object('pawn', pawn_obj)
    return scene, pawn_mesh


def get_depth_image_from_3d_model(scene, camera_intrinsics, image_height, image_width, camera_pose, z_near, z_far,
                                  point_light_strength, ambient_strength):
    # Add a point light source to the scene
    pointlight = PointLight(location=camera_pose[:3, 3], color=np.array([1.0, 1.0, 1.0]),
                            strength=point_light_strength)
    scene.add_light('point_light', pointlight)

    # Add lighting to the scene
    # Create an ambient light
    ambient = AmbientLight(
        color=np.array([1.0, 1.0, 1.0]),
        strength=ambient_strength
    )
    # Add the lights to the scene
    scene.ambient_light = ambient  # only one ambient light per scene

    # Add a camera to the scene
    # Set up camera intrinsics
    ci = CameraIntrinsics(
        frame='camera',
        fx=camera_intrinsics[0, 0],
        fy=camera_intrinsics[1, 1],
        cx=camera_intrinsics[0, 2],
        cy=camera_intrinsics[1, 2],
        skew=0.0,
        height=image_height,
        width=image_width
    )
    # Set up the camera pose (z axis faces away from scene, x to right, y up)
    cp = RigidTransform(
        rotation=camera_pose[:3, :3],
        translation=camera_pose[:3, 3],
        from_frame='camera',
        to_frame='world'
    )
    # Create a VirtualCamera
    camera = VirtualCamera(ci, cp, z_near=z_near, z_far=z_far)
    # Add the camera to the scene
    scene.camera = camera
    # Render raw numpy arrays containing color and depth
    color_image_raw, depth_image_raw = scene.render(render_color=True, front_and_back=True)
    return color_image_raw, depth_image_raw
