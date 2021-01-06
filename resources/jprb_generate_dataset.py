import imageio
import numpy as np
import random
import h5py
import time
import sys
import re
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from cdp4_data_collection import CDP4DataCollection
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from scipy.spatial.transform import Rotation
from cv2 import fillConvexPoly, fillPoly
from glob import glob

global toto_c
toto_c = 0

# Some global variables
class TrainingRoom():

	# May be this function should also build the walls and create the camera object
	def __init__(self, camera_params, image_dimensions, n_sequences_per_scene, n_frames_per_sequence=20):

		# Training_room parameters
		self.table_height = 0.78  # should check in sdf
		self.object_bounds_on  = {'x': [-0.55, 0.55], 'y': [-1.60, 0.30]}  # rough space for both tables
		self.object_bounds_off = {'x': [-0.55,-0.25], 'y': [-1.60,-0.40]}  # hole between both tables

		# Camera parameters
		self.image_dimensions = image_dimensions
		self.data_collector = CDP4DataCollection(camera_params)
		self.n_frames_per_sequence = n_frames_per_sequence
		self.n_cameras = camera_params['n_cameras']
		self.camera_type = camera_params['camera_type']
		self.camera_near_distance = camera_params['near_distance']
		self.n_sequences_per_scene = n_sequences_per_scene
		self.n_sequences_per_camera = n_sequences_per_scene/self.n_cameras
		self.camera_speed_states = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for c in range(self.n_cameras)]
		self.camera_speed_range_t = [0.003, 0.005]  # tangential speed
		self.camera_speed_range_r = [0.5,0.6]  # radial speed, > 1 for inward spiral, < 1 for outward spiral
		self.camera_speed_t = np.random.uniform(self.camera_speed_range_t[0], self.camera_speed_range_t[1])
		self.camera_speed_r = np.random.uniform(self.camera_speed_range_r[0], self.camera_speed_range_r[1])
		self.camera_look_at = [0.2, -0.6, self.table_height]
		self.camera_heights = [0.4, 0.8]  # above table height (for now); in meteres
		self.min_camera_radius = 0.5
		self.max_camera_radius = 1.0

		# Objects in the scene variables
		self.object_shapes = ['table', 'eric_armL', 'eric_armR', 'eric_calfL', 'eric_calfR', 'eric_footL', 'eric_footR', 'eric_forearmL', 'eric_forearmR',
			'eric_handL', 'eric_handR', 'eric_head', 'eric_legL', 'eric_legR', 'eric_torso', 'hammer', 'spanner', 'screwdriver', 'sphere', 'box', 'cylinder']
		self.object_presence_prob = {'table': 1.0, 'eric_armL': 1.0, 'eric_armR': 1.0, 'eric_calfL': 1.0, 'eric_calfR': 1.0, 'eric_footL': 1.0, 'eric_footR': 1.0, 'eric_forearmL': 1.0, 'eric_forearmR': 1.0,
			'eric_handL': 1.0, 'eric_handR': 1.0, 'eric_head': 1.0, 'eric_legL': 1.0, 'eric_legR': 1.0, 'eric_torso': 1.0, 'hammer': 0.3, 'spanner': 0.3, 'screwdriver': 0.3, 'sphere': 0.15, 'box': 0.15, 'cylinder': 0.15}
		self.object_numbers = {'table': 2, 'eric_armL': 1, 'eric_armR': 1, 'eric_calfL': 1, 'eric_calfR': 1, 'eric_footL': 1, 'eric_footR': 1, 'eric_forearmL': 1, 'eric_forearmR': 1,
			'eric_handL': 1, 'eric_handR': 1, 'eric_head': 1, 'eric_legL': 1, 'eric_legR': 1, 'eric_torso': 1, 'hammer': 5, 'spanner': 5, 'screwdriver': 5, 'sphere': 5, 'box': 5, 'cylinder': 5}
		self.object_min_scales = {'table': 1.00, 'eric_armL': 1.00, 'eric_armR': 1.00, 'eric_calfL': 1.00, 'eric_calfR': 1.00, 'eric_footL': 1.00, 'eric_footR': 1.00, 'eric_forearmL': 1.00, 'eric_forearmR': 1.00,
			'eric_handL': 1.00, 'eric_handR': 1.00, 'eric_head': 1.00, 'eric_legL': 1.00, 'eric_legR': 1.00, 'eric_torso': 1.00, 'hammer': 0.25, 'spanner': 0.75, 'screwdriver': 1.00, 'sphere': 0.08, 'box': 0.05, 'cylinder': 0.06}
		self.object_max_scales = {'table': 1.00, 'eric_armL': 1.00, 'eric_armR': 1.00, 'eric_calfL': 1.00, 'eric_calfR': 1.00, 'eric_footL': 1.00, 'eric_footR': 1.00, 'eric_forearmL': 1.00, 'eric_forearmR': 1.00,
			'eric_handL': 1.00, 'eric_handR': 1.00, 'eric_head': 1.00, 'eric_legL': 1.00, 'eric_legR': 1.00, 'eric_torso': 1.00, 'hammer': 1.00, 'spanner': 1.50, 'screwdriver': 2.00, 'sphere': 0.30, 'box': 0.20, 'cylinder': 0.25}
		self.object_names = ['_'.join((shape, str(n))) for shape in self.object_shapes for n in range(self.object_numbers[shape])]
		self.object_statics = [True if shape in ['table', 'eric', 'eric_armL', 'eric_armR', 'eric_calfL', 'eric_calfR', 'eric_footL', 'eric_footR', 'eric_forearmL', 'eric_forearmR',
			'eric_handL', 'eric_handR', 'eric_head', 'eric_legL', 'eric_legR', 'eric_torso'] else False for shape in self.object_shapes for n in range(self.object_numbers[shape])]
		self.object_instances = [i+1 for i, shape in enumerate(self.object_shapes) for n in range(self.object_numbers[shape])]  # 0 is no object
		self.object_instances_prob = [self.object_presence_prob[name[:-2]] for name in self.object_names]
		self.object_poses = [self.data_collector.get_object_pose(n) for n in self.object_names]
		self.object_visibility = np.ones((len(self.object_names),), dtype=int)
		self.object_scales = [self.data_collector.get_object_scale(n) for n in self.object_names]
		self.vertex_index_meshes, self.triangle_index_meshes, self.posed_meshes, self.physical_scales = self.load_object_meshes()

		# Classification and segmentation dataset variables
		data_images_shape = (self.n_sequences_per_scene, self.n_frames_per_sequence) + self.image_dimensions
		self.segment_image = np.zeros(self.image_dimensions[:-1], dtype=np.uint8)
		self.segment_layer = np.zeros(self.segment_image.shape, dtype=np.bool)
		self.distance_image = np.zeros(self.segment_image.shape)
		self.distance_layer = np.zeros(self.segment_image.shape)
		self.data_rgb = np.zeros(data_images_shape, dtype=np.uint8)
		self.data_dvs = np.zeros(data_images_shape, dtype=np.uint8)
		self.data_lbl = {'pos_3D': np.zeros((self.n_sequences_per_scene, len(self.object_names), 3)),
						 'ori_3D': np.zeros((self.n_sequences_per_scene, len(self.object_names), 3)),
		                 'scale_3D': np.zeros((self.n_sequences_per_scene, len(self.object_names), 3)),
		                 'visibles': np.zeros((self.n_sequences_per_scene, len(self.object_names),)),
		                 'segments': np.zeros(data_images_shape[:-1], dtype=np.uint8)}

	# Select random object positions for the new scene
	def choose_new_object_poses(self):

		# Go through all moving objects to modify their positions and scales
		for (name, static, visible, pose, scale, phys_scale) in zip(self.object_names, self.object_statics,
			self.object_visibility, self.object_poses, self.object_scales, self.physical_scales):
			shape = name[:-2]
			if not static:  # some objects are always there

				# Set new positions for visble objects (checking for collision)
				if visible:
					bad_pos = True
					n_trials = 0
					while bad_pos and n_trials < 100:
						scale_x, scale_y, scale_z = np.random.uniform(
							self.object_min_scales[shape], self.object_max_scales[shape], (3,))
						if shape == 'cylinder':
							scale_y = scale_x
						if shape in ['sphere', 'spanner', 'screwdriver', 'hammer']:
							scale_y = scale_x
							scale_z = scale_x
						dest_x = np.random.uniform(self.object_bounds_on['x'][0], self.object_bounds_on['x'][1])
						dest_y = np.random.uniform(self.object_bounds_on['y'][0], self.object_bounds_on['y'][1])
						dest_z = phys_scale[-1]*scale_z/2 + 0.024 + self.table_height  # on the table
						dest_roll, dest_pitch, dest_yaw = 0.0, 0.0, 2*np.pi*np.random.random()
						bad_pos = False
						n_trials += 1
						bad_pos = self.check_bad_positions(name, dest_yaw, dest_x, dest_y, scale_x, scale_y, phys_scale)
					if n_trials == 100:
						visible = False  # make object invisible if no good position is found
						self.object_visibility[self.object_names.index(name)] = False  # less weird way?

				# Set new positions for unvisible objects
				if not visible:  # no 'else' because some names might go through both conditions
					dest_x, dest_y, dest_z = 0.0, 0.0, -1.0
					scale_x, scale_y, scale_z = [self.object_min_scales[shape]]*3
					dest_roll, dest_pitch, dest_yaw = 0.0, 0.0, 0.0

				# Update new states
				pose.position.x = dest_x
				pose.position.y = dest_y
				pose.position.z = dest_z
				ori_4 = quaternion_from_euler(dest_roll, dest_pitch, dest_yaw)
				pose.orientation.x = ori_4[0]
				pose.orientation.y = ori_4[1]
				pose.orientation.z = ori_4[2]
				pose.orientation.w = ori_4[3]
				scale.x = scale_x
				scale.y = scale_y
				scale.z = scale_z

	# Check if new object would overlap with other existing objects
	def check_bad_positions(self, this_name, this_yaw, this_pos_x, this_pos_y, this_scale_x, this_scale_y, this_phys):
		if self.object_bounds_off != None:
			if (self.object_bounds_off['x'][0] < this_pos_x < self.object_bounds_off['x'][1]\
				and self.object_bounds_off['y'][0] < this_pos_y < self.object_bounds_off['y'][1]):
				return True
		rot = Rotation.from_euler('z', -this_yaw, degrees=False)
		this_pos_x, this_pos_y, _ = rot.apply([this_pos_x, this_pos_y, 0.0])
		this_xmin = this_pos_x - this_scale_x*this_phys[0]/2.0
		this_xmax = this_pos_x + this_scale_x*this_phys[0]/2.0
		this_ymin = this_pos_y - this_scale_y*this_phys[1]/2.0
		this_ymax = this_pos_y + this_scale_y*this_phys[1]/2.0
		for (other_name, other_pose, other_scale, other_phys) in\
			zip(self.object_names, self.object_poses, self.object_scales, self.physical_scales):
			if other_name != this_name:
				other_pos_x, other_pos_y = other_pose.position.x, other_pose.position.y
				_, _, other_yaw = ori3_pose = euler_from_quaternion([
					other_pose.orientation.x, other_pose.orientation.y, other_pose.orientation.z, other_pose.orientation.w])
				rot_pro = Rotation.from_euler('z', -other_yaw, degrees=False)
				rot_ant = Rotation.from_euler('z', other_yaw-this_yaw, degrees=False)
				other_pos_x, other_pos_y, _ = rot_pro.apply([other_pos_x, other_pos_y, 0.0])
				other_x1 = other_pos_x - other_scale.x*other_phys[0]/2.0
				other_x2 = other_pos_x + other_scale.x*other_phys[0]/2.0
				other_y1 = other_pos_y - other_scale.y*other_phys[1]/2.0
				other_y2 = other_pos_y + other_scale.y*other_phys[1]/2.0
				other_PA = rot_ant.apply([other_x1, other_y1, 0.0])
				other_PB = rot_ant.apply([other_x1, other_y2, 0.0])
				other_PC = rot_ant.apply([other_x2, other_y1, 0.0])
				other_PD = rot_ant.apply([other_x2, other_y2, 0.0])
				for other_P in [other_PA, other_PB, other_PC, other_PD]:
					if (this_xmin < other_P[0] < this_xmax and this_ymin < other_P[1] < this_ymax):
						return True
		return False

	# Select new random position and angle for the camera
	def choose_new_camera_pose(self):
		radius = np.random.uniform(self.min_camera_radius, self.max_camera_radius)
		angle = 2*np.pi*np.random.random()
		dest_x = self.camera_look_at[0] + radius*np.cos(angle)
		dest_y = self.camera_look_at[1] + radius*np.sin(angle)
		dest_z = self.camera_look_at[2] + np.random.uniform(self.camera_heights[0], self.camera_heights[1])
		dest_roll = 0.0
		dest_pitch = np.arctan2(dest_z - self.camera_look_at[2], radius)  # points towards table plane
		dest_yaw = np.pi + angle
		return dest_x, dest_y, dest_z, dest_roll, dest_pitch, dest_yaw


	# Circular motion around the center
	def update_cameras_positions_and_speeds(self):
		global toto_c
		toto_c +=1
		for camera_id in range(self.n_cameras):

			# Update camera position and orientation according to its speeds
			v_x, v_y, v_z, v_roll, v_pitch, v_yaw = self.camera_speed_states[camera_id]
			print("Moving Camera to: v_x=%0.6f | v_y=%0.6f | v_z=%0.6f | v_roll=%0.6f | v_pitch=%0.6f | v_yaw=%0.6f " % (v_x, v_y, v_z, v_roll, v_pitch, v_yaw))
			self.data_collector.move_camera(camera_id, v_x, v_y, v_z, v_roll, v_pitch, v_yaw)

			# Update the speeds to obtain a circular (or spiralic) motion
			pose = self.data_collector.get_object_pose('camera_%02i' % (camera_id,))
			r_x = self.camera_look_at[0] - pose.position.x
			r_y = self.camera_look_at[1] - pose.position.y
			norm2_r = r_x**2 + r_y**2
			norm2_v = v_x**2 + v_y**2
			factor = norm2_v/norm2_r  # a_c = (v**2/r)*r_unit = (v**2/r**2)*r_vect
			v_x += self.camera_speed_r*factor*r_x
			v_y += self.camera_speed_r*factor*r_y
			if self.camera_speed_r != 1.0:  # keep same v_norm
				new_norm2_v = v_x**2 + v_y**2
				v_x = v_x*(norm2_v/new_norm2_v)**(0.5)
				v_y = v_y*(norm2_v/new_norm2_v)**(0.5)
			self.camera_speed_states[camera_id][0:2] = v_x, v_y


	# For each caemra, initiate new sequences of frames in a given scene
	def reset_cameras(self):
		for camera_id in range(self.n_cameras):

			# Set camera position and angles
			dest_x, dest_y, dest_z, dest_roll, dest_pitch, dest_yaw = self.choose_new_camera_pose()
			pose = self.data_collector.get_object_pose('camera_%02i' % (camera_id,))
			ori3_pose = euler_from_quaternion([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
			x = dest_x - pose.position.x
			y = dest_y - pose.position.y
			z = dest_z - pose.position.z
			roll = dest_roll - ori3_pose[0]
			pitch = dest_pitch - ori3_pose[1]
			yaw = dest_yaw - ori3_pose[2]
			self.data_collector.move_camera(camera_id, x, y, z, roll, pitch, yaw)

			# Compute the correct velocities to turn around the center
			self.camera_speed_t = np.random.uniform(self.camera_speed_range_t[0], self.camera_speed_range_t[1])
			self.camera_speed_r = np.random.uniform(self.camera_speed_range_r[0], self.camera_speed_range_r[1])
			sense = np.random.choice([-1, 1])
			r_x = dest_x - self.camera_look_at[0]
			r_y = dest_y - self.camera_look_at[1]
			v_x = self.camera_speed_t*sense*r_y
			v_y = -self.camera_speed_t*sense*r_x
			v_z = 0.0
			v_roll = 0.0
			v_pitch = 0.0
			norm_r = (r_x**2 + r_y**2)**(0.5)
			norm_v = (v_x**2 + v_y**2)**(0.5)
			v_yaw = -sense*(norm_v/norm_r)

			# Update speed variables of each camera
			# time.sleep(0.1/self.n_cameras)  # might be problematic, max be only need once per scene?
			self.camera_speed_states[camera_id] = [v_x, v_y, v_z, v_roll, v_pitch, v_yaw]

	# Re-shuffle all objects in the scene
	def reset_scene(self):
		self.object_visibility = [np.random.choice([1, 0], p=[p, 1-p]) for p in self.object_instances_prob]
		self.choose_new_object_poses()
		for name, pose, scale in zip(self.object_names, self.object_poses, self.object_scales):
			self.data_collector.set_object_pose(name, pose, False)
			self.data_collector.set_object_scale(name, scale.x, scale.y, scale.z)

	# Update the labels after each scene reset (numpy format, special values for non-visible objects)
	def update_scene_labels(self):
		positions = [[p.position.x, p.position.y, p.position.z] for p in self.object_poses]
		orientations = [euler_from_quaternion([p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]) for p in self.object_poses]
		scales = [[s.x, s.y, s.z] for s in self.object_scales]
		for pos, ori, scl, visible in zip(positions, orientations, scales, self.object_visibility):
			if not visible:
				pos, ori, scl = ([-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0])
		self.data_lbl['pos_3D'] = np.array([positions]*self.n_sequences_per_scene)
		self.data_lbl['ori_3D'] = np.array([orientations]*self.n_sequences_per_scene)
		self.data_lbl['scale_3D'] = np.array([scales]*self.n_sequences_per_scene)
		self.data_lbl['visibles'] = np.array([self.object_visibility]*self.n_sequences_per_scene)

	# Create lists containing all voxel positions (and other useful arrays) inside every object
	def load_object_meshes(self, use_low_poly=True):
		all_vertices_meshes = {shape: np.zeros((0, 3), dtype=float) for shape in self.object_shapes}
		all_triangles_meshes = {shape: np.zeros((0, 3), dtype=int) for shape in self.object_shapes}
		all_physical_scales = {shape: None for shape in self.object_shapes}
		for shape in self.object_shapes:
			dae_file_path = './dae_files/%s_low.dae' % (shape,) if use_low_poly else './dae_files/%s.dae' % (shape,)
			with open(dae_file_path) as f:  # us
				s = f.read()
				vertices_info = re.findall(r'<float_array.+?mesh-positions-array.+?>(.+?)</float_array>', s)
				transform_info = re.findall(r'<matrix sid="transform">(.+?)</matrix>.+?<instance_geometry', s, flags=re.DOTALL)  # better way?
				triangles_info = re.findall(r'<triangles.+?<p>(.+?)</p>.+?</triangles>', s, flags=re.DOTALL)
				if len(triangles_info) == 0:
					triangles_info = re.findall(r'<polylist.+?<p>(.+?)</p>.+?</polylist>', s, flags=re.DOTALL)
				for part_id in range(len(vertices_info)):
					transform_matrix = np.array([float(n) for n in transform_info[part_id].split(' ')]).reshape((4, 4))
					vertices_temp = np.array([float(n) for n in vertices_info[part_id].split(' ')])
					vertices_temp = np.reshape(vertices_temp, (vertices_temp.shape[0]/3, 3))
					vertices_temp = np.dot(transform_matrix, np.c_[vertices_temp, np.ones(vertices_temp.shape[0])].T)[:-1].T
					triangles_temp = np.array([int(n) for n in triangles_info[part_id].split(' ')])[::3]
					triangles_temp = np.reshape(triangles_temp, (triangles_temp.shape[0]/3, 3))
					triangles_temp = triangles_temp + all_vertices_meshes[shape].shape[0]  # shift triangle indices
					all_vertices_meshes[shape] = np.vstack((all_vertices_meshes[shape], vertices_temp))
					all_triangles_meshes[shape] = np.vstack((all_triangles_meshes[shape], triangles_temp))
				min_pos = [all_vertices_meshes[shape][:, d].min() for d in range(3)]
				max_pos = [all_vertices_meshes[shape][:, d].max() for d in range(3)]
				correction = 2.0 if shape in ['sphere', 'box', 'cylinder'] else 1.0  # these .dae files actually do not match the actual objects
				all_physical_scales[shape] = [(max_pos[d] - min_pos[d])/correction for d in range(3)]

		# Better way?
		vertices_meshes_list = []
		triangles_meshes_list = []
		physical_scales_list = []
		posed_meshes_list = [None for name in self.object_names]
		for name in self.object_names:
			shape = name[:-2]
			vertices_meshes_list.append(all_vertices_meshes[shape])
			triangles_meshes_list.append(all_triangles_meshes[shape])
			physical_scales_list.append(all_physical_scales[shape])
		return (np.array(lst) for lst in [vertices_meshes_list, triangles_meshes_list, posed_meshes_list, physical_scales_list])

	# Transform the basic mesh coordinates with actual object psotion, scale and orientation
	def update_object_meshes(self):
		for i, (name, pos, scale, ori, visible) in enumerate(zip(self.object_names,
				self.data_lbl['pos_3D'][0], self.data_lbl['scale_3D'][0],
				self.data_lbl['ori_3D'][0], self.data_lbl['visibles'][0])):
			if visible:
				rot = Rotation.from_euler('z', ori[-1])  # or ('xyz', ori)
				scl = scale/2.0 if any([shape in name for shape in ['sphere', 'box', 'cylinder']]) else scale
				self.posed_meshes[i] = rot.apply(scl*self.vertex_index_meshes[i]) + pos

	# Compute distance of all object vertices to the camera
	def compute_distances_to_cam(self, camera_id, vertices):
		cam = self.data_collector.cam_transform[camera_id]
		cam_pos = [cam.pos_x_m, cam.pos_y_m, cam.elevation_m]
		cam_vector = np.array([l-p for (p,l) in zip(cam_pos, self.camera_look_at)])
		distances = vertices - cam_pos
		proj_dist = np.dot(cam_vector/np.linalg.norm(cam_vector), distances.T)
		norm_dist = np.linalg.norm(distances, axis=1)
		norm_dist[proj_dist < self.camera_near_distance] = np.nan
		return norm_dist

	# Move camera around the scene and take screenshots
	def generate_data_subset(self):
		for sequence_id in range(self.n_sequences_per_camera):
			self.reset_cameras()
			for frame_id in range(self.n_frames_per_sequence):
				self.update_cameras_positions_and_speeds()
				for camera_id in range(self.n_cameras):
					sample_id = sequence_id*self.n_cameras + camera_id
					sequence_sample = self.data_collector.capture_image(camera_id, ensure_timing=True)
					self.data_rgb[sample_id, frame_id] = sequence_sample[0]
					self.data_dvs[sample_id, frame_id] = sequence_sample[1]
					self.segment_image[:] = 0
					self.distance_image[:] = np.inf
					for (visible, vertices, triangles_idx, segment_idx) in zip(
						self.object_visibility, self.posed_meshes, self.triangle_index_meshes, self.object_instances):
						if visible:
							self.segment_layer[:] = 0
							self.distance_layer[:] = np.inf
							vertices_2D = np.array(self.data_collector.cam_transform[camera_id].imageFromSpace(vertices))
							if len(vertices_2D) > 0:
								triangles_2D = np.take(vertices_2D, triangles_idx, axis=0).astype(int)
								distances_to_cam_vertices = self.compute_distances_to_cam(camera_id, vertices)
								distances_to_cam_triangles = np.take(distances_to_cam_vertices, triangles_idx, axis=0).max(axis=1)
								for distance, triangle_2D in sorted(zip(distances_to_cam_triangles, triangles_2D), key=lambda x: x[0])[::-1]:
									if not np.isnan(distance):
										fillConvexPoly(self.distance_layer, triangle_2D, distance)
								self.segment_image[self.distance_layer < self.distance_image] = segment_idx
								self.distance_image = np.minimum(self.distance_image, self.distance_layer)
					self.data_lbl['segments'][sample_id, frame_id] = self.segment_image
		self.data_dvs[:, 0] = 0  # difference between 1st frame and previous one makes no sense
		if plot_gifs:  # with last camera output (just to plot an example)
			self.record_sequence_gif()

	# Save an example gif of object segmentation labelling (uses 1sr camera only)
	def record_sequence_gif(self):
		gif_frames = []
		for frame_id in range(self.n_frames_per_sequence):
			fig, ax = plt.subplots(dpi=150)
			fig.subplots_adjust(hspace=0.5)
			plt.subplot(1,3,1)
			plt.title('Sample\nrgb frame')
			plt.imshow(self.data_rgb[0, frame_id]/255.0)
			plt.axis('off')
			plt.subplot(1,3,2)
			plt.title('Sample\ndvs frame')
			plt.imshow(self.data_dvs[0, frame_id]/255.0)
			plt.axis('off')
			plt.subplot(1,3,3)
			plt.title('Segmentation\nlabelling')
			plt.imshow(self.data_lbl['segments'][0, frame_id], vmin=0, vmax=max(self.object_instances))
			plt.axis('off')
			fig.canvas.draw()  # draw the canvas, cache the renderer
			gif_frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
			gif_frames.append(gif_frame.reshape(fig.canvas.get_width_height()[::-1] + (3,)))
			plt.close()
		imageio.mimsave('./segment_examples/sample_%02i_%02i.gif' % (file_name_index, scene_index+1,), gif_frames, fps=24)

# Generate the whole dataset 1907
if __name__ == '__main__':

	# Camera and scene parameters
	camera_params = {'name': 'robot'}
	camera_params['n_cameras'] = 1
	camera_params['camera_type'] = 'both'  # 'rgb', 'dvs', 'both'
	with open('../room.sdf') as sdf:  # !!!all camera parameters should be the same!!!
		text = sdf.read()
		first_cam_text = text.split('<sensor name="camera" type="camera">')[1].split('</sensor>')[0]
		height = int(first_cam_text.split('<height>')[1].split('</height>')[0])
		width = int(first_cam_text.split('<width>')[1].split('</width>')[0])
		h_fov = float(first_cam_text.split('<horizontal_fov>')[1].split('</horizontal_fov>')[0])
		v_fov = h_fov*float(height)/float(width)
		n_cameras_max = int(text.split('<model name="camera_')[-1].split('">')[0])+1
		update_rate = float(first_cam_text.split('<update_rate>')[1].split('</update_rate>')[0])
		near_distance = float(first_cam_text.split('<near>')[1].split('</near>')[0])
	camera_params['camera_resolution'] = (height, width)
	camera_params['focal_length_px'] = 0.5003983220157445*width  # ??? not sure.. but it works
	camera_params['update_rate'] = update_rate
	camera_params['near_distance'] = near_distance
	if camera_params['n_cameras'] > n_cameras_max:
		print('Too many cameras selected: number of cameras set to %s.' % (n_cameras_max,))
		camera_params['n_cameras'] = n_cameras_max

	# Dataset parameters
	plot_gifs = True  # record segmentation labelling .gif examples
	dataset_output_dir = './datasets'
	dataset_output_name = 'training_room_dataset'  # a number is added to avoid overwriting
	n_color_channels = 3
	image_dimensions = camera_params['camera_resolution'] + (n_color_channels,)
	n_frames_per_sequence = 1  # 20
	n_sequences_per_scene = 1
	assert n_sequences_per_scene % camera_params['n_cameras'] == 0,\
		'Error: n_sequences_per_scene must be a multiple of n_cameras.'
	n_samples_per_dataset = 1  # ~16 GB for uncompressed np.array((64000, 20, 64, 64, 3), dtype=np.uint8)
	n_scenes_per_dataset = int(n_samples_per_dataset/n_sequences_per_scene)
	if float(n_samples_per_dataset)/n_sequences_per_scene - n_scenes_per_dataset > 0:
		n_scenes_per_dataset += 1  # 1 partial run to finish the sequence samples
	training_room = TrainingRoom(camera_params, image_dimensions, n_sequences_per_scene, n_frames_per_sequence)

	# Create datasets to be filled by the NRP simulation
	starting_time = time.time()
	dataset_dims_images = (n_samples_per_dataset, n_frames_per_sequence,) + image_dimensions
	dataset_dims_lbl_3D = (n_samples_per_dataset, len(training_room.object_names), 3)
	dataset_dims_lbl_1D = (n_samples_per_dataset, len(training_room.object_names))
	dataset_dims_lbl_seg = dataset_dims_images[:-1]
	chunk_dims_image = (1,) + dataset_dims_images[1:]
	chunk_dims_seg = (1,) + dataset_dims_images[1:-1]
	file_name_index = len(glob('%s/%s_*.h5' % (dataset_output_dir, dataset_output_name,))) + 1
	dataset_output_name = '%s/%s_%02i.h5' % (dataset_output_dir, dataset_output_name, file_name_index)
	with h5py.File(dataset_output_name, 'w') as f:
		f.create_dataset('rgb_samples', shape=dataset_dims_images, dtype='uint8', chunks=chunk_dims_image, compression='gzip')
		f.create_dataset('dvs_samples', shape=dataset_dims_images, dtype='uint8', chunks=chunk_dims_image, compression='gzip')
		f.create_dataset('lbl_pos_3D', shape=dataset_dims_lbl_3D, dtype='float64')
		f.create_dataset('lbl_ori_3D', shape=dataset_dims_lbl_3D, dtype='float64')
		f.create_dataset('lbl_scale_3D', shape=dataset_dims_lbl_3D, dtype='float64')
		f.create_dataset('lbl_visibles', shape=dataset_dims_lbl_1D, dtype='int64')
		f.create_dataset('lbl_segments', shape=dataset_dims_lbl_seg, chunks=chunk_dims_seg, dtype='uint8', compression='gzip')

		# Fill the dataset with the generated sequences of frames and corresponding labels
		remaining_indexes = np.array(range(n_samples_per_dataset))
		for scene_index in range(n_scenes_per_dataset):
			first_id = scene_index*n_sequences_per_scene
			last_id = min((scene_index+1)*n_sequences_per_scene, n_samples_per_dataset)
			indexes_to_fill = np.random.choice(remaining_indexes, size=(last_id-first_id,), replace=False)
			remaining_indexes = np.delete(remaining_indexes, [np.where(remaining_indexes==idx) for idx in indexes_to_fill])
			sys.stdout.write('\rCreating dataset (%i/%i sequences generated)' % (first_id, n_samples_per_dataset))
			sys.stdout.flush()
			training_room.reset_scene()
			training_room.update_scene_labels()
			training_room.update_object_meshes()
			training_room.generate_data_subset()
			for i, sample_id in enumerate(indexes_to_fill):
				f['rgb_samples'][sample_id] = training_room.data_rgb[i]
				f['dvs_samples'][sample_id] = training_room.data_dvs[i]
				f['lbl_pos_3D'][sample_id] = training_room.data_lbl['pos_3D'][i]
				f['lbl_ori_3D'][sample_id] = training_room.data_lbl['ori_3D'][i]
				f['lbl_scale_3D'][sample_id] = training_room.data_lbl['scale_3D'][i]
				f['lbl_visibles'][sample_id] = training_room.data_lbl['visibles'][i]
				f['lbl_segments'][sample_id] = training_room.data_lbl['segments'][i]

	# Goodbye message
	n_minutes = int((time.time() - starting_time)/60) + 1
	print('\rDataset created in %i minutes (%i/%i sequences generated)' % (n_minutes, n_samples_per_dataset, n_samples_per_dataset))
