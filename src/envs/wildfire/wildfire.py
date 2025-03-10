import numpy as np
# from scipy.ndimage import convolve
import gymnasium as gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec
# import matplotlib.animation as animation
# from matplotlib.patches import Rectangle
import os, string, random
from types import SimpleNamespace

matplotlib.set_loglevel("warning")

import logging
logging.getLogger("PIL").setLevel(logging.WARNING)


# np.set_printoptions(threshold=sys.maxsize)

class Agents:
    def __init__ (self, n_agents, x_len = 100, y_len = 100, desired_comm_dist = 30, r_collision_avoidance = 5, fire_origin = None, int_type1 = np.int8, int_type2 = np.int16):
        self.n_agents = n_agents
        self.x_len = x_len
        self.y_len = y_len
        self.desired_comm_dist = desired_comm_dist
        self.r_collision_avoidance = r_collision_avoidance
        self.int_type1 = int_type1
        self.int_type2 = int_type2
        self.reset_position(fire_origin)
        self.reset_params()

    def choose_agent_init_pos(self, fire_origin):
        x, y = fire_origin

        # Determine the opposite boundary based on fire location
        possible_x = 5 if x > self.x_len // 2 else self.x_len - 6
        possible_y = 5 if y > self.y_len // 2 else self.y_len - 6

        # Randomly choose x or y boundary as the fixed coordinate
        if np.random.rand() > 0.5:
            base_x, base_y = possible_x, np.random.randint(0, self.y_len)
        else:
            base_x, base_y = np.random.randint(0, self.x_len), possible_y

        return base_x, base_y
    
    def initialize_uav_positions(self, fire_origin):
        base_x, base_y = self.choose_agent_init_pos(fire_origin)

        # Define the spacing based on the collision radius (5)
        spacing = self.r_collision_avoidance + 1  # Slightly larger than the collision radius (5)
        n_per_row = int(np.ceil(np.sqrt(self.n_agents)))  # Approximate square layout
        total_spacing = (n_per_row - 1) * spacing  # Max required spacing for UAVs

        # Adjust base_x to ensure all UAVs fit within boundaries
        min_x = 0
        max_x = self.x_len - 1
        base_x = max(min_x + total_spacing // 2, min(base_x, max_x - total_spacing // 2))

        # Adjust base_y to ensure all UAVs fit within boundaries
        min_y = 0
        max_y = self.y_len - 1
        base_y = max(min_y + total_spacing // 2, min(base_y, max_y - total_spacing // 2))

        # Now distribute UAVs in a grid pattern near the base point
        start_x = base_x - (n_per_row // 2) * spacing
        start_y = base_y - (n_per_row // 2) * spacing

        self.positions = []
        for i in range(self.n_agents):
            row, col = divmod(i, n_per_row)  # Arrange in a grid pattern
            new_x = start_x + row * spacing
            new_y = start_y + col * spacing

            self.positions.append((new_x, new_y))

        self.positions = np.array(self.positions).astype(self.int_type2)

        return self.positions

    def reset_position(self, fire_origin):
        # self.positions = np.hstack([np.random.randint(0, self.x_len, (self.n_agents, 1)), np.random.randint(0, self.y_len, (self.n_agents, 1))]).astype(self.int_type2)
        # self.positions = np.array([self.choose_agent_init_pos(fire_origin)] * self.n_agents).astype(self.int_type2)
        # self.positions += np.column_stack((np.arange(self.n_agents), np.zeros(self.n_agents, dtype=int)))
        self.positions = self.initialize_uav_positions(fire_origin)
    
    def reset_params(self):
        self.belief_map = np.full((self.n_agents, self.x_len, self.y_len), 0, dtype=self.int_type1)
        self.coverage_map = np.full((self.n_agents, self.x_len, self.y_len), 0, dtype=self.int_type2)
        self.last_bmap = np.full((self.n_agents, self.x_len, self.y_len), 0, dtype=self.int_type1)
        self.last_cmap = np.full((self.n_agents, self.x_len, self.y_len), 0, dtype=self.int_type2)
        self.last_action = np.full(self.n_agents, 0, dtype=self.int_type1)
        self.nei_adj_mtx = np.full((self.n_agents, self.n_agents), 0, dtype=self.int_type1)
        self.rel_dist = np.full((self.n_agents, self.n_agents, 2), 0, dtype=self.int_type2)

        self.update_rel_mtx()

    def reset(self, fire_origin):
        self.reset_position(fire_origin)
        self.reset_params()

    def reset_debug_rand_pos(self):
        self.n_agents = 3
        self.reset()

    def reset_debug_set_pos(self, pos):
        if self.n_agents == pos.shape[0]:
            self.positions = pos
        else:
            print("Set position shape in compatibility, #agents=", self.n_agents, ", input position shape:", pos.shape)
        self.reset_params()

    def get_position(self, agent_id = -1):
        if agent_id == -1:
            return self.positions.copy()
        else:
            return self.positions[agent_id, :].copy()

    def update_rel_mtx(self):
        self.rel_dist = (self.positions[:, np.newaxis, :] - self.positions[np.newaxis, :, :])
        self.nei_adj_mtx = ( np.linalg.norm(self.rel_dist, axis=2) <= self.desired_comm_dist ).astype(self.int_type1)
        np.fill_diagonal(self.nei_adj_mtx, 0)
        self.rel_dist = self.rel_dist * self.nei_adj_mtx[:, :, np.newaxis]

    def copy_maps(self):
        self.last_bmap = self.belief_map.copy()
        self.last_cmap = self.coverage_map.copy()


class WildfireEnvironment(gym.Env):
  
    def __init__ (
        self,
        map_name = "circular",
        episode_limit = 320,
        desired_comm_dist = 30,
        r_collision_avoidance = 5,
        n_agents = 3,
        kick_start_timesteps = 12,
        t_env_update = 4,
        x_len = 100,
        y_len = 100,
        # fire_x_start = 48,
        # fire_y_start = 48, 
        # fire_x_end = 52, 
        # fire_y_end = 52, 
        flame_min = 15, 
        flame_max = 20, 
        seed = None,
        dmax = 2.5, 
        K = 0.05, 
        d_sight = 10,
        c_v = 10,
        est_fire_noise = 5,
        rho = 0.25,
        rho_n = 0.15,
    ):
        super().__init__()

        self.map_name = map_name

        self.x_len = x_len
        self.y_len = y_len
        # self.fire_x_start = fire_x_start
        # self.fire_y_start = fire_y_start
        # self.fire_x_end = fire_x_end
        # self.fire_y_end = fire_y_end
        self.flame_min = flame_min
        self.flame_max = flame_max

        self.dmax = dmax
        self.K = K
        self.d_sight = d_sight
        self.fire_count = 0
        self.r_collision_avoidance = r_collision_avoidance
        self.desired_comm_dist = desired_comm_dist
        self.t_env_update = t_env_update
        self.kick_start_timesteps = kick_start_timesteps
        self.episode_limit = episode_limit
        self.c_v = c_v
        self.est_fire_noise = est_fire_noise
        self.rho = rho
        self.rho_n = rho_n

        self.int_type1 = np.int8
        self.int_type2 = np.int16
        if np.max([255, self.x_len, self.y_len, self.episode_limit]) > np.iinfo(np.int16).max:
            self.int_type2 = np.int32

        # self.fire_origin = np.array([np.random.randint(int(self.x_len/4), int((self.x_len*3)/4)), np.random.randint(int(self.y_len/4), int((self.y_len*3)/4))]).astype(self.int_type2)
        self.fire_origin = np.array([46, 27])

        self.n_agents = n_agents
        self.agents = Agents(self.n_agents, self.x_len, self.y_len, self.desired_comm_dist, self.r_collision_avoidance, self.fire_origin, self.int_type1, self.int_type2)
        self.agents_obs = np.full((self.n_agents, self.get_obs_size()), -1)
        self.agents_states = np.full((self.n_agents, self.get_state_size()), -1)
        # self.agents_possible_actions = np.full((self.n_agents, 9), 1)
        self.agents_action_scores = np.full((self.n_agents, 9), 0)

        self.debugging = False
        print("Debugging status: ", self.debugging)
        self.test_case_id = 6
        self.test_case_coords = [[[40, 40], [53, 18], [65, 40]], [[40, 59], [18, 46], [40, 34]], [[59, 59], [46, 81], [34, 59]], [[59, 40], [81, 53], [59, 65]], [[40, 65], [40, 40], [65, 40]], [[65, 59], [40, 59], [40, 34]], [[59, 34], [59, 59], [34, 59]], [[34, 40], [59, 40], [59, 65]], [[50, 60], [65, 40], [80, 20]], [[60, 49], [40, 34], [20, 19]], [[49, 39], [34, 59], [19, 79]], [[39, 50], [59, 65], [79, 80]], [[50, 60], [65, 40], [30, 20]], [[60, 49], [40, 34], [20, 69]], [[49, 39], [34, 59], [69, 79]], [[39, 50], [59, 65], [79, 30]], [[50, 60], [80, 30], [30, 20]], [[60, 49], [30, 19], [20, 69]], [[49, 39], [19, 69], [69, 79]], [[39, 50], [69, 80], [79, 30]], [[0, 0], [0, 21], [21, 0]], [[0, 99], [21, 99], [0, 78]], [[99, 99], [99, 78], [78, 99]], [[99, 0], [78, 0], [99, 21]], [[0, 0], [0, 25], [25, 0]], [[0, 99], [25, 99], [0, 74]], [[99, 99], [99, 74], [74, 99]], [[99, 0], [74, 0], [99, 25]], [[20, 0], [45, 0], [70, 0]], [[0, 79], [0, 54], [0, 29]], [[79, 99], [54, 99], [29, 99]], [[99, 20], [99, 45], [99, 70]], [[10, 0], [45, 0], [80, 0]], [[0, 89], [0, 54], [0, 19]], [[89, 99], [54, 99], [19, 99]], [[99, 10], [99, 45], [99, 80]], [[0, 0], [0, 25], [25, 25]], [[0, 99], [25, 99], [25, 74]], [[99, 99], [99, 74], [74, 74]], [[99, 0], [74, 0], [74, 25]], [[0, 20], [0, 45], [22, 33]], [[20, 99], [45, 99], [33, 77]], [[99, 79], [99, 54], [77, 66]], [[79, 0], [54, 0], [66, 22]], [[20, 0], [45, 0], [80, 0]], [[0, 79], [0, 54], [0, 19]], [[79, 99], [54, 99], [19, 99]], [[99, 20], [99, 45], [99, 80]], [[20, 0], [45, 0], [40, 30]], [[0, 79], [0, 54], [30, 59]], [[79, 99], [54, 99], [59, 69]], [[99, 20], [99, 45], [69, 40]], [[0, 40], [20, 25], [25, 50]], [[40, 99], [25, 79], [50, 74]], [[99, 59], [79, 74], [74, 49]], [[59, 0], [74, 20], [49, 25]], [[0, 40], [20, 25], [40, 40]], [[40, 99], [25, 79], [40, 59]], [[99, 59], [79, 74], [59, 59]], [[59, 0], [74, 20], [59, 40]], [[0, 40], [20, 25], [50, 50]], [[40, 99], [25, 79], [50, 49]], [[99, 59], [79, 74], [49, 49]], [[59, 0], [74, 20], [49, 50]], [[0, 40], [30, 10], [50, 50]], [[40, 99], [10, 69], [50, 49]], [[99, 59], [69, 89], [49, 49]], [[59, 0], [89, 30], [49, 50]], [[0, 40], [30, 0], [50, 50]], [[40, 99], [0, 69], [50, 49]], [[99, 59], [69, 99], [49, 49]], [[59, 0], [99, 30], [49, 50]], [[0, 0], [0, 0], [0, 0]]]

        self.total_rew = 0
        self.recorded_fire_map = np.full((self.x_len, self.y_len), 0, dtype=self.int_type2)

        self.env_render_color = {
            "field" : (34/255, 139/255, 34/255),
            "fire" : (255/255, 94/255, 5/255),
            "burnt" : (25/255, 25/255, 25/255),
            "drones" : [(1/255, 148/255, 202/255), (255/255, 0/255, 0/255), (139/255, 69/255, 19/255), (0/255, 0/255, 255/255), (127/255, 0/255, 255/255), (255/255, 0/255, 255/255), (255/255, 0/255, 127/255), (128/255, 128/255, 128/255), (139/255, 0/255, 0/255), (85/255, 107/255, 47/255)],
            "drone_trajs" : [(112/255, 207/255, 238/255), (255/255, 102/255, 102/255), (160/255, 82/255, 45/255), (102/255, 102/255, 255/255), (178/255, 102/255, 255/255), (255/255, 102/255, 255/255), (255/255, 102/255, 178/255), (192/255, 192/255, 192/255), (165/255, 42/255, 42/255), (107/255, 142/255, 35/255)]
        }

        # self.binary_val_record = []
        # self.fire_set_record = []
        # self.fire_off_record = []
        # self.whole_fire_set_record = []
        self.map_record = []
        

    def reset(self, test_case_id = 1):

        self.test_case_id = test_case_id

        self.timestep = 0

        self.binary_val = np.full([self.x_len, self.y_len], 0, dtype=self.int_type1)
        self.fuel_map = np.random.randint(self.flame_min, self.flame_max+1, [self.x_len, self.y_len])
        self.fire_set = set()
        self.fire_off = set()
        self.whole_fire_set = set()

        # self.fire_origin = np.array([np.random.randint(int(self.x_len/4), int((self.x_len*3)/4)), np.random.randint(int(self.y_len/4), int((self.y_len*3)/4))]).astype(self.int_type2)

        # for i in range(self.fire_origin[0] - 2, self.fire_origin[0] + 2):
        #     for j in range(self.fire_origin[1] - 2, self.fire_origin[1] + 2):
        #         self.binary_val[i][j] = 1
        #         self.fire_set.add((i,j))
        #         self.whole_fire_set.add((i,j))

        # for _ in range(self.kick_start_timesteps):
        #     self.simStep()

        # self.binary_val_record.append(self.binary_val)
        # self.fire_set_record.append(list(self.fire_set))
        # self.fire_off_record.append(list(self.fire_off))
        # self.whole_fire_set_record.append(list(self.whole_fire_set))

        self.binary_val_ep_record = np.load("binary_val_ep_record.npy", allow_pickle=True)
        self.fire_set_ep_record = np.load("fire_set_ep_record.npy", allow_pickle=True)
        self.fire_off_ep_record = np.load("fire_off_ep_record.npy", allow_pickle=True)
        self.whole_fire_set_ep_record = np.load("whole_fire_set_ep_record.npy", allow_pickle=True)

        self.binary_val = np.array(self.binary_val_ep_record[self.timestep], dtype=self.int_type1)
        self.fire_set = set(map(tuple, self.fire_set_ep_record[self.timestep]))
        self.fire_off = set(map(tuple, self.fire_off_ep_record[self.timestep]))
        self.whole_fire_set = set(map(tuple, self.whole_fire_set_ep_record[self.timestep]))

        self.map_record = [self.binary_val]

        # self.fire_origin = np.array([52, 65])

        self.agents.reset(self.fire_origin)
        # self.agents_obs = np.full((self.n_agents, self.get_obs_size()), -1)
        # self.agents_states = np.full((self.n_agents, self.get_state_size()), -1)
        # self.agents_possible_actions = np.full((self.n_agents, 9), 1)
        # self.agents_action_scores = np.full((self.n_agents, 9), 0)
        
        if self.debugging:
            print("Debugging test_case_id: ", test_case_id)
            self.agents.reset_debug_set_pos(np.array(self.test_case_coords[self.test_case_id-1]))

        self.map_unionize()
        self.agents.update_rel_mtx()
        # self.agents_possible_actions = self.update_movement_dir()
        self.agents_action_scores = self.calculate_action_scores()
        self.agents_obs = self.update_obs()
        self.agents_states = self.update_states()

        self.total_rew = 0
        self.recorded_fire_map = np.full((self.x_len, self.y_len), 0, dtype=self.int_type2)

        self.trajectories = [self.agents.get_position()]
        self.fire_trajectories = [{"fire_on" : np.array(list(self.fire_set)), "fire_off" : np.array(list(self.fire_off))}]
        self.map_history = [{"belief_map" : self.agents.belief_map.copy(), "coverage_map" : self.agents.coverage_map.copy()}]
        self.agents_action_scores_history = [self.agents_action_scores]
        # self.uav_views = []
        ### Calling reward() to initialize uav_views[0].
        _, _ = self.reward()

    def map_unionize(self):
        ## Call this function before calculate_action_score
        latest_agent_indices = np.argmax(self.agents.coverage_map, axis=0)
        latest_fire_values = self.agents.belief_map[latest_agent_indices, np.arange(self.x_len)[:, np.newaxis], np.arange(self.y_len)]
        latest_coverage_values = self.agents.coverage_map[latest_agent_indices, np.arange(self.x_len)[:, np.newaxis], np.arange(self.y_len)]
        for agent_idx in range(self.n_agents):
            self.agents.belief_map[agent_idx] = latest_fire_values
            self.agents.coverage_map[agent_idx] = latest_coverage_values

        self.recorded_fire_map += latest_fire_values

    def calculate_action_scores(self):

        positions = self.agents.get_position()

        action_scores = np.full((self.n_agents, 9), 0)

        x_view_min = np.maximum(positions[:, 0] - self.d_sight, 0)
        x_view_max = np.minimum(positions[:, 0] + self.d_sight, self.x_len - 1)
        y_view_min = np.maximum(positions[:, 1] - self.d_sight, 0)
        y_view_max = np.minimum(positions[:, 1] + self.d_sight, self.y_len - 1)

        for agent_id in range(self.n_agents):
        
            grid_current = self.agents.belief_map[agent_id, x_view_min[agent_id]:x_view_max[agent_id], y_view_min[agent_id]:y_view_max[agent_id]]
            grid_last = self.agents.last_bmap[agent_id, x_view_min[agent_id]:x_view_max[agent_id], y_view_min[agent_id]:y_view_max[agent_id]]

            expanded_map = np.zeros((self.x_len * 2, self.y_len * 2))
            expanded_map[(x_view_min[agent_id] + self.x_len):(x_view_max[agent_id] + self.x_len), (y_view_min[agent_id] + self.y_len):(y_view_max[agent_id] + self.y_len)] = np.clip(grid_current - grid_last, 0, None)
            grid_diff = expanded_map[(positions[agent_id, 0] - self.d_sight + self.x_len):(positions[agent_id, 0] + self.d_sight + self.x_len), (positions[agent_id, 1] - self.d_sight + self.y_len):(positions[agent_id, 1] + self.d_sight + self.y_len)].copy()

            grid_dim = grid_diff.shape
            x_half, y_half = grid_dim[0] // 2, grid_dim[1] // 2
            x_quat, y_quat = grid_dim[0] // 4, grid_dim[1] // 4

            action_scores[agent_id, 0] = np.sum(grid_diff[x_quat:3*x_quat, y_quat:3*y_quat])

            action_scores[agent_id, 1] = np.sum(np.triu(grid_diff[:x_half, :y_half])) + np.sum(np.tril(np.rot90(grid_diff[:x_half, y_half:]), k=1))
            
            action_scores[agent_id, 2] = np.sum(grid_diff[:x_half, y_half:])  # Top-right
            
            action_scores[agent_id, 3] = np.sum(np.triu(np.rot90(grid_diff[:x_half, y_half:]), k=1)) + np.sum(np.triu(grid_diff[x_half:, y_half:]))
            
            action_scores[agent_id, 4] = np.sum(grid_diff[x_half:, y_half:])  # Bottom-right
            
            action_scores[agent_id, 5] = np.sum(np.tril(grid_diff[x_half:, y_half:])) + np.sum(np.triu(np.rot90(grid_diff[x_half:, :y_half]), k=1))
            
            action_scores[agent_id, 6] = np.sum(grid_diff[x_half:, :y_half])  # Bottom-left
            
            action_scores[agent_id, 7] = np.sum(np.tril(np.rot90(grid_diff[x_half:, :y_half]), k=1)) + np.sum(np.tril(grid_diff[:x_half, :y_half]))
            
            action_scores[agent_id, 8] = np.sum(grid_diff[:x_half, :y_half])  # Top-left

        return action_scores

    def update_movement_dir(self):
        
        positions = self.agents.get_position()
        new_mv_dir = np.full((self.n_agents, 9), 1)

        for agent_id in range(self.n_agents):
            [x, y] = positions[agent_id]
            if x == 0:
                new_mv_dir[agent_id, 8] = 0
                new_mv_dir[agent_id, 1] = 0
                new_mv_dir[agent_id, 2] = 0
            if y == 0:
                new_mv_dir[agent_id, 6] = 0
                new_mv_dir[agent_id, 7] = 0
                new_mv_dir[agent_id, 8] = 0
            if x == self.x_len-1:
                new_mv_dir[agent_id, 4] = 0
                new_mv_dir[agent_id, 5] = 0
                new_mv_dir[agent_id, 6] = 0
            if y == self.y_len-1:
                new_mv_dir[agent_id, 2] = 0
                new_mv_dir[agent_id, 3] = 0
                new_mv_dir[agent_id, 4] = 0

        ## Later, Set 0 if the agents are about to collide.

        return new_mv_dir
    
    def update_obs(self):
        # position: (n X 2)
        # movement_direction: (n X 9)
        # unit_features: ( n X (n-1)*4 ) || rel_x, rel_y, euc_dist, last_action
        # Fire discovery status: (n X n)
        # action_scores: (n X 9)

        # Estimated fire location (noised)
        est_fire_coords = np.round(np.random.normal(self.fire_origin, self.est_fire_noise)).astype(self.int_type2)
        est_fire_rel_coords = np.tile(est_fire_coords, (self.n_agents, 1)) - self.agents.get_position()
        est_fire_euc_dist = np.linalg.norm(est_fire_rel_coords, axis=1)[:, np.newaxis]

        action_score_feats = self.agents_action_scores
        fire_disc_idx = np.sum(action_score_feats, axis=1) > 0
        nei_mask = self.agents.nei_adj_mtx.copy()
        nei_mask[fire_disc_idx, fire_disc_idx] = 1
        fire_disc_mask = fire_disc_idx + fire_disc_idx[:, np.newaxis]
        fire_disc_status = fire_disc_mask * nei_mask

        agents_obs = []
        for agent_id in range(self.n_agents):
            # movement_feats = self.agents_possible_actions[agent_id]
            rel_pos = np.delete(self.agents.rel_dist[agent_id], agent_id, axis = 0) / np.mean([self.x_len, self.y_len])
            euc_dist = np.linalg.norm(rel_pos, axis=1)[:, np.newaxis] / np.linalg.norm([self.x_len, self.y_len])
            self_feats = np.hstack((rel_pos, euc_dist, np.delete(self.agents.last_action, agent_id, axis = 0)[:, np.newaxis] / self.get_total_actions()))

            # fire_disc_status = [0]
            # if np.sum(action_score_feats) > 0:
            #     fire_disc_status[0] = 1

            agents_obs.append(np.concatenate(
                    (
                        self.agents.get_position(agent_id) / np.mean([self.x_len, self.y_len]),
                        # movement_feats,
                        self_feats.flatten(),
                        fire_disc_status[agent_id],
                        action_score_feats[agent_id] / np.max([self.x_len, self.y_len]),
                        est_fire_rel_coords[agent_id] / np.mean([self.x_len, self.y_len]),
                        est_fire_euc_dist[agent_id] / np.linalg.norm([self.x_len, self.y_len]),
                        [self.timestep / self.episode_limit],
                    )
                )
            )

        return np.array(agents_obs)

    def update_states(self):
        # Positions: (n X 2)
        # # Unit_features: (n X 3) || rel_x, rel_y, euc_dist
        # Action_scores: (n X 9)
        # Last_actions: (n X 1)
        # Closest fire features: (n X 3) || rel_x, rel_y, euc_dist
        # Neighbor adjacency matrix: (n X n)
        # Relative_XY with all agents: (n X n X 2) -> (n X n*2)
        # Euc. distance with all agents: (n X n)
        # Fire center and radius: 3 - Added later in get_state() function.

        positions = self.agents.get_position()
        # center = np.array([[np.floor(self.x_len/2), np.floor(self.y_len/2)]] * self.n_agents)
        # rel_pos = center - positions
        # euc_dist = np.linalg.norm(rel_pos, axis = 1)[:, np.newaxis]

        # action_marks = np.column_stack((np.arange(self.n_agents), self.agents.last_action))
        # prev_actions = np.full((self.n_agents, 9), 0)
        # prev_actions[action_marks[:,0], action_marks[:,1]] = 1

        rel_dist_closest_fire = self.find_closest_fire(positions)

        # print(rel_pos.shape)
        # print(euc_dist.shape)
        # print(self.agents_action_scores.shape)
        # print(self.agents.last_action)
        # print(self.agents.last_action[:, np.newaxis].shape)
        # print(rel_dist_closest_fire.shape)
        # print(self.agents.nei_adj_mtx.shape)
        # print(self.agents.rel_dist.reshape(self.n_agents, (self.n_agents * 2)).shape)
        # print(np.linalg.norm(self.agents.rel_dist, axis=2).shape)
        agent_states = np.hstack([
            positions / np.mean([self.x_len, self.y_len]), 
            self.agents_action_scores / np.max([self.x_len, self.y_len]), 
            self.agents.last_action[:, np.newaxis] / self.get_total_actions(), 
            rel_dist_closest_fire / np.mean([self.x_len, self.y_len]), 
            self.agents.nei_adj_mtx, 
            self.agents.rel_dist.reshape(self.n_agents, (self.n_agents * 2)) / np.mean([self.x_len, self.y_len]), 
            np.linalg.norm(self.agents.rel_dist, axis=2) / np.linalg.norm([self.x_len, self.y_len]),
            ])
        return np.append(agent_states, (self.timestep / self.episode_limit))
    
    def find_closest_fire(self, positions):
        fire_coords = np.array(list(self.fire_set))
        distances = np.linalg.norm(positions[:, np.newaxis] - fire_coords, axis=2)

        closest_indices = np.argmin(distances, axis=1)

        closest_coords = fire_coords[closest_indices]
        closest_distances = np.min(distances, axis=1)[:, np.newaxis]

        # Calculate relative x & y
        relative_coords = closest_coords - positions

        return np.hstack([relative_coords, closest_distances])
    
    def get_positions(self, agent_id = None):
        if agent_id is None:
            return self.agents.get_position()
        else:
            return self.agents.get_position(agent_id)
    
    def get_neighbors(self, agent_id = None):
        if agent_id is None:
            return np.argwhere(np.triu(self.agents.nei_adj_mtx))
        else:
            return np.where(self.agents.nei_adj_mtx[agent_id] > 0)[0]
    
    def get_obs_agent(self, agent_id):
        return self.agents_obs[agent_id]

    def get_obs(self):
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return np.array(agents_obs)
    
    def get_avail_agent_actions(self, agent_id):
        # return self.agents_possible_actions[agent_id]
        return np.full(9, 1)

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return np.array(avail_actions)
    
    def get_state(self): 
        return np.expand_dims(np.append(self.agents_states.flatten(), 
                [self.fire_origin[0], self.fire_origin[1], 
                 np.mean(np.linalg.norm(np.array([self.fire_origin] * len(self.fire_set)) - np.array(list(self.fire_set)), axis = 1))
                ]),
            axis = 0
        )
    
    def get_unit_by_id(self, a_id):
        pos = self.get_positions(agent_id=a_id)
        return SimpleNamespace(pos=SimpleNamespace(x=pos[0], y=pos[1]))

    def get_best_dir(self, agent_pos):
        """
        Returns an integer 1..8 for the best direction from UAV (i_u,j_u) to fire (i_f,j_f).
        If (i_f == i_u) and (j_f == j_u), you could return 0 or something else.
        """
        di, dj = self.fire_origin - agent_pos
        if di == 0 and dj == 0:
            return 0  # or "stay"
        
        # sign-based approach
        vertical = 0  # -1 for north, +1 for south, 0 for none
        horizontal = 0  # -1 for west, +1 for east, 0 for none
        
        if di < 0:
            vertical = -1  # north
        elif di > 0:
            vertical = +1  # south
        
        if dj < 0:
            horizontal = -1  # west
        elif dj > 0:
            horizontal = +1  # east
        
        # map (vertical, horizontal) to direction index
        # 1=north,2=NE,3=east,4=SE,5=south,6=SW,7=west,8=NW
        direction_map = {
            (-1, 0): 1,  # north
            (-1, +1): 2, # north-east
            (0, +1): 3,  # east
            (+1, +1): 4, # south-east
            (+1, 0): 5,  # south
            (+1, -1): 6, # south-west
            (0, -1): 7,  # west
            (-1, -1): 8  # north-west
        }
        return direction_map.get((vertical, horizontal), 0)  # fallback 0=stay


    def get_action_probs(self):
        """ 
        Returns an action from 0..8 according to a discrete 'Gaussian-like' distribution 
        around the best direction (1..8).
        """
        action_probs = []
        for agent_pos in self.get_positions():
            best_dir = self.get_best_dir(agent_pos)
            
            # If best_dir is 0, we can either stay or pick some default distribution
            # For now, let's handle it gracefully:
            if best_dir == 0:
                # Means UAV is already at (i_f, j_f).
                # Could just pick stay with prob=1, or do a uniform distribution across all actions, etc.
                # Let's do uniform here for illustration:
                return np.random.randint(0, 9)
            
            # Probability array for 9 actions
            probs = [0.0] * 9
            
            # Solve for gamma: self.rho + 2self.rho_n + 6gamma = 1 => gamma = (1 - self.rho - 2self.rho_n)/6
            remainder = 1.0 - self.rho - 2 * self.rho_n
            if remainder < 0:
                # If self.rho/self.rho_n are chosen badly (sum >1), fallback to uniform or clip them
                return np.random.randint(0, 9)
            gamma = remainder / 6.0
            
            # best_dir gets self.rho
            probs[best_dir] = self.rho
            
            # neighbors in a ring
            left_neighbor = ((best_dir - 1 - 1) % 8) + 1  # subtract 1 for 1..8 range, do mod 8, add 1 back
            right_neighbor = ((best_dir - 1 + 1) % 8) + 1
            probs[left_neighbor] = self.rho_n
            probs[right_neighbor] = self.rho_n
            
            # all others (including 0=stay)
            for action in range(9):
                if action not in (best_dir, left_neighbor, right_neighbor):
                    probs[action] = gamma
            
            action_probs.append(probs)
            
        return np.array(action_probs)
    

    # def get_dynamic_weights(self):
    #     dist_matrix = np.linalg.norm(self.agents.rel_dist, axis=2)
    #     dyn_w_mtx = np.zeros_like(self.agents.nei_adj_mtx).astype(np.float64)
    #     for i in range(self.n_agents):
    #         nei_coords = np.where(self.agents.nei_adj_mtx[i] > 0)[0]
    #         N = len(nei_coords)
    #         if N > 0:
    #             for nei_coord in nei_coords:
    #                 cw1 = ( (2 * self.c_v) / ((self.desired_comm_dist**2) * N ) )
    #                 cw1 = np.random.rand() * cw1

    #                 V = ( ( dist_matrix[i, nei_coord] ** 2 ) + self.c_v ) / (self.desired_comm_dist ** 2)
    #                 dyn_w_mtx[i, nei_coord] = (cw1 / V)
    #             dyn_w_mtx[i, i] = (1 - np.sum(dyn_w_mtx[i, :]))
    #     return dyn_w_mtx

    # def simStep(self):
    #     kernel = np.ones((5, 5))
    #     kernel[2, 2] = 0

    #     fire_matrix = np.zeros((self.x_len, self.y_len), dtype=int)
    #     for i, j in self.fire_set:
    #         fire_matrix[i, j] = 1

    #     fire_neighbors = np.array(convolve(self.binary_val, kernel, mode='constant', cval=0) > 0).astype(self.int_type1)
    #     fire_neighbors[self.binary_val == 1] = 0

    #     potential_neighbors = np.argwhere(fire_neighbors)
    #     self.B = potential_neighbors

    #     for i, j in self.fire_set:
    #         if self.fuel_map[i, j] > 0:
    #             self.fuel_map[i, j] -= 1
    #         else:
    #             self.fire_off.add((i, j))

    #     # pre_univ_fire_count = len(self.whole_fire_set)

    #     Pnmkl_matrix = np.ones_like(fire_neighbors, dtype=float)

    #     for ni, nj in potential_neighbors:
    #         # Get the slice of the 5x5 neighborhood around the potential fire cell
    #         i_slice = slice(max(ni - 2, 0), min(ni + 3, self.x_len))
    #         j_slice = slice(max(nj - 2, 0), min(nj + 3, self.y_len))

    #         fire_in_slice = fire_matrix[i_slice, j_slice]
    #         d_nmkl_squared = (np.arange(i_slice.start, i_slice.stop)[:, None] - ni) ** 2 + (np.arange(j_slice.start, j_slice.stop) - nj) ** 2

    #         # Ignore distances that are larger than `dmax`
    #         mask = d_nmkl_squared < 2.5 ** 2
    #         fire_distances = d_nmkl_squared[(fire_in_slice & mask).astype(bool)]

    #         # # Wind Code
    #         # self.w_dir = [0, 45, 90, 135, 180, 225, 270, 315]
    #         # self.w_speed = [0.0, 0.5, 1.0]
    #         # self.wind = [np.random.choice(self.w_dir), self.w_speed[2]]
    #         # d_nmkl_angle = np.degrees(np.arctan2(j - l, i - k))
    #         # vec_diff = d_nmkl_angle - self.wind[0]
    #         # if vec_diff < 0:
    #         #     vec_diff += 360
    #         # P_nmkl_w = (self.wind[1]*d_nmkl*np.cos(vec_diff*np.pi/180))

    #         if fire_distances.size > 0:
    #             Pnmkl_matrix[ni, nj] = np.prod(1 - np.clip(self.K * (1 / fire_distances), 0, 1))

    #     ignition_chance = np.random.uniform(size=fire_neighbors.shape)
    #     new_fires = (1 - Pnmkl_matrix) >= ignition_chance
        
    #     # plt.imshow(new_fires, cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
    #     # plt.colorbar()  # Add a colorbar to show the mapping
    #     # plt.show()

    #     ignition_chance = np.random.uniform(size=fire_neighbors.shape)
    #     new_fires = (1 - Pnmkl_matrix) >= ignition_chance

    #     # 7. Update the fire set
    #     new_fire_coords = np.argwhere(new_fires)
    #     for ni, nj in new_fire_coords:
    #         self.fire_set.add((ni, nj))
    #         self.whole_fire_set.add((ni, nj))
    #         self.binary_val[ni, nj] = 1

    #     # Remove burned out fires
    #     for ind_fire_off in self.fire_off:
    #         self.fire_set.discard(ind_fire_off)
    #         self.binary_val[ind_fire_off[0], ind_fire_off[1]] = 0

    #     # # Update fire count
    #     # self.fire_count = len(self.whole_fire_set) - pre_univ_fire_count

    def step(self, actions):

        terminated = False
        info = {}
        
        self.timestep += 1

        # if self.timestep % self.t_env_update == 0 :
        #     self.simStep()

        # self.binary_val_record.append(self.binary_val)
        # self.fire_set_record.append(list(self.fire_set))
        # self.fire_off_record.append(list(self.fire_off))
        # self.whole_fire_set_record.append(list(self.whole_fire_set))

        self.binary_val = np.array(self.binary_val_ep_record[self.timestep], dtype=self.int_type1)
        self.fire_set = set(map(tuple, self.fire_set_ep_record[self.timestep]))
        self.fire_off = set(map(tuple, self.fire_off_ep_record[self.timestep]))
        self.whole_fire_set = set(map(tuple, self.whole_fire_set_ep_record[self.timestep]))

        self.map_record.append(self.binary_val)

        actions_int = [int(a) for a in actions]

        self.agents.coverage_map = (self.agents.coverage_map - 1).clip(min = 0)

        ### Assign action_array to agents.last_action
        self.agents.last_action = np.array(actions_int)

        ### Based on the action_array, create the change matrix and sum it with agents.position
        action_ch = np.array([[0, 0], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]).astype(self.int_type1)
        action_pr = action_ch[actions_int]
        self.agents.positions += action_pr

        ### Retrieve indices of position going out of bound
        min_boundary = np.array([0, 0])
        max_boundary = np.array([self.x_len, self.y_len]) - 1
        neg_rew_idx = np.any((self.agents.positions < min_boundary) | (self.agents.positions > max_boundary), axis=1)

        ### Clip the agent positions
        self.agents.positions = np.clip(self.agents.positions, min_boundary, max_boundary)

        ### Re-calculate relative distance, euc dist of agents. (Adjacency matrix)
        self.agents.update_rel_mtx()

        ### Calculate rewards; Assign -1 if any of the agent's euc dist < collision distance
        rew, rew_report = self.reward()

        for i, idx in enumerate(neg_rew_idx):
            if idx:
                rew[i] = -5
                terminated = True
                rew_report[i][3] = -5
                # info["boundary_crossed"] = True
                # info["boundary_crossed_agent_id"] = i

        if self.timestep >= ( self.episode_limit - 1 ):
            terminated = True
            # info["episode_limit"] = True
            # np.save("binary_val_ep_record.npy", np.array(self.binary_val_record, dtype=object))
            # np.save("fire_set_ep_record.npy", np.array(self.fire_set_record, dtype=object))
            # np.save("fire_off_ep_record.npy", np.array(self.fire_off_record, dtype=object))
            # np.save("whole_fire_set_ep_record.npy", np.array(self.whole_fire_set_record, dtype=object))
            # print("Saved the wildfire episode.")

        # print("Reward report:\n", rew_report)
        self.total_rew += np.sum(rew)

        ### Update everything as a part of last step in Step() here.
        # Update position (Done already!)
        # Update Maps
        # Update last actions (Done already!)
        # Update adjacency matrix (Done already!)
        # Update Relative distance (Done already!)
        # Update action_scores, movement_dir, obs, state
        self.map_unionize()

        # self.agents_possible_actions = self.update_movement_dir()
        self.agents_action_scores = self.calculate_action_scores()
        self.agents_obs = self.update_obs()
        self.agents_states = self.update_states()

        self.trajectories.append(self.agents.get_position())
        # self.fire_trajectories.append({"fire_on" : np.array(list(self.fire_set)), "fire_off" : np.array(list(self.fire_off))})
        # self.map_history.append({"belief_map" : self.agents.belief_map.copy(), "coverage_map" : self.agents.coverage_map.copy()})
        # self.agents_action_scores_history.append(self.agents_action_scores)

        return rew, terminated, self.get_stats()
    
    def reshape_action_scores_to_display(self, action_scores):
        out = np.zeros((3, 3))

        out[0, 0] = action_scores[8]
        out[0, 1] = action_scores[1]
        out[0, 2] = action_scores[2]
        out[1, 0] = action_scores[7]
        out[1, 1] = action_scores[0]
        out[1, 2] = action_scores[3]
        out[2, 0] = action_scores[6]
        out[2, 1] = action_scores[5]
        out[2, 2] = action_scores[4]

        return out

    def generate_random_keyword(self, length=8):
        """Generates a random keyword of specified length."""
        characters = string.ascii_letters + string.digits
        return ''.join(random.choice(characters) for i in range(length))

    def generate_multiple_keywords(self, num_keywords=3, length=8):
        """Generates a list of random keywords."""
        return [self.generate_random_keyword(length) for _ in range(num_keywords)]

    
    def render(self, ep_n, n_ts = 0, path = "", test_mode = False):
        # print(self.fire_origin)
        # print(self.trajectories)

        mode = "tr"
        if test_mode:
            mode = "te"

        whole_fire_set_map = np.zeros((100, 100))
        for (wx, wy) in self.whole_fire_set:
            whole_fire_set_map[wx, wy] = 1
        
        keyword = self.generate_random_keyword()
        # np.save("i_trajs/" + keyword + "_" + mode + "_fire_origin.npy", np.array(self.fire_origin, dtype=object))
        np.save("i_trajs/" + keyword + "_" + mode + "_trajectories.npy", np.array(self.trajectories, dtype=object))
        max_div = 1
        if np.max(self.recorded_fire_map) > 0:
            max_div = np.max(self.recorded_fire_map)
        # arrays = [self.recorded_fire_map/max_div, whole_fire_set_map, self.binary_val]
        # filenames = ["images/" + keyword + "_" + mode + "_image1_" + str(np.sum(self.recorded_fire_map.clip(max=1))) + ".png", "images/" + keyword + "_" + mode + "_image2_" + str(len(self.whole_fire_set)) + ".png", "images/" + keyword + "_" + mode + "_image3_" + str(np.sum(self.binary_val)) + ".png"]

        # # Save each image
        # for arr, filename in zip(arrays, filenames):
        #     plt.imsave(filename, arr, cmap="gray", vmin=0, vmax=1)  # Black & white
        plt.imsave("images/" + keyword + "_" + mode + "_image1_" + str(np.sum(self.recorded_fire_map.clip(max=1))) + ".png", self.recorded_fire_map/max_div, cmap="gray", vmin=0, vmax=1)  # Black & white

        ## keywords = self.generate_multiple_keywords(num_keywords=320)
        ## for idx in range(320):
        ##     plt.imsave("images/binary_val_"+str(idx)+"_"+keywords[idx]+".png", np.array(self.binary_val_ep_record[idx], dtype=self.int_type1), cmap="gray", vmin=0, vmax=1)  # Black & white
        ## plt.imsave("images/recorded_fire_map_" + str(np.sum(self.recorded_fire_map.clip(max=1))) +"_"+ self.generate_random_keyword()+".png", np.array(self.recorded_fire_map/self.episode_limit), cmap="gray", vmin=0, vmax=1)  # Black & white
        ## print("Images saved successfully!")
            
        ## # Initialize figure for Matplotlib
        ## fig = plt.figure(figsize=(13, 9))

        # # Create a GridSpec layout
        # gs = GridSpec(3, 4, figure=fig, width_ratios=[2, 1, 1, 1], height_ratios=[1, 1, 1])

        # # Left half: scatter plots
        # ax_left = fig.add_subplot(gs[:2, :2])
        # ax_left.set_title("Episode: " + str(ep_n))
        # ax_left.set_facecolor((0, 0, 0))
        # ax_left.set_xlim(0, self.y_len)
        # ax_left.set_ylim(0, self.x_len)
        # ax_left.set_aspect('equal')

        # map_axs = []
        # for i in range(2):
        #     map_axs.append(fig.add_subplot(gs[2, i]))
        #     map_axs[-1].set_aspect('equal')
        #     # map_axs[-1].imshow(self.map_history[0]["belief_map"][i], cmap='binary', interpolation='none')

        # view_axs = []
        # for i in range(3):
        #     view_axs.append(fig.add_subplot(gs[i, 2]))
        #     view_axs[-1].set_aspect('equal')
        #     # view_axs[-1].imshow(self.map_history[0]["belief_map"][i], cmap='binary', interpolation='none')
            
        # # Right half: tables
        # table_axs = []
        # for i in range(3):
        #     table_ax = fig.add_subplot(gs[i, 3:])
        #     table_ax.axis("tight")
        #     table_ax.axis("off")
        #     table_axs.append(table_ax)

        # def update(frame):
        #     ax_left.clear()
        #     ax_left.set_title("Episode: " + str(ep_n) + ", Ts: " + str(frame+1))
        #     ax_left.set_facecolor((0, 0, 0))
        #     ax_left.set_xlim(0, self.y_len)
        #     ax_left.set_ylim(0, self.x_len)
        #     ax_left.set_aspect('equal')

        #     # Draw the forest area (green background)
        #     rect = Rectangle((0, 0), self.y_len, self.x_len, color=self.env_render_color["field"], zorder=0)
        #     ax_left.add_patch(rect)

        #     # Plot fire
        #     fire_coords = np.array(list(self.fire_trajectories[frame]["fire_on"]))
        #     if len(fire_coords) > 0:
        #         ax_left.scatter(fire_coords[:, 1], np.abs( fire_coords[:, 0] - self.x_len ), color=self.env_render_color["fire"], label="Fire", s=10, zorder=1)

        #     # Plot burnt forest
        #     burnt_coords = np.array(list(self.fire_trajectories[frame]["fire_off"]))
        #     if len(burnt_coords) > 0:
        #         ax_left.scatter(burnt_coords[:, 1] , np.abs( burnt_coords[:, 0] - self.x_len ), color=self.env_render_color["burnt"], label="Burnt", s=10, zorder=1)

        #     # Plot UAV trajectories
        #     if len(self.trajectories) > 1 and frame > 0:
        #         print(frame, list(self.trajectories[frame]))
        #         for traj in self.trajectories[:frame]:
        #             for clr_i, (x, y) in enumerate(traj):
        #                 # UAV trajectories 
        #                 clr_id = clr_i%len(self.env_render_color["drone_trajs"])
        #                 ax_left.scatter(y, self.x_len - x, color=self.env_render_color["drone_trajs"][clr_id], label="UAV", s=10, zorder=2)
            
        #     # Plot UAV positions and their view areas
        #     for clr_i, (x, y) in enumerate(self.trajectories[frame]):
        #         x = self.x_len - x
        #         # UAV position
        #         clr_id = clr_i%len(self.env_render_color["drones"])
        #         ax_left.scatter(y, x, color=self.env_render_color["drones"][clr_id], label="UAV", s=10, zorder=2)

        #         # UAV view area
        #         x_view_min = max(0, x - 1 - self.d_sight)
        #         x_view_max = min(self.x_len, x - 1 + self.d_sight + 1)
        #         y_view_min = max(0, y - 1 - self.d_sight)
        #         y_view_max = min(self.y_len, y - 1 + self.d_sight + 1)

        #         rect = Rectangle((y_view_min, x_view_min), y_view_max - y_view_min, x_view_max - x_view_min,
        #                         edgecolor="yellow", fill=False, linewidth=1, zorder=3)
        #         ax_left.add_patch(rect)

        #     map_axs[0].clear()
        #     map_axs[0].set_aspect('equal')
        #     map_axs[0].imshow(self.map_history[frame]["belief_map"][1], cmap='binary', interpolation='none')
            
        #     map_axs[1].clear()
        #     map_axs[1].set_aspect('equal')
        #     map_axs[1].imshow(self.map_history[frame]["coverage_map"][1], cmap='gray', interpolation='none')
            
        #     for i in range(3):
        #         view_axs[i].clear()
        #         view_axs[i].set_aspect('equal')
        #         view_axs[i].imshow(self.uav_views[frame][i], cmap='binary', interpolation='none')

        #         # Update right tables
        #         grid_data = self.reshape_action_scores_to_display(self.agents_action_scores_history[frame][i])
        #         table_axs[i].clear()
        #         table_axs[i].axis("tight")
        #         table_axs[i].axis("off")
        #         table = table_axs[i].table(
        #             cellText= grid_data,
        #             loc="center",
        #             cellLoc="center",
        #             bbox=[0, 0, 1, 1]
        #         )
        #         table.auto_set_font_size(False)
        #         table.set_fontsize(12)

        #         # Highlight the highest value in blue
        #         # grid = grid_data
        #         # max_pos = np.unravel_index(np.argmax(grid), grid.shape)
        #         # table[(max_pos[0] + 1, max_pos[1])].set_facecolor("lightblue")

        #         # # Highlight custom cell in red if provided
        #         # if highlight_cells and i < len(highlight_cells):
        #         #     highlight_pos = highlight_cells[i]
        #         #     table[(highlight_pos[0] + 1, highlight_pos[1])].set_facecolor("lightcoral")

        # # Create the animation
        # ani = animation.FuncAnimation(fig, update, frames=len(self.trajectories), repeat=True, interval=500)

        # print("Saving rendered video at ", os.path.join(path, "animation_at_ep" + str(ep_n) + ".mp4"))

        # # # Save the animation
        # ani.save(os.path.join(path, "animation_at_ep" + str(ep_n) + ".mp4"), writer="ffmpeg", fps=2)
        
        # # plt.show()

        # print("Saved render video at ", os.path.join(path, "animation_at_ep" + str(ep_n) + ".mp4"))


    def reward(self):
        rewards = np.full(self.n_agents, 0)

        self.agents.copy_maps()

        ### Use slicing for x_view_min, x_view_max, y_view_min, y_view_max and clip within boundaries.
        positions = self.agents.get_position()
        x_view_min = np.maximum(positions[:, 0] - self.d_sight, 0)
        x_view_max = np.minimum(positions[:, 0] + self.d_sight, self.x_len)
        y_view_min = np.maximum(positions[:, 1] - self.d_sight, 0)
        y_view_max = np.minimum(positions[:, 1] + self.d_sight, self.y_len)

        nei_collision_adj_mtx = ( np.linalg.norm(self.agents.rel_dist, axis=2) <= self.r_collision_avoidance ).astype(self.int_type1)
        np.fill_diagonal(nei_collision_adj_mtx, 0)
        nei_collision_adj_mtx = nei_collision_adj_mtx * self.agents.nei_adj_mtx

        # uav_view = []

        rewards_report = np.full((self.n_agents, 4), 0)
        ### Slice binary_val and layer it n times
        for agent_id in range(self.n_agents):
        
            fire_map = self.binary_val[x_view_min[agent_id]:x_view_max[agent_id], y_view_min[agent_id]:y_view_max[agent_id]]
            belief_map = self.agents.belief_map[agent_id, x_view_min[agent_id]:x_view_max[agent_id], y_view_min[agent_id]:y_view_max[agent_id]]
            
            ### Perform counting
            rewards[agent_id] += np.sum(np.clip(fire_map - belief_map, 0, None))
            rewards_report[agent_id][0] = np.sum(np.clip(fire_map - belief_map, 0, None))
            
            ### update belief maps with binary_val_extended
            self.agents.belief_map[agent_id, x_view_min[agent_id]:x_view_max[agent_id], y_view_min[agent_id]:y_view_max[agent_id]] = fire_map

            # uav_view.append(fire_map)
            
            ### Try similar with coverage maps
            self.agents.coverage_map[agent_id, x_view_min[agent_id]:x_view_max[agent_id], y_view_min[agent_id]:y_view_max[agent_id]] = np.full((x_view_max[agent_id]-x_view_min[agent_id], y_view_max[agent_id]-y_view_min[agent_id]), 255, dtype=self.int_type2)
        
            ### Reward if neighbor
            # rewards[agent_id] += np.sum(self.agents.nei_adj_mtx[agent_id]) * 2
            # rewards_report[agent_id][1] = np.sum(self.agents.nei_adj_mtx[agent_id]) * 2
            
            ## Punish if neighbor gets into collision radius
            rewards[agent_id] += np.sum(nei_collision_adj_mtx[agent_id]) * -2
            rewards_report[agent_id][2] = np.sum(nei_collision_adj_mtx[agent_id]) * -2

        # self.uav_views.append(uav_view)

        return rewards, rewards_report
    
    def get_stats(self):
        stats = {
            "coverage": ( np.sum(self.recorded_fire_map.clip(max=1)) / len(self.whole_fire_set) ) * 100,
            # "coverage_at_static": ( np.sum(self.recorded_fire_map.clip(max=1)) / len(self.fire_set) ) * 100,
            "total_reward": self.total_rew,
        }
        return stats
    
    def get_obs_size(self):
        # position: 2
        # # movement_direction: 9
        # unit_features: (n-1)*4 || rel_x, rel_y, euc_dist, last_action
        # Fire discovery status: n
        # action_scores: 9
        # Relative x&y amd euc_norm w.r.to assumed fire coordinates: 3
        
        obs_size = 2 + ( ( self.n_agents - 1 ) * 4 ) + self.n_agents + 9 + 3 + 1
        # obs_size = 2 + 9 + 3
        return obs_size
    
    def get_state_size(self):
        # Positions: (n X 2)
        # # Unit_features: (n X 3) || rel_x, rel_y, euc_dist
        # Action_scores: (n X 9)
        # Last_actions: (n X 1)
        # Closest fire features: (n X 3) || rel_x, rel_y, euc_dist
        # Neighbor adjacency matrix: (n X n)
        # Relative_XY with neighbors: (n X n X 2) -> (n X n*2)
        # Euc. distance with neighbors: (n X n)
        # Fire center and radius: 3
        
        state_size = ( 2 + 9 + 1 + 3 + self.n_agents + (self.n_agents * 2) + self.n_agents )
        # state_size = ( 2 + 9 + 3 )
        return state_size * self.n_agents + 3 + 1
    
    def get_total_actions(self):
        return 9

    def seed(self):
        return 
    
    def close(self):
        return 
    
    def save_replay(self):
        return 

    def unit_sight_range(self, agent_id):
        """Returns the sight range for an agent."""
        return self.desired_comm_dist
    
    def get_env_info(self):
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
        }
        return env_info