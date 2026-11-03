"""独立的三维捕食者与猎物动力学，使用旋转向量控制。"""

import numpy as np
from gym import spaces
from .pps import PredatorPreySwarmEnv
from .putils.putils import get_mass, get_sizes
from .putils.putils3d import (get_dist_b2b_batch, get_dist_b2w,
                              get_focused_batch, make_periodic,
                              rotate_headings, vector_norm)


class PredatorPreySwarm3DEnv(PredatorPreySwarmEnv):
    spatial_dim = 3
    observation_layout = "njp3d-v1"
    action_layout = "rotation-vector-thrust-v1"
    _L = 2
    _act_dim_pursuer = 4
    _act_dim_escaper = 4
    _angle_p_max = .05
    _angle_e_max = .05
    _render_traj = False
    _penalize_control_effort = False
    _energy_enabled = True
    _prey_death_enabled = False
    _capture_cooldown_steps = 0
    _spawn_mode = "uniform"
    _prey_spawn_radius = 1.0
    _predator_spawn_inner = 1.5
    _predator_spawn_outer = 1.8
    param_list = PredatorPreySwarmEnv.param_list + [
        "n_o", "p_o", "heading", "dp", "ddp", "energy", "energy_enabled",
        "prey_death_enabled", "spawn_mode", "prey_spawn_radius",
        "predator_spawn_inner", "predator_spawn_outer", "capture_cooldown_steps"]

    def __init__(self, n_p=16, n_e=160, n_o=0, **parameters):
        self._n_p, self._n_e, self._n_o = n_p, n_e, n_o
        self.viewer = None
        self.optimization_flags = dict(collision=True, observation=True, trajectory=True)
        self._trajectory_buffer = None
        self._trajectory_cursor = 0
        self._trajectory_mode = True
        self.seed()
        for name, value in parameters.items():
            if name not in self.param_list:
                raise TypeError("未知的三维环境参数：" + name)
            setattr(self, name, value)
        self.__reinit__()

    def __reinit__(self):
        for name in ("n_p", "n_e", "n_o"):
            value = getattr(self, "_" + name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(name + " 必须为非负整数")
        if self._n_p + self._n_e == 0:
            raise ValueError("至少需要一个智能体")
        if self._dynamics_mode != "Polar" or self._billiards_mode:
            raise ValueError("三维环境仅支持 Polar 旋转向量动力学")
        if not self._obstacles_cannot_move:
            raise ValueError("三维环境仅支持静态球形障碍物")
        if self._capture_cooldown_steps and not self._prey_death_enabled:
            raise ValueError("capture_cooldown_steps 需要启用 prey_death_enabled")
        if not np.isfinite(self._L) or self._L <= 0:
            raise ValueError("L 必须是有限的正数")
        for name in ("m_p", "m_e", "m_o", "size_p", "size_e", "size_o", "dt"):
            value = getattr(self, "_" + name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(name + " 必须是有限的正数")
        for name in ("linVel_p_max", "linVel_e_max", "linAcc_p_max", "linAcc_e_max",
                     "angle_p_max", "angle_e_max", "FoV_p", "FoV_e"):
            value = getattr(self, "_" + name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(name + " 必须是有限的非负数")
        if self._traj_len < 1:
            raise ValueError("traj_len 必须为正数")
        self._validate_spawn()
        self._n_pe = self._n_p + self._n_e
        self._n_peo = self._n_pe + self._n_o
        if self._obstacles_is_constant and hasattr(self, "_p_o"):
            self._validate_xyz("p_o", self._p_o, self._n_o)
        self._collision_pair_indices = None
        self._observation_workspace = None
        self._trajectory_buffer = None
        self._trajectory_cursor = 0
        self._reset_capture_state()
        self._m = get_mass(self._m_p, self._m_e, self._m_o,
                           self._n_p, self._n_e, self._n_o).astype(np.float64)
        self._size, self._sizes = get_sizes(
            self._size_p, self._size_e, self._size_o, self._n_p, self._n_e, self._n_o)
        self._size, self._sizes = self._size.astype(np.float64), self._sizes.astype(np.float64)
        self._linAcc_p_min = self._linAcc_e_min = 0
        self.max_energy_p = self.max_energy_e = 1000.
        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()

    def set_reference_mode(self, enabled=True):
        """切换向量化的全配对、全排序参考路径，用于验证。"""
        for key in self.optimization_flags:
            self.optimization_flags[key] = not enabled

    def _get_observation_space(self):
        self._topo_n_p = self._topo_n_p2p + self._topo_n_p2e
        self._topo_n_e = self._topo_n_e2p + self._topo_n_e2e
        self.obs_dim_pursuer = 10 + 6 * self._topo_n_p
        self.obs_dim_escaper = 10 + 6 * self._topo_n_e
        return spaces.Box(low=-np.inf, high=np.inf,
                          shape=(max(self.obs_dim_pursuer, self.obs_dim_escaper), self._n_pe),
                          dtype=np.float32)

    def reset(self):
        self._reset_capture_state()
        max_size = np.max(self._size)
        if self._spawn_mode == "cluster_shell":
            self._reset_cluster_positions()
            self._d_b2b_center, self.d_b2b_edge, self._is_collide_b2b = get_dist_b2b_batch(
                self._p, self._L, self._is_periodic, self._sizes)
        else:
            # 保留现有实验的随机抽样与拒绝采样顺序。
            for attempt in range(100):
                self._p = np.random.uniform(-self._L + max_size, self._L - max_size,
                                             (3, self._n_peo))
                if self._obstacles_is_constant:
                    if not hasattr(self, "_p_o"):
                        raise ValueError("固定障碍物需要形状为 (3, n_o) 的 p_o")
                    self._p[:, self._n_pe:] = self._p_o
                self._d_b2b_center, self.d_b2b_edge, self._is_collide_b2b = get_dist_b2b_batch(
                    self._p, self._L, self._is_periodic, self._sizes)
                if not self._is_collide_b2b.any():
                    break
        self._dp = np.zeros((3, self._n_peo))
        self._ddp = np.zeros((3, self._n_peo))
        self._energy = np.concatenate((np.full(self._n_p, self.max_energy_p),
                                        np.full(self._n_e, self.max_energy_e)))[None, :]
        # 均匀方位角和均匀 Z 坐标构成球面均匀分布，无需拒绝采样。
        azimuth = np.random.uniform(-np.pi, np.pi, self._n_peo)
        z = np.random.uniform(-1., 1., self._n_peo)
        radial = np.sqrt(1 - z ** 2)
        self._heading = np.vstack((radial * np.cos(azimuth), radial * np.sin(azimuth), z))
        self._last_action = np.zeros((4, self._n_pe))
        self._trajectory_buffer = None
        self._trajectory_cursor = 0
        self._trajectory_mode = bool(self.optimization_flags["trajectory"])
        if self._render_traj:
            # 初始位置重复填充时只产生零长度线段，
            # 避免在历史不足时生成连接世界原点的虚假轨迹。
            self._trajectory_buffer = np.repeat(self._p[None, :, :], self._traj_len, axis=0)
        if not self._is_periodic:
            self.d_b2w, self.is_collide_b2w = get_dist_b2w(self._p, self._size, self._L)
        return self._get_obs()

    def _validate_spawn(self):
        if self._spawn_mode not in ("uniform", "cluster_shell"):
            raise ValueError("spawn_mode 必须为 uniform 或 cluster_shell")
        if self._spawn_mode == "uniform":
            return
        values = (self._prey_spawn_radius, self._predator_spawn_inner,
                  self._predator_spawn_outer)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("出生区域半径必须有限")
        if not (0 < values[0] < values[1] < values[2]):
            raise ValueError("必须满足 0 < prey_spawn_radius < predator_spawn_inner < predator_spawn_outer")
        if (values[0] + self._size_e > self._L
                or values[2] + self._size_p > self._L):
            raise ValueError("出生区域加上个体半径后必须完全位于场地内")

    def _reset_cluster_positions(self):
        """按球体或球壳体积均匀提出出生位置，并拒绝初始重叠。"""
        self._validate_spawn()
        self._p = np.empty((3, self._n_peo), dtype=np.float64)
        placed = []
        # 先放置静态障碍物，确保全部智能体出生时避开障碍物。
        for index in range(self._n_pe, self._n_peo):
            for attempt in range(10000):
                if self._obstacles_is_constant:
                    if not hasattr(self, "_p_o"):
                        raise ValueError("固定障碍物需要形状为 (3, n_o) 的 p_o")
                    candidate = self._p_o[:, index - self._n_pe]
                else:
                    candidate = np.random.uniform(-self._L + self._size_o,
                                                  self._L - self._size_o, 3)
                if self._spawn_is_clear(candidate, index, placed):
                    break
                if self._obstacles_is_constant:
                    raise ValueError("cluster_shell 出生模式中的固定障碍物存在重叠")
            else:
                raise ValueError("无法放置互不重叠的障碍物")
            self._p[:, index] = candidate
            placed.append(index)
        for index in range(self._n_pe):
            inner, outer = ((self._predator_spawn_inner, self._predator_spawn_outer)
                            if index < self._n_p else (0., self._prey_spawn_radius))
            for attempt in range(10000):
                azimuth = np.random.uniform(-np.pi, np.pi)
                z = np.random.uniform(-1., 1.)
                radius = np.random.uniform(inner ** 3, outer ** 3) ** (1. / 3.)
                radial = np.sqrt(1. - z * z)
                candidate = radius * np.array([radial * np.cos(azimuth),
                                               radial * np.sin(azimuth), z])
                if self._spawn_is_clear(candidate, index, placed):
                    self._p[:, index] = candidate
                    placed.append(index)
                    break
            else:
                raise ValueError("无法在出生区域内放置互不重叠的智能体")

    def _spawn_is_clear(self, candidate, index, placed):
        if not placed:
            return True
        delta = self._p[:, placed] - candidate[:, None]
        if self._is_periodic:
            make_periodic(delta, self._L)
        return bool(np.all(vector_norm(delta) >= self._size[index] + self._size[placed]))

    spawn_mode = property(lambda self: self._spawn_mode,
                          lambda self, value: setattr(self, "_spawn_mode", value))
    prey_spawn_radius = property(lambda self: self._prey_spawn_radius,
                                lambda self, value: setattr(self, "_prey_spawn_radius", value))
    predator_spawn_inner = property(lambda self: self._predator_spawn_inner,
                                   lambda self, value: setattr(self, "_predator_spawn_inner", value))
    predator_spawn_outer = property(lambda self: self._predator_spawn_outer,
                                   lambda self, value: setattr(self, "_predator_spawn_outer", value))

    def _reset_capture_state(self):
        self._alive = np.ones(self._n_pe, dtype=bool)
        self._alive_at_step_start = self._alive.copy()
        self._capture_contacts = np.zeros((self._n_e, self._n_p), dtype=bool)
        self._capture_remaining = np.zeros(self._n_p, dtype=np.int64)

    @property
    def capture_cooldown_steps(self):
        return self._capture_cooldown_steps

    @capture_cooldown_steps.setter
    def capture_cooldown_steps(self, value):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError("capture_cooldown_steps 必须为非负整数")
        self._capture_cooldown_steps = int(value)

    @property
    def capture_cooldown_remaining(self):
        return self._capture_remaining.copy()

    @property
    def prey_death_enabled(self):
        return self._prey_death_enabled

    @prey_death_enabled.setter
    def prey_death_enabled(self, enabled):
        if not isinstance(enabled, (bool, np.bool_)):
            raise ValueError("prey_death_enabled 必须为布尔值")
        self._prey_death_enabled = bool(enabled)

    @property
    def alive_mask(self):
        """智能体槽位保持固定，调用方获得独立的存活掩码。"""
        return self._alive.copy()

    @property
    def capture_contacts(self):
        """记录上一步中新捕获的猎物，每只猎物仅归功于一个捕食者。"""
        return self._capture_contacts.copy()

    def _capture_prey(self):
        """在原有的积分前奖励采样时刻判定接触。"""
        active = np.ones(self._n_peo, dtype=bool)
        active[:self._n_pe] = self._alive
        self._is_collide_b2b &= active[:, None] & active[None, :]
        contact = self._is_collide_b2b[self._n_p:self._n_pe, :self._n_p]
        if self._capture_cooldown_steps:
            # 优先处理距离最短的可用接触；每个已冷却捕食者最多捕获一只猎物。
            prey, predators = np.nonzero(contact & (self._capture_remaining == 0)[None, :])
            distances = self._d_b2b_center[self._n_p + prey, predators]
            order = np.lexsort((predators, prey, distances))
            used_prey, used_predators = set(), set()
            accepted = []
            for index in order:
                p, hunter = int(prey[index]), int(predators[index])
                if p not in used_prey and hunter not in used_predators:
                    accepted.append((p, hunter))
                    used_prey.add(p); used_predators.add(hunter)
            for p, hunter in accepted:
                self._capture_contacts[p, hunter] = True
                self._capture_remaining[hunter] = self._capture_cooldown_steps
            if accepted:
                dead = self._n_p + np.asarray([p for p, _ in accepted])
                self._alive[dead] = False
                self._dp[:, dead] = self._ddp[:, dead] = 0
                self._is_collide_b2b[dead, :] = False
                self._is_collide_b2b[:, dead] = False
            return
        caught = np.flatnonzero(contact.any(axis=1))
        if len(caught):
            distances = self._d_b2b_center[self._n_p + caught, :self._n_p]
            # 距离完全相同时，argmin 确定性地选择最小编号。
            owners = np.argmin(np.where(contact[caught], distances, np.inf), axis=1)
            self._capture_contacts[caught, owners] = True
            dead = self._n_p + caught
            self._alive[dead] = False
            self._dp[:, dead] = 0
            self._ddp[:, dead] = 0
            self._is_collide_b2b[dead, :] = False
            self._is_collide_b2b[:, dead] = False

    def _get_obs(self):
        # 返回的观测独立拥有存储空间，内部工作区可以复用。
        self.obs = np.zeros(self.observation_space.shape)
        positions, velocities = self._p[:, :self._n_pe].T, self._dp[:, :self._n_pe].T
        headings = self._heading[:, :self._n_pe].T
        optimized = self.optimization_flags["observation"]
        shape = (self._n_pe, self._n_pe, 3)
        workspace = self._observation_workspace if optimized else None
        if workspace is None or workspace[0] != shape:
            workspace = (shape, np.empty(shape), np.empty(shape))
            if optimized:
                self._observation_workspace = workspace
        relative_positions, relative_headings = workspace[1:]
        np.subtract(positions[None, :, :], positions[:, None, :], out=relative_positions)
        if self._is_periodic:
            make_periodic(relative_positions, self._L, optimized)
        np.subtract(headings[None, :, :], headings[:, None, :], out=relative_headings)
        if self._prey_death_enabled:
            # 将死亡槽位的候选距离设为无穷，在选择邻居前排除它们，
            # 避免占用较远存活邻居的位置，同时保证输出有限。
            relative_positions[:, ~self._alive, :] = np.inf
        own_state = np.concatenate((positions, velocities), axis=1)
        populations = (slice(0, self._n_p), slice(self._n_p, self._n_pe))
        groups = ((self._FoV_p, (self._topo_n_p2p, self._topo_n_p2e), self.max_energy_p),
                  (self._FoV_e, (self._topo_n_e2p, self._topo_n_e2e), self.max_energy_e))
        for species, (observers, (field, widths, max_energy)) in enumerate(zip(populations, groups)):
            count = observers.stop - observers.start
            if count == 0:
                continue
            features = [own_state[observers, None, :]]
            for neighbor_species, (neighbors, width) in enumerate(zip(populations, widths)):
                position, heading = get_focused_batch(
                    relative_positions[observers, neighbors],
                    relative_headings[observers, neighbors], field, width,
                    species == neighbor_species, optimized=optimized)
                features.append(np.concatenate((position, heading), axis=2))
            features = np.concatenate(features, axis=1).reshape(count, -1).T
            energy_row = features.shape[0]
            self.obs[:energy_row, observers] = features
            self.obs[energy_row, observers] = (2 / max_energy * self._energy[0, observers] - 1
                                               if self._energy_enabled else 1.)
            self.obs[energy_row + 1:energy_row + 4, observers] = self._heading[:, observers]
        if self._prey_death_enabled:
            self.obs[:, ~self._alive] = 0
        return self.obs

    def _get_ball_collision_force(self):
        optimized = self.optimization_flags["collision"]
        cached = self._collision_pair_indices if optimized else None
        if cached is None or cached[0] != self._n_peo:
            cached = (self._n_peo, *np.tril_indices(self._n_peo, k=-1))
            if optimized:
                self._collision_pair_indices = cached
        rows, columns = cached[1:]
        if optimized:
            contact = self._is_collide_b2b[rows, columns]
            rows, columns = rows[contact], columns[contact]
        delta = self._p[:, columns] - self._p[:, rows]
        if self._is_periodic:
            make_periodic(delta, self._L)
        distances = self._d_b2b_center[rows, columns]
        direction = np.zeros_like(delta)
        np.divide(delta, distances[None, :], out=direction, where=distances[None, :] != 0)
        direction[0, distances == 0] = 1.
        pair_force = (self._is_collide_b2b[rows, columns]
                      * self.d_b2b_edge[rows, columns] * self._k_ball * (-direction))
        forces = np.zeros((self._n_peo, 3, self._n_peo))
        forces[rows, :, columns] = pair_force.T
        forces[columns, :, rows] = -pair_force.T
        return forces.sum(axis=2).T

    def _nearest_action(self, observers, targets, escape=False):
        count = observers.stop - observers.start
        rotation = np.zeros((3, count))
        if targets.stop <= targets.start or count == 0:
            return rotation, np.zeros((1, count))
        candidates = np.arange(targets.start, targets.stop)
        if self._prey_death_enabled:
            candidates = candidates[self._alive[candidates]]
            if not len(candidates):
                return rotation, np.zeros((1, count))
        nearest = np.argmin(self._d_b2b_center[observers][:, candidates], axis=1)
        delta = self._p[:, candidates[nearest]] - self._p[:, observers]
        if escape:
            delta = -delta
        if self._is_periodic:
            make_periodic(delta, self._L)
        lengths = vector_norm(delta)
        direction = np.zeros_like(delta)
        np.divide(delta, lengths[None, :], out=direction, where=lengths[None, :] > 0)
        heading = self._heading[:, observers]
        axes = np.cross(heading.T, direction.T).T
        sine = vector_norm(axes)
        cosine = np.clip(np.sum(heading * direction, axis=0), -1., 1.)
        regular = sine > 1e-15
        axes[:, regular] /= sine[regular]
        opposite = (~regular) & (cosine < 0) & (lengths > 0)
        if opposite.any():
            h = heading[:, opposite]
            basis = np.eye(3)[:, np.argmin(np.abs(h), axis=0)]
            anti_axes = np.cross(h.T, basis.T).T
            axes[:, opposite] = anti_axes / vector_norm(anti_axes)[None, :]
        angles = np.arctan2(sine, cosine)
        angles[lengths == 0] = 0
        limit = self._angle_p_max if observers.start == 0 else self._angle_e_max
        rotation = axes * np.minimum(angles, limit)[None, :]
        force = self._linAcc_p_max if observers.start == 0 else self._linAcc_e_max
        return rotation, np.full((1, count), force)

    def _physical_action(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape not in ((4, self._n_pe), (4, self._n_peo)):
            raise ValueError("三维动作形状必须为 (4, n_agents) 或 (4, n_agents_and_obstacles)")
        if not np.isfinite(action).all():
            raise ValueError("三维动作必须有限")
        rotation, thrust = np.zeros((3, self._n_pe)), np.zeros((1, self._n_pe))
        populations = (slice(0, self._n_p), slice(self._n_p, self._n_pe))
        for species, (observers, strategy, angle_max, force_max) in enumerate(zip(
                populations, (self._pursuer_strategy, self._escaper_strategy),
                (self._angle_p_max, self._angle_e_max),
                (self._linAcc_p_max, self._linAcc_e_max))):
            count = observers.stop - observers.start
            if strategy == "static":
                continue
            if strategy == "nearest":
                rotation[:, observers], thrust[:, observers] = self._nearest_action(
                    observers, populations[1 - species], escape=species == 1)
                continue
            normalized = (np.random.uniform(-1, 1, (4, count))
                          if strategy == "random" else action[:, observers])
            heading = self._heading[:, observers]
            axes = normalized[:3] - heading * np.sum(normalized[:3] * heading, axis=0, keepdims=True)
            axes /= np.maximum(1., vector_norm(axes))[None, :]
            rotation[:, observers] = axes * angle_max
            thrust[:, observers] = (np.clip(normalized[3:4], -1, 1) + 1) * force_max / 2
        if self._prey_death_enabled:
            rotation[:, ~self._alive] = 0
            thrust[:, ~self._alive] = 0
        return rotation, thrust

    def _limit_velocity(self):
        for population, maximum in ((slice(0, self._n_p), self._linVel_p_max),
                                     (slice(self._n_p, self._n_pe), self._linVel_e_max)):
            speed = vector_norm(self._dp[:, population])
            factor = np.ones_like(speed)
            np.divide(maximum, speed, out=factor, where=speed > maximum)
            self._dp[:, population] *= factor[None, :]

    def step(self, action):
        if self._prey_death_enabled:
            self._alive_at_step_start = self._alive.copy()
            self._capture_contacts.fill(False)
            if self._capture_cooldown_steps:
                np.maximum(self._capture_remaining - 1, 0, out=self._capture_remaining)
        for _ in range(self._n_frames):
            self._d_b2b_center, self.d_b2b_edge, self._is_collide_b2b = get_dist_b2b_batch(
                self._p, self._L, self._is_periodic, self._sizes)
            if self._prey_death_enabled:
                self._capture_prey()
            sphere_force = self._get_ball_collision_force()
            rotation, thrust = self._physical_action(action)
            self._last_action = np.concatenate((rotation, thrust), axis=0)
            self._heading[:, :self._n_pe] = rotate_headings(self._heading[:, :self._n_pe], rotation)
            control = np.zeros((3, self._n_peo))
            control[:, :self._n_pe] = self._heading[:, :self._n_pe] * thrust
            force = self._sensitivity * control + sphere_force - self._c_aero * self._dp
            if not self._is_periodic:
                self.d_b2w, self.is_collide_b2w = get_dist_b2w(self._p, self._size, self._L)
                contact = self.is_collide_b2w.reshape(3, 2, self._n_peo)
                penetration = (self.is_collide_b2w * self.d_b2w).reshape(3, 2, self._n_peo)
                wall_force = (penetration[:, 0] - penetration[:, 1]) * self._k_wall
                damping = -(contact[:, 0].astype(float) + contact[:, 1]) * self._dp * self._c_wall
                force = force + wall_force + damping
            self._ddp = force / self._m
            self._dp += self._ddp * self._dt
            self._dp[:, self._n_pe:] = 0
            if self._prey_death_enabled:
                self._dp[:, :self._n_pe][:, ~self._alive] = 0
                self._ddp[:, :self._n_pe][:, ~self._alive] = 0
            self._limit_velocity()
            if self._energy_enabled:
                self._dp[:, :self._n_pe][:, self._energy[0] < .5] = 0
                speeds = vector_norm(self._dp[:, :self._n_pe])[None, :]
                self._energy -= speeds
                self._energy[speeds < .01] += .1
                np.minimum(self._energy[:, :self._n_p], self.max_energy_p,
                           out=self._energy[:, :self._n_p])
                np.minimum(self._energy[:, self._n_p:], self.max_energy_e,
                           out=self._energy[:, self._n_p:])
            else:
                self._energy[:, :self._n_p] = self.max_energy_p
                self._energy[:, self._n_p:] = self.max_energy_e
            self._p += self._dp * self._dt
            if self._obstacles_is_constant:
                self._p[:, self._n_pe:] = self._p_o
            if self._is_periodic:
                make_periodic(self._p, self._L)
            if self._render_traj:
                self._append_trajectory()
        return self._get_obs(), self._get_reward(self._last_action), self._get_done(), self._get_info()

    def _get_reward(self, action):
        # 接触及辅助惩罚与维度无关；在加入三维控制代价后
        # 再应用奖励共享，避免使用父类的二维控制代价分支。
        penalize = self._penalize_control_effort
        mode = self._reward_sharing_mode
        self._penalize_control_effort = False
        self._reward_sharing_mode = "individual"
        distances = self._d_b2b_center
        contacts = self._is_collide_b2b
        if self._prey_death_enabled and self._capture_cooldown_steps:
            # 冷却期间的物理接触不产生新的捕获。
            self._is_collide_b2b = contacts.copy()
            self._is_collide_b2b[:self._n_p, self._n_p:self._n_pe] = False
            self._is_collide_b2b[self._n_p:self._n_pe, :self._n_p] = False
        if self._prey_death_enabled and self._penalize_distance:
            self._d_b2b_center = distances.copy()
            dead = np.flatnonzero(~self._alive)
            self._d_b2b_center[dead, :] = 0
            self._d_b2b_center[:, dead] = 0
        try:
            reward = super()._get_reward(action)
        finally:
            self._penalize_control_effort = penalize
            self._reward_sharing_mode = mode
            self._d_b2b_center = distances
            self._is_collide_b2b = contacts
        if self._prey_death_enabled:
            reward[:, ~self._alive] = 0
            reward[:, :self._n_p] += 5. * self._capture_contacts.sum(axis=0)
            reward[:, self._n_p:self._n_pe] -= self._capture_contacts.any(axis=1)
        if penalize:
            penalty = .1 * vector_norm(action[:3]) + .01 * np.abs(action[3])
            if self._prey_death_enabled:
                penalty[~self._alive] = 0
            reward -= penalty[None, :]
        for population in (slice(0, self._n_p), slice(self._n_p, self._n_pe)):
            if population.stop == population.start:
                continue
            eligible = (self._alive_at_step_start[population] if self._prey_death_enabled
                        else np.ones(population.stop - population.start, dtype=bool))
            values = reward[:, population]
            if eligible.any() and mode == "sharing_mean":
                values[:, eligible] = values[:, eligible].mean()
            elif eligible.any() and mode == "sharing_max":
                values[:, eligible] = values[:, eligible].max()
        return reward

    def _get_done(self):
        if not self._prey_death_enabled:
            return super()._get_done()
        done = (~self._alive)[None, :].copy()
        done[:, :self._n_p] = not self._alive[self._n_p:].any()
        return done

    def _get_info(self):
        if self._prey_death_enabled:
            prey, owners = np.nonzero(self._capture_contacts)
            return {"captured_prey": prey.tolist(), "capture_owners": owners.tolist(),
                    "alive_prey_count": int(self._alive[self._n_p:].sum()),
                    "all_prey_caught": not self._alive[self._n_p:].any()}
        # 保留旧版公开 info 的形状，不修改距离矩阵。
        return np.array([None, None, None]).reshape(3, 1)

    @property
    def _p_traj(self):
        if self._trajectory_buffer is None:
            return None
        start = self._trajectory_cursor if self._trajectory_mode else 0
        return np.concatenate((self._trajectory_buffer[start:], self._trajectory_buffer[:start]), axis=0)

    def _append_trajectory(self):
        optimized = bool(self.optimization_flags["trajectory"])
        if self._trajectory_buffer is None:
            self._trajectory_buffer = np.repeat(self._p[None, :, :], self._traj_len, axis=0)
            self._trajectory_cursor = 0
        if optimized != self._trajectory_mode:
            self._trajectory_buffer = self._p_traj
            self._trajectory_cursor = 0
            self._trajectory_mode = optimized
        if optimized:
            self._trajectory_buffer[self._trajectory_cursor] = self._p
            self._trajectory_cursor = (self._trajectory_cursor + 1) % self._traj_len
        else:
            self._trajectory_buffer = np.concatenate(
                (self._trajectory_buffer[1:], self._p[None, :, :]), axis=0)

    def render(self, mode="human"):
        if mode not in ("human", "rgb_array"):
            raise ValueError("不支持的渲染模式：" + str(mode))
        if self.viewer is None:
            from .rendering3d import Viewer3D
            self.viewer = Viewer3D(640, 640)
            if hasattr(self, "camera_view"):
                self.viewer.set_view(self.camera_view)
        return self.viewer.render(self, return_rgb_array=mode == "rgb_array")

    @staticmethod
    def _validate_xyz(name, value, count):
        if not isinstance(value, np.ndarray) or value.ndim != 2 or value.shape != (3, count):
            raise TypeError("%s 必须是形状为 (3, %d) 的 ndarray" % (name, count))
        if not np.isfinite(value).all():
            raise ValueError(name + " 必须有限")

    def _set_xyz(self, name, value):
        self._validate_xyz(name, value, self._n_p + self._n_e + self._n_o)
        setattr(self, "_" + name, np.array(value, dtype=np.float64, copy=True))

    p = property(lambda self: self._p, lambda self, value: self._set_xyz("p", value))
    dp = property(lambda self: self._dp, lambda self, value: self._set_xyz("dp", value))
    ddp = property(lambda self: self._ddp, lambda self, value: self._set_xyz("ddp", value))

    @property
    def heading(self):
        return self._heading

    @heading.setter
    def heading(self, value):
        self._validate_xyz("heading", value, self._n_p + self._n_e + self._n_o)
        lengths = vector_norm(value)
        if np.any(lengths == 0):
            raise ValueError("朝向向量必须非零")
        self._heading = value.astype(np.float64) / lengths[None, :]

    @property
    def theta(self):
        raise AttributeError("三维姿态使用单位朝向向量，不使用 theta")

    @theta.setter
    def theta(self, value):
        raise AttributeError("三维姿态使用单位朝向向量，不使用 theta")

    @property
    def dynamics_mode(self):
        return self._dynamics_mode

    @dynamics_mode.setter
    def dynamics_mode(self, value):
        if value != "Polar":
            raise ValueError("三维环境仅支持 Polar 旋转向量动力学")
        self._dynamics_mode = value

    @property
    def billiards_mode(self):
        return False

    @billiards_mode.setter
    def billiards_mode(self, value):
        if value:
            raise ValueError("三维环境不支持台球动力学")
        self._billiards_mode = False

    @property
    def n_o(self):
        return self._n_o

    @n_o.setter
    def n_o(self, value):
        self._n_o = value

    @property
    def p_o(self):
        return self._p_o

    @p_o.setter
    def p_o(self, value):
        self._validate_xyz("p_o", value, self._n_o)
        self._p_o = np.array(value, dtype=np.float64, copy=True)

    @property
    def energy_enabled(self):
        return self._energy_enabled

    @energy_enabled.setter
    def energy_enabled(self, value):
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError("energy_enabled 必须为布尔值")
        self._energy_enabled = bool(value)

    @property
    def energy(self):
        return self._energy

    @energy.setter
    def energy(self, value):
        if not isinstance(value, np.ndarray) or value.shape != (1, self._n_p + self._n_e):
            raise TypeError("energy 必须是形状为 (1, n_agents) 的 ndarray")
        if not np.isfinite(value).all():
            raise ValueError("energy 必须有限")
        self._energy = np.array(value, dtype=np.float64, copy=True)

    @property
    def m_o(self):
        return self._m_o

    @m_o.setter
    def m_o(self, value):
        self._m_o = value

    @property
    def sensitivity(self):
        return self._sensitivity

    @sensitivity.setter
    def sensitivity(self, value):
        if not np.isfinite(value) or value < 0:
            raise ValueError("sensitivity 必须是有限的非负数")
        self._sensitivity = value

    @property
    def act_dim_pursuer(self):
        return 4

    @act_dim_pursuer.setter
    def act_dim_pursuer(self, value):
        if value != 4:
            raise ValueError("三维旋转向量与推力动作必须包含四个分量")

    @property
    def act_dim_escaper(self):
        return 4

    @act_dim_escaper.setter
    def act_dim_escaper(self, value):
        if value != 4:
            raise ValueError("三维旋转向量与推力动作必须包含四个分量")

    @property
    def obstacles_cannot_move(self):
        return True

    @obstacles_cannot_move.setter
    def obstacles_cannot_move(self, value):
        if not value:
            raise ValueError("三维环境仅支持静态球形障碍物")
        self._obstacles_cannot_move = True

    @property
    def obstacles_is_constant(self):
        return self._obstacles_is_constant

    @obstacles_is_constant.setter
    def obstacles_is_constant(self, value):
        self._obstacles_is_constant = bool(value)

    @property
    def n_frames(self):
        return self._n_frames

    @n_frames.setter
    def n_frames(self, value):
        if not isinstance(value, int) or value < 1:
            raise ValueError("n_frames 必须为正整数")
        self._n_frames = value
