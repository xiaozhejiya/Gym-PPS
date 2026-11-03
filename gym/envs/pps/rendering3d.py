"""独立的三维查看器，导入时不创建 OpenGL 上下文。

无显示器录制时，请在创建查看器之前设置 pyglet.options['headless'] = True。
渲染不会修改仿真器或其随机状态。
"""

import math

import numpy as np


def _sphere_mesh(rings=8, sectors=16):
    """返回可复用的单位球面顶点及三角形索引。"""
    latitude = np.linspace(-np.pi / 2, np.pi / 2, rings + 1)
    longitude = np.linspace(0, 2 * np.pi, sectors, endpoint=False)
    points = np.empty((rings + 1, sectors, 3), dtype=np.float32)
    points[:, :, 0] = np.cos(latitude[:, None]) * np.cos(longitude[None, :])
    points[:, :, 1] = np.cos(latitude[:, None]) * np.sin(longitude[None, :])
    points[:, :, 2] = np.sin(latitude[:, None])
    triangles = []
    for ring in range(rings):
        for sector in range(sectors):
            a = ring * sectors + sector
            b = ring * sectors + (sector + 1) % sectors
            triangles.extend((a, b, a + sectors, b, b + sectors, a + sectors))
    return points.reshape(-1, 3), np.asarray(triangles, dtype=np.uint32)


class Viewer3D:
    """支持环绕、缩放和几何体复用的透视查看器。

    左键拖动旋转，滚轮缩放，R 复位，1/2/3/4 分别选择 XY、XZ、
    YZ 或默认斜视视角。相机不会自动旋转。
    """

    def __init__(self, width=640, height=640, visible=True):
        import pyglet
        from pyglet import gl

        self._pyglet, self._gl = pyglet, gl
        self.width, self.height = int(width), int(height)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("查看器的宽度和高度必须为正数")
        config = gl.Config(double_buffer=True, depth_size=24)
        self.window = pyglet.window.Window(
            width=self.width, height=self.height, visible=visible,
            caption="Gym-PPS 三维仿真", config=config, resizable=True,
        )
        self.isopen = True
        self._half_width = 1.0
        self._mesh, self._mesh_indices = _sphere_mesh()
        self._geometry_key = None
        self._spheres = self._headings = self._trails = None
        self._trail_key = None
        self._box = self._axes = None
        self.reset_camera()
        self.window.push_handlers(
            on_close=self._on_close, on_resize=self._on_resize,
            on_mouse_drag=self._on_mouse_drag, on_mouse_scroll=self._on_mouse_scroll,
            on_key_press=self._on_key_press,
        )
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LEQUAL)
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)

    def reset_camera(self):
        self.azimuth = 45.0
        self.elevation = 25.0
        self.zoom = 1.0

    def set_view(self, view):
        """选择可复现的视角，不修改仿真状态。"""
        angles = {"xy": (0.0, 90.0), "xz": (-90.0, 0.0),
                  "yz": (0.0, 0.0), "oblique": (45.0, 25.0)}
        if view not in angles:
            raise ValueError("view 必须为 xy、xz、yz 或 oblique")
        self.azimuth, self.elevation = angles[view]
        self.zoom = 1.0

    def _on_close(self):
        self.close()
        return self._pyglet.event.EVENT_HANDLED

    def _on_resize(self, width, height):
        self.width, self.height = width, max(1, height)

    def _on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if buttons & self._pyglet.window.mouse.LEFT:
            self.azimuth += dx * 0.4
            self.elevation = float(np.clip(self.elevation + dy * 0.4, -89.5, 89.5))

    def _on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        self.zoom = float(np.clip(self.zoom * math.exp(-scroll_y * 0.1), 0.35, 5.0))

    def _on_key_press(self, symbol, modifiers):
        key = self._pyglet.window.key
        if symbol == key.R:
            self.reset_camera()
        else:
            views = {key._1: "xy", key._2: "xz", key._3: "yz", key._4: "oblique"}
            if symbol in views:
                self.set_view(views[symbol])

    def _make_vertices(self, points, colors):
        return self._pyglet.graphics.vertex_list(
            len(points), ("v3f/dynamic", points.reshape(-1)),
            ("c3B/static", colors.reshape(-1)),
        )

    @staticmethod
    def _delete(*geometries):
        for geometry in geometries:
            if geometry is not None:
                geometry.delete()

    def _prepare_geometry(self, env):
        count = env._n_peo
        key = (count, env._n_p, env._n_e, float(env._L))
        if key == self._geometry_key:
            return
        self._delete(self._spheres, self._headings, self._box, self._axes)
        self._geometry_key = key
        self._trail_key = None
        self._half_width = float(env._L)
        self._body_colors = np.empty((count, 3), dtype=np.uint8)
        self._body_colors[:] = (140, 140, 140)
        self._body_colors[:env._n_p] = (255, 128, 0)
        self._body_colors[env._n_p:env._n_pe] = (0, 85, 198)
        vertices_per_body = len(self._mesh)
        self._body_vertices = np.empty((count, vertices_per_body, 3), dtype=np.float32)
        # 固定顶点明暗使球体轮廓清晰，无需依赖光照状态。
        shade = 0.65 + 0.35 * np.clip(
            self._mesh @ np.array((0.3, -0.4, 0.866), dtype=np.float32), 0, 1)
        body_colors = np.asarray(self._body_colors[:, None, :] * shade[None, :, None],
                                 dtype=np.uint8)
        indices = (self._mesh_indices[None, :]
                   + np.arange(count, dtype=np.uint32)[:, None] * vertices_per_body)
        self._spheres = self._pyglet.graphics.vertex_list_indexed(
            count * vertices_per_body, indices.reshape(-1).tolist(),
            ("v3f/dynamic", self._body_vertices.reshape(-1)),
            ("c3B/static", body_colors.reshape(-1)),
        )
        self._heading_vertices = np.empty((env._n_pe, 2, 3), dtype=np.float32)
        heading_colors = np.repeat(self._body_colors[:env._n_pe], 2, axis=0)
        self._headings = self._make_vertices(self._heading_vertices.reshape(-1, 3), heading_colors)
        corners = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1)
                            for z in (-1, 1)], dtype=np.float32) * env._L
        edges = [(i, j) for i in range(8) for j in range(i)
                 if np.count_nonzero(corners[i] != corners[j]) == 1]
        box_vertices = corners[np.asarray(edges)].reshape(-1, 3)
        self._box = self._make_vertices(box_vertices, np.full(box_vertices.shape, 185, np.uint8))
        # 坐标轴略超出场地边框，并使用通用的 XYZ 配色。
        axes = np.zeros((3, 2, 3), dtype=np.float32)
        axes[:, 0, :] = -env._L
        axes[:, 1, :] = -env._L
        for axis in range(3):
            axes[axis, 1, axis] = env._L * 1.15
        axis_colors = np.repeat(np.array(((190, 35, 35), (35, 145, 35), (35, 70, 190)), np.uint8), 2, axis=0)
        self._axes = self._make_vertices(axes.reshape(-1, 3), axis_colors)

    def _update_bodies(self, env):
        sizes = np.asarray(env._size).reshape(-1)
        np.multiply(self._mesh[None, :, :], sizes[:, None, None], out=self._body_vertices)
        np.add(self._body_vertices, env._p.T[:, None, :], out=self._body_vertices)
        if getattr(env, "_prey_death_enabled", False):
            dead = np.flatnonzero(~env._alive)
            # 使用退化三角形隐藏死亡个体，避免重新分配几何缓冲。
            self._body_vertices[dead] = env._p[:, dead].T[:, None, :]
        self._spheres.vertices = self._body_vertices.reshape(-1)
        self._heading_vertices[:, 0, :] = env._p[:, :env._n_pe].T
        np.multiply(env._heading[:, :env._n_pe].T,
                    (sizes[:env._n_pe] * 2.6)[:, None], out=self._heading_vertices[:, 1, :])
        np.add(self._heading_vertices[:, 1, :], self._heading_vertices[:, 0, :],
               out=self._heading_vertices[:, 1, :])
        if getattr(env, "_prey_death_enabled", False):
            self._heading_vertices[dead, 1] = self._heading_vertices[dead, 0]
        self._headings.vertices = self._heading_vertices.reshape(-1)

    def _update_trails(self, env):
        trajectory = getattr(env, "_p_traj", None) if env._render_traj else None
        length = 0 if trajectory is None else len(trajectory)
        key = (length, env._n_peo)
        if key != self._trail_key:
            self._delete(self._trails)
            self._trails = None
            self._trail_key = key
            self._trail_vertices = np.empty((max(0, length - 1), env._n_peo, 2, 3), dtype=np.float32)
            if length > 1:
                colors = np.empty(self._trail_vertices.shape[:-1] + (4,), dtype=np.uint8)
                colors[..., :3] = self._body_colors[None, :, None, :]
                # 较早的历史轨迹逐渐淡入背景；
                # 最新线段也保持较低显著度，以突出个体和朝向。
                alpha = np.linspace(0.025, 0.20, length) * 255
                colors[:, :, 0, 3] = alpha[:-1, None]
                colors[:, :, 1, 3] = alpha[1:, None]
                self._trails = self._pyglet.graphics.vertex_list(
                    self._trail_vertices.size // 3,
                    ("v3f/dynamic", self._trail_vertices.reshape(-1)),
                    ("c4B/static", colors.reshape(-1)),
                )
        if length <= 1:
            return
        self._trail_vertices[:, :, 0, :] = trajectory[:-1].transpose(0, 2, 1)
        self._trail_vertices[:, :, 1, :] = trajectory[1:].transpose(0, 2, 1)
        if env._is_periodic:
            # 退化线段不产生像素，并保留固定缓冲区。
            crossing = np.any(np.abs(self._trail_vertices[:, :, 1, :]
                                     - self._trail_vertices[:, :, 0, :]) > env._L, axis=2)
            self._trail_vertices[:, :, 1, :][crossing] = self._trail_vertices[:, :, 0, :][crossing]
        if getattr(env, "_prey_death_enabled", False):
            dead = np.flatnonzero(~env._alive)
            self._trail_vertices[:, dead, 1, :] = self._trail_vertices[:, dead, 0, :]
        self._trails.vertices = self._trail_vertices.reshape(-1)

    def _set_camera(self):
        gl = self._gl
        framebuffer_width, framebuffer_height = self.window.get_framebuffer_size()
        gl.glViewport(0, 0, framebuffer_width, framebuffer_height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        distance = 5.5 * self._half_width * self.zoom
        gl.gluPerspective(42.0, framebuffer_width / max(1, framebuffer_height),
                          self._half_width * 0.02, self._half_width * 60.0)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        azimuth, elevation = math.radians(self.azimuth), math.radians(self.elevation)
        eye = (distance * math.cos(elevation) * math.cos(azimuth),
               distance * math.cos(elevation) * math.sin(azimuth),
               distance * math.sin(elevation))
        up = (0.0, 1.0, 0.0) if abs(self.elevation) >= 89.9 else (0.0, 0.0, 1.0)
        gl.gluLookAt(*eye, 0.0, 0.0, 0.0, *up)

    def render(self, env, return_rgb_array=False):
        if not self.isopen:
            return False
        gl = self._gl
        self.window.switch_to()
        self.window.dispatch_events()
        if not self.isopen:
            return False
        self._prepare_geometry(env)
        self._update_bodies(env)
        self._update_trails(env)
        self._set_camera()
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glLineWidth(1.0)
        self._box.draw(gl.GL_LINES)
        self._axes.draw(gl.GL_LINES)
        self._spheres.draw(gl.GL_TRIANGLES)
        gl.glLineWidth(2.0)
        self._headings.draw(gl.GL_LINES)
        if self._trails is not None and env._render_traj:
            gl.glLineWidth(1.0)
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            # 透明历史轨迹不能遮挡实体或其他轨迹。
            gl.glDepthMask(gl.GL_FALSE)
            try:
                self._trails.draw(gl.GL_LINES)
            finally:
                gl.glDepthMask(gl.GL_TRUE)
                gl.glDisable(gl.GL_BLEND)
        result = self.isopen
        if return_rgb_array:
            image = self._pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
            pixels = np.frombuffer(image.get_data("RGBA", image.width * 4), dtype=np.uint8)
            result = pixels.reshape(image.height, image.width, 4)[::-1, :, :3].copy()
        self.window.flip()
        return result

    def close(self):
        if self.window is not None:
            self.window.switch_to()
            self._delete(self._spheres, self._headings, self._trails, self._box, self._axes)
            self._spheres = self._headings = self._trails = self._box = self._axes = None
            self.window.close()
            self.window = None
        self.isopen = False
