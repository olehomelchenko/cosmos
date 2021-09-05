import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import pandas as pd


class Universe:
    def __init__(
        self,
        G: float,
        N: int,
        dt: float,
        pos_std: int = 1,
        mas_std: int = 1,
        acc_std: int = 1,
        seed: int = None,
        heavy: bool = False,
        obj: np.array = None,
    ) -> None:
        self.G = G
        self.N = N
        self.dt = dt
        self.heavy = heavy
        self.seed = seed
        self.pos_std = pos_std
        self.mas_std = mas_std
        self.acc_std = acc_std
        self.img = None
        self.current_iteration = 0
        self.current_dt = 0

        if obj is None:
            self.obj = self.generate_objects()
            print("generating objects...")
        else:
            print("assigning external object")
            self.obj = obj

        self.obj_df = self.get_obj_df()

        self.generate_plot()
        self.borders = self.get_borders()
        self.scatter = None
        self.quiver = None

    def generate_objects(self) -> np.array:
        if self.seed:
            np.random.seed(self.seed)
        positions = self.pos_std * np.random.randn(self.N, 2)
        masses = self.mas_std * np.abs(np.random.randn(self.N, 1))
        if self.heavy:
            masses = 1.1 ** (masses * (100.0 / np.max(masses)))
            masses = 1000 * masses / np.max(masses)
        acc = self.acc_std * np.random.randn(self.N, 2)
        return np.c_[positions, masses, acc]
        return pd.DataFrame(arr, columns=["x", "y", "m", "dx", "dy"])

    def get_object_radius(self, objects: np.array) -> np.array:
        return np.power(objects / (3 / 4 * np.pi), 1 / 3)

    def get_obj_df(self):
        return pd.DataFrame(self.obj, columns=["x", "y", "m", "dx", "dy"])

    def plot_altair(self, x_domain=None, y_domain=None):
        df = self.get_obj_df()
        x_domain = x_domain or np.percentile(df["x"], [0, 100])
        y_domain = y_domain or np.percentile(df["y"], [0, 100])
        x = (
            alt.Chart(df)
            .mark_circle(clip=True)
            .encode(
                alt.X("x", scale=alt.Scale(domain=(x_domain))),
                alt.Y("y", scale=alt.Scale(domain=(y_domain))),
                size="m",
            )
            .interactive()
        )
        return x

    def get_com(self):
        arr = self.obj
        com = np.average(arr[:, :2], weights=arr[:, 2], axis=0)
        return com

    def get_borders(self):
        return np.percentile(self.obj[:, :2], [0, 100], axis=0).T

    def generate_plot(self):
        obj = self.obj
        if not self.img:
            self.img = plt.subplots()
            fig, ax = self.img
            scatter = ax.scatter(
                obj[:, 0], obj[:, 1], s=self.get_object_radius(obj[:, 2]), alpha=0.8
            )
            # quiver = ax.quiver(obj[:, 0], obj[:, 1], obj[:, 3], obj[:, 4], alpha=0.2)
            # self.img = fig, ax

    def plot_bodies(
        self, i=None, ax_obj=None, quiver=True, xlim=None, ylim=None, auto_center=None
    ):
        obj = self.obj
        com = self.get_com()
        fig, ax = self.img
        if ax_obj:
            ax = ax_obj
        ax.clear()
        ax.set_xlim(xlim[0] + com[0], xlim[1] + com[0]) if xlim else None
        ax.set_ylim(ylim[0] + com[1], ylim[1] + com[1]) if ylim else None
        scatter = ax.scatter(
            obj[:, 0], obj[:, 1], s=self.get_object_radius(obj[:, 2]), alpha=0.8
        )
        if quiver:
            ax.quiver(obj[:, 0], obj[:, 1], obj[:, 3], obj[:, 4], alpha=0.2)
        ax.plot(*self.get_com(), "r+")
        self.img = fig, ax
        return ax

    def calc_force(self, a, b):
        a_x, a_y, a_m = a
        b_x, b_y, b_m = b
        dx, dy = b_x - a_x, b_y - a_y

        r = (dx ** 2 + dy ** 2) ** 0.5
        F = self.G * a_m * b_m / r ** 2

        fx_1 = F * dx / r if dx != 0 else 0
        fy_1 = F * dy / r if dy != 0 else 0

        return fx_1, fy_1

    def calc_force_alt(self, obj):
        # print(obj)
        i, j = obj[:2]
        a, b = obj[2:]
        a_x, a_y, a_m = a[:3]
        b_x, b_y, b_m = b[:3]
        dx, dy = b_x - a_x, b_y - a_y

        r = (dx ** 2 + dy ** 2) ** 0.5
        F = self.G * a_m * b_m / r ** 2

        fx_1 = F * dx / r if dx != 0 else 0
        fy_1 = F * dy / r if dy != 0 else 0

        return [i, j, [fx_1, fy_1]]

    def get_delta_pos(self, F, m, pos, d_pos, dt):
        dd = F / m
        new_pos = pos + d_pos * dt + dd * (dt ** 2) * 0.5
        new_d_pos = d_pos + dd * dt
        return new_pos, new_d_pos

    def move_body(self, obj, forces, dt):
        b_x, b_y, b_m, b_dx, b_dy = obj
        Fx, Fy = forces
        new_x, new_dx = self.get_delta_pos(Fx, b_m, b_x, b_dx, dt)
        new_y, new_dy = self.get_delta_pos(Fy, b_m, b_y, b_dy, dt)
        return [new_x, new_y, b_m, new_dx, new_dy]

    def move_all_bodies(self, forces):
        obj = self.obj
        dt = self.dt
        new_pos = np.zeros((len(obj), 5))
        for i, o in enumerate(obj):
            neww = self.move_body(o, forces[i], dt)
            new_pos[i] = neww
        return new_pos

    def get_total_forces_alt(self):
        obj = self.obj
        final_forces = np.zeros((len(obj), 2))
        forces = []
        for i, pt_i in enumerate(obj):
            for j, pt_j in enumerate(obj):
                if i <= j:
                    continue
                # forces.append([i, j, self.calc_force(pt_i[0:3], pt_j[0:3])])
                forces.append([i, j, pt_i, pt_j])
        forces = np.apply_along_axis(self.calc_force_alt, 1, forces)
        for l in forces:
            f = list(l[2])
            final_forces[l[0]] += f
            final_forces[l[1]] -= f
        return final_forces

    def get_total_forces(self):
        obj = self.obj
        forces = np.zeros((len(obj), 2))
        for i, pt_i in enumerate(obj):
            for j, pt_j in enumerate(obj):
                if i <= j:
                    continue
                force = self.calc_force(pt_i[0:3], pt_j[0:3])
                forces[i] += list(force)
                forces[j] -= list(force)
        return forces

    def calc_positions(self, alt=False):
        if alt:
            forces = self.get_total_forces_alt()
        else:
            forces = self.get_total_forces()
        new_positions = self.move_all_bodies(forces)
        self.current_iteration += 1
        self.current_dt += self.dt
        self.borders = self.get_borders()
        return new_positions

    def iterate(self, N, alt=False):
        for _ in range(N):
            self.obj = self.calc_positions(alt=alt)
