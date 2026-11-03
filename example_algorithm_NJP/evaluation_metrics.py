"""Read-only behavioral diagnostics; no random draws or changes to the simulator."""
import numpy as np


class Diagnostics:
    def __init__(self, env):
        self.env = env
        self.previous_rotation = None
        self.rows = []

    def add(self, actions, rewards):
        env = self.env
        row = {}
        death = getattr(env.unwrapped, "prey_death_enabled", False)
        alive = env.unwrapped.alive_mask if death else None
        contacts = (env.unwrapped.capture_contacts if death else
                    env.unwrapped._is_collide_b2b[env.n_p:env.n_pe, :env.n_p])
        row["contact_pairs"] = int(contacts.sum())
        if death:
            row["captures"] = int(contacts.any(axis=1).sum())
            row["alive_prey"] = int(alive[env.n_p:env.n_pe].sum())
        rotations = getattr(env.unwrapped, "_last_action", None)
        rotations = rotations[:3] if rotations is not None else None
        for name, group in (("predator", slice(0, env.n_p)), ("prey", slice(env.n_p, env.n_pe))):
            strategy = getattr(env.unwrapped, "pursuer_strategy" if name == "predator" else "escaper_strategy", "input")
            velocity = env.dp[:, group]
            living = alive[group] if death else None
            if death:
                velocity = velocity[:, living]
            speeds = np.linalg.norm(velocity, axis=0)
            energy = env.energy[0, group]
            if death:
                energy = energy[living]
            policy_actions = actions[name][:, living] if death else actions[name]
            data = {"reward": float(np.mean(rewards[name])),
                    "mean_speed": float(speeds.mean()) if len(speeds) else 0.,
                    "moving_fraction": float(np.mean(speeds > .01)) if len(speeds) else 0.,
                    "mean_energy": float(energy.mean()) if len(energy) else 0.,
                    "low_energy_fraction": float(np.mean(energy < .5)) if len(energy) else 0.,
                    # Rule controllers replace the supplied actions internally;
                    # their placeholder input has no policy saturation meaning.
                    "action_saturation": (float(np.mean(np.abs(policy_actions) >= .99))
                                          if policy_actions.size else 0.) if strategy == "input" else None}
            if death:
                data["alive_count"] = int(living.sum())
            if rotations is not None:
                current = rotations[:, group]
                if death:
                    current = current[:, living]
                if self.previous_rotation is not None:
                    previous = self.previous_rotation[:, group]
                    if death:
                        previous = previous[:, living]
                    valid = ((np.linalg.norm(current, axis=0) > 1e-10)
                             & (np.linalg.norm(previous, axis=0) > 1e-10))
                    data["turn_reversal_fraction"] = (float(np.mean(valid & (np.sum(current * previous, axis=0) < 0)))
                                                       if len(valid) else 0.)
                else:
                    data["turn_reversal_fraction"] = 0.
            positions = env.p[:, group].T
            data["mean_nearest_distance"] = None
            data["mean_pair_distance"] = None
            if death:
                positions = positions[living]
                data["local_heading_alignment"] = None
                data["moving_heading_alignment"] = None
            if len(positions) > 1:
                delta = positions[None] - positions[:, None]
                if env.is_periodic:
                    delta[delta > env.L] -= 2 * env.L
                    delta[delta < -env.L] += 2 * env.L
                distances = np.linalg.norm(delta, axis=2)
                np.fill_diagonal(distances, np.inf)
                data["mean_nearest_distance"] = float(distances.min(axis=1).mean())
                data["mean_pair_distance"] = float(distances[np.triu_indices(len(positions), 1)].mean())
                nearest = np.argmin(distances, axis=1)
                headings = env.heading[:, group].T
                if death:
                    headings = headings[living]
                alignment = np.sum(headings * headings[nearest], axis=1)
                data["local_heading_alignment"] = float(alignment.mean())
                moving = (speeds > .01) & (speeds[nearest] > .01)
                data["moving_heading_alignment"] = float(alignment[moving].mean()) if moving.any() else None
            row[name] = data
        if rotations is not None:
            self.previous_rotation = rotations.copy()
        self.rows.append(row)

    def summary(self):
        result = {"steps": len(self.rows), "mean_contact_pairs": float(np.mean([r["contact_pairs"] for r in self.rows]))}
        if self.rows and "captures" in self.rows[0]:
            result["captures"] = sum(r["captures"] for r in self.rows)
            result["surviving_prey"] = self.rows[-1]["alive_prey"]
            result["prey_survival_fraction"] = self.rows[-1]["alive_prey"] / self.env.n_e if self.env.n_e else 0.
        for name in ("predator", "prey"):
            result[name] = {}
            for metric in self.rows[0][name]:
                values = [r[name][metric] for r in self.rows if r[name][metric] is not None]
                result[name][metric] = float(np.mean(values)) if values else None
        return result
