"""Centre of mass of the Go2 from joint angles alone; NumPy only.

The training-side estimator asks MuJoCo for the centre of mass and its
Jacobian. The robot should not need a simulator, so the same rigid-body tree
is exported once as plain numbers (see `extract_tree`, PC only) and evaluated
here. Everything is expressed in the base frame; index 0 is the base.

Tree fields, one row per body, parents always before children:
    parent   index of the parent body, -1 for the base
    pos/quat pose in the parent frame at zero joint angle (quat is w, x, y, z)
    mass     kg
    ipos     centre of mass in the body frame
    joint    index into the 12 joint angles, or -1 for a fixed body
    axis     hinge axis in the body frame
    anchor   a point on the hinge axis in the body frame
plus `feet`: body index and local offset of the four foot-sphere centres.
"""

import numpy as np


def quat_to_matrix(quat):
    """Rotation matrix of a unit quaternion (w, x, y, z)."""
    w, x, y, z = np.asarray(quat, dtype=float) / np.linalg.norm(quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _axis_rotation(axis, angle):
    """Rodrigues rotation about a unit axis."""
    x, y, z = axis
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


class KinematicModel:
    def __init__(self, tree):
        self.parent = np.asarray(tree["parent"], dtype=int)
        self.pos = np.asarray(tree["pos"], dtype=float)
        self.rot = np.array([quat_to_matrix(q) for q in tree["quat"]])
        self.mass = np.asarray(tree["mass"], dtype=float)
        self.ipos = np.asarray(tree["ipos"], dtype=float)
        self.joint = np.asarray(tree["joint"], dtype=int)
        self.axis = np.asarray(tree["axis"], dtype=float)
        self.anchor = np.asarray(tree["anchor"], dtype=float)
        self.foot_body = np.asarray(tree["feet"]["body"], dtype=int)
        self.foot_offset = np.asarray(tree["feet"]["offset"], dtype=float)
        self.total_mass = float(self.mass.sum())
        n = len(self.parent)
        if self.parent[0] != -1 or (self.parent[1:] >= np.arange(1, n)).any():
            raise ValueError("kinematic tree must list parents before children")
        if sorted(self.joint[self.joint >= 0]) != list(range(12)):
            raise ValueError("kinematic tree must contain the 12 leg joints once")

    def _forward(self, q):
        """Pose of every body frame in the base frame."""
        n = len(self.parent)
        rotation = np.empty((n, 3, 3))
        position = np.empty((n, 3))
        rotation[0], position[0] = np.eye(3), 0.0
        for b in range(1, n):
            p = self.parent[b]
            r = rotation[p] @ self.rot[b]
            origin = position[p] + rotation[p] @ self.pos[b]
            j = self.joint[b]
            if j >= 0:
                turn = _axis_rotation(self.axis[b], q[j])
                # The hinge turns the body about its anchor, not its origin.
                origin = origin + r @ (self.anchor[b] - turn @ self.anchor[b])
                r = r @ turn
            rotation[b], position[b] = r, origin
        return rotation, position

    def com(self, q):
        """Centre of mass (3,) and its joint Jacobian (3, 12), base frame."""
        q = np.asarray(q, dtype=float)
        rotation, position = self._forward(q)
        centres = position + np.einsum("bij,bj->bi", rotation, self.ipos)
        # Mass and first moment of each body's subtree, children into parents.
        sub_mass = self.mass.copy()
        sub_moment = self.mass[:, None] * centres
        for b in range(len(self.parent) - 1, 0, -1):
            sub_mass[self.parent[b]] += sub_mass[b]
            sub_moment[self.parent[b]] += sub_moment[b]
        jacobian = np.zeros((3, 12))
        for b in np.flatnonzero(self.joint >= 0):
            axis = rotation[b] @ self.axis[b]
            anchor = position[b] + rotation[b] @ self.anchor[b]
            # Turning the hinge moves only its subtree, about the anchor.
            arm = sub_moment[b] - sub_mass[b] * anchor
            jacobian[:, self.joint[b]] = np.cross(axis, arm) / self.total_mass
        return sub_moment[0] / self.total_mass, jacobian

    def feet(self, q):
        """Centres of the four foot spheres (4, 3), base frame; FL, FR, RL, RR."""
        rotation, position = self._forward(np.asarray(q, dtype=float))
        return position[self.foot_body] + np.einsum(
            "bij,bj->bi", rotation[self.foot_body], self.foot_offset
        )

    def feet_motion(self, q):
        """Foot centres (4, 3) and how the joints move them, base frame.

        Returns centres, linear Jacobians (4, 3, 12) of the centres, and
        angular Jacobians (4, 3, 12) of the links the feet are fixed to.
        """
        rotation, position = self._forward(np.asarray(q, dtype=float))
        centres = position[self.foot_body] + np.einsum(
            "bij,bj->bi", rotation[self.foot_body], self.foot_offset
        )
        linear = np.zeros((4, 3, 12))
        angular = np.zeros((4, 3, 12))
        for foot, body in enumerate(self.foot_body):
            b = body
            while b > 0:  # every hinge between this foot and the base
                j = self.joint[b]
                if j >= 0:
                    axis = rotation[b] @ self.axis[b]
                    anchor = position[b] + rotation[b] @ self.anchor[b]
                    linear[foot, :, j] = np.cross(axis, centres[foot] - anchor)
                    angular[foot, :, j] = axis
                b = self.parent[b]
        return centres, linear, angular


def extract_tree(model, joint_names, foot_geoms):
    """Export a MuJoCo Go2 model as the plain tree above. PC side only."""
    base = model.body("base").id
    order = [base]
    for b in range(model.nbody):
        if b != base and model.body_parentid[b] in order:
            order.append(b)
    if len(order) != model.nbody - 1:
        raise ValueError("expected every body except the world below the base")
    index = {b: i for i, b in enumerate(order)}
    joint_of = {model.joint(n).id: i for i, n in enumerate(joint_names)}
    tree = dict(
        parent=[], pos=[], quat=[], mass=[], ipos=[], joint=[], axis=[], anchor=[]
    )
    for b in order:
        tree["parent"].append(-1 if b == base else index[int(model.body_parentid[b])])
        tree["pos"].append(model.body_pos[b].tolist())
        tree["quat"].append(model.body_quat[b].tolist())
        tree["mass"].append(float(model.body_mass[b]))
        tree["ipos"].append(model.body_ipos[b].tolist())
        joints = [
            j
            for j in range(
                model.body_jntadr[b], model.body_jntadr[b] + model.body_jntnum[b]
            )
            if b != base
        ]
        if len(joints) > 1 or any(j not in joint_of for j in joints):
            raise ValueError("expected at most one actuated hinge per body")
        if joints:
            j = joints[0]
            tree["joint"].append(joint_of[j])
            tree["axis"].append(model.jnt_axis[j].tolist())
            tree["anchor"].append(model.jnt_pos[j].tolist())
        else:
            tree["joint"].append(-1)
            tree["axis"].append([0.0, 0.0, 1.0])
            tree["anchor"].append([0.0, 0.0, 0.0])
    # The base pose is the root of the tree, not an offset from the world.
    tree["pos"][0], tree["quat"][0] = [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]
    tree["feet"] = dict(
        body=[index[int(model.geom_bodyid[g])] for g in foot_geoms],
        offset=[model.geom_pos[g].tolist() for g in foot_geoms],
    )
    return tree
