import numpy as np
import json


def rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def parse_pose_string(pose_string):
    """
    Parse pose string to motions list.
    Format: "w-3, right-0.5, d-4"
    - w: forward movement
    - s: backward movement
    - a: left movement
    - d: right movement
    - up: pitch up rotation
    - down: pitch down rotation
    - left: yaw left rotation
    - right: yaw right rotation
    - number after dash: duration in latents

    Args:
        pose_string: str, comma-separated pose commands

    Returns:
        list of dict: motions for generate_camera_trajectory_local
    """
    # Movement amount per frame
    # forward_speed = 0.08  # units per frame
    # yaw_speed = np.deg2rad(3)  # radians per frame
    # pitch_speed = np.deg2rad(3)  # radians per frame
    # forward_speed = 0.005 # units per frame
    # yaw_speed = np.deg2rad(0.5)  # radians per frame
    # pitch_speed = np.deg2rad(0.5)  # radians per frame

    forward_speed = 0.001 # units per frame
    yaw_speed = np.deg2rad(0.2)  # radians per frame
    pitch_speed = np.deg2rad(0.2)  # radians per frame

    motions = []
    commands = [cmd.strip() for cmd in pose_string.split(",")]

    for cmd in commands:
        if not cmd:
            continue

        parts = cmd.split("-")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid pose command: {cmd}. Expected format: 'action-duration'"
            )

        action = parts[0].strip()
        try:
            duration = float(parts[1].strip())
        except ValueError:
            raise ValueError(f"Invalid duration in command: {cmd}")

        num_frames = int(duration)

        # Parse action and create motion dict
        if action == "w":
            # Forward
            for _ in range(num_frames):
                motions.append({"forward": forward_speed})
        elif action == "s":
            # Backward
            for _ in range(num_frames):
                motions.append({"forward": -forward_speed})
        elif action == "a":
            # Left
            for _ in range(num_frames):
                motions.append({"right": -forward_speed})
        elif action == "d":
            # Right
            for _ in range(num_frames):
                motions.append({"right": forward_speed})
        elif action == "up":
            # Pitch up
            for _ in range(num_frames):
                motions.append({"pitch": pitch_speed})
        elif action == "down":
            # Pitch down
            for _ in range(num_frames):
                motions.append({"pitch": -pitch_speed})
        elif action == "left":
            # Yaw left
            for _ in range(num_frames):
                motions.append({"yaw": -yaw_speed})
        elif action == "right":
            # Yaw right
            for _ in range(num_frames):
                motions.append({"yaw": yaw_speed})
        else:
            raise ValueError(
                f"Unknown action: {action}. Supported actions: w, s, a, d, up, down, left, right"
            )

    return motions

def generate_camera_trajectory_local(motions, initial_pose=None):
    """
    motions: list of dict
             {"forward": 1.0}, {"yaw": np.pi/2}, {"pitch": np.pi/6}, {"right": 1.0}
             - forward: Translation (Forward or Backward)
             - yaw:   Rotate (Left or Right)
             - pitch: Rotate (Up or Down)
             - right: Translation (Right or Left)
             - third_yaw: Third Perspective Rotate (Left or Right)
    """

    poses = []
    if initial_pose is not None:
        T = np.array(initial_pose, dtype=float)
    else:
        T = np.eye(4)
    poses.append(T.copy())

    for move in motions:
        # Rotate (Left or Right)
        if "yaw" in move:
            R = rot_y(move["yaw"])
            T[:3, :3] = T[:3, :3] @ R

        # Rotate (Up or Down)
        if "pitch" in move:
            R = rot_x(move["pitch"])
            T[:3, :3] = T[:3, :3] @ R

        # Translation (Z-direction of the camera's local coordinate system)
        forward = move.get("forward", 0.0)
        if forward != 0:
            local_t = np.array([0, 0, forward])
            world_t = T[:3, :3] @ local_t
            T[:3, 3] += world_t

        # Translation (Z-direction of the camera's local coordinate system)
        right = move.get("right", 0.0)
        if right != 0:
            local_t = np.array([right, 0, 0])
            world_t = T[:3, :3] @ local_t
            T[:3, 3] += world_t

        # Third Perspective Rotate (Left or Right)
        third_yaw = move.get("third_yaw", 0.0)
        if third_yaw != 0:
            theta = -third_yaw
            C = np.array([[1, 0.0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -1.0], [0, 0, 0, 1]])
            c_origin = C.copy()
            # Rotation around the Y-axis
            R_y = np.array(
                [
                    [np.cos(theta), 0, np.sin(theta)],
                    [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)],
                ]
            )
            # Translation
            C[:3, :3] = C[:3, :3] @ R_y
            C[:3, 3] = R_y @ C[:3, 3]
            c_inv = np.linalg.inv(c_origin)
            c_relative = c_inv @ C
            T = T @ c_relative

        poses.append(T.copy())

    return poses


def pose_string_to_json(pose_string, initial_pose=None):
    """
    Convert pose string to pose JSON format.

    Args:
        pose_string: str, comma-separated pose commands
        initial_pose: array-like, 4x4 initial camera pose matrix

    Returns:
        dict: pose JSON with extrinsic and intrinsic parameters
    """
    motions = parse_pose_string(pose_string)
    poses = generate_camera_trajectory_local(motions, initial_pose=initial_pose)

    # Default intrinsic matrix (from generate_custom_trajectory.py)
    intrinsic = [
        [300.0, 0.0, 320.0],
        [0.0, 300.0, 180.0],
        [0.0, 0.0, 1.0],
    ]

    pose_json = {}
    for i, p in enumerate(poses):
        pose_json[str(i)] = {"extrinsic": p.tolist(), "K": intrinsic}

    return pose_json

if __name__ == "__main__":
    # Examples: Forward 0.08 * 16 -> Right Rotate 3 degree * 16
    motions = []
    for i in range(15):
        motions.append({"forward": 0.08})

    for i in range(16):
        motions.append({"yaw": np.deg2rad(3)})

    intrinsic = [
        [969.6969696969696, 0.0, 960.0],
        [0.0, 969.6969696969696, 540.0],
        [0.0, 0.0, 1.0],
    ]

    poses = generate_camera_trajectory_local(motions)
    custom_c2w = {}
    for i, p in enumerate(poses):
        custom_c2w[str(i)] = {"extrinsic": p.tolist(), "K": intrinsic}
        json.dump(
            custom_c2w,
            open("./assets/pose/pose.json", "w"),
            indent=4,
            ensure_ascii=False,
        )
