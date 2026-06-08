# -*- coding: utf-8 -*-
"""AI Image Pose Character Tool v1.6 Pro Robot.

Single-file Maya tool for image-to-procedural robot pose.
Keeps the Clean UI features while removing obsolete callbacks, hidden solver UI, and dead code.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time

import maya.cmds as cmds


TOOL_ROOT = (
    os.path.dirname(os.path.abspath(__file__))
    if "__file__" in globals()
    else r"E:\ITi\Secitions\25.AI\maya_pose_ai"
)
EXTERNAL_PYTHON = os.path.join(TOOL_ROOT, ".venv", "Scripts", "python.exe")
EXTRACT_SCRIPT = os.path.join(TOOL_ROOT, "extract_pose_image.py")
POSE_JSON = os.path.join(TOOL_ROOT, "pose.json")

UI_NAME = "AI_IMAGE_POSE_CHARACTER_TOOL_UI_V16_PRO_ROBOT"
SOURCE_ROOT = "AI_Source_Skeleton_GRP"
SOURCE_SET = "AI_Source_HIK_Joints_SET"
LANDMARK_GRP = "AI_Image_Landmarks_GRP"
BOY_MODEL_GRP = "AI_Robot_Character_GRP"   # renamed for clarity
BOY_RIG_GRP = "AI_Boy_Rig_GRP"
BACKGROUND_GRP = "AI_Background_Image_GRP"
CONTROL_GRP = "AI_Viewport_Controls_GRP"
IK_GRP = "AI_IK_Handles_GRP"


REQUIRED_LANDMARKS = [
    "nose",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_foot_index", "right_foot_index",
]

JOINTS = {
    "hips": "AI_Hips",
    "spine": "AI_Spine",
    "chest": "AI_Chest",
    "neck": "AI_Neck",
    "head": "AI_Head",
    "left_arm": "AI_LeftArm",
    "left_forearm": "AI_LeftForeArm",
    "left_hand": "AI_LeftHand",
    "right_arm": "AI_RightArm",
    "right_forearm": "AI_RightForeArm",
    "right_hand": "AI_RightHand",
    "left_upleg": "AI_LeftUpLeg",
    "left_leg": "AI_LeftLeg",
    "left_foot": "AI_LeftFoot",
    "left_toe": "AI_LeftToeBase",
    "right_upleg": "AI_RightUpLeg",
    "right_leg": "AI_RightLeg",
    "right_foot": "AI_RightFoot",
    "right_toe": "AI_RightToeBase",
}

JOINT_POINT_MAP = {
    "hips": "hip_center",
    "spine": "spine_mid",
    "chest": "shoulder_center",
    "neck": "neck",
    "head": "head",
    "left_arm": "left_shoulder",
    "left_forearm": "left_elbow",
    "left_hand": "left_wrist",
    "right_arm": "right_shoulder",
    "right_forearm": "right_elbow",
    "right_hand": "right_wrist",
    "left_upleg": "left_hip",
    "left_leg": "left_knee",
    "left_foot": "left_ankle",
    "left_toe": "left_foot_index",
    "right_upleg": "right_hip",
    "right_leg": "right_knee",
    "right_foot": "right_ankle",
    "right_toe": "right_foot_index",
}

PARENTS = {
    "spine": "hips",
    "chest": "spine",
    "neck": "chest",
    "head": "neck",
    "left_arm": "chest",
    "left_forearm": "left_arm",
    "left_hand": "left_forearm",
    "right_arm": "chest",
    "right_forearm": "right_arm",
    "right_hand": "right_forearm",
    "left_upleg": "hips",
    "left_leg": "left_upleg",
    "left_foot": "left_leg",
    "left_toe": "left_foot",
    "right_upleg": "hips",
    "right_leg": "right_upleg",
    "right_foot": "right_leg",
    "right_toe": "right_foot",
}

DEFAULT_SLIDERS = {
    "height": 10.0,
    "depth": 0.25,
    "character_scale": 1.0,
    "landmark_size": 0.08,
    "pose_match": 1.0,
    "endpoint_bias": 0.15,
    "body_x": 0.0,
    "body_y": 0.0,
    "body_z": 0.0,
    "head_x": 0.0,
    "head_y": 0.0,
    "left_arm_x": 0.0,
    "left_arm_y": 0.0,
    "right_arm_x": 0.0,
    "right_arm_y": 0.0,
    "left_leg_x": 0.0,
    "left_leg_y": 0.0,
    "right_leg_x": 0.0,
    "right_leg_y": 0.0,
}


class ToolState(object):
    def __init__(self):
        self.image_path = ""
        self.pose_json_path = POSE_JSON
        self.pose_status = "Not generated"
        self.confidence = None
        self.quality = "Unknown"
        self.source_points = None          # raw visual skeleton from image / manual edit
        self.base_points = None            # fixed-length driver points used by the robot
        self.current_points = None
        self.loaded_pose_mode = False
        self.sliders = dict(DEFAULT_SLIDERS)
        self.log_lines = []
        self.warnings = []
        self.last_action = "Tool loaded"
        self.show_skeleton = True
        self.show_controls = True
        self._updating_from_controls = False
        self._script_jobs = []
        self._active_control = None
        self._rebuild_queued = False


STATE = ToolState()


def timestamp():
    return time.strftime("%H:%M:%S")


def log(message, warning=False):
    line = "[%s] %s: %s" % (timestamp(), "WARNING" if warning else "INFO", message)
    STATE.log_lines.append(line)
    STATE.log_lines = STATE.log_lines[-120:]
    if warning:
        STATE.warnings.append(message)
        STATE.warnings = STATE.warnings[-20:]
        cmds.warning(message)
    else:
        print(message)
    refresh_ui()


def set_action(message):
    STATE.last_action = message
    log(message)


def ui_exists(name, kind="control"):
    try:
        if kind == "window":
            return cmds.window(name, exists=True)
        if kind == "text":
            return cmds.text(name, exists=True)
        if kind == "textField":
            return cmds.textField(name, exists=True)
        if kind == "scrollField":
            return cmds.scrollField(name, exists=True)
        if kind == "floatSliderGrp":
            return cmds.floatSliderGrp(name, exists=True)
        return cmds.control(name, exists=True)
    except Exception:
        return False


def refresh_ui():
    if ui_exists("aiBoyImagePathField", "textField"):
        cmds.textField("aiBoyImagePathField", e=True, text=STATE.image_path)
    if ui_exists("aiBoyPoseJsonField", "textField"):
        cmds.textField("aiBoyPoseJsonField", e=True, text=STATE.pose_json_path)
    if ui_exists("aiBoyStatusLabel", "text"):
        conf = "n/a" if STATE.confidence is None else "%.3f" % STATE.confidence
        cmds.text(
            "aiBoyStatusLabel",
            e=True,
            label="%s | Confidence: %s | Quality: %s" % (STATE.pose_status, conf, STATE.quality),
        )
    if ui_exists("aiBoyDiagnostics", "scrollField"):
        text = [
            "Selected image: %s" % (STATE.image_path or "None"),
            "Pose JSON: %s" % STATE.pose_json_path,
            "Status: %s" % STATE.pose_status,
            "Last action: %s" % STATE.last_action,
            "Warnings: %s" % (", ".join(STATE.warnings[-5:]) if STATE.warnings else "None"),
            "",
            "Notes:",
            "- No AI camera is created.",
            "- The image is a real textured polygon plane in the scene.",
            "- The robot is fully procedural (no skinCluster, no bind).",
            "- Move/rotate any rig control to update the robot live.",
            "- Left controls: blue/cyan  |  Right controls: red/orange  |  Center: yellow.",
            "",
            "Log:",
        ]
        text.extend(STATE.log_lines[-80:])
        cmds.scrollField("aiBoyDiagnostics", e=True, text="\n".join(text))


def update_state_from_ui():
    for key in DEFAULT_SLIDERS:
        control = "aiBoy_%s" % key
        if ui_exists(control, "floatSliderGrp"):
            STATE.sliders[key] = cmds.floatSliderGrp(control, q=True, value=True)


def with_undo(label, func):
    update_state_from_ui()
    cmds.undoInfo(openChunk=True)
    try:
        func()
    except Exception as exc:
        log("%s failed: %s" % (label, exc), warning=True)
    finally:
        cmds.undoInfo(closeChunk=True)
        refresh_ui()


def v_add(a, b):
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def v_sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def v_mul(a, s):
    return [a[0] * s, a[1] * s, a[2] * s]


def v_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def v_len(v):
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def v_norm(v):
    l = v_len(v)
    if l < 1e-9:
        return [0.0, 1.0, 0.0]
    return v_mul(v, 1.0 / l)


def midpoint(a, b):
    return [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5]


def lerp(a, b, t):
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]


def rotate_point_euler(point, pivot, rotation_degrees):
    rx, ry, rz = [math.radians(float(v)) for v in rotation_degrees]
    x, y, z = v_sub(point, pivot)
    cy, sy = math.cos(rx), math.sin(rx)
    y, z = y * cy - z * sy, y * sy + z * cy
    cy, sy = math.cos(ry), math.sin(ry)
    x, z = x * cy + z * sy, -x * sy + z * cy
    cy, sy = math.cos(rz), math.sin(rz)
    x, y = x * cy - y * sy, x * sy + y * cy
    return v_add(pivot, [x, y, z])


def rotate_points_around(points, names, pivot, rotation_degrees):
    for name in names:
        if name in points and isinstance(points[name], list):
            points[name] = rotate_point_euler(points[name], pivot, rotation_degrees)


def safe_delete(node):
    if cmds.objExists(node):
        try:
            cmds.delete(node)
        except Exception:
            pass

class SuspendRefresh(object):
    def __enter__(self):
        try:
            cmds.refresh(suspend=True)
        except Exception:
            pass
        return self

    def __exit__(self, *_):
        try:
            cmds.refresh(suspend=False)
            cmds.refresh(force=True)
        except Exception:
            pass


def safe_name(text):
    base = os.path.splitext(os.path.basename(text))[0]
    base = re.sub(r"[^a-zA-Z0-9_]+", "_", base).strip("_")
    if not base:
        base = "image"
    if base[0].isdigit():
        base = "img_" + base
    return base


def remove_old_ai_cameras():
    for cam in cmds.ls("AI_Pose_Camera*", type="transform") or []:
        shapes = cmds.listRelatives(cam, shapes=True, type="camera") or []
        if shapes:
            try:
                cmds.delete(cam)
            except Exception:
                pass


def enable_textured_viewports():
    for panel in cmds.getPanel(type="modelPanel") or []:
        try:
            cmds.modelEditor(panel, e=True, displayTextures=True)
            cmds.modelEditor(panel, e=True, displayAppearance="smoothShaded")
        except Exception:
            pass


def load_pose_json(path=None):
    path = path or STATE.pose_json_path
    if not os.path.exists(path):
        raise RuntimeError("pose.json not found: %s" % path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not data.get("ok"):
        raise RuntimeError(data.get("error", "Pose JSON is not ok"))
    return data


def get_pose_landmarks(data):
    landmarks = data.get("landmarks")
    if not isinstance(landmarks, dict):
        raise RuntimeError("Invalid pose.json schema – expected data['landmarks'] as a dict.")
    missing = [name for name in REQUIRED_LANDMARKS if name not in landmarks]
    if missing:
        raise RuntimeError("Missing pose landmarks: " + ", ".join(missing))
    return landmarks


def landmark_confidence(lm):
    visibility = float(lm.get("visibility", 1.0))
    presence = float(lm.get("presence", visibility))
    return min(visibility, presence)


def update_pose_status_from_file():
    data = load_pose_json(STATE.pose_json_path)
    landmarks = get_pose_landmarks(data)
    values = [landmark_confidence(landmarks[name]) for name in REQUIRED_LANDMARKS]
    STATE.confidence = float(
        data.get("average_visibility", sum(values) / max(len(values), 1))
    )
    STATE.quality = "Good" if STATE.confidence >= 0.75 else "Warning"
    STATE.pose_status = "Pose ready"


def normalized_point(lm, aspect=1.0):
    return [
        (float(lm.get("x", 0.5)) - 0.5) * aspect,
        0.5 - float(lm.get("y", 0.5)),
        0.0,
    ]


def world_point(lm):
    return [float(lm.get("x", 0.0)), -float(lm.get("y", 0.0)), -float(lm.get("z", 0.0))]


def build_pose_points(data, fixed=True):
    """Build pose points from MediaPipe landmarks.

    fixed=False returns the raw source skeleton that visually follows the image.
    fixed=True returns a fixed-length driver pose so the robot never stretches.
    """
    aspect = float(data.get("width", 1.0) or 1.0) / max(
        float(data.get("height", 1.0) or 1.0), 1e-6
    )
    landmarks = get_pose_landmarks(data)
    points = {name: normalized_point(landmarks[name], aspect=aspect) for name in REQUIRED_LANDMARKS}

    world_landmarks = (
        data.get("world_landmarks")
        if isinstance(data.get("world_landmarks"), dict)
        else {}
    )
    if world_landmarks:
        world_points = {
            name: world_point(world_landmarks[name])
            for name in REQUIRED_LANDMARKS
            if name in world_landmarks
        }
        if "left_hip" in world_points and "right_hip" in world_points:
            hip_world = midpoint(world_points["left_hip"], world_points["right_hip"])
            for name in REQUIRED_LANDMARKS:
                if name in world_points:
                    points[name][2] = (
                        world_points[name][2] - hip_world[2]
                    ) * STATE.sliders["depth"]

    hip_center = midpoint(points["left_hip"], points["right_hip"])
    shoulder_center = midpoint(points["left_shoulder"], points["right_shoulder"])
    foot_center = midpoint(points["left_foot_index"], points["right_foot_index"])

    source_height = max(v_len(v_sub(points["nose"], foot_center)), 0.001)
    scale = STATE.sliders["height"] / source_height
    root = [0.0, STATE.sliders["height"] * 0.48, 0.0]

    result = {}
    for name in REQUIRED_LANDMARKS:
        relative = v_sub(points[name], hip_center)
        result[name] = v_add(root, v_mul(relative, scale))

    result["hip_center"] = root
    result["shoulder_center"] = v_add(root, v_mul(v_sub(shoulder_center, hip_center), scale))
    result["spine_mid"] = midpoint(result["hip_center"], result["shoulder_center"])
    result["neck"] = lerp(result["shoulder_center"], result["nose"], 0.35)
    result["head"] = result["nose"]
    if fixed:
        return retarget_to_fixed_proportions(result)
    return result


def _dir_between(a, b, fallback=(0.0, 1.0, 0.0)):
    d = v_sub(b, a)
    if v_len(d) < 1e-6:
        return list(fallback)
    return v_norm(d)


def _safe_perp_from_mid(root, mid, end, main_dir, fallback=(0.0, 0.0, 1.0)):
    to_mid = v_sub(mid, root)
    projected = v_mul(main_dir, v_dot(to_mid, main_dir))
    side = v_sub(to_mid, projected)
    if v_len(side) < 1e-5:
        side = list(fallback)
    return v_norm(side)


def _solve_two_bone_fixed(root, src_mid, src_end, len_a, len_b, fallback_pole=(0.0, 0.0, 1.0)):
    """Return mid/end preserving the two bone lengths.

    The end follows the source direction as closely as possible, but distance is
    clamped to the physically reachable range. This is the core no-stretch fix.
    """
    target_dir = _dir_between(root, src_end, fallback=(1.0, 0.0, 0.0))
    desired_dist = v_len(v_sub(src_end, root))
    max_d = max(len_a + len_b - 1e-4, 1e-4)
    min_d = max(abs(len_a - len_b) + 1e-4, 1e-4)
    d = max(min(desired_dist, max_d), min_d)
    end = v_add(root, v_mul(target_dir, d))

    a = (len_a * len_a - len_b * len_b + d * d) / max(2.0 * d, 1e-6)
    h2 = max(len_a * len_a - a * a, 0.0)
    h = math.sqrt(h2)
    pole = _safe_perp_from_mid(root, src_mid, src_end, target_dir, fallback=fallback_pole)
    mid = v_add(v_add(root, v_mul(target_dir, a)), v_mul(pole, h))
    return mid, end


def _solve_two_bone_angle_copy(root, src_root, src_mid, src_end, len_a, len_b,
                               fallback_a=(1.0, 0.0, 0.0), fallback_b=(1.0, 0.0, 0.0)):
    """Copy the actual image bone directions with fixed segment lengths.

    This is closer visually than forcing the end effector to a distant image
    point. It preserves the silhouette/action of the photo without stretching the
    robot limbs. If a landmark is noisy, fallback directions keep the limb stable.
    """
    dir_a = _dir_between(src_root, src_mid, fallback=fallback_a)
    dir_b = _dir_between(src_mid, src_end, fallback=fallback_b)
    mid = v_add(root, v_mul(dir_a, len_a))
    end = v_add(mid, v_mul(dir_b, len_b))
    return mid, end


def _blend_point(a, b, t):
    t = max(0.0, min(1.0, float(t)))
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]


def _solve_two_bone_pose_match(root, src_root, src_mid, src_end, len_a, len_b,
                               fallback_pole=(0.0, 0.0, 1.0)):
    """Best visual two-bone solve for image pose matching, with no stretching.

    - angle-copy gives the most faithful pose/silhouette.
    - endpoint IK gives stable reach when the detection is noisy.
    The pose_match slider blends toward angle-copy by default.
    """
    fallback_a = _dir_between(root, src_end, fallback=(1.0, 0.0, 0.0))
    angle_mid, angle_end = _solve_two_bone_angle_copy(
        root, src_root, src_mid, src_end, len_a, len_b,
        fallback_a=fallback_a, fallback_b=fallback_a
    )
    ik_mid, ik_end = _solve_two_bone_fixed(
        root, src_mid, src_end, len_a, len_b, fallback_pole=fallback_pole
    )
    match = STATE.sliders.get("pose_match", 1.0)
    mid = _blend_point(ik_mid, angle_mid, match)
    end = _blend_point(ik_end, angle_end, match)

    bias = STATE.sliders.get("endpoint_bias", 0.15)
    if bias > 0.0:
        mid = _blend_point(mid, ik_mid, bias)
        end = _blend_point(end, ik_end, bias)
    return mid, end


def _make_source_with_derived(points):
    """Ensure all derived pose keys exist before solving."""
    p = json.loads(json.dumps(points or {}))
    if "left_hip" in p and "right_hip" in p:
        p["hip_center"] = midpoint(p["left_hip"], p["right_hip"])
    if "left_shoulder" in p and "right_shoulder" in p:
        p["shoulder_center"] = midpoint(p["left_shoulder"], p["right_shoulder"])
    if "hip_center" in p and "shoulder_center" in p:
        p["spine_mid"] = midpoint(p["hip_center"], p["shoulder_center"])
    if "nose" not in p and "head" in p:
        p["nose"] = p["head"]
    if "neck" not in p and "shoulder_center" in p and "nose" in p:
        p["neck"] = lerp(p["shoulder_center"], p["nose"], 0.35)
    if "head" not in p and "nose" in p:
        p["head"] = p["nose"]
    return p

def _natural_lengths(height):
    h = max(float(height), 1.0)
    return {
        "spine": h * 0.32,
        "neck": h * 0.055,
        "head": h * 0.110,
        "shoulder_width": h * 0.30,
        "hip_width": h * 0.19,
        "upper_arm": h * 0.175,
        "forearm": h * 0.170,
        "thigh": h * 0.235,
        "shin": h * 0.235,
        "foot": h * 0.085,
    }


def retarget_to_fixed_proportions(source_points):
    """Convert source pose into a fixed-length robot driver pose.

    v1.5 uses a more faithful pose-direction solver:
    - the source skeleton supplies bone directions and body action
    - the robot keeps its natural bone lengths
    - limbs copy photo angles instead of stretching toward far landmarks
    This gives a much closer pose match without the long spaghetti-limb problem.
    """
    if not source_points:
        return {}
    src = _make_source_with_derived(source_points)
    L = _natural_lengths(STATE.sliders.get("height", 10.0))

    hip = src.get("hip_center", midpoint(src["left_hip"], src["right_hip"]))
    shoulder_src = src.get("shoulder_center", midpoint(src["left_shoulder"], src["right_shoulder"]))

    spine_dir = _dir_between(hip, shoulder_src, fallback=(0.0, 1.0, 0.0))
    shoulder_dir = _dir_between(src["left_shoulder"], src["right_shoulder"], fallback=(1.0, 0.0, 0.0))
    hip_dir = _dir_between(src["left_hip"], src["right_hip"], fallback=shoulder_dir)

    out = {}
    out["hip_center"] = list(hip)
    out["shoulder_center"] = v_add(out["hip_center"], v_mul(spine_dir, L["spine"]))
    out["spine_mid"] = midpoint(out["hip_center"], out["shoulder_center"])

    out["left_shoulder"] = v_add(out["shoulder_center"], v_mul(shoulder_dir, -L["shoulder_width"] * 0.5))
    out["right_shoulder"] = v_add(out["shoulder_center"], v_mul(shoulder_dir, L["shoulder_width"] * 0.5))
    out["left_hip"] = v_add(out["hip_center"], v_mul(hip_dir, -L["hip_width"] * 0.5))
    out["right_hip"] = v_add(out["hip_center"], v_mul(hip_dir, L["hip_width"] * 0.5))

    neck_dir = _dir_between(src.get("shoulder_center", shoulder_src), src.get("neck", shoulder_src), fallback=spine_dir)
    head_dir = _dir_between(src.get("neck", shoulder_src), src.get("head", src.get("nose", shoulder_src)), fallback=spine_dir)
    out["neck"] = v_add(out["shoulder_center"], v_mul(neck_dir, L["neck"]))
    out["head"] = v_add(out["neck"], v_mul(head_dir, L["head"]))

    nose_dir = _dir_between(src.get("head", src.get("nose", out["head"])), src.get("nose", src.get("head", out["head"])), fallback=head_dir)
    out["nose"] = v_add(out["head"], v_mul(nose_dir, max(L["head"] * 0.25, 0.05)))

    out["left_elbow"], out["left_wrist"] = _solve_two_bone_pose_match(
        out["left_shoulder"], src["left_shoulder"], src["left_elbow"], src["left_wrist"],
        L["upper_arm"], L["forearm"], fallback_pole=(0.0, 0.0, 1.0)
    )
    out["right_elbow"], out["right_wrist"] = _solve_two_bone_pose_match(
        out["right_shoulder"], src["right_shoulder"], src["right_elbow"], src["right_wrist"],
        L["upper_arm"], L["forearm"], fallback_pole=(0.0, 0.0, 1.0)
    )

    out["left_knee"], out["left_ankle"] = _solve_two_bone_pose_match(
        out["left_hip"], src["left_hip"], src["left_knee"], src["left_ankle"],
        L["thigh"], L["shin"], fallback_pole=(0.0, 0.0, -1.0)
    )
    out["right_knee"], out["right_ankle"] = _solve_two_bone_pose_match(
        out["right_hip"], src["right_hip"], src["right_knee"], src["right_ankle"],
        L["thigh"], L["shin"], fallback_pole=(0.0, 0.0, -1.0)
    )

    out["left_foot_index"] = v_add(
        out["left_ankle"],
        v_mul(_dir_between(src["left_ankle"], src["left_foot_index"], fallback=(0.0, -0.2, 1.0)), L["foot"])
    )
    out["right_foot_index"] = v_add(
        out["right_ankle"],
        v_mul(_dir_between(src["right_ankle"], src["right_foot_index"], fallback=(0.0, -0.2, 1.0)), L["foot"])
    )

    return out

def apply_live_offsets(points):
    p = json.loads(json.dumps(points))
    s = STATE.sliders

    body_offset = [s["body_x"], s["body_y"], s["body_z"]]
    for name in p:
        if isinstance(p[name], list):
            p[name] = v_add(p[name], body_offset)

    head_offset = [s["head_x"], s["head_y"], 0.0]
    for name in ["nose", "neck", "head"]:
        if name in p:
            p[name] = v_add(p[name], head_offset)

    left_arm = [s["left_arm_x"], s["left_arm_y"], 0.0]
    right_arm = [s["right_arm_x"], s["right_arm_y"], 0.0]
    for name, factor in [("left_elbow", 0.55), ("left_wrist", 1.0)]:
        p[name] = v_add(p[name], v_mul(left_arm, factor))
    for name, factor in [("right_elbow", 0.55), ("right_wrist", 1.0)]:
        p[name] = v_add(p[name], v_mul(right_arm, factor))

    left_leg = [s["left_leg_x"], s["left_leg_y"], 0.0]
    right_leg = [s["right_leg_x"], s["right_leg_y"], 0.0]
    for name, factor in [("left_knee", 0.55), ("left_ankle", 0.85), ("left_foot_index", 1.0)]:
        p[name] = v_add(p[name], v_mul(left_leg, factor))
    for name, factor in [("right_knee", 0.55), ("right_ankle", 0.85), ("right_foot_index", 1.0)]:
        p[name] = v_add(p[name], v_mul(right_leg, factor))

    p["hip_center"] = midpoint(p["left_hip"], p["right_hip"])
    p["shoulder_center"] = midpoint(p["left_shoulder"], p["right_shoulder"])
    p["spine_mid"] = midpoint(p["hip_center"], p["shoulder_center"])
    p["neck"] = lerp(p["shoulder_center"], p["nose"], 0.35)
    p["head"] = p["nose"]
    return retarget_to_fixed_proportions(p)


def make_image_material(image_path):
    base = safe_name(image_path)
    mat = "AI_%s_Image_MAT" % base
    sg = mat + "SG"
    file_node = mat + "_file"
    for n in [mat, sg, file_node]:
        if cmds.objExists(n):
            try:
                cmds.delete(n)
            except Exception:
                pass
    mat = cmds.shadingNode("lambert", asShader=True, name=mat)
    file_node = cmds.shadingNode("file", asTexture=True, name=file_node)
    cmds.setAttr(file_node + ".fileTextureName", image_path, type="string")
    cmds.connectAttr(file_node + ".outColor", mat + ".color", force=True)
    sg = cmds.sets(renderable=True, noSurfaceShader=True, empty=True, name=sg)
    cmds.connectAttr(mat + ".outColor", sg + ".surfaceShader", force=True)
    return sg


def create_image_background():
    if not STATE.image_path or not os.path.exists(STATE.image_path):
        return
    safe_delete(BACKGROUND_GRP)
    remove_old_ai_cameras()

    group = cmds.group(empty=True, name=BACKGROUND_GRP)
    base = safe_name(STATE.image_path)
    mesh_name = "%s_ImagePlane" % base

    width = 16.0
    height = 9.0
    try:
        data = load_pose_json(STATE.pose_json_path)
        iw = float(data.get("width", 16.0) or 16.0)
        ih = float(data.get("height", 9.0) or 9.0)
        aspect = iw / max(ih, 1.0)
        height = STATE.sliders["height"] * 1.15
        width = height * aspect
    except Exception:
        pass

    plane = cmds.polyPlane(
        name=mesh_name, width=width, height=height,
        subdivisionsX=1, subdivisionsY=1, axis=(0, 0, 1)
    )[0]
    cmds.xform(plane, ws=True, t=[0, STATE.sliders["height"] * 0.46, -0.35])
    sg = make_image_material(STATE.image_path)
    cmds.sets(plane, edit=True, forceElement=sg)
    try:
        shape = (cmds.listRelatives(plane, shapes=True) or [None])[0]
        if shape:
            cmds.setAttr(shape + ".doubleSided", 1)
    except Exception:
        pass
    cmds.parent(plane, group)
    for node in [plane, group]:
        if cmds.objExists(node):
            try:
                cmds.setAttr(node + ".overrideEnabled", 1)
                cmds.setAttr(node + ".overrideDisplayType", 2)  # Reference: visible but not directly selectable
            except Exception:
                pass
    enable_textured_viewports()
    log("Created non-selectable viewport image background: %s" % mesh_name)


_ROBOT_MAT_CACHE = {}



def make_robot_materials():
    """Create/cache pro robot materials. Returns {key: shadingGroup}."""
    specs = {
        "armor":  ("AI_Robot_Gunmetal_MAT", (0.055, 0.075, 0.095), (0.55, 0.70, 0.82), 0.78),
        "body":   ("AI_Robot_Silver_MAT",   (0.48, 0.56, 0.62),  (0.85, 0.92, 1.00), 0.62),
        "panel":  ("AI_Robot_Panel_MAT",    (0.16, 0.22, 0.29),  (0.45, 0.62, 0.78), 0.55),
        "dark":   ("AI_Robot_Dark_MAT",     (0.018,0.022,0.028), (0.15, 0.18, 0.22), 0.30),
        "joint":  ("AI_Robot_Joint_MAT",    (0.09, 0.10, 0.115), (0.45, 0.48, 0.52), 0.50),
        "accent": ("AI_Robot_Cyan_MAT",     (0.00, 0.74, 0.94),  (0.72, 0.96, 1.00), 0.95),
        "warning":("AI_Robot_Orange_MAT",   (1.00, 0.36, 0.05),  (1.00, 0.70, 0.40), 0.75),
        "face":   ("AI_Robot_Face_MAT",     (0.01, 0.018,0.026), (0.12, 0.22, 0.35), 0.40),
        "eye":    ("AI_Robot_Eye_MAT",      (0.00, 1.00, 0.88),  (1.00, 1.00, 1.00), 1.00),
    }
    out = {}
    for key, (mat_name, color, spec_color, spec_power) in specs.items():
        sg_name = mat_name + "SG"
        cached = _ROBOT_MAT_CACHE.get(mat_name)
        if cached and cmds.objExists(cached):
            out[key] = cached
            continue
        for n in [mat_name, sg_name]:
            if cmds.objExists(n):
                try:
                    cmds.delete(n)
                except Exception:
                    pass
        mat = cmds.shadingNode("blinn", asShader=True, name=mat_name)
        cmds.setAttr(mat + ".color", color[0], color[1], color[2], type="double3")
        cmds.setAttr(mat + ".specularColor", spec_color[0], spec_color[1], spec_color[2], type="double3")
        cmds.setAttr(mat + ".eccentricity", max(0.03, 1.0 - spec_power))
        sg = cmds.sets(renderable=True, noSurfaceShader=True, empty=True, name=sg_name)
        cmds.connectAttr(mat + ".outColor", sg + ".surfaceShader", force=True)
        _ROBOT_MAT_CACHE[mat_name] = sg
        out[key] = sg
    return out


def assign_mat(node, sg):
    try:
        cmds.sets(node, edit=True, forceElement=sg)
    except Exception:
        pass


def _del_hist(node):
    try:
        cmds.delete(node, constructionHistory=True)
    except Exception:
        pass


def _aim_y_to_vector(node, start, end):
    tmp = node + "_aimTgt_TMP"
    safe_delete(tmp)
    tmp = cmds.spaceLocator(name=tmp)[0]
    cmds.xform(tmp, ws=True, t=end)
    try:
        cons = cmds.aimConstraint(tmp, node, aimVector=(0, 1, 0), upVector=(0, 0, 1), worldUpType="vector", worldUpVector=(0, 0, 1))
        if cons:
            cmds.delete(cons)
    except Exception:
        pass
    safe_delete(tmp)


def make_oriented_box(name, start, end, width, depth, sg, parent):
    length = max(v_len(v_sub(end, start)), 0.001)
    node = cmds.polyCube(name=name, width=width, height=length, depth=depth, sx=1, sy=1, sz=1)[0]
    cmds.xform(node, ws=True, t=midpoint(start, end))
    _aim_y_to_vector(node, start, end)
    assign_mat(node, sg)
    _del_hist(node)
    cmds.parent(node, parent)
    return node


def make_flat_box(name, pos, sx, sy, sz, sg, parent):
    node = cmds.polyCube(name=name, width=sx, height=sy, depth=sz, sx=1, sy=1, sz=1)[0]
    cmds.xform(node, ws=True, t=pos)
    assign_mat(node, sg)
    _del_hist(node)
    cmds.parent(node, parent)
    return node


def make_joint_ball(name, pos, radius, sg, parent):
    node = cmds.polySphere(name=name, radius=radius, subdivisionsX=18, subdivisionsY=10)[0]
    cmds.xform(node, ws=True, t=pos)
    assign_mat(node, sg)
    _del_hist(node)
    cmds.parent(node, parent)
    return node


def make_cylinder_between(name, start, end, radius, sg, parent, sides=18):
    length = max(v_len(v_sub(end, start)), 0.001)
    node = cmds.polyCylinder(name=name, radius=radius, height=length, subdivisionsX=sides, subdivisionsY=1)[0]
    cmds.xform(node, ws=True, t=midpoint(start, end))
    _aim_y_to_vector(node, start, end)
    assign_mat(node, sg)
    _del_hist(node)
    cmds.parent(node, parent)
    return node


def make_robot_head(pts, h, scale, mats, root):
    head = pts["head"]
    hs = h * 0.138 * scale
    make_flat_box("AI_RobotHead_MainHelmet", head, hs * 0.82, hs * 0.95, hs * 0.70, mats["body"], root)
    make_flat_box("AI_RobotHead_CrownArmor", v_add(head, [0, hs * 0.28, 0]), hs * 0.98, hs * 0.22, hs * 0.80, mats["armor"], root)
    make_flat_box("AI_RobotHead_JawArmor", v_add(head, [0, -hs * 0.36, hs * 0.02]), hs * 0.68, hs * 0.18, hs * 0.55, mats["panel"], root)
    make_flat_box("AI_RobotHead_FaceGlass", v_add(head, [0, hs * 0.02, hs * 0.39]), hs * 0.58, hs * 0.36, hs * 0.035, mats["face"], root)
    make_flat_box("AI_RobotHead_VisorGlow", v_add(head, [0, hs * 0.10, hs * 0.425]), hs * 0.49, hs * 0.105, hs * 0.025, mats["eye"], root)
    for sgn, label in [(-1, "Left"), (1, "Right")]:
        make_flat_box("AI_RobotHead_%sEarPod" % label, v_add(head, [sgn * hs * 0.53, hs * 0.02, 0]), hs * 0.12, hs * 0.34, hs * 0.26, mats["dark"], root)
        make_flat_box("AI_RobotHead_%sEarGlow" % label, v_add(head, [sgn * hs * 0.60, hs * 0.02, 0]), hs * 0.025, hs * 0.18, hs * 0.16, mats["accent"], root)
    make_cylinder_between("AI_RobotHead_AntennaStem", v_add(head, [0, hs * 0.42, 0]), v_add(head, [0, hs * 0.76, 0]), hs * 0.035, mats["accent"], root, sides=12)
    make_joint_ball("AI_RobotHead_AntennaTip", v_add(head, [0, hs * 0.80, 0]), hs * 0.075, mats["eye"], root)
    make_cylinder_between("AI_RobotHead_NeckPiston", pts["neck"], head, hs * 0.12, mats["joint"], root, sides=16)


def make_robot_torso(pts, h, scale, mats, root):
    hip, spine, chest = pts["hip_center"], pts["spine_mid"], pts["shoulder_center"]
    shoulder_span = max(v_len(v_sub(pts["left_shoulder"], pts["right_shoulder"])), h * 0.22)
    cw = max(shoulder_span * 0.92, h * 0.25) * scale
    td = h * 0.115 * scale
    make_cylinder_between("AI_RobotTorso_SpineHydraulic", hip, chest, cw * 0.16, mats["joint"], root, sides=18)
    make_oriented_box("AI_RobotTorso_AbdomenCore", hip, spine, cw * 0.46, td * 0.82, mats["body"], root)
    make_oriented_box("AI_RobotTorso_ChestArmor", spine, chest, cw * 0.88, td * 1.05, mats["armor"], root)
    make_flat_box("AI_RobotTorso_ChestFace", v_add(lerp(spine, chest, 0.62), [0, 0, td * 0.58]), cw * 0.66, h * 0.080 * scale, td * 0.10, mats["panel"], root)
    make_flat_box("AI_RobotTorso_CoreLight", v_add(spine, [0, 0, td * 0.66]), h * 0.070 * scale, h * 0.070 * scale, h * 0.018 * scale, mats["eye"], root)
    make_flat_box("AI_RobotTorso_Collar", v_add(chest, [0, h * 0.030 * scale, 0]), cw * 0.98, h * 0.040 * scale, td * 0.88, mats["panel"], root)
    make_flat_box("AI_RobotTorso_BackReactor", v_add(spine, [0, h * 0.025 * scale, -td * 0.62]), cw * 0.42, h * 0.18 * scale, td * 0.38, mats["dark"], root)
    for sgn, label in [(-1, "Left"), (1, "Right")]:
        make_flat_box("AI_RobotTorso_%sSideVent" % label, v_add(spine, [sgn * cw * 0.40, 0, td * 0.25]), cw * 0.10, h * 0.15 * scale, td * 0.08, mats["accent"], root)


def make_robot_pelvis(pts, h, scale, mats, root):
    hip, lhip, rhip = pts["hip_center"], pts["left_hip"], pts["right_hip"]
    hip_span = max(v_len(v_sub(lhip, rhip)), h * 0.13)
    pw = max(hip_span * 1.25, h * 0.20) * scale
    td = h * 0.105 * scale
    make_oriented_box("AI_RobotPelvis_CrossBeam", lhip, rhip, pw * 0.50, td * 0.90, mats["armor"], root)
    make_flat_box("AI_RobotPelvis_CoreBlock", hip, pw * 0.76, h * 0.082 * scale, td * 1.05, mats["panel"], root)
    make_flat_box("AI_RobotPelvis_BeltGlow", v_add(hip, [0, h * 0.052 * scale, td * 0.56]), pw * 0.54, h * 0.022 * scale, td * 0.08, mats["accent"], root)


def make_robot_shoulder_pad(name, shoulder, elbow, arm_w, sgn, mats, root):
    pad_pos = v_add(shoulder, [sgn * arm_w * 0.46, arm_w * 0.20, 0])
    make_flat_box(name + "_MainPad", pad_pos, arm_w * 2.30, arm_w * 1.04, arm_w * 1.48, mats["armor"], root)
    make_flat_box(name + "_Trim", v_add(pad_pos, [0, arm_w * 0.62, arm_w * 0.04]), arm_w * 1.70, arm_w * 0.17, arm_w * 1.10, mats["accent"], root)
    make_flat_box(name + "_SideCap", v_add(pad_pos, [sgn * arm_w * 0.80, 0, 0]), arm_w * 0.28, arm_w * 0.76, arm_w * 1.12, mats["panel"], root)


def make_robot_limb_segment(name, start, end, width, depth_ratio, mats, root):
    make_cylinder_between(name + "_InnerPiston", start, end, width * 0.23, mats["joint"], root, sides=14)
    make_oriented_box(name + "_ArmorA", start, end, width * 0.92, width * depth_ratio, mats["body"], root)
    make_oriented_box(name + "_ArmorPlate", start, end, width * 0.58, width * 0.22, mats["armor"], root)
    try:
        plate = name + "_ArmorPlate"
        if cmds.objExists(plate):
            t = cmds.xform(plate, q=True, ws=True, t=True)
            cmds.xform(plate, ws=True, t=[t[0], t[1], t[2] + width * 0.20])
    except Exception:
        pass


def make_robot_hand(name, wrist, arm_w, mats, root):
    make_flat_box(name + "_Palm", wrist, arm_w * 1.15, arm_w * 0.78, arm_w * 1.04, mats["dark"], root)
    make_flat_box(name + "_BackPlate", v_add(wrist, [0, 0, arm_w * 0.58]), arm_w * 0.92, arm_w * 0.19, arm_w * 0.17, mats["accent"], root)
    for i, x in enumerate([-0.32, 0.0, 0.32]):
        make_flat_box(name + "_Finger_%d" % i, v_add(wrist, [x * arm_w, -arm_w * 0.56, arm_w * 0.22]), arm_w * 0.16, arm_w * 0.36, arm_w * 0.15, mats["body"], root)


def make_robot_foot(name, ankle, toe, leg_w, mats, root):
    make_oriented_box(name + "_BootBase", ankle, toe, leg_w * 1.08, leg_w * 1.25, mats["dark"], root)
    make_flat_box(name + "_ToeArmor", toe, leg_w * 1.22, leg_w * 0.45, leg_w * 0.78, mats["armor"], root)
    make_flat_box(name + "_Sole", v_add(toe, [0, -leg_w * 0.28, 0]), leg_w * 1.30, leg_w * 0.18, leg_w * 0.92, mats["joint"], root)
    make_flat_box(name + "_AnkleGlow", v_add(ankle, [0, -leg_w * 0.16, leg_w * 0.30]), leg_w * 0.82, leg_w * 0.22, leg_w * 0.14, mats["accent"], root)


def create_skeleton_fit_character(points):
    """Build a polished procedural robot from pose points. No skinCluster, no mesh bind."""
    safe_delete(BOY_MODEL_GRP)
    root = cmds.group(empty=True, name=BOY_MODEL_GRP)
    mats = make_robot_materials()
    h = max(STATE.sliders.get("height", 10.0), 1.0)
    scale = max(STATE.sliders.get("character_scale", 1.0), 0.25)
    arm_w = h * 0.052 * scale
    leg_w = h * 0.062 * scale

    make_robot_head(points, h, scale, mats, root)
    make_robot_torso(points, h, scale, mats, root)
    make_robot_pelvis(points, h, scale, mats, root)

    for side, sgn, keys in [
        ("Left", -1.0, ("left_shoulder", "left_elbow", "left_wrist")),
        ("Right", 1.0, ("right_shoulder", "right_elbow", "right_wrist")),
    ]:
        shoulder, elbow, wrist = [points[k] for k in keys]
        make_robot_shoulder_pad("AI_Robot_%sShoulder" % side, shoulder, elbow, arm_w, sgn, mats, root)
        make_joint_ball("AI_Robot_%sShoulderJoint" % side, shoulder, arm_w * 0.72, mats["joint"], root)
        make_robot_limb_segment("AI_Robot_%sUpperArm" % side, shoulder, elbow, arm_w * 0.98, 0.82, mats, root)
        make_joint_ball("AI_Robot_%sElbowJoint" % side, elbow, arm_w * 0.80, mats["joint"], root)
        make_robot_limb_segment("AI_Robot_%sForeArm" % side, elbow, wrist, arm_w * 0.84, 0.74, mats, root)
        make_robot_hand("AI_Robot_%sHand" % side, wrist, arm_w, mats, root)

    for side, keys in [
        ("Left", ("left_hip", "left_knee", "left_ankle", "left_foot_index")),
        ("Right", ("right_hip", "right_knee", "right_ankle", "right_foot_index")),
    ]:
        hip_pt, knee, ankle, toe = [points[k] for k in keys]
        make_joint_ball("AI_Robot_%sHipJoint" % side, hip_pt, leg_w * 0.72, mats["joint"], root)
        make_robot_limb_segment("AI_Robot_%sThigh" % side, hip_pt, knee, leg_w * 0.98, 0.84, mats, root)
        make_joint_ball("AI_Robot_%sKneeJoint" % side, knee, leg_w * 0.78, mats["joint"], root)
        make_flat_box("AI_Robot_%sKneeCap" % side, v_add(knee, [0, 0, leg_w * 0.62]), leg_w * 0.88, leg_w * 0.32, leg_w * 0.22, mats["accent"], root)
        make_robot_limb_segment("AI_Robot_%sShin" % side, knee, ankle, leg_w * 0.86, 0.76, mats, root)
        make_joint_ball("AI_Robot_%sAnkleJoint" % side, ankle, leg_w * 0.58, mats["joint"], root)
        make_robot_foot("AI_Robot_%sFoot" % side, ankle, toe, leg_w, mats, root)

    return root

def create_joint(name, position, parent=None, radius=0.15):
    if cmds.objExists(name):
        cmds.delete(name)
    cmds.select(clear=True)
    joint = cmds.joint(name=name, position=position, radius=radius)
    if parent and cmds.objExists(parent):
        try:
            cmds.parent(joint, parent)
        except Exception:
            pass
    return joint


def create_skeleton_at(points):
    safe_delete(SOURCE_ROOT)
    safe_delete(BOY_RIG_GRP)
    rig_group = cmds.group(empty=True, name=BOY_RIG_GRP)
    source_group = cmds.group(empty=True, name=SOURCE_ROOT)

    for key in [
        "hips", "spine", "chest", "neck", "head",
        "left_arm", "left_forearm", "left_hand",
        "right_arm", "right_forearm", "right_hand",
        "left_upleg", "left_leg", "left_foot", "left_toe",
        "right_upleg", "right_leg", "right_foot", "right_toe",
    ]:
        parent_key = PARENTS.get(key)
        parent = JOINTS[parent_key] if parent_key else None
        point_name = JOINT_POINT_MAP[key]
        create_joint(JOINTS[key], points[point_name], parent=parent, radius=0.18)

    try:
        cmds.parent(JOINTS["hips"], source_group)
    except Exception:
        pass
    try:
        cmds.parent(source_group, rig_group)
    except Exception:
        pass

    if cmds.objExists(SOURCE_SET):
        cmds.delete(SOURCE_SET)
    all_joints = [j for j in JOINTS.values() if cmds.objExists(j)]
    if all_joints:
        cmds.sets(all_joints, name=SOURCE_SET)
    return all_joints


def set_skeleton_positions(points):
    for key, joint in JOINTS.items():
        point_name = JOINT_POINT_MAP[key]
        if cmds.objExists(joint) and point_name in points:
            try:
                cmds.xform(joint, ws=True, t=points[point_name])
            except Exception:
                pass
    try:
        cmds.refresh(force=True)
    except Exception:
        pass


def clean_tool_scene():
    kill_control_script_jobs()
    for node in [BOY_MODEL_GRP, BOY_RIG_GRP, SOURCE_ROOT, LANDMARK_GRP, BACKGROUND_GRP, CONTROL_GRP, IK_GRP]:
        safe_delete(node)
    remove_old_ai_cameras()
    STATE.source_points = None
    STATE.base_points = None
    STATE.current_points = None
    STATE.pose_status = "Cleaned"
    set_action("Cleaned all AI Pose tool objects from the scene.")


def create_landmarks(points):
    safe_delete(LANDMARK_GRP)
    group = cmds.group(empty=True, name=LANDMARK_GRP)
    size = STATE.sliders["landmark_size"]
    for name in REQUIRED_LANDMARKS:
        loc = cmds.spaceLocator(name="AI_LM_%s_LOC" % name)[0]
        cmds.xform(loc, ws=True, t=points[name])
        for axis in "XYZ":
            cmds.setAttr(loc + ".localScale" + axis, size)
        try:
            cmds.parent(loc, group)
        except Exception:
            pass


def color_shape(node, color_index):
    for shape in cmds.listRelatives(node, shapes=True, fullPath=True) or []:
        try:
            cmds.setAttr(shape + ".overrideEnabled", 1)
            cmds.setAttr(shape + ".overrideColor", color_index)
        except Exception:
            pass


def _add_curve_shape(target, temp_curve):
    shapes = cmds.listRelatives(temp_curve, shapes=True, fullPath=True) or []
    for shape in shapes:
        try:
            cmds.parent(shape, target, add=True, shape=True, relative=True)
        except Exception:
            pass
    try:
        cmds.delete(temp_curve)
    except Exception:
        pass


def create_large_dual_ring_control(name, pos, radius=0.50, color=17, rings=3):
    """Create a control with 2 or 3 visible rings for important joints.
    rings=3 gives a full 3-axis sphere look for head and body.
    """
    if cmds.objExists(name):
        cmds.delete(name)
    ctrl = cmds.circle(name=name, normal=(0, 1, 0), radius=radius,
                       constructionHistory=False)[0]
    cmds.xform(ctrl, ws=True, t=pos)
    r2 = cmds.circle(name=name + "_shB_TMP", normal=(1, 0, 0), radius=radius * 0.96,
                     constructionHistory=False)[0]
    _add_curve_shape(ctrl, r2)
    if rings >= 3:
        r3 = cmds.circle(name=name + "_shC_TMP", normal=(0, 0, 1), radius=radius * 1.04,
                         constructionHistory=False)[0]
        _add_curve_shape(ctrl, r3)
    color_shape(ctrl, color)
    if cmds.objExists(CONTROL_GRP):
        cmds.parent(ctrl, CONTROL_GRP)
    return ctrl


def create_box_control(name, pos, size=0.45, color=17):
    if cmds.objExists(name):
        cmds.delete(name)
    pts = [
        (-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1), (-1, -1, -1),
        (-1, 1, -1), (1, 1, -1), (1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1),
        (1, 1, 1), (-1, 1, 1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1),
    ]
    pts = [(x * size, y * size, z * size) for x, y, z in pts]
    ctrl = cmds.curve(name=name, degree=1, point=pts)
    cmds.xform(ctrl, ws=True, t=pos)
    color_shape(ctrl, color)
    if cmds.objExists(CONTROL_GRP):
        cmds.parent(ctrl, CONTROL_GRP)
    return ctrl


def create_viewport_controls_and_ik(points):
    safe_delete(CONTROL_GRP)
    safe_delete(IK_GRP)
    cmds.group(empty=True, name=CONTROL_GRP)
    cmds.group(empty=True, name=IK_GRP)

    h = max(STATE.sliders.get("height", 10.0), 1.0)
    cs = h * 0.048   # base control size

    create_box_control("AI_CTRL_Body", points["hip_center"], cs * 1.80, 17)
    create_large_dual_ring_control("AI_CTRL_Head", points["head"], cs * 1.95, 17, rings=3)

    create_large_dual_ring_control("AI_CTRL_LeftShoulder",  points["left_shoulder"],  cs * 1.00, 6,  rings=2)
    create_large_dual_ring_control("AI_CTRL_LeftElbow",     points["left_elbow"],     cs * 0.90, 18, rings=2)
    create_large_dual_ring_control("AI_CTRL_LeftHand_IK",   points["left_wrist"],     cs * 1.05, 6,  rings=2)
    create_large_dual_ring_control("AI_CTRL_LeftHip",       points["left_hip"],       cs * 1.00, 6,  rings=2)
    create_large_dual_ring_control("AI_CTRL_LeftKnee",      points["left_knee"],      cs * 0.92, 18, rings=2)
    create_box_control("AI_CTRL_LeftFoot_IK",  points["left_foot_index"],  cs * 1.08, 6)

    create_large_dual_ring_control("AI_CTRL_RightShoulder", points["right_shoulder"], cs * 1.00, 13, rings=2)
    create_large_dual_ring_control("AI_CTRL_RightElbow",    points["right_elbow"],    cs * 0.90, 18, rings=2)
    create_large_dual_ring_control("AI_CTRL_RightHand_IK",  points["right_wrist"],    cs * 1.05, 13, rings=2)
    create_large_dual_ring_control("AI_CTRL_RightHip",      points["right_hip"],      cs * 1.00, 13, rings=2)
    create_large_dual_ring_control("AI_CTRL_RightKnee",     points["right_knee"],     cs * 0.92, 18, rings=2)
    create_box_control("AI_CTRL_RightFoot_IK", points["right_foot_index"], cs * 1.08, 13)


    if cmds.objExists(BOY_RIG_GRP):
        try:
            cmds.parent(CONTROL_GRP, BOY_RIG_GRP)
            cmds.parent(IK_GRP, BOY_RIG_GRP)
        except Exception:
            pass

    install_control_script_jobs()
    set_skeleton_visibility(STATE.show_skeleton)
    log("Stable controls created. Hidden Maya IK handles are disabled to stop cross-control motion.")


_CTRL_POINT_MAP = {
    "AI_CTRL_Body":           "hip_center",
    "AI_CTRL_Head":           "head",
    "AI_CTRL_LeftShoulder":   "left_shoulder",
    "AI_CTRL_RightShoulder":  "right_shoulder",
    "AI_CTRL_LeftElbow":      "left_elbow",
    "AI_CTRL_RightElbow":     "right_elbow",
    "AI_CTRL_LeftHand_IK":    "left_wrist",
    "AI_CTRL_RightHand_IK":   "right_wrist",
    "AI_CTRL_LeftHip":        "left_hip",
    "AI_CTRL_RightHip":       "right_hip",
    "AI_CTRL_LeftKnee":       "left_knee",
    "AI_CTRL_RightKnee":      "right_knee",
    "AI_CTRL_LeftFoot_IK":    "left_foot_index",
    "AI_CTRL_RightFoot_IK":   "right_foot_index",
}


def sync_controls_to_points(points, preserve_active_control=None):
    """Move each control to its matching point position.
    Skip the currently-dragged control so it doesn't jump back.
    """
    for ctrl_name, point_key in _CTRL_POINT_MAP.items():
        if ctrl_name == preserve_active_control:
            continue
        if not cmds.objExists(ctrl_name):
            continue
        if point_key not in points:
            continue
        try:
            cmds.xform(ctrl_name, ws=True, t=points[point_key])
        except Exception:
            pass


def get_world_pos(node):
    return cmds.xform(node, q=True, ws=True, t=True)


def control_pos(name, fallback):
    if cmds.objExists(name):
        try:
            return list(get_world_pos(name))
        except Exception:
            return list(fallback)
    return list(fallback)


def get_world_rot(node):
    return cmds.xform(node, q=True, ws=True, rotation=True)


def control_rot(name, fallback=(0.0, 0.0, 0.0)):
    if cmds.objExists(name):
        try:
            return list(get_world_rot(name))
        except Exception:
            return list(fallback)
    return list(fallback)


def _solver_start_points():
    """Always solve from the original image/rest pose, not from the last result.

    The v1.2 bug was cumulative rotation: a control with rotateZ=10 would rotate
    the already-rotated points again on every scriptJob tick, causing endless
    spinning and sometimes a Maya crash. Solving from base_points makes rotations
    absolute and stable.
    """
    base = STATE.base_points or STATE.current_points or {}
    return json.loads(json.dumps(base))


def _hide_ik_group():
    if cmds.objExists(IK_GRP):
        try:
            cmds.setAttr(IK_GRP + ".visibility", 0)
        except Exception:
            pass


def points_from_viewport_controls():
    """Read current control positions/rotations and derive updated pose points.

    v1.3 fixes:
    - solve from the image/base pose every time, not from the previous result
    - use absolute control rotations, so Rotate does not accumulate forever
    - no hidden IK handles fight the point solver
    """
    src = _solver_start_points()
    if not src:
        return {}

    def move_chain(root_key, child_keys, ctrl_name):
        old_pos = src.get(root_key)
        if old_pos is None:
            return
        new_pos = control_pos(ctrl_name, old_pos)
        delta = v_sub(new_pos, old_pos)
        src[root_key] = new_pos
        for key in child_keys:
            if key in src:
                src[key] = v_add(src[key], delta)

    old_hip = src.get("hip_center", [0, 0, 0])
    new_hip = control_pos("AI_CTRL_Body", old_hip)
    delta = v_sub(new_hip, old_hip)
    for k, v in list(src.items()):
        if isinstance(v, list):
            src[k] = v_add(v, delta)

    body_rot = control_rot("AI_CTRL_Body")
    if any(abs(r) > 0.001 for r in body_rot):
        rotate_points_around(
            src,
            [k for k, v in src.items() if isinstance(v, list)],
            new_hip, body_rot
        )

    move_chain("left_shoulder",  ["left_elbow", "left_wrist"], "AI_CTRL_LeftShoulder")
    move_chain("right_shoulder", ["right_elbow", "right_wrist"], "AI_CTRL_RightShoulder")
    move_chain("left_hip",  ["left_knee", "left_ankle", "left_foot_index"], "AI_CTRL_LeftHip")
    move_chain("right_hip", ["right_knee", "right_ankle", "right_foot_index"], "AI_CTRL_RightHip")

    src["head"] = control_pos("AI_CTRL_Head", src.get("head", [0, 0, 0]))
    src["left_elbow"]  = control_pos("AI_CTRL_LeftElbow",    src.get("left_elbow",  [0, 0, 0]))
    src["right_elbow"] = control_pos("AI_CTRL_RightElbow",   src.get("right_elbow", [0, 0, 0]))
    src["left_wrist"]  = control_pos("AI_CTRL_LeftHand_IK",  src.get("left_wrist",  [0, 0, 0]))
    src["right_wrist"] = control_pos("AI_CTRL_RightHand_IK", src.get("right_wrist", [0, 0, 0]))
    src["left_knee"]   = control_pos("AI_CTRL_LeftKnee",     src.get("left_knee",   [0, 0, 0]))
    src["right_knee"]  = control_pos("AI_CTRL_RightKnee",    src.get("right_knee",  [0, 0, 0]))
    src["left_foot_index"]  = control_pos("AI_CTRL_LeftFoot_IK",  src.get("left_foot_index",  [0, 0, 0]))
    src["right_foot_index"] = control_pos("AI_CTRL_RightFoot_IK", src.get("right_foot_index", [0, 0, 0]))

    rotate_points_around(src, ["left_elbow", "left_wrist"],
                         src["left_shoulder"], control_rot("AI_CTRL_LeftShoulder"))
    rotate_points_around(src, ["right_elbow", "right_wrist"],
                         src["right_shoulder"], control_rot("AI_CTRL_RightShoulder"))

    rotate_points_around(src, ["left_knee", "left_ankle", "left_foot_index"],
                         src["left_hip"], control_rot("AI_CTRL_LeftHip"))
    rotate_points_around(src, ["right_knee", "right_ankle", "right_foot_index"],
                         src["right_hip"], control_rot("AI_CTRL_RightHip"))

    head_rot = control_rot("AI_CTRL_Head")
    if "neck" in src:
        src["head"] = rotate_point_euler(src["head"], src["neck"], head_rot)
    src["nose"] = v_add(src["head"], [0, 0, max(STATE.sliders.get("height", 10.0) * 0.03, 0.14)])
    src["nose"] = rotate_point_euler(src["nose"], src["head"], head_rot)

    rotate_points_around(src, ["left_wrist"],  src["left_elbow"],  control_rot("AI_CTRL_LeftElbow"))
    rotate_points_around(src, ["right_wrist"], src["right_elbow"], control_rot("AI_CTRL_RightElbow"))
    rotate_points_around(src, ["left_wrist"],  src["left_elbow"],  control_rot("AI_CTRL_LeftHand_IK"))
    rotate_points_around(src, ["right_wrist"], src["right_elbow"], control_rot("AI_CTRL_RightHand_IK"))

    src["left_ankle"]  = lerp(src["left_knee"],  src["left_foot_index"],  0.72)
    src["right_ankle"] = lerp(src["right_knee"], src["right_foot_index"], 0.72)
    rotate_points_around(src, ["left_ankle", "left_foot_index"],
                         src["left_knee"], control_rot("AI_CTRL_LeftKnee"))
    rotate_points_around(src, ["right_ankle", "right_foot_index"],
                         src["right_knee"], control_rot("AI_CTRL_RightKnee"))
    rotate_points_around(src, ["left_foot_index"],  src["left_ankle"],  control_rot("AI_CTRL_LeftFoot_IK"))
    rotate_points_around(src, ["right_foot_index"], src["right_ankle"], control_rot("AI_CTRL_RightFoot_IK"))

    src["hip_center"]       = midpoint(src["left_hip"], src["right_hip"])
    src["shoulder_center"]  = midpoint(src["left_shoulder"], src["right_shoulder"])
    src["spine_mid"]        = midpoint(src["hip_center"], src["shoulder_center"])
    src["neck"]             = lerp(src["shoulder_center"], src["head"], 0.35)
    return retarget_to_fixed_proportions(src)


def kill_all_ai_pose_script_jobs():
    """Kill legacy auto-update scriptJobs from older versions of this tool.

    Older builds installed attributeChange, SelectionChanged, and DragRelease
    jobs. DragRelease is global in Maya, so even clicking the image plane could
    trigger a full robot rebuild and look like an endless background loading
    loop. This function removes those old jobs aggressively but only when their
    command text references this tool/control names.
    """
    killed = 0
    try:
        jobs = cmds.scriptJob(listJobs=True) or []
    except Exception:
        jobs = []
    needles = [
        "AI_CTRL_",
        "ai_pose_on_control",
        "ai_pose_request_control_apply",
        "rebuild_character_from_current_skeleton",
        "AI_IMAGE_POSE_CHARACTER_TOOL_UI",
        "DragRelease",
    ]
    for item in jobs:
        try:
            text = str(item)
            if not any(n in text for n in needles):
                continue
            job_id = int(text.split(":", 1)[0].strip())
            if cmds.scriptJob(exists=job_id):
                cmds.scriptJob(kill=job_id, force=True)
                killed += 1
        except Exception:
            pass
    STATE._script_jobs = []
    if killed:
        print("[AI Pose] Killed %d legacy auto-update scriptJobs." % killed)
    return killed


def kill_control_script_jobs():
    for job in list(getattr(STATE, "_script_jobs", [])):
        try:
            if cmds.scriptJob(exists=job):
                cmds.scriptJob(kill=job, force=True)
        except Exception:
            pass
    STATE._script_jobs = []


def rebuild_character_from_current_skeleton(active_control=None):
    if STATE._updating_from_controls:
        return
    STATE._updating_from_controls = True
    STATE._rebuild_queued = False
    if active_control:
        STATE._active_control = active_control
    try:
        with SuspendRefresh():
            pts = points_from_viewport_controls()
            if not pts:
                return
            STATE.current_points = pts
            set_skeleton_positions(pts)
            create_skeleton_fit_character(pts)
            create_landmarks(pts)
            sync_controls_to_points(pts, preserve_active_control=STATE._active_control)
            _hide_ik_group()
            enable_textured_viewports()
        refresh_ui()
    finally:
        STATE._updating_from_controls = False

def _selected_ai_control():
    """Return the selected viewport control if the current selection contains one."""
    controls = set(_CTRL_POINT_MAP.keys())
    selection = cmds.ls(selection=True, long=False) or []
    for item in selection:
        short = item.split("|")[-1]
        if short in controls:
            return short
    selection_long = cmds.ls(selection=True, long=True) or []
    for item in selection_long:
        parents = cmds.listRelatives(item, parent=True, fullPath=False) or []
        for parent in parents:
            short = parent.split("|")[-1]
            if short in controls:
                return short
    return None


def install_control_script_jobs():
    """Manual-safe mode: do not install ANY viewport auto-update scriptJobs.

    This fixes the Maya background-loading freeze. The previous performance fix
    still used DragRelease, which fires globally after viewport mouse actions,
    including clicking the background image. Now controls never rebuild
    automatically. Move/rotate controls freely, then press Apply Control Changes.
    """
    kill_all_ai_pose_script_jobs()
    STATE._active_control = None
    _hide_ik_group()
    log("Safe edit mode: legacy background jobs removed. Use Update Character after viewport edits.")


def set_skeleton_visibility(value):
    STATE.show_skeleton = bool(value)
    for node in [SOURCE_ROOT, LANDMARK_GRP]:
        if cmds.objExists(node):
            try:
                cmds.setAttr(node + ".visibility", 1 if STATE.show_skeleton else 0)
            except Exception:
                pass
    _apply_control_visibility()
    refresh_ui()


def _apply_control_visibility():
    vis = 1 if STATE.show_controls else 0
    if cmds.objExists(CONTROL_GRP):
        try:
            cmds.setAttr(CONTROL_GRP + ".visibility", vis)
        except Exception:
            pass
    _hide_ik_group()


def toggle_skeleton_visibility():
    set_skeleton_visibility(not STATE.show_skeleton)
    set_action("Skeleton: %s" % ("ON" if STATE.show_skeleton else "OFF"))


def toggle_controls_visibility():
    STATE.show_controls = not STATE.show_controls
    _apply_control_visibility()
    set_action("Controls: %s" % ("ON" if STATE.show_controls else "OFF"))


def build_character_from_base_points(rebuild_controls=True):
    if not STATE.base_points:
        raise RuntimeError("No pose points available. Browse an image or load a pose first.")
    current = apply_live_offsets(STATE.base_points)
    STATE.current_points = current

    with SuspendRefresh():
        remove_old_ai_cameras()
        create_image_background()
        if rebuild_controls or not cmds.objExists(JOINTS["hips"]):
            kill_control_script_jobs()
            safe_delete(CONTROL_GRP)
            safe_delete(IK_GRP)
            create_skeleton_at(current)
        else:
            set_skeleton_positions(current)

        set_skeleton_positions(current)
        create_skeleton_fit_character(current)
        create_landmarks(current)

        if rebuild_controls or not cmds.objExists(CONTROL_GRP):
            create_viewport_controls_and_ik(current)
        else:
            sync_controls_to_points(current)

        set_skeleton_visibility(STATE.show_skeleton)
        _apply_control_visibility()
        enable_textured_viewports()
    set_action("Robot built with v1.5 exact-direction fixed-length solver. Move/rotate controls to refine.")

def copy_character_with_controls():
    if not cmds.objExists(BOY_MODEL_GRP):
        raise RuntimeError("No character to copy yet.")
    base = "AI_Copied_Character_GRP"
    index = 1
    name = base
    while cmds.objExists(name):
        index += 1
        name = "%s_%02d" % (base, index)
    group = cmds.group(empty=True, name=name)
    for node in [BOY_MODEL_GRP, SOURCE_ROOT, CONTROL_GRP, IK_GRP]:
        if cmds.objExists(node):
            dup = cmds.duplicate(node, renameChildren=True)[0]
            cmds.parent(dup, group)
    set_action("Copied character: %s" % group)
    return group


def extract_pose_json():
    if not STATE.image_path:
        raise RuntimeError("Choose an image first.")
    if not os.path.exists(STATE.image_path):
        raise RuntimeError("Selected image does not exist: %s" % STATE.image_path)
    if not os.path.exists(EXTERNAL_PYTHON):
        raise RuntimeError("External Python not found: %s" % EXTERNAL_PYTHON)
    if not os.path.exists(EXTRACT_SCRIPT):
        raise RuntimeError("extract_pose_image.py not found: %s" % EXTRACT_SCRIPT)
    cmd = [EXTERNAL_PYTHON, EXTRACT_SCRIPT, "--image", STATE.image_path,
           "--out", STATE.pose_json_path]
    log("Running extractor: %s" % " ".join(cmd))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        STATE.pose_status = "Failed"
        raise RuntimeError("Pose extraction failed. See Script Editor output.")
    update_pose_status_from_file()


def extract_and_apply():
    extract_pose_json()
    data = load_pose_json(STATE.pose_json_path)
    STATE.source_points = build_pose_points(data, fixed=False)
    STATE.base_points = retarget_to_fixed_proportions(STATE.source_points)
    reset_sliders_to_defaults(update_ui=True)
    safe_delete(BOY_MODEL_GRP)
    safe_delete(BOY_RIG_GRP)
    safe_delete(SOURCE_ROOT)
    build_character_from_base_points(rebuild_controls=True)


def browse_image():
    result = cmds.fileDialog2(
        fileMode=1, caption="Choose Pose Image",
        fileFilter="Images (*.jpg *.jpeg *.png *.bmp)"
    )
    if not result:
        return
    STATE.image_path = result[0]
    STATE.loaded_pose_mode = False
    set_action("Selected image: %s" % STATE.image_path)
    extract_and_apply()


def save_pose():
    if not STATE.current_points:
        raise RuntimeError("No current pose to save.")
    result = cmds.fileDialog2(fileMode=0, caption="Save Pose JSON",
                               fileFilter="JSON (*.json)")
    if not result:
        return
    path = result[0]
    if not path.lower().endswith(".json"):
        path += ".json"
    data = {
        "ok": True,
        "type": "ai_robot_pose_v14",
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "image_path": STATE.image_path,
        "settings": STATE.sliders,
        "source_points": STATE.source_points or STATE.current_points,
        "points": STATE.current_points,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    set_action("Saved pose JSON: %s" % path)


def load_pose():
    result = cmds.fileDialog2(fileMode=1, caption="Load Pose JSON",
                               fileFilter="JSON (*.json)")
    if not result:
        return
    path = result[0]
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("type") not in ["ai_boy_pose_v05", "ai_boy_pose_v08", "ai_robot_pose_v14"] or \
       not isinstance(data.get("points"), dict):
        raise RuntimeError("Not a valid AI Pose JSON from this tool.")
    STATE.image_path = data.get("image_path", "")
    STATE.source_points = data.get("source_points", data["points"])
    STATE.loaded_pose_mode = True
    for key, value in data.get("settings", {}).items():
        if key in STATE.sliders:
            STATE.sliders[key] = value
    STATE.base_points = retarget_to_fixed_proportions(data["points"])
    STATE.current_points = STATE.base_points
    set_slider_ui_values()
    safe_delete(BOY_MODEL_GRP)
    safe_delete(BOY_RIG_GRP)
    safe_delete(SOURCE_ROOT)
    build_character_from_base_points(rebuild_controls=True)
    set_action("Loaded pose: %s" % path)


def reset_sliders_to_defaults(update_ui=True):
    for key, value in DEFAULT_SLIDERS.items():
        STATE.sliders[key] = value
    if update_ui:
        set_slider_ui_values()


def set_slider_ui_values():
    for key, value in STATE.sliders.items():
        control = "aiBoy_%s" % key
        if ui_exists(control, "floatSliderGrp"):
            try:
                cmds.floatSliderGrp(control, e=True, value=value)
            except Exception:
                pass


def reset_pos():
    if not STATE.base_points:
        raise RuntimeError("No image pose available to reset to.")
    reset_sliders_to_defaults(update_ui=True)
    build_character_from_base_points(rebuild_controls=True)
    set_action("Reset to original image pose.")


def cb_browse(*_):
    with_undo("Browse Image", browse_image)


def cb_reset(*_):
    with_undo("Reset Pos", reset_pos)


def cb_save(*_):
    with_undo("Save Pose", save_pose)


def cb_load(*_):
    with_undo("Load Pose", load_pose)


def cb_toggle_skeleton(*_):
    with_undo("Show/Hide Skeleton", toggle_skeleton_visibility)


def cb_toggle_controls(*_):
    with_undo("Show/Hide Controls", toggle_controls_visibility)


def cb_apply_controls(*_):
    with_undo("Apply Control Changes", lambda: rebuild_character_from_current_skeleton(_selected_ai_control() or STATE._active_control))


def cb_clean_tool_scene(*_):
    with_undo("Clean Tool Scene", clean_tool_scene)


def cb_copy_character(*_):
    with_undo("Copy Character", copy_character_with_controls)


def section(label):
    cmds.frameLayout(label=label, collapsable=False, marginWidth=10, marginHeight=8)
    cmds.columnLayout(adjustableColumn=True, rowSpacing=6)


def end_section():
    cmds.setParent("..")
    cmds.setParent("..")


def build_ui():
    kill_all_ai_pose_script_jobs()
    if cmds.window(UI_NAME, exists=True):
        cmds.deleteUI(UI_NAME)

    remove_old_ai_cameras()

    window = cmds.window(
        UI_NAME,
        title="AI Image Pose Character Tool v1.6 Pro Robot",
        widthHeight=(620, 650),
        sizeable=True,
    )
    cmds.scrollLayout(childResizable=True)
    cmds.columnLayout(adjustableColumn=True, rowSpacing=12, columnAttach=("both", 12))

    cmds.text(
        label="AI Image Pose Character Tool",
        align="center",
        height=34,
        font="boldLabelFont",
    )
    cmds.text(
        label="Browse an image, then edit the generated robot using the viewport controls.",
        align="center",
    )

    section("1) Reference Image")
    cmds.rowLayout(
        numberOfColumns=2,
        adjustableColumn=1,
        columnWidth2=(470, 120),
        columnAttach2=("both", "both"),
    )
    cmds.textField("aiBoyImagePathField", text=STATE.image_path, editable=False)
    cmds.button(
        label="Browse Image",
        height=36,
        backgroundColor=(0.18, 0.38, 0.55),
        command=cb_browse,
    )
    cmds.setParent("..")
    cmds.text("aiBoyStatusLabel", label=STATE.pose_status, align="left")
    end_section()

    section("2) Viewport Editing")
    cmds.text(
        label=(
            "Controls: Blue = left side  |  Red = right side  |  Yellow = body/head\n"
            "Move or rotate the rig controls in the viewport, then update the robot once."
        ),
        align="left",
    )
    cmds.rowColumnLayout(
        numberOfColumns=3,
        columnWidth=[(1, 190), (2, 190), (3, 190)],
        columnSpacing=[(1, 6), (2, 6), (3, 6)],
        rowSpacing=(1, 6),
    )
    cmds.button(
        label="Update Character",
        height=40,
        backgroundColor=(0.18, 0.44, 0.24),
        command=cb_apply_controls,
    )
    cmds.button(
        label="Show / Hide Skeleton",
        height=40,
        backgroundColor=(0.28, 0.28, 0.40),
        command=cb_toggle_skeleton,
    )
    cmds.button(
        label="Show / Hide Controls",
        height=40,
        backgroundColor=(0.28, 0.38, 0.28),
        command=cb_toggle_controls,
    )
    cmds.setParent("..")
    end_section()

    section("3) Pose Tools")
    cmds.rowColumnLayout(
        numberOfColumns=3,
        columnWidth=[(1, 190), (2, 190), (3, 190)],
        columnSpacing=[(1, 6), (2, 6), (3, 6)],
        rowSpacing=(1, 6),
    )
    cmds.button(
        label="Reset Pose",
        height=38,
        backgroundColor=(0.42, 0.30, 0.18),
        command=cb_reset,
    )
    cmds.button(
        label="Save Pose",
        height=38,
        backgroundColor=(0.20, 0.38, 0.24),
        command=cb_save,
    )
    cmds.button(
        label="Load Pose",
        height=38,
        backgroundColor=(0.20, 0.28, 0.48),
        command=cb_load,
    )
    cmds.setParent("..")
    cmds.rowColumnLayout(
        numberOfColumns=2,
        columnWidth=[(1, 285), (2, 285)],
        columnSpacing=[(1, 6), (2, 6)],
        rowSpacing=(1, 6),
    )
    cmds.button(
        label="Copy Character Snapshot",
        height=36,
        backgroundColor=(0.40, 0.30, 0.16),
        command=cb_copy_character,
    )
    cmds.button(
        label="Clean Scene",
        height=36,
        backgroundColor=(0.42, 0.22, 0.22),
        command=cb_clean_tool_scene,
    )
    cmds.setParent("..")
    end_section()

    section("4) Status / Log")
    cmds.scrollField(
        "aiBoyDiagnostics",
        editable=False,
        wordWrap=False,
        height=230,
        text="",
    )
    end_section()

    cmds.setParent("..")
    cmds.setParent("..")
    cmds.showWindow(window)
    refresh_ui()
    log(
        "Tool v1.5 Clean UI ready. Solver settings and Stop Background buttons were removed from the UI. "
        "The reference image is non-selectable, legacy background jobs are killed on startup, and viewport editing uses one clean Update Character action."
    )

build_ui()
