# -*- coding: utf-8 -*-
"""
AI Image Pose Character Tool v29 Final Stable

A production-oriented Mixamo mode for the existing AI pose project.

What this tool does:
- Imports a real Mixamo FBX character, preferably the Character FBX downloaded with Skin.
- Extracts pose landmarks from a reference image using the existing external MediaPipe extractor.
- Scans Mixamo skeleton joints by stable joint names.
- Captures rest pose for safe reset.
- Creates IK handles and visible IK/PV controls for arms and legs.
- Builds an AI source guide from the image pose.
- Applies the source pose to the Mixamo character through IK targets, not mesh deformation.

Important:
- Use a true Mixamo FBX: Format = FBX for Maya, Skin = With Skin.
- Do not use OBJ, and do not use a random custom rig for this version.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time

import maya.cmds as cmds

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
TOOL_ROOT = (
    os.path.dirname(os.path.abspath(__file__))
    if "__file__" in globals()
    else r"E:\ITi\Secitions\25.AI\maya_pose_ai"
)
PROJECT_ROOTS = [
    TOOL_ROOT,
    r"E:\ITi\Secitions\25.AI\FinalProject",
]
EXTERNAL_PYTHON = os.path.join(TOOL_ROOT, ".venv", "Scripts", "python.exe")
EXTRACT_SCRIPT = os.path.join(TOOL_ROOT, "extract_pose_image.py")
POSE_JSON = os.path.join(TOOL_ROOT, "pose.json")

UI_NAME = "AI_IMAGE_POSE_CHARACTER_TOOL_V28_UI"
IMPORT_NAMESPACE_BASE = "mixamoCharacter"
TOOL_GRP = "AI_Mixamo_Character_Tool_GRP"
SOURCE_GRP = "AI_Mixamo_Source_Guide_GRP"
TARGET_GRP = "AI_Mixamo_Source_Targets_GRP"
IK_GRP = "AI_Mixamo_IK_Handles_GRP"
CTRL_GRP = "AI_Mixamo_Controls_GRP"
BACKGROUND_GRP = "AI_Mixamo_Background_Image_GRP"
NINJA_MATERIAL_NAME = "AI_Character_Auto_Texture_MAT"
NINJA_MATERIAL_SG = "AI_Character_Auto_Texture_SG"

# -----------------------------------------------------------------------------
# Pose landmarks / Mixamo mapping
# -----------------------------------------------------------------------------
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

MIXAMO_ALIASES = {
    "hips": ["Hips", "mixamorig:Hips", "mixamorig_Hips"],
    "spine": ["Spine", "mixamorig:Spine"],
    "spine1": ["Spine1", "Spine2", "Chest", "mixamorig:Spine1", "mixamorig:Spine2"],
    "neck": ["Neck", "mixamorig:Neck"],
    "head": ["Head", "mixamorig:Head"],
    "left_arm": ["LeftArm", "mixamorig:LeftArm"],
    "left_forearm": ["LeftForeArm", "mixamorig:LeftForeArm"],
    "left_hand": ["LeftHand", "mixamorig:LeftHand"],
    "right_arm": ["RightArm", "mixamorig:RightArm"],
    "right_forearm": ["RightForeArm", "mixamorig:RightForeArm"],
    "right_hand": ["RightHand", "mixamorig:RightHand"],
    "left_upleg": ["LeftUpLeg", "mixamorig:LeftUpLeg"],
    "left_leg": ["LeftLeg", "mixamorig:LeftLeg"],
    "left_foot": ["LeftFoot", "mixamorig:LeftFoot"],
    "left_toe": ["LeftToeBase", "LeftToe", "mixamorig:LeftToeBase"],
    "right_upleg": ["RightUpLeg", "mixamorig:RightUpLeg"],
    "right_leg": ["RightLeg", "mixamorig:RightLeg"],
    "right_foot": ["RightFoot", "mixamorig:RightFoot"],
    "right_toe": ["RightToeBase", "RightToe", "mixamorig:RightToeBase"],
}

REQUIRED_MIXAMO = [
    "hips", "spine", "head",
    "left_arm", "left_forearm", "left_hand",
    "right_arm", "right_forearm", "right_hand",
    "left_upleg", "left_leg", "left_foot",
    "right_upleg", "right_leg", "right_foot",
]

# -----------------------------------------------------------------------------
# State
# -----------------------------------------------------------------------------
class ToolState(object):
    def __init__(self):
        self.image_path = ""
        self.character_path = ""
        self.pose_json_path = POSE_JSON
        self.pose_points = None
        self.targets = {}
        self.joints = {}
        self.rest = {}
        self.ik = {}
        self.namespace = ""
        self.pose_status = "Not generated"
        self.confidence = None
        self.quality = "Unknown"
        self.pose_scale = 1.0
        self.depth_scale = 0.65
        self.mirror = False
        self.follow_root = True
        self.keep_feet_grounded = True
        self.show_source = False
        self.strong_retarget = True
        self.match_strength = 1.0
        self.body_align = True
        self.body_align_strength = 1.0
        self.auto_materials = True
        self.initial_image_pose = {}
        self.initial_pose_image_path = ""
        self.copy_index = 1
        self.log_lines = []
        self.warnings = []
        self.last_action = "Ready"

STATE = ToolState()

# -----------------------------------------------------------------------------
# UI / logging
# -----------------------------------------------------------------------------
def timestamp():
    return time.strftime("%H:%M:%S")


def log(message, warning=False):
    line = "[%s] %s: %s" % (timestamp(), "WARNING" if warning else "INFO", message)
    STATE.log_lines.append(line)
    STATE.log_lines = STATE.log_lines[-160:]
    if warning:
        STATE.warnings.append(message)
        STATE.warnings = STATE.warnings[-20:]
        try:
            cmds.warning(message)
        except Exception:
            pass
    else:
        print("[AIImagePoseV28] " + str(message))
    refresh_ui()


def set_action(message):
    STATE.last_action = message
    log(message)


def ui_exists(name, kind="control"):
    try:
        if kind == "window": return cmds.window(name, exists=True)
        if kind == "text": return cmds.text(name, exists=True)
        if kind == "textField": return cmds.textField(name, exists=True)
        if kind == "checkBox": return cmds.checkBox(name, exists=True)
        if kind == "floatField": return cmds.floatField(name, exists=True)
        if kind == "scrollField": return cmds.scrollField(name, exists=True)
        return cmds.control(name, exists=True)
    except Exception:
        return False


def refresh_ui():
    if ui_exists("mnImagePath", "textField"):
        cmds.textField("mnImagePath", e=True, text=STATE.image_path)
    if ui_exists("mnCharacterPath", "textField"):
        cmds.textField("mnCharacterPath", e=True, text=STATE.character_path)
    if ui_exists("mnStatus", "text"):
        conf = "n/a" if STATE.confidence is None else "%.3f" % STATE.confidence
        cmds.text("mnStatus", e=True, label="%s | Confidence: %s | Quality: %s" % (STATE.pose_status, conf, STATE.quality))
    if ui_exists("mnPoseScale", "floatField"):
        cmds.floatField("mnPoseScale", e=True, value=STATE.pose_scale)
    if ui_exists("mnDepthScale", "floatField"):
        cmds.floatField("mnDepthScale", e=True, value=STATE.depth_scale)
    if ui_exists("mnMirror", "checkBox"):
        cmds.checkBox("mnMirror", e=True, value=STATE.mirror)
    if ui_exists("mnRoot", "checkBox"):
        cmds.checkBox("mnRoot", e=True, value=STATE.follow_root)
    if ui_exists("mnGround", "checkBox"):
        cmds.checkBox("mnGround", e=True, value=STATE.keep_feet_grounded)
    if ui_exists("mnShowSource", "checkBox"):
        cmds.checkBox("mnShowSource", e=True, value=STATE.show_source)
    if ui_exists("mnStrongRetarget", "checkBox"):
        cmds.checkBox("mnStrongRetarget", e=True, value=STATE.strong_retarget)
    if ui_exists("mnMatchStrength", "floatField"):
        cmds.floatField("mnMatchStrength", e=True, value=STATE.match_strength)
    if ui_exists("mnBodyAlign", "checkBox"):
        cmds.checkBox("mnBodyAlign", e=True, value=STATE.body_align)
    if ui_exists("mnBodyAlignStrength", "floatField"):
        cmds.floatField("mnBodyAlignStrength", e=True, value=STATE.body_align_strength)
    if ui_exists("mnAutoMaterials", "checkBox"):
        cmds.checkBox("mnAutoMaterials", e=True, value=STATE.auto_materials)
    if ui_exists("mnJointPreview", "scrollField"):
        if STATE.joints:
            lines = []
            for key in sorted(MIXAMO_ALIASES):
                node = STATE.joints.get(key, "")
                status = "OK" if node else "MISSING"
                lines.append("%-14s %-8s %s" % (key, status, node))
            cmds.scrollField("mnJointPreview", e=True, text="\n".join(lines))
        else:
            cmds.scrollField("mnJointPreview", e=True, text="No skeleton scanned yet.")
    if ui_exists("mnLog", "scrollField"):
        text = [
            "Image: %s" % (STATE.image_path or "None"),
            "Character: %s" % (STATE.character_path or "None"),
            "Last action: %s" % STATE.last_action,
            "Warnings: %s" % (", ".join(STATE.warnings[-5:]) if STATE.warnings else "None"),
            "",
            "Log:",
        ]
        text.extend(STATE.log_lines[-100:])
        cmds.scrollField("mnLog", e=True, text="\n".join(text))


def read_ui():
    if ui_exists("mnPoseScale", "floatField"):
        STATE.pose_scale = float(cmds.floatField("mnPoseScale", q=True, value=True))
    if ui_exists("mnDepthScale", "floatField"):
        STATE.depth_scale = float(cmds.floatField("mnDepthScale", q=True, value=True))
    if ui_exists("mnMirror", "checkBox"):
        STATE.mirror = bool(cmds.checkBox("mnMirror", q=True, value=True))
    if ui_exists("mnRoot", "checkBox"):
        STATE.follow_root = bool(cmds.checkBox("mnRoot", q=True, value=True))
    if ui_exists("mnGround", "checkBox"):
        STATE.keep_feet_grounded = bool(cmds.checkBox("mnGround", q=True, value=True))
    if ui_exists("mnShowSource", "checkBox"):
        STATE.show_source = bool(cmds.checkBox("mnShowSource", q=True, value=True))
    if ui_exists("mnStrongRetarget", "checkBox"):
        STATE.strong_retarget = bool(cmds.checkBox("mnStrongRetarget", q=True, value=True))
    if ui_exists("mnMatchStrength", "floatField"):
        STATE.match_strength = float(cmds.floatField("mnMatchStrength", q=True, value=True))
    if ui_exists("mnBodyAlign", "checkBox"):
        STATE.body_align = bool(cmds.checkBox("mnBodyAlign", q=True, value=True))
    if ui_exists("mnBodyAlignStrength", "floatField"):
        STATE.body_align_strength = float(cmds.floatField("mnBodyAlignStrength", q=True, value=True))
    if ui_exists("mnAutoMaterials", "checkBox"):
        STATE.auto_materials = bool(cmds.checkBox("mnAutoMaterials", q=True, value=True))


def read_ui_values():
    """Compatibility alias. v27 accidentally called read_ui_values while only read_ui existed."""
    return read_ui()


def with_undo(label, func):
    read_ui()
    cmds.undoInfo(openChunk=True)
    try:
        return func()
    except Exception as exc:
        log("%s failed: %s" % (label, exc), warning=True)
    finally:
        cmds.undoInfo(closeChunk=True)
        refresh_ui()

# -----------------------------------------------------------------------------
# Math / scene helpers
# -----------------------------------------------------------------------------
def exists(node):
    return bool(node) and cmds.objExists(node)


def safe_delete(node):
    if exists(node):
        try: cmds.delete(node)
        except Exception: pass


def leaf(node):
    return (node or "").split("|")[-1]


def leaf_no_ns(node):
    return leaf(node).split(":")[-1]


def clean_name(name):
    return (name or "").lower().replace(":", "_").replace("-", "_")


def safe_name(text):
    base = os.path.splitext(os.path.basename(text))[0]
    base = re.sub(r"[^a-zA-Z0-9_]+", "_", base).strip("_") or "image"
    if base[0].isdigit(): base = "img_" + base
    return base


def ws_pos(node):
    return [float(v) for v in cmds.xform(node, q=True, ws=True, t=True)]


def set_ws_pos(node, pos):
    cmds.xform(node, ws=True, t=[float(pos[0]), float(pos[1]), float(pos[2])])


def ws_rot(node):
    return [float(v) for v in cmds.xform(node, q=True, ws=True, rotation=True)]


def local_trs(node):
    """Return local channel values. Safer for hierarchical joint restore than world xform."""
    data = {}
    for attr in ["translate", "rotate", "scale", "jointOrient"]:
        if cmds.objExists(node + "." + attr):
            try:
                data[attr] = [float(v) for v in cmds.getAttr(node + "." + attr)[0]]
            except Exception:
                pass
    return data


def set_local_trs(node, data):
    """Restore local joint/object channels without fighting parent hierarchy."""
    if not exists(node) or not data:
        return False
    for attr in ["translate", "rotate", "scale", "jointOrient"]:
        vals = data.get(attr)
        if vals is None or not cmds.objExists(node + "." + attr):
            continue
        try:
            for axis, value in zip("XYZ", vals):
                plug = "%s.%s%s" % (node, attr, axis)
                if cmds.objExists(plug):
                    try:
                        if not cmds.getAttr(plug, lock=True):
                            cmds.setAttr(plug, float(value))
                    except Exception:
                        cmds.setAttr(plug, float(value))
        except Exception:
            pass
    return True


def ordered_joint_pose_items(snapshot):
    """Root-to-leaf order avoids children getting double-transformed during reset."""
    def depth(item):
        key, data = item
        node = STATE.joints.get(key) or data.get("node", "")
        if not exists(node):
            return 9999
        parents = cmds.listRelatives(node, allParents=True, fullPath=True) or []
        return len(parents[0].split("|")) if parents else 0
    return sorted(snapshot.items(), key=depth)


def v_add(a, b): return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
def v_sub(a, b): return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
def v_mul(a, s): return [a[0] * s, a[1] * s, a[2] * s]
def v_dot(a, b): return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
def v_len(v): return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
def midpoint(a, b): return [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5]


def normalize(v, fallback=(0, 0, 1)):
    length = v_len(v)
    if length < 1e-8:
        return [float(fallback[0]), float(fallback[1]), float(fallback[2])]
    return [v[0] / length, v[1] / length, v[2] / length]


def v_cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def clamp(value, low, high):
    return max(low, min(high, value))


def rotate_joint_world_to_vector(joint, child, target_parent, target_child, strength=1.0):
    if not (exists(joint) and exists(child)):
        return 0
    joint_pos = ws_pos(joint)
    child_pos = ws_pos(child)
    current = normalize(v_sub(child_pos, joint_pos))
    target = normalize(v_sub(target_child, target_parent))
    dot = clamp(v_dot(current, target), -1.0, 1.0)
    angle = math.degrees(math.acos(dot)) * clamp(float(strength), 0.0, 1.0)
    if angle < 0.05:
        return 0
    axis = v_cross(current, target)
    if v_len(axis) < 1e-8:
        axis = v_cross(current, [0, 1, 0])
        if v_len(axis) < 1e-8:
            axis = v_cross(current, [1, 0, 0])
    axis = normalize(axis)
    try:
        cmds.rotate(
            axis[0] * angle,
            axis[1] * angle,
            axis[2] * angle,
            joint,
            relative=True,
            objectSpace=False,
            worldSpace=True,
            pivot=joint_pos,
        )
        return 1
    except Exception:
        return 0



def rotate_joint_world_line_to_line(joint, left_node, right_node, target_left, target_right, strength=1.0):
    """Rotate a parent joint so a left-right body line matches the source line.

    This fixes the common sideways roll/tilt problem where the limbs are close
    but the torso feels crooked. It aligns the actual Mixamo shoulder/hip line
    with the shoulder/hip line coming from the image pose.
    """
    if not (exists(joint) and exists(left_node) and exists(right_node)):
        return 0
    pivot = ws_pos(joint)
    current = normalize(v_sub(ws_pos(right_node), ws_pos(left_node)), fallback=(1,0,0))
    target = normalize(v_sub(target_right, target_left), fallback=(1,0,0))
    dot = clamp(v_dot(current, target), -1.0, 1.0)
    angle = math.degrees(math.acos(dot)) * clamp(float(strength), 0.0, 1.0)
    if angle < 0.05:
        return 0
    axis = v_cross(current, target)
    if v_len(axis) < 1e-8:
        axis = v_cross(current, [0, 1, 0]) or [0,0,1]
    axis = normalize(axis, fallback=(0,0,1))
    try:
        cmds.rotate(axis[0]*angle, axis[1]*angle, axis[2]*angle, joint,
                    relative=True, objectSpace=False, worldSpace=True, pivot=pivot)
        return 1
    except Exception:
        return 0


def apply_body_line_alignment(targets):
    """Extra torso/hip correction pass after bone retargeting."""
    if not STATE.body_align:
        return 0
    j = STATE.joints
    moved = 0
    s = clamp(float(STATE.body_align_strength), 0.0, 1.0)
    # Hip roll line, then shoulder roll line. These two corrections remove most
    # of the "crooked character" look without touching mesh vertices.
    if all(k in targets for k in ["left_hip", "right_hip"]):
        moved += rotate_joint_world_line_to_line(j.get("hips"), j.get("left_upleg"), j.get("right_upleg"), targets["left_hip"], targets["right_hip"], s)
    if all(k in targets for k in ["left_shoulder", "right_shoulder"]):
        parent = j.get("spine1") or j.get("spine")
        moved += rotate_joint_world_line_to_line(parent, j.get("left_arm"), j.get("right_arm"), targets["left_shoulder"], targets["right_shoulder"], s)
    # Body center direction correction.
    if all(k in targets for k in ["hips", "chest"]):
        moved += rotate_joint_world_to_vector(j.get("hips"), j.get("spine"), targets["hips"], targets["chest"], s)
    return moved

def enable_textured_viewports():
    for panel in cmds.getPanel(type="modelPanel") or []:
        try:
            cmds.modelEditor(panel, e=True, displayTextures=True)
            cmds.modelEditor(panel, e=True, displayAppearance="smoothShaded")
        except Exception:
            pass


def suspend_refresh(func):
    try:
        cmds.refresh(suspend=True)
        return func()
    finally:
        try:
            cmds.refresh(suspend=False)
            cmds.refresh(force=True)
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Character materials
# -----------------------------------------------------------------------------
def texture_search_dirs():
    roots = []
    for root in PROJECT_ROOTS + [TOOL_ROOT]:
        if root and root not in roots:
            roots.append(root)
        for sub in ["textures", "Textures", "materials", "Materials", "maps", "Maps", "resources", "Resources", "assets", "Assets"]:
            p = os.path.join(root, sub)
            if p not in roots:
                roots.append(p)
    if STATE.character_path:
        char_dir = os.path.dirname(STATE.character_path)
        for p in [char_dir, os.path.join(char_dir, "textures"), os.path.join(char_dir, "Textures"), os.path.dirname(char_dir)]:
            if p and p not in roots:
                roots.append(p)
    return [p for p in roots if p and os.path.isdir(p)]


def find_texture_file(kind):
    names = {
        "diffuse": ["Ch24_1001_Diffuse.png", "diffuse", "albedo", "basecolor", "base_color", "color"],
        "specular": ["Ch24_1001_Specular.png", "specular", "spec"],
        "glossiness": ["Ch24_1001_Glossiness.png", "glossiness", "gloss", "roughness"],
        "normal": ["Ch24_1001_Normal.png", "normal", "nrm"],
    }.get(kind, [])
    # Exact names first.
    for d in texture_search_dirs():
        for n in names:
            if n.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                p = os.path.join(d, n)
                if os.path.exists(p):
                    return p
    # Recursive fuzzy search.
    for d in texture_search_dirs():
        try:
            for dirpath, _, files in os.walk(d):
                for f in files:
                    low = f.lower()
                    if not low.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                        continue
                    if any(token.lower() in low for token in names):
                        return os.path.join(dirpath, f)
        except Exception:
            pass
    return ""


def ninja_meshes():
    meshes = []
    for shape in cmds.ls(type="mesh", long=True) or []:
        if shape.startswith("|AI_") or "AI_Mixamo" in shape:
            continue
        if STATE.namespace and (STATE.namespace + ":") not in shape:
            continue
        parent = (cmds.listRelatives(shape, parent=True, fullPath=True) or [""])[0]
        if parent:
            meshes.append(parent)
    # Fallback: all non-tool meshes.
    if not meshes:
        for shape in cmds.ls(type="mesh", long=True) or []:
            if "AI_Mixamo" in shape or "AI_Character" in shape:
                continue
            parent = (cmds.listRelatives(shape, parent=True, fullPath=True) or [""])[0]
            if parent:
                meshes.append(parent)
    return sorted(set(meshes))


def connect_if_attr(src, dst):
    try:
        if cmds.objExists(dst):
            cmds.connectAttr(src, dst, force=True)
            return True
    except Exception:
        pass
    return False


def create_file_node(name, path, raw=False):
    safe_delete(name)
    node = cmds.shadingNode("file", asTexture=True, name=name)
    cmds.setAttr(node + ".fileTextureName", path, type="string")
    if raw:
        try: cmds.setAttr(node + ".colorSpace", "Raw", type="string")
        except Exception: pass
    return node


def apply_ninja_materials():
    """Auto-assign the uploaded Character texture set to the imported character."""
    meshes = ninja_meshes()
    if not meshes:
        raise RuntimeError("No Character meshes found to assign materials.")
    diffuse = find_texture_file("diffuse")
    specular = find_texture_file("specular")
    normal = find_texture_file("normal")
    gloss = find_texture_file("glossiness")

    if not diffuse:
        log("Diffuse texture not found. Put Ch24_1001_Diffuse.png beside the script or in a textures folder.", warning=True)

    safe_delete(NINJA_MATERIAL_NAME)
    safe_delete(NINJA_MATERIAL_SG)
    try:
        mat = cmds.shadingNode("aiStandardSurface", asShader=True, name=NINJA_MATERIAL_NAME)
        color_attr = mat + ".baseColor"
        spec_attr = mat + ".specularColor"
        rough_attr = mat + ".specularRoughness"
        normal_attr = mat + ".normalCamera"
    except Exception:
        mat = cmds.shadingNode("phong", asShader=True, name=NINJA_MATERIAL_NAME)
        color_attr = mat + ".color"
        spec_attr = mat + ".specularColor"
        rough_attr = mat + ".cosinePower"
        normal_attr = mat + ".normalCamera"

    sg = cmds.sets(renderable=True, noSurfaceShader=True, empty=True, name=NINJA_MATERIAL_SG)
    cmds.connectAttr(mat + ".outColor", sg + ".surfaceShader", force=True)

    if diffuse:
        tex = create_file_node("AI_Character_Diffuse_FILE", diffuse, raw=False)
        connect_if_attr(tex + ".outColor", color_attr)
    else:
        try: cmds.setAttr(color_attr, 0.05, 0.045, 0.13, type="double3")
        except Exception: pass

    if specular:
        spec = create_file_node("AI_Character_Specular_FILE", specular, raw=True)
        connect_if_attr(spec + ".outColor", spec_attr)
    if gloss:
        gl = create_file_node("AI_Character_Glossiness_FILE", gloss, raw=True)
        # For Arnold this is not mathematically inverted, but gives visible texture response.
        connect_if_attr(gl + ".outAlpha", rough_attr)
    if normal:
        ntex = create_file_node("AI_Character_Normal_FILE", normal, raw=True)
        bump = "AI_Character_Normal_BUMP"
        safe_delete(bump)
        bump = cmds.shadingNode("bump2d", asUtility=True, name=bump)
        try: cmds.setAttr(bump + ".bumpInterp", 1)
        except Exception: pass
        try: cmds.setAttr(bump + ".bumpDepth", 0.18)
        except Exception: pass
        connect_if_attr(ntex + ".outAlpha", bump + ".bumpValue")
        connect_if_attr(bump + ".outNormal", normal_attr)

    # Assign to both transforms and mesh shapes. Some imported FBX files ignore
    # assignment on parent transforms, so assigning shapes directly is safer.
    assigned = 0
    for m in meshes:
        try:
            cmds.sets(m, e=True, forceElement=sg)
            assigned += 1
        except Exception:
            pass
        for shape in cmds.listRelatives(m, shapes=True, fullPath=True) or []:
            try:
                cmds.sets(shape, e=True, forceElement=sg)
                assigned += 1
            except Exception:
                pass
    try:
        cmds.setAttr(mat + ".diffuse", 1.0)
    except Exception:
        pass
    try:
        cmds.setAttr(mat + ".specular", 0.35)
    except Exception:
        pass
    enable_textured_viewports()
    log("Applied Character materials to %d mesh assignments. Diffuse=%s" % (assigned, diffuse or "missing"))
    set_action("Character material applied automatically.")
    return sg

# -----------------------------------------------------------------------------
# Character file discovery / import
# -----------------------------------------------------------------------------
def auto_find_ninja_fbx():
    names = []
    for root in PROJECT_ROOTS:
        for sub in ["", "characters", "Resources", "resources", "assets", "models"]:
            base = os.path.join(root, sub)
            if not os.path.isdir(base):
                continue
            try:
                for dirpath, _, files in os.walk(base):
                    for f in files:
                        if f.lower().endswith(".fbx"):
                            path = os.path.join(dirpath, f)
                            score = 0
                            low = f.lower()
                            if "ninja" in low: score += 10
                            if "mixamo" in low: score += 5
                            names.append((score, path))
            except Exception:
                pass
    if not names:
        return ""
    names.sort(key=lambda x: (-x[0], x[1].lower()))
    return names[0][1]


def browse_image():
    result = cmds.fileDialog2(fileMode=1, caption="Choose Reference Pose Image", fileFilter="Images (*.jpg *.jpeg *.png *.bmp *.webp)")
    if not result:
        return
    STATE.image_path = result[0]
    STATE.initial_image_pose = {}
    STATE.initial_pose_image_path = ""
    set_action("Selected image: %s" % STATE.image_path)


def browse_character():
    result = cmds.fileDialog2(fileMode=1, caption="Choose Mixamo Character FBX", fileFilter="Mixamo Character (*.fbx);;Maya Scene (*.ma *.mb)")
    if not result:
        return
    STATE.character_path = result[0]
    STATE.joints = {}
    STATE.rest = {}
    STATE.ik = {}
    STATE.initial_image_pose = {}
    STATE.initial_pose_image_path = ""
    set_action("Selected character: %s" % STATE.character_path)


def load_auto_ninja_path():
    path = auto_find_ninja_fbx()
    if not path:
        raise RuntimeError("No FBX found in project folders. Browse the character manually.")
    STATE.character_path = path
    set_action("Auto-found character: %s" % path)


def unique_namespace(base):
    if not cmds.namespace(exists=base):
        return base
    i = 1
    while cmds.namespace(exists="%s%d" % (base, i)):
        i += 1
    return "%s%d" % (base, i)


def import_character():
    if not STATE.character_path:
        auto_path = auto_find_ninja_fbx()
        if auto_path:
            STATE.character_path = auto_path
    if not STATE.character_path or not os.path.exists(STATE.character_path):
        raise RuntimeError("Choose a valid rigged Mixamo FBX character first.")
    ext = os.path.splitext(STATE.character_path)[1].lower()
    ns = unique_namespace(IMPORT_NAMESPACE_BASE)
    STATE.namespace = ns
    if ext == ".fbx":
        try:
            cmds.loadPlugin("fbxmaya", quiet=True)
        except Exception:
            pass
        before = set(cmds.ls(long=True) or [])
        cmds.file(STATE.character_path, i=True, type="FBX", ignoreVersion=True, mergeNamespacesOnClash=False, namespace=ns, options="fbx")
        after = set(cmds.ls(long=True) or [])
        imported = list(after - before)
    else:
        imported = cmds.file(STATE.character_path, i=True, ignoreVersion=True, mergeNamespacesOnClash=False, namespace=ns, preserveReferences=False, returnNewNodes=True) or []
    ensure_tool_groups()
    log("Imported Mixamo Character character into namespace: %s" % ns)
    scan_mixamo_skeleton()
    capture_rest_pose()
    if STATE.auto_materials:
        try:
            apply_ninja_materials()
        except Exception as exc:
            log("Auto material setup failed: %s" % exc, warning=True)
    create_ik_solver()
    return imported

# -----------------------------------------------------------------------------
# Skeleton scanning / calibration
# -----------------------------------------------------------------------------
def find_joint_alias(aliases):
    candidates = cmds.ls(type="joint", long=True) or []
    # Prefer the currently imported character namespace if known, so old rigs in scene do not confuse scanning.
    if STATE.namespace:
        ns_candidates = [n for n in candidates if (":" + STATE.namespace + ":" in n) or (leaf(n).startswith(STATE.namespace + ":"))]
        if ns_candidates:
            candidates = ns_candidates
    exact = set(clean_name(a.split(":")[-1]) for a in aliases)
    for node in candidates:
        if clean_name(leaf_no_ns(node)) in exact:
            return node
    for node in candidates:
        name = clean_name(leaf_no_ns(node))
        for alias in aliases:
            alias_clean = clean_name(alias.split(":")[-1])
            if alias_clean and alias_clean in name:
                return node
    return ""


def scan_mixamo_skeleton():
    joints = {}
    for key, aliases in MIXAMO_ALIASES.items():
        joints[key] = find_joint_alias(aliases)
    STATE.joints = joints
    missing_required = [k for k in REQUIRED_MIXAMO if not joints.get(k)]
    if missing_required:
        log("Mixamo scan finished, but missing required joints: %s" % ", ".join(missing_required), warning=True)
    else:
        log("Mixamo skeleton scan OK. Required joints found.")
    refresh_ui()
    return joints


def capture_rest_pose():
    if not STATE.joints:
        scan_mixamo_skeleton()
    rest = {}
    for key, node in STATE.joints.items():
        if exists(node):
            try:
                pos = ws_pos(node)
                rot = ws_rot(node)
                rest[key] = {
                    "node": node,
                    "local": local_trs(node),
                    "translate": pos,
                    "rotate": rot,
                    "pos": pos,
                    "rot": rot,
                    "parent": (cmds.listRelatives(node, parent=True, fullPath=True) or [""])[0],
                }
            except Exception:
                pass
    STATE.rest = rest
    if rest:
        log("Captured Mixamo rest pose for %d joints." % len(rest))
    else:
        log("No joints captured. Is this a real Mixamo rig?", warning=True)
    return rest


def validate_mixamo_ready():
    if not STATE.joints or not exists(STATE.joints.get("hips")):
        scan_mixamo_skeleton()
    missing = [k for k in REQUIRED_MIXAMO if not exists(STATE.joints.get(k))]
    if missing:
        # One more full scene rescan in case the FBX imported under a different namespace.
        old_ns = STATE.namespace
        STATE.namespace = ""
        scan_mixamo_skeleton()
        STATE.namespace = old_ns
        missing = [k for k in REQUIRED_MIXAMO if not exists(STATE.joints.get(k))]
    if missing:
        raise RuntimeError("Required Mixamo joints missing: %s. Make sure the character is a Mixamo FBX with Skin." % ", ".join(missing))
    if not STATE.rest:
        capture_rest_pose()

# -----------------------------------------------------------------------------
# Pose extraction / target conversion
# -----------------------------------------------------------------------------
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
        raise RuntimeError("Invalid pose.json schema. Expected data['landmarks'] as dict.")
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
    vals = [landmark_confidence(landmarks[n]) for n in REQUIRED_LANDMARKS]
    STATE.confidence = float(data.get("average_visibility", sum(vals) / max(len(vals), 1)))
    STATE.quality = "Good" if STATE.confidence >= 0.75 else "Warning"
    STATE.pose_status = "Pose ready"


def normalized_point(lm, aspect=1.0):
    return [(float(lm.get("x", 0.5)) - 0.5) * aspect, 0.5 - float(lm.get("y", 0.5)), 0.0]


def world_point(lm):
    return [float(lm.get("x", 0.0)), -float(lm.get("y", 0.0)), -float(lm.get("z", 0.0))]


def mirror_landmark_points(points):
    p = json.loads(json.dumps(points))
    pairs = [
        ("left_shoulder", "right_shoulder"), ("left_elbow", "right_elbow"), ("left_wrist", "right_wrist"),
        ("left_hip", "right_hip"), ("left_knee", "right_knee"), ("left_ankle", "right_ankle"),
        ("left_foot_index", "right_foot_index"),
    ]
    for a, b in pairs:
        if a in p and b in p:
            p[a], p[b] = p[b], p[a]
    for k, v in list(p.items()):
        if isinstance(v, list) and len(v) >= 3:
            v[0] = -v[0]
    return p


def build_pose_points(data):
    aspect = float(data.get("width", 1.0) or 1.0) / max(float(data.get("height", 1.0) or 1.0), 1e-6)
    landmarks = get_pose_landmarks(data)
    pts = {name: normalized_point(landmarks[name], aspect=aspect) for name in REQUIRED_LANDMARKS}
    world_landmarks = data.get("world_landmarks") if isinstance(data.get("world_landmarks"), dict) else {}
    if world_landmarks and "left_hip" in world_landmarks and "right_hip" in world_landmarks:
        try:
            world_points = {n: world_point(world_landmarks[n]) for n in REQUIRED_LANDMARKS if n in world_landmarks}
            hip_z = midpoint(world_points["left_hip"], world_points["right_hip"])[2]
            for n in REQUIRED_LANDMARKS:
                if n in world_points:
                    pts[n][2] = (world_points[n][2] - hip_z) * STATE.depth_scale
        except Exception:
            pass
    if STATE.mirror:
        pts = mirror_landmark_points(pts)
    hip_center = midpoint(pts["left_hip"], pts["right_hip"])
    shoulder_center = midpoint(pts["left_shoulder"], pts["right_shoulder"])
    foot_center = midpoint(pts["left_ankle"], pts["right_ankle"])
    src_height = max(v_len(v_sub(pts["nose"], foot_center)), 0.001)
    scale = 10.0 / src_height
    root = [0.0, 4.8, 0.0]
    out = {}
    for n in REQUIRED_LANDMARKS:
        out[n] = v_add(root, v_mul(v_sub(pts[n], hip_center), scale))
    out["hip_center"] = root
    out["shoulder_center"] = v_add(root, v_mul(v_sub(shoulder_center, hip_center), scale))
    out["spine_mid"] = midpoint(out["hip_center"], out["shoulder_center"])
    out["neck"] = [
        out["shoulder_center"][0] * 0.72 + out["nose"][0] * 0.28,
        out["shoulder_center"][1] * 0.72 + out["nose"][1] * 0.28,
        out["shoulder_center"][2] * 0.72 + out["nose"][2] * 0.28,
    ]
    out["head"] = out["nose"]
    out["foot_center"] = midpoint(out["left_ankle"], out["right_ankle"])
    return out


def extract_pose_json():
    if not STATE.image_path:
        raise RuntimeError("Choose an image first.")
    if not os.path.exists(STATE.image_path):
        raise RuntimeError("Selected image does not exist: %s" % STATE.image_path)
    if not os.path.exists(EXTERNAL_PYTHON):
        raise RuntimeError("External Python not found: %s" % EXTERNAL_PYTHON)
    if not os.path.exists(EXTRACT_SCRIPT):
        raise RuntimeError("extract_pose_image.py not found: %s" % EXTRACT_SCRIPT)
    cmd = [EXTERNAL_PYTHON, EXTRACT_SCRIPT, "--image", STATE.image_path, "--out", STATE.pose_json_path]
    log("Running pose extractor...")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        STATE.pose_status = "Failed"
        raise RuntimeError("Pose extraction failed. See Script Editor output.")
    update_pose_status_from_file()
    STATE.pose_points = build_pose_points(load_pose_json(STATE.pose_json_path))
    set_action("Extracted image pose JSON.")


def rig_rest_feet_center():
    r = STATE.rest
    if "left_foot" in r and "right_foot" in r:
        return midpoint(r["left_foot"]["pos"], r["right_foot"]["pos"])
    return [0.0, 0.0, 0.0]


def pose_feet_center(pts):
    return midpoint(pts["left_ankle"], pts["right_ankle"])


def scale_from_pose_to_rig(pts):
    r = STATE.rest
    values = []
    rig_feet = rig_rest_feet_center()
    if "head" in r:
        rig_h = max(r["head"]["pos"][1] - rig_feet[1], 0.1)
        pose_h = max(pts["head"][1] - pose_feet_center(pts)[1], 0.1)
        values.append(rig_h / pose_h)
    if "left_hand" in r and "right_hand" in r:
        rig_span = v_len(v_sub(r["left_hand"]["pos"], r["right_hand"]["pos"]))
        pose_span = max(v_len(v_sub(pts["left_wrist"], pts["right_wrist"])), 0.1)
        values.append(rig_span / pose_span)
    scale = sum(values) / len(values) if values else 1.0
    return max(0.05, min(scale * STATE.pose_scale, 100.0))


def pose_to_rig(point, pts, anchor, scale):
    rel = v_sub(point, pose_feet_center(pts))
    rel[2] *= STATE.depth_scale
    return v_add(anchor, v_mul(rel, scale))


def compute_targets():
    if not STATE.pose_points:
        raise RuntimeError("No image pose points. Press Extract Pose first.")
    pts = STATE.pose_points
    anchor = rig_rest_feet_center()
    scale = scale_from_pose_to_rig(pts)
    targets = {
        "hips": pose_to_rig(pts["hip_center"], pts, anchor, scale),
        "chest": pose_to_rig(pts["shoulder_center"], pts, anchor, scale),
        "head": pose_to_rig(pts["head"], pts, anchor, scale),
        "left_shoulder": pose_to_rig(pts["left_shoulder"], pts, anchor, scale),
        "right_shoulder": pose_to_rig(pts["right_shoulder"], pts, anchor, scale),
        "left_elbow": pose_to_rig(pts["left_elbow"], pts, anchor, scale),
        "right_elbow": pose_to_rig(pts["right_elbow"], pts, anchor, scale),
        "left_hand": pose_to_rig(pts["left_wrist"], pts, anchor, scale),
        "right_hand": pose_to_rig(pts["right_wrist"], pts, anchor, scale),
        "left_hip": pose_to_rig(pts["left_hip"], pts, anchor, scale),
        "right_hip": pose_to_rig(pts["right_hip"], pts, anchor, scale),
        "left_knee": pose_to_rig(pts["left_knee"], pts, anchor, scale),
        "right_knee": pose_to_rig(pts["right_knee"], pts, anchor, scale),
        "left_foot": pose_to_rig(pts["left_ankle"], pts, anchor, scale),
        "right_foot": pose_to_rig(pts["right_ankle"], pts, anchor, scale),
        "left_toe": pose_to_rig(pts["left_foot_index"], pts, anchor, scale),
        "right_toe": pose_to_rig(pts["right_foot_index"], pts, anchor, scale),
    }
    if STATE.keep_feet_grounded:
        if "left_foot" in STATE.rest:
            targets["left_foot"][1] = STATE.rest["left_foot"]["pos"][1]
        if "right_foot" in STATE.rest:
            targets["right_foot"][1] = STATE.rest["right_foot"]["pos"][1]
    STATE.targets = targets
    return targets, scale

# -----------------------------------------------------------------------------
# Visual target guide and background
# -----------------------------------------------------------------------------
def ensure_tool_groups():
    if not exists(TOOL_GRP): cmds.group(empty=True, name=TOOL_GRP)
    for grp in [SOURCE_GRP, TARGET_GRP, IK_GRP, CTRL_GRP]:
        if not exists(grp):
            cmds.group(empty=True, name=grp)
            try: cmds.parent(grp, TOOL_GRP)
            except Exception: pass


def make_curve(name, a, b, parent=None):
    if exists(name): safe_delete(name)
    crv = cmds.curve(name=name, d=1, p=[a, b])
    if parent and exists(parent):
        try: cmds.parent(crv, parent)
        except Exception: pass
    return crv


def build_source_guide():
    targets, scale = compute_targets()
    ensure_tool_groups()
    safe_delete(SOURCE_GRP)
    safe_delete(TARGET_GRP)
    cmds.group(empty=True, name=SOURCE_GRP)
    cmds.group(empty=True, name=TARGET_GRP)
    try:
        cmds.parent(SOURCE_GRP, TOOL_GRP)
        cmds.parent(TARGET_GRP, TOOL_GRP)
    except Exception:
        pass
    for name, pos in targets.items():
        loc = cmds.spaceLocator(name="AI_Character_SRC_%s_LOC" % name)[0]
        set_ws_pos(loc, pos)
        for ax in "XYZ": cmds.setAttr(loc + ".localScale" + ax, max(scale * 0.04, 0.12))
        try: cmds.parent(loc, TARGET_GRP)
        except Exception: pass
    pairs = [
        ("left_shoulder", "left_elbow"), ("left_elbow", "left_hand"),
        ("right_shoulder", "right_elbow"), ("right_elbow", "right_hand"),
        ("left_hip", "left_knee"), ("left_knee", "left_foot"), ("left_foot", "left_toe"),
        ("right_hip", "right_knee"), ("right_knee", "right_foot"), ("right_foot", "right_toe"),
        ("hips", "chest"), ("chest", "head"), ("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
    ]
    for i, (a, b) in enumerate(pairs):
        if a in targets and b in targets:
            make_curve("AI_Character_Source_%02d_CRV" % i, targets[a], targets[b], SOURCE_GRP)
    try:
        cmds.setAttr(SOURCE_GRP + ".visibility", 1 if STATE.show_source else 0)
        cmds.setAttr(TARGET_GRP + ".visibility", 1 if STATE.show_source else 0)
    except Exception:
        pass
    log("Built AI source guide. scale=%.3f" % scale)


def make_image_material(image_path):
    mat = "AI_Character_%s_Image_MAT" % safe_name(image_path)
    sg = mat + "SG"
    file_node = mat + "_file"
    for node in [mat, sg, file_node]:
        safe_delete(node)
    mat = cmds.shadingNode("lambert", asShader=True, name=mat)
    file_node = cmds.shadingNode("file", asTexture=True, name=file_node)
    cmds.setAttr(file_node + ".fileTextureName", image_path, type="string")
    cmds.connectAttr(file_node + ".outColor", mat + ".color", force=True)
    sg = cmds.sets(renderable=True, noSurfaceShader=True, empty=True, name=sg)
    cmds.connectAttr(mat + ".outColor", sg + ".surfaceShader", force=True)
    return sg


def create_image_background():
    # v22: Do not create the selected image in the viewport.
    # The image is used only for pose extraction.
    return

# -----------------------------------------------------------------------------
# IK controls / solver

# -----------------------------------------------------------------------------
# IK controls / solver
# -----------------------------------------------------------------------------
def color_shape(node, color_index):
    for shape in cmds.listRelatives(node, shapes=True, fullPath=True) or []:
        try:
            cmds.setAttr(shape + ".overrideEnabled", 1)
            cmds.setAttr(shape + ".overrideColor", color_index)
        except Exception:
            pass


def make_locator_control(name, pos, size=0.35, color=17):
    if exists(name): safe_delete(name)
    loc = cmds.spaceLocator(name=name)[0]
    set_ws_pos(loc, pos)
    for ax in "XYZ": cmds.setAttr(loc + ".localScale" + ax, size)
    color_shape(loc, color)
    try: cmds.parent(loc, CTRL_GRP)
    except Exception: pass
    return loc


def create_ik_handle(name, start_joint, end_joint):
    if exists(name): safe_delete(name)
    try:
        handle, effector = cmds.ikHandle(name=name, sj=start_joint, ee=end_joint, solver="ikRPsolver")
    except Exception:
        handle, effector = cmds.ikHandle(name=name, sj=start_joint, ee=end_joint, solver="ikSCsolver")
    try:
        cmds.parent(handle, IK_GRP)
        cmds.setAttr(handle + ".visibility", 0)
    except Exception:
        pass
    return handle


def create_ik_solver():
    validate_mixamo_ready()
    ensure_tool_groups()
    safe_delete(IK_GRP)
    safe_delete(CTRL_GRP)
    cmds.group(empty=True, name=IK_GRP)
    cmds.group(empty=True, name=CTRL_GRP)
    try:
        cmds.parent(IK_GRP, TOOL_GRP)
        cmds.parent(CTRL_GRP, TOOL_GRP)
    except Exception:
        pass
    j = STATE.joints
    r = STATE.rest
    ik = {}
    specs = [
        ("left_arm", "left_arm", "left_hand", "left_hand", "left_forearm", 6, "AI_Character_LeftHand_IK", "AI_Character_LeftElbow_PV"),
        ("right_arm", "right_arm", "right_hand", "right_hand", "right_forearm", 13, "AI_Character_RightHand_IK", "AI_Character_RightElbow_PV"),
        ("left_leg", "left_upleg", "left_foot", "left_foot", "left_leg", 6, "AI_Character_LeftFoot_IK", "AI_Character_LeftKnee_PV"),
        ("right_leg", "right_upleg", "right_foot", "right_foot", "right_leg", 13, "AI_Character_RightFoot_IK", "AI_Character_RightKnee_PV"),
    ]
    for sem, start_key, end_key, ctrl_rest, pv_rest, color, ik_name, pv_name in specs:
        handle = create_ik_handle(ik_name + "_HANDLE", j[start_key], j[end_key])
        ctrl = make_locator_control(ik_name, r[ctrl_rest]["pos"], 0.45, color)
        pv = make_locator_control(pv_name, r[pv_rest]["pos"], 0.40, 18)
        try: cmds.pointConstraint(ctrl, handle, mo=False)
        except Exception as exc: log("Point constraint failed for %s: %s" % (sem, exc), warning=True)
        try: cmds.poleVectorConstraint(pv, handle)
        except Exception as exc: log("Pole vector failed for %s: %s" % (sem, exc), warning=True)
        ik[sem] = {"handle": handle, "ctrl": ctrl, "pv": pv}
    STATE.ik = ik
    log("Created Mixamo IK solver: %s" % ", ".join(sorted(ik)))


def pv_from_chain(a, b, c, fallback=(0, 0, 1), scale=1.0, factor=1.4):
    ac = v_sub(c, a)
    ab = v_sub(b, a)
    ac_len2 = max(v_dot(ac, ac), 1e-8)
    proj = v_add(a, v_mul(ac, v_dot(ab, ac) / ac_len2))
    bend = v_sub(b, proj)
    return v_add(b, v_mul(normalize(bend, fallback), max(v_len(ac), scale) * factor))



def apply_strong_bone_retarget(targets):
    """Rotate existing Mixamo joints so bone directions match the image source pose."""
    j = STATE.joints
    moved = 0
    strength = clamp(float(STATE.match_strength), 0.0, 1.0)

    if STATE.follow_root and exists(j.get("hips")) and targets.get("hips"):
        try:
            set_ws_pos(j["hips"], targets["hips"])
            moved += 1
        except Exception:
            pass

    chain_pairs = [
        ("hips", "spine", "hips", "chest"),
        ("spine", "spine1", "hips", "chest"),
        ("spine1", "neck", "hips", "head"),
        ("neck", "head", "chest", "head"),

        ("left_arm", "left_forearm", "left_shoulder", "left_elbow"),
        ("left_forearm", "left_hand", "left_elbow", "left_hand"),
        ("right_arm", "right_forearm", "right_shoulder", "right_elbow"),
        ("right_forearm", "right_hand", "right_elbow", "right_hand"),

        ("left_upleg", "left_leg", "left_hip", "left_knee"),
        ("left_leg", "left_foot", "left_knee", "left_foot"),
        ("left_foot", "left_toe", "left_foot", "left_toe"),
        ("right_upleg", "right_leg", "right_hip", "right_knee"),
        ("right_leg", "right_foot", "right_knee", "right_foot"),
        ("right_foot", "right_toe", "right_foot", "right_toe"),
    ]

    for _pass in range(5):
        moved += apply_body_line_alignment(targets)
        for joint_key, child_key, target_a, target_b in chain_pairs:
            if target_a in targets and target_b in targets:
                moved += rotate_joint_world_to_vector(
                    j.get(joint_key),
                    j.get(child_key),
                    targets[target_a],
                    targets[target_b],
                    strength=strength,
                )
    moved += apply_body_line_alignment(targets)
    return moved



def final_pose_polish(targets):
    """Final rotation cleanup after IK targets are placed.

    This is designed for the common "crooked body" result: limbs reach roughly
    the right area, but torso/head/shoulders are tilted differently from the
    image. It applies small safe rotations on the existing Mixamo skeleton only.
    """
    if not STATE.body_align:
        return 0
    j = STATE.joints
    moved = 0
    s = clamp(float(STATE.body_align_strength), 0.0, 1.0)
    # Strong torso lines first.
    for _ in range(2):
        moved += apply_body_line_alignment(targets)
    # Head and neck line. These are small but visually important.
    if all(k in targets for k in ["chest", "head"]):
        moved += rotate_joint_world_to_vector(j.get("neck"), j.get("head"), targets["chest"], targets["head"], s)
    # Re-aim limb bones after IK handles move endpoints. Lower strength avoids violent flips.
    mini = [
        ("left_arm", "left_forearm", "left_shoulder", "left_elbow"),
        ("left_forearm", "left_hand", "left_elbow", "left_hand"),
        ("right_arm", "right_forearm", "right_shoulder", "right_elbow"),
        ("right_forearm", "right_hand", "right_elbow", "right_hand"),
        ("left_upleg", "left_leg", "left_hip", "left_knee"),
        ("left_leg", "left_foot", "left_knee", "left_foot"),
        ("right_upleg", "right_leg", "right_hip", "right_knee"),
        ("right_leg", "right_foot", "right_knee", "right_foot"),
    ]
    for joint_key, child_key, target_a, target_b in mini:
        if target_a in targets and target_b in targets:
            moved += rotate_joint_world_to_vector(
                j.get(joint_key), j.get(child_key), targets[target_a], targets[target_b], strength=s * 0.45
            )
    return moved

def apply_pose_to_character():
    validate_mixamo_ready()
    if not STATE.ik:
        create_ik_solver()
    targets, scale = compute_targets()
    build_source_guide()
    ik = STATE.ik
    j = STATE.joints
    moved = 0

    if STATE.strong_retarget:
        moved += apply_strong_bone_retarget(targets)
    else:
        # Move root first. This keeps limbs near the image action while feet IK can still stabilize the pose.
        if STATE.follow_root and exists(j.get("hips")):
            try:
                set_ws_pos(j["hips"], targets["hips"])
                moved += 1
            except Exception as exc:
                log("Could not move hips: %s" % exc, warning=True)

    def set_ctrl(sem, pos):
        nonlocal moved
        if sem in ik and exists(ik[sem]["ctrl"]):
            set_ws_pos(ik[sem]["ctrl"], pos)
            moved += 1

    def set_pv(sem, pos):
        nonlocal moved
        if sem in ik and exists(ik[sem]["pv"]):
            set_ws_pos(ik[sem]["pv"], pos)
            moved += 1

    set_ctrl("left_leg", targets["left_foot"])
    set_ctrl("right_leg", targets["right_foot"])
    set_pv("left_leg", pv_from_chain(targets["left_hip"], targets["left_knee"], targets["left_foot"], fallback=(0, 0, 1), scale=scale, factor=1.25))
    set_pv("right_leg", pv_from_chain(targets["right_hip"], targets["right_knee"], targets["right_foot"], fallback=(0, 0, 1), scale=scale, factor=1.25))
    set_ctrl("left_arm", targets["left_hand"])
    set_ctrl("right_arm", targets["right_hand"])
    set_pv("left_arm", pv_from_chain(targets["left_shoulder"], targets["left_elbow"], targets["left_hand"], fallback=(0, 0, 1), scale=scale, factor=1.15))
    set_pv("right_arm", pv_from_chain(targets["right_shoulder"], targets["right_elbow"], targets["right_hand"], fallback=(0, 0, 1), scale=scale, factor=1.15))

    moved += final_pose_polish(targets)
    moved += final_pose_polish(targets)

    try:
        cmds.refresh(force=True)
    except Exception:
        pass
    set_action("Applied image pose to character with strong retarget. Operations=%d | scale=%.3f" % (moved, scale))



def capture_current_character_pose():
    """Capture current local + world transforms for every scanned character joint.

    v26 stores local channels too. Restoring local channels is much safer than
    setting child joints in world space inside an already-rotated hierarchy.
    """
    if not STATE.joints:
        scan_mixamo_skeleton()
    pose = {}
    for key, node in STATE.joints.items():
        if exists(node):
            try:
                pos = ws_pos(node)
                rot = ws_rot(node)
                pose[key] = {
                    "node": node,
                    "local": local_trs(node),
                    "translate": pos,
                    "rotate": rot,
                    "pos": pos,
                    "rot": rot,
                    "parent": (cmds.listRelatives(node, parent=True, fullPath=True) or [""])[0],
                }
            except Exception:
                pass
    return pose


def restore_character_pose_snapshot(snapshot, label="pose"):
    if not snapshot:
        return False
    if not STATE.joints:
        scan_mixamo_skeleton()
    ensure_rest_pose_has_world_aliases()
    # Remove IK constraints while restoring joints. Recreate after restore.
    safe_delete(IK_GRP)
    safe_delete(CTRL_GRP)
    STATE.ik = {}
    for key, data in ordered_joint_pose_items(snapshot):
        node = STATE.joints.get(key) or data.get("node")
        if not exists(node):
            continue
        try:
            if data.get("local"):
                set_local_trs(node, data.get("local"))
            else:
                # Backward-compatible fallback for older JSON/snapshots.
                cmds.xform(node, ws=True, t=data.get("translate", data.get("pos", [0, 0, 0])))
                cmds.xform(node, ws=True, rotation=data.get("rotate", data.get("rot", [0, 0, 0])))
        except Exception as exc:
            log("Could not restore %s: %s" % (key, exc), warning=True)
    STATE.ik = {}
    create_ik_solver()
    try:
        cmds.refresh(force=True)
    except Exception:
        pass
    set_action("Reset character to %s." % label)
    return True


def capture_initial_image_pose_once(force=False):
    """Store the first generated pose for the current image.

    Reset uses this snapshot. It is captured once per Browse Image so manual
    changes after generation do not overwrite the reset target.
    """
    if not STATE.image_path:
        return
    if (not force) and STATE.initial_image_pose and STATE.initial_pose_image_path == STATE.image_path:
        return
    STATE.initial_image_pose = capture_current_character_pose()
    STATE.initial_pose_image_path = STATE.image_path
    log("Captured reset pose from the first generated image pose.")


def current_character_meshes():
    """Return visible character mesh transforms robustly.

    The FBX can import meshes as Ch24/Body/Character while joints are in a mixamo namespace.
    This function intentionally does not require a namespace match.
    """
    bad_tokens = ["ai_mixamo", "ai_imagepose", "ai_hik", "source_guide", "imageplane", "reference_image", "copy_"]
    out = []
    for shape in cmds.ls(type="mesh", long=True) or []:
        try:
            if cmds.getAttr(shape + ".intermediateObject"):
                continue
        except Exception:
            pass
        parent = (cmds.listRelatives(shape, parent=True, fullPath=True) or [""])[0]
        if not parent or not exists(parent):
            continue
        low = parent.lower()
        if any(tok in low for tok in bad_tokens):
            continue
        if leaf_no_ns(parent).lower().startswith("ai_"):
            continue
        out.append(parent)
    # Prefer meshes that are skinned to the current character skeleton, but fall back to all non-tool meshes.
    skinned = []
    character_joint_set = set([j for j in STATE.joints.values() if exists(j)])
    for mesh in out:
        try:
            history = cmds.listHistory(mesh, pruneDagObjects=True) or []
            skins = cmds.ls(history, type="skinCluster") or []
            if not skins:
                continue
            infl = set(cmds.skinCluster(skins[0], q=True, influence=True) or [])
            if infl & character_joint_set:
                skinned.append(mesh)
        except Exception:
            pass
    final = skinned or out
    seen, clean = set(), []
    for node in final:
        if node not in seen and exists(node):
            seen.add(node)
            clean.append(node)
    return clean


def copy_current_pose_snapshot():
    """Create a selectable copy of the character in its current visible pose."""
    meshes = current_character_meshes()
    if not meshes and STATE.character_path:
        # If the user presses Copy before Generate imported the character, generate once first.
        try:
            generate_all()
            meshes = current_character_meshes()
        except Exception:
            meshes = current_character_meshes()
    if not meshes:
        # Last resort: any visible mesh that is not an AI helper.
        meshes = []
        for shape in cmds.ls(type="mesh", long=True) or []:
            parent = (cmds.listRelatives(shape, parent=True, fullPath=True) or [""])[0]
            if parent and exists(parent) and not leaf_no_ns(parent).lower().startswith("ai_"):
                meshes.append(parent)
    if not meshes:
        raise RuntimeError("No character meshes found to copy. Press Generate Pose first, then Copy Pose.")
    grp_name = "AI_ImagePose_Copy_%02d_GRP" % STATE.copy_index
    while exists(grp_name):
        STATE.copy_index += 1
        grp_name = "AI_ImagePose_Copy_%02d_GRP" % STATE.copy_index
    grp = cmds.group(empty=True, name=grp_name)
    copied = []
    for mesh in meshes:
        try:
            # rr=True gives only top roots. ic=False avoids reconnecting the old skinCluster.
            dup = cmds.duplicate(mesh, returnRootsOnly=True, inputConnections=False, upstreamNodes=False)[0]
            dup = cmds.rename(dup, "AI_ImagePose_Copy_%02d_%s" % (STATE.copy_index, clean_name(leaf_no_ns(mesh))))
            try:
                cmds.delete(dup, constructionHistory=True)
            except Exception:
                pass
            try:
                for attr in ["translateX", "translateY", "translateZ", "rotateX", "rotateY", "rotateZ", "scaleX", "scaleY", "scaleZ"]:
                    plug = dup + "." + attr
                    if cmds.objExists(plug):
                        try: cmds.setAttr(plug, lock=False)
                        except Exception: pass
            except Exception:
                pass
            cmds.parent(dup, grp)
            copied.append(dup)
        except Exception as exc:
            log("Could not copy mesh %s: %s" % (mesh, exc), warning=True)
    if not copied:
        safe_delete(grp)
        raise RuntimeError("Could not create a pose copy from the current character meshes.")
    # Move the snapshot slightly aside so it is easy to see/select, like the robot tool.
    try:
        cmds.xform(grp, ws=True, t=[2.0 * STATE.copy_index, 0, 0])
    except Exception:
        pass
    STATE.copy_index += 1
    set_action("Copied current pose to %s. It is visible/selectable in viewport and Outliner." % grp)
    return grp


def save_current_pose_json():
    if not STATE.joints:
        raise RuntimeError("No character skeleton found. Generate a pose first.")
    result = cmds.fileDialog2(
        fileMode=0,
        caption="Save Current Character Pose JSON",
        startingDirectory=TOOL_ROOT,
        fileFilter="JSON Files (*.json)",
    )
    if not result:
        return
    path = result[0]
    if not path.lower().endswith(".json"):
        path += ".json"
    pose = capture_current_character_pose()
    data = {
        "ok": True,
        "tool": "AI Image Pose v29",
        "type": "character_pose_snapshot",
        "image_path": STATE.image_path,
        "character_path": STATE.character_path,
        "namespace": STATE.namespace,
        "settings": {
            "pose_scale": STATE.pose_scale,
            "depth_scale": STATE.depth_scale,
            "mirror": STATE.mirror,
            "keep_feet_grounded": STATE.keep_feet_grounded,
            "match_strength": STATE.match_strength,
            "fix_body_tilt": STATE.body_align,
            "body_tilt_strength": STATE.body_align_strength,
        },
        "joints": pose,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    set_action("Saved current pose JSON: %s" % path)



def ensure_rest_pose_has_world_aliases():
    """Repair old rest dictionaries that were missing pos/rot aliases.

    v26 failed with KeyError 'pos' because some solver functions use rest[...]['pos'].
    v29 guarantees every rest item has pos and rot before Generate/Reset/IK.
    """
    if not STATE.joints:
        scan_mixamo_skeleton()
    if not STATE.rest:
        capture_rest_pose()
    for key, node in STATE.joints.items():
        if not exists(node):
            continue
        STATE.rest.setdefault(key, {"node": node})
        item = STATE.rest[key]
        if "pos" not in item:
            item["pos"] = item.get("translate") or ws_pos(node)
        if "rot" not in item:
            item["rot"] = item.get("rotate") or ws_rot(node)
        if "translate" not in item:
            item["translate"] = item["pos"]
        if "rotate" not in item:
            item["rotate"] = item["rot"]
    return STATE.rest


def restore_character_default_pose():
    """Restore the imported character default/rest pose.

    User changed reset behavior back to the original expected workflow:
    Reset Pose should return the character to the default pose it had when
    imported from the Mixamo/character file, not to the generated image pose.
    """
    if not STATE.rest:
        capture_rest_pose()
    if not STATE.rest:
        raise RuntimeError("No default/rest pose captured. Browse Character and Generate Pose once first.")
    ensure_rest_pose_has_world_aliases()
    return restore_character_pose_snapshot(STATE.rest, "the default character pose")


def restore_first_image_pose_like_robot():
    # Kept as a compatibility helper for old saved UI callbacks, but Reset no longer uses it.
    return restore_character_default_pose()


def reset_character_pose():
    """Reset to the character default/rest pose from import."""
    if not STATE.joints or not exists(STATE.joints.get("hips")):
        if STATE.character_path:
            setup_ninja_character()
        else:
            raise RuntimeError("No character loaded. Browse Character first.")
    return restore_character_default_pose()


def bake_pose_to_character():
    # Kept internally for compatibility. UI now uses Copy Pose instead.
    if not STATE.joints:
        raise RuntimeError("No Mixamo skeleton scanned.")
    joints = [n for n in STATE.joints.values() if exists(n)]
    if joints:
        cmds.setKeyframe(joints, attribute=["translateX", "translateY", "translateZ", "rotateX", "rotateY", "rotateZ"])
    set_action("Baked current character pose to keyframes on %d joints." % len(joints))

# -----------------------------------------------------------------------------
# High-level workflow
# -----------------------------------------------------------------------------
def setup_ninja_character():
    if not STATE.character_path:
        found = auto_find_ninja_fbx()
        if found:
            STATE.character_path = found
            log("Auto-selected character: %s" % found)
    if not STATE.character_path or not os.path.exists(STATE.character_path):
        raise RuntimeError("No character file found. Press Browse Character and choose a rigged Mixamo FBX.")
    return suspend_refresh(import_character)


def generate_all():
    def _run():
        read_ui_values()
        if not STATE.character_path:
            setup_ninja_character()
        # If no valid skeleton is currently in the scene, import/reuse the chosen character before solving.
        if not STATE.joints or not exists(STATE.joints.get("hips")):
            setup_ninja_character()
        validate_mixamo_ready()
        ensure_rest_pose_has_world_aliases()
        if STATE.auto_materials:
            try:
                apply_ninja_materials()
            except Exception as exc:
                log("Auto material setup failed: %s" % exc, warning=True)
        extract_pose_json()
        # Rebuild IK every generate so old broken controls never block the new pose.
        safe_delete(IK_GRP)
        safe_delete(CTRL_GRP)
        STATE.ik = {}
        create_ik_solver()
        apply_pose_to_character()
        capture_initial_image_pose_once(force=False)
        set_action("Generated pose from image. Reset Pose returns the character default pose.")
    return suspend_refresh(_run)


def clean_tool():
    for node in [TOOL_GRP, SOURCE_GRP, TARGET_GRP, IK_GRP, CTRL_GRP, BACKGROUND_GRP]:
        safe_delete(node)
    STATE.targets = {}
    STATE.ik = {}
    set_action("Cleaned AI Image Pose helper objects. Character remains in scene.")


def toggle_source_visibility():
    STATE.show_source = not STATE.show_source
    for grp in [SOURCE_GRP, TARGET_GRP]:
        if exists(grp):
            try: cmds.setAttr(grp + ".visibility", 1 if STATE.show_source else 0)
            except Exception: pass
    set_action("AI source visibility: %s" % ("ON" if STATE.show_source else "OFF"))


def toggle_controls_visibility():
    if exists(CTRL_GRP):
        current = True
        try: current = bool(cmds.getAttr(CTRL_GRP + ".visibility"))
        except Exception: pass
        try: cmds.setAttr(CTRL_GRP + ".visibility", 0 if current else 1)
        except Exception: pass
        set_action("Character controls visibility toggled.")
    else:
        set_action("No controls yet. Import Character first.")

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
def section(label):
    cmds.frameLayout(label=label, collapsable=False, marginWidth=10, marginHeight=8)
    cmds.columnLayout(adjustableColumn=True, rowSpacing=6)


def end_section():
    cmds.setParent("..")
    cmds.setParent("..")


def build_ui():
    if cmds.window(UI_NAME, exists=True):
        cmds.deleteUI(UI_NAME)
    if not STATE.character_path:
        STATE.character_path = auto_find_ninja_fbx()

    window = cmds.window(UI_NAME, title="AI Image Pose", widthHeight=(560, 560), sizeable=True)
    cmds.scrollLayout(childResizable=True)
    cmds.columnLayout(adjustableColumn=True, rowSpacing=10, columnAttach=("both", 10))

    cmds.text(label="AI Image Pose", align="center", height=34, font="boldLabelFont")
    cmds.text(label="Browse image + character, generate pose, then copy/save snapshots.", align="center")

    section("1) Reference Image")
    cmds.rowLayout(numberOfColumns=2, adjustableColumn=1, columnWidth2=(410, 120))
    cmds.textField("mnImagePath", text=STATE.image_path, editable=False)
    cmds.button(label="Browse Image", height=34, backgroundColor=(0.18, 0.38, 0.55), command=lambda *_: with_undo("Browse Image", browse_image))
    cmds.setParent("..")
    cmds.text("mnStatus", label=STATE.pose_status, align="left")
    end_section()

    section("2) Character")
    cmds.rowLayout(numberOfColumns=2, adjustableColumn=1, columnWidth2=(410, 120))
    cmds.textField("mnCharacterPath", text=STATE.character_path, editable=False)
    cmds.button(label="Browse Character", height=34, backgroundColor=(0.30, 0.28, 0.45), command=lambda *_: with_undo("Browse Character", browse_character))
    cmds.setParent("..")
    end_section()

    section("3) Pose")
    cmds.button(label="Generate Pose", height=46, backgroundColor=(0.18, 0.50, 0.28), command=lambda *_: with_undo("Generate Pose", generate_all))
    cmds.rowColumnLayout(numberOfColumns=3, columnWidth=[(1, 170), (2, 170), (3, 170)], columnSpacing=[(1, 6), (2, 6), (3, 6)])
    cmds.button(label="Reset Pose", height=34, backgroundColor=(0.42, 0.30, 0.18), command=lambda *_: with_undo("Reset Pose", reset_character_pose))
    cmds.button(label="Copy Pose", height=34, backgroundColor=(0.20, 0.34, 0.46), command=lambda *_: with_undo("Copy Pose", copy_current_pose_snapshot))
    cmds.button(label="Save Pose", height=34, backgroundColor=(0.18, 0.42, 0.26), command=lambda *_: with_undo("Save Pose", save_current_pose_json))
    cmds.setParent("..")
    end_section()

    cmds.frameLayout(label="Advanced Fine Tune", collapsable=True, collapse=True, marginWidth=10, marginHeight=8)
    cmds.columnLayout(adjustableColumn=True, rowSpacing=6)
    cmds.rowLayout(numberOfColumns=6, columnWidth6=(80, 70, 70, 70, 90, 70))
    cmds.text(label="Scale")
    cmds.floatField("mnPoseScale", value=STATE.pose_scale, minValue=0.1, maxValue=3.0)
    cmds.text(label="Depth")
    cmds.floatField("mnDepthScale", value=STATE.depth_scale, minValue=0.0, maxValue=3.0)
    cmds.text(label="Match")
    cmds.floatField("mnMatchStrength", value=STATE.match_strength, minValue=0.0, maxValue=1.0)
    cmds.setParent("..")
    cmds.rowLayout(numberOfColumns=4, columnWidth4=(120, 120, 140, 80))
    cmds.checkBox("mnMirror", label="Mirror L/R", value=STATE.mirror)
    cmds.checkBox("mnGround", label="Keep Feet", value=STATE.keep_feet_grounded)
    cmds.checkBox("mnBodyAlign", label="Fix Body Tilt", value=STATE.body_align)
    cmds.floatField("mnBodyAlignStrength", value=STATE.body_align_strength, minValue=0.0, maxValue=1.0)
    cmds.setParent("..")
    cmds.text(label="Tip: open only if left/right is reversed or the pose needs small tuning.", align="left")
    cmds.checkBox("mnRoot", label="Move Hips Root", value=STATE.follow_root, visible=False)
    cmds.checkBox("mnShowSource", label="Show AI Source", value=STATE.show_source, visible=False)
    cmds.checkBox("mnStrongRetarget", label="Strong Retarget", value=STATE.strong_retarget, visible=False)
    cmds.checkBox("mnAutoMaterials", label="Auto Materials", value=STATE.auto_materials, visible=False)
    cmds.setParent("..")
    cmds.setParent("..")

    section("4) Status / Log")
    cmds.scrollField("mnLog", editable=False, wordWrap=False, height=170, text="")
    end_section()

    cmds.setParent("..")
    cmds.setParent("..")
    cmds.showWindow(window)
    refresh_ui()
    log("AI Image Pose v29 stable ready. Browse Image, Browse Character, then Generate Pose.")

build_ui()
