import numpy as np


def vector_normalize(v: np.ndarray):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def vector_project_on_plane(v: np.ndarray, n: np.ndarray):
    v_proj_n = (v @ n) / (np.linalg.norm(n)) * n
    return v - v_proj_n


def vector_vector_angle(v1: np.ndarray, v2: np.ndarray):
    va = v1 / np.linalg.norm(v1)
    vb = v2 / np.linalg.norm(v2)
    angle = np.arccos(va @ vb) * 180 / np.pi
    return angle


def vector_plane_angle(v: np.ndarray, n: np.ndarray):
    angle = vector_vector_angle(v, n)
    return 90 - angle


def vector_reflect(v: np.ndarray, n: np.ndarray):
    v1 = v / np.linalg.norm(v)
    n1 = n / np.linalg.norm(n)
    v_r = v1 - 2 * (v1 @ n1) * n1
    return v_r / np.linalg.norm(v_r)


def clip(value: float, min_v: float, max_v: float):
    return min(max_v, max(min_v, value))
