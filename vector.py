import numpy as np

def vector_from_points(p1, p2):
    """Create a vector from two points."""
    return np.array([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z])

def calculate_angle(v1, v2):
    """Calculate the angle between two vectors."""
    dot_product = np.dot(v1, v2)
    magnitude_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    cos_angle = dot_product / magnitude_product
    return np.arccos(cos_angle)

def project_vector_onto_plane(vector, plane_normal):
    """Project a vector onto a plane defined by its normal vector."""
    plane_normal_normalized = plane_normal / np.linalg.norm(plane_normal)
    dot_product = np.dot(vector, plane_normal_normalized)
    return vector - dot_product * plane_normal_normalized