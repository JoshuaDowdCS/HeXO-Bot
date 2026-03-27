# utils.py

AXIAL_DIRECTIONS = [
    (1, 0), (1, -1), (0, -1),
    (-1, 0), (-1, 1), (0, 1)
]

def get_neighbors(q, r):
    """Return all 6 neighbors of (q, r)."""
    return [(q + dq, r + dr) for dq, dr in AXIAL_DIRECTIONS]

def axial_distance(p1, p2):
    """Calculate axial distance between two (q, r) points."""
    q1, r1 = p1
    q2, r2 = p2
    return (abs(q1 - q2) + abs(q1 + r1 - q2 - r2) + abs(r1 - r2)) // 2

def get_cells_within_distance(center, dist):
    """Return all (q, r) cells within a given distance from a center (q, r)."""
    cells = []
    qc, rc = center
    for dq in range(0 - dist, dist + 1):
        for dr in range(max(0 - dist, (0 - dq) - dist), min(dist, (0 - dq) + dist) + 1):
            cells.append((qc + dq, rc + dr))
    return cells
