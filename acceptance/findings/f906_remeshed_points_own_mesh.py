"""F906 (review G11): after remeshing, every point of an orbit curve keeps the mesh it was
computed on, and is a solution there.

Oracle, closed form: c' = -w s + c f, s' = w c + s f, f = 1 - c^2 - s^2, w = 1 + b c has the
cycle (c, s) = ((cos phi - b) / (1 - b cos phi), sqrt(1 - b^2) sin phi / (1 - b cos phi)),
T = 2 pi / sqrt(1 - b^2). With remesh_every = 2, point 2 read on its own mesh (data["mesh"])
matches the closed form to the collocation accuracy, and its collocation residual is at the
corrector tolerance (0.2.4: field error 0.40, residual 5.4 on the current mesh). Control:
remesh_every = None.
"""
import numpy as np
from _acceptance import check, passed, start

start(906, "remeshed orbit curve: historical points keep their own mesh")
import conatus.expert as C  # noqa: E402
from conatus.bifurcation import CollocationMesh, PeriodicOrbitProblem, continue_orbits  # noqa: E402


def exact(tau, b, phi0=0.0):
    phi = 2 * np.pi * tau + phi0
    return np.column_stack(((np.cos(phi) - b) / (1 - b * np.cos(phi)),
                            np.sqrt(1 - b * b) * np.sin(phi) / (1 - b * np.cos(phi))))


def rhs(X, P):
    c, s = X["c"], X["s"]
    f, w = 1 - c * c - s * s, 1 + P["b"] * c
    return [-w * s + c * f, w * c + s * f]


for remesh in (None, 2):
    pb = PeriodicOrbitProblem(C.ir_system(C.ode_problem(["c", "s"], {"b": 0.5}, rhs), ["b"]), "b",
                              CollocationMesh.uniform(20, 4))
    z = np.r_[exact(pb.mesh.times(), 0.5).ravel(), 2 * np.pi / np.sqrt(0.75), 0.5]
    cfg = C.ContinuationSettings(ds=0.025, ds_min=1e-8, ds_max=0.025, tolerance=1e-11, max_points=6)
    cv = continue_orbits(pb, z, cfg, remesh_every=remesh, tests=())
    for i, pt in enumerate(cv.points):
        own = pb.on_mesh(pt.data["mesh"])
        U, T, b = own.split(pt.z)
        ph = np.arctan2(np.sqrt(1 - b * b) * U[0, 1], U[0, 0] + b)
        err = float(np.max(np.abs(U - exact(own.mesh.times(), b, ph))))
        own.set_reference(U)
        res = float(np.max(np.abs(own.residual(pt.z)[:own.nu - 2])))
        check(err <= 1e-4 and abs(T - 2 * np.pi / np.sqrt(1 - b * b)) <= 1e-6,
              f"remesh {remesh}, point {i}: field error {err:.2e}, T {T}")
        check(res <= 1e-9, f"remesh {remesh}, point {i}: residual {res:.2e} on its own mesh")
passed("every point of a remeshed curve matches the closed-form cycle on its own mesh")
