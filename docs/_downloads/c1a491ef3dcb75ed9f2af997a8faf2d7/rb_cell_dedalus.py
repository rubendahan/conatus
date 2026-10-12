"""A Dedalus 3 script, written for tutorial 14 of Conatus (not taken from the Dedalus examples).

Onset of Rayleigh-Benard convection between two no-slip walls, periodic sideways, in diffusive
units (length: the depth; time: depth^2 / thermal diffusivity). The temperature is the conduction
profile 1 - z plus a deviation th, which vanishes on both walls; the velocity is u, the pressure p.
With these units the equations are

    dt(u) - Pr lap(u) + grad(p) - Ra Pr th e_z = -u.grad(u),   div(u) = 0,
    dt(th) - lap(th) - u.e_z = -u.grad(th)          (u.e_z: advection of the conduction profile).

Dedalus wants first-order equations in z here: each gradient carries the tau term of the bottom
wall, each evolution equation the tau term of the top wall, and a constant tau closes the pressure
gauge. The time step is fixed and the start is one deterministic roll, so that another code can
repeat the run exactly.

Tutorial 14 never runs this file: it gives it to ``cn.from_dedalus``, which reads it without
executing it. ``t14b_dedalus_check.py`` runs it with Dedalus (if installed) to compare.
Run with Dedalus: python rb_cell_dedalus.py
"""
import numpy as np
import dedalus.public as d3

# Physics and numerics
Ra = 3000.0          # above the no-slip onset, Ra_c = 1708
Pr = 7.0             # water
period = 2.0         # close to the critical wavelength 2.016
N_x, N_z = 24, 16
tstep = 5e-4
steps = 40

# Coordinates, bases (quadratic terms on a 3/2 grid) and unknowns
plane = d3.CartesianCoordinates('x', 'z')
mesh = d3.Distributor(plane, dtype=np.float64)
horiz = d3.RealFourier(plane['x'], size=N_x, bounds=(0, period), dealias=1.5)
vert = d3.ChebyshevT(plane['z'], size=N_z, bounds=(0, 1), dealias=1.5)
u = mesh.VectorField(plane, name='u', bases=(horiz, vert))
p = mesh.Field(name='p', bases=(horiz, vert))
th = mesh.Field(name='th', bases=(horiz, vert))
u_bot = mesh.VectorField(plane, name='u_bot', bases=horiz)    # tau terms, one per wall condition
u_top = mesh.VectorField(plane, name='u_top', bases=horiz)
th_bot = mesh.Field(name='th_bot', bases=horiz)
th_top = mesh.Field(name='th_top', bases=horiz)
gauge = mesh.Field(name='gauge')                               # constant tau of the pressure

# Tau terms live in the first-derivative basis of z; the bottom ones enter through the gradients
e_x, e_z = plane.unit_vector_fields(mesh)
dz_basis = vert.derivative_basis(1)
tau_z = lambda f: d3.Lift(f, dz_basis, -1)  # noqa: E731 (a Dedalus idiom)
Gu = d3.grad(u) + e_z*tau_z(u_bot)
Gth = d3.grad(th) + e_z*tau_z(th_bot)

onset = d3.IVP([u, p, th, u_bot, u_top, th_bot, th_top, gauge], namespace=locals())
onset.add_equation("dt(u) - Pr*div(Gu) + grad(p) - Ra*Pr*th*e_z + tau_z(u_top) = -u@grad(u)")
onset.add_equation("trace(Gu) + gauge = 0")
onset.add_equation("dt(th) - div(Gth) - u@e_z + tau_z(th_top) = -u@grad(th)")
onset.add_equation("u(z=0) = 0")
onset.add_equation("th(z=0) = 0")
onset.add_equation("u(z=1) = 0")
onset.add_equation("th(z=1) = 0")
onset.add_equation("integ(p) = 0")

# Fixed-step SBDF2 run from one roll of the deviation
run = onset.build_solver(d3.SBDF2)
run.stop_iteration = steps
x, z = mesh.local_grids(horiz, vert)
th['g'] = 1e-3*np.cos(2*np.pi*x/period)*np.sin(np.pi*z)
while run.proceed:
    run.step(tstep)
