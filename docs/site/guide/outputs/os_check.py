import conatus as cn

m = cn.Model(cn.Box(y=cn.walls(-1, 1)), name="Orr-Sommerfeld", Re=1e4, alpha=1.0)
m.scalar("v")
m.given(U=lambda y: 1 - y**2)
m.eigenvalue("c", growth="imag")
m.define("L2(f) = dy(dy(f)) - alpha**2*f")
m.equations("L2(L2(v))/(1j*alpha*Re) - U*L2(v) + dy(dy(U))*v + c*L2(v) = 0")
m.at("walls", "v = 0", "dy(v) = 0")
res = m.eigenvalues(N=64, drift=96)          # drift test without its tolerance
