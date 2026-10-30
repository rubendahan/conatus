"""F778 (audit of 0.2.1, N62): spatial differences of opposite sign contradict q_inf + C e^{-a N}
and the study is not trusted. Oracle: 1 + (-1/2)^N has alternating differences; 1 + 2^-N does not."""
from _acceptance import check, passed, start

start(778, "spatial uncertainty model and alternating differences")
import conatus as cn  # noqa: E402

dts = [0.04, 0.02, 0.01]
bad = cn.uncertainty(lambda N, dt: 1 + (-0.5) ** N + dt ** 2, N=[2, 3, 4], dt=dts, scheme_order=2)
check(not bad.trusted and bad.spatial.flags, f"alternating: trusted={bad.trusted}")
good = cn.uncertainty(lambda N, dt: 1 + 0.5 ** N + dt ** 2, N=[2, 4, 5], dt=dts, scheme_order=2)
check(good.trusted and good.interval[0] <= 1 <= good.interval[1], f"monotone: {good.spatial.flags}")
passed("alternating flagged; monotone control trusted and its interval holds the limit")
