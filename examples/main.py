import ptmpc as pt
import casadi as ca
import numpy as np

NX = 2
NU = 1
NG = 2
NGN = 2
T = 1.0
N = 10
M = 5
lbu = -1.0
ubu = 1.0
tau = 1.0

x = ca.SX.sym('x', NX, 1)
u = ca.SX.sym('u', NU, 1)

Q = np.eye(NX)
R = np.eye(NU)
QN = np.eye(NX)

lc  = 1.0/2.0*ca.mtimes(x.T, ca.mtimes(Q, x)) + 1.0/2.0*ca.mtimes(u.T, ca.mtimes(R, u))
lcN = 1.0/2.0*ca.mtimes(x.T, ca.mtimes(Q, x))
g = u - ubu 
g = -u + lbu 
gN = [] 
fc = ca.vertcat(-x[0] + 0.1 * x[1], -x[1] + u[0])

dims = pt.ocp.OcpDims(NX, NU, NG, NGN, N, M)
ocp = pt.ocp.Ocp(dims, x, u, lc, lcN, g, gN, fc, T, tau)
