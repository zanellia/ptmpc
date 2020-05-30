import ptmpc as pt
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

NX = 2
NU = 1
NG = 0
NGN = 0
T = 1.0
# N = 20
N = 2
# M = 2
M = 0
lbu = -100.0
ubu = 100.0
tau = 0.1
niter = 100

x0 = np.array([[2.0], [-1.0]])

x = ca.MX.sym('x', NX, 1)
u = ca.MX.sym('u', NU, 1)

Q = 1.0*np.eye(NX)
R = 1.0*np.eye(NU)
QN = 1.0*Q

lc  = 1.0/2.0*ca.mtimes(x.T, ca.mtimes(Q, x)) + 1.0/2.0*ca.mtimes(u.T, ca.mtimes(R, u))
lcN = 1.0/2.0*ca.mtimes(x.T, ca.mtimes(QN, x))
# g = ca.vertcat(u - ubu, -u + lbu)
g = []
gN = [] 
fc = ca.vertcat(-x[0] + 0.1 * x[1], -0.5*x[1] + u[0])

dims = pt.ocp.OcpDims(NX, NU, NG, NGN, N, M)
ocp = pt.ocp.Ocp(dims, x, u, lc, lcN, g, gN, fc, T, tau)

# solve OCP
ocp.update_x0(x0)
sol = ocp.eval()

# partially tightened RTI
ocp.linearize()
for i in range(niter):

    ocp.eliminate_s_lam()
    ocp.update_vectors_stage_M()
    ocp.backward_riccati()
    ocp.forward_riccati()
    ocp.expand_solution()
    ocp.compute_qp_res()
    ocp.primal_dual_step()
    import pdb; pdb.set_trace()
    # ocp.pt_rti()

plt.figure()
plt.subplot(211)
x1 = sol['x'][0::NX+NU]
x2 = sol['x'][1::NX+NU]
plt.plot(np.linspace(0,T, N+1), x1)
plt.plot(np.linspace(0,T, N+1), x2)
legend=[r"x_1", r"x_2"]
plt.legend(legend)
plt.grid()
plt.ylabel(r"$x$")
plt.subplot(212)
u = sol['x'][2::NX+NU]
plt.step(np.linspace(0,T, N), u)
plt.grid()
plt.xlabel(r"$t$")
plt.ylabel(r"$u$")
plt.show()
