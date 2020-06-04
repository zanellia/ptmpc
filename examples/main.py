import ptmpc as pt
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

NX = 2
NU = 1
NG = 2
NGN = 0
T = 10.0
N = 10
# M = 2
M = N
lbu = -1.0
ubu = 1.0
tau = 1.01
niter = 100

SOLVE_DENSE = False 

x0 = np.array([[4.0], [3.0]])

x = ca.MX.sym('x', NX, 1)
u = ca.MX.sym('u', NU, 1)

Q = 10.0*np.eye(NX)
R = 1.0*np.eye(NU)
QN = 1.0*Q

lc  = 1.0/2.0*ca.mtimes(x.T, ca.mtimes(Q, x)) + 1.0/2.0*ca.mtimes(u.T, ca.mtimes(R, u))
lcN = 1.0/2.0*ca.mtimes(x.T, ca.mtimes(QN, x))
g = ca.vertcat(u[0] - ubu, -u[0] + lbu)
gN = [] 
fc = ca.vertcat(-x[1] + 0.1*np.sin(x[1]), u[0] + x[0] -0.1*x[1])
# fc = ca.vertcat(-x[1], u[0] + x[0] -0.1*x[1])

dims = pt.ocp.OcpDims(NX, NU, NG, NGN, N, M)
ocp = pt.ocp.Ocp(dims, x, u, lc, lcN, g, gN, fc, T, tau, print_level=2)

# solve OCP
ocp.update_x0(x0)
sol = ocp.eval()

if SOLVE_DENSE:
    sol_dense = pt.auxiliary.solve_dense_nonlinear_system(ocp, \
        newton_iters=20, alpha=0.5)

# partially tightened RTI
for i in range(niter):

    ocp.pt_rti()

plt.figure()
plt.subplot(211)
x1 = sol['x'][0::NX+NU]
x2 = sol['x'][1::NX+NU]
plt.plot(np.linspace(0,T, N+1), x1)
plt.plot(np.linspace(0,T, N+1), x2)

if SOLVE_DENSE:
    x1 = sol_dense[NX::NX+NX+NU+NG+NG]
    x2 = sol_dense[NX+1::NX+NX+NU+NG+NG]
    plt.plot(np.linspace(0,T, N+1), x1, '--')
    plt.plot(np.linspace(0,T, N+1), x2, '--')

pt_sol_x = np.vstack(ocp.x)
pt_sol_u = np.vstack(ocp.u)
pt_x1 = pt_sol_x[0::NX]
pt_x2 = pt_sol_x[1::NX]
plt.plot(np.linspace(0,T,N+1), pt_x1, 'o')
plt.plot(np.linspace(0,T,N+1), pt_x2, 'o')
legend=[r"$x_1$", r"$x_2$"]
plt.legend(legend)
plt.grid()
plt.ylabel(r"$x$")
plt.subplot(212)
u1 = sol['x'][NX::NX+NU]
plt.step(np.linspace(0,T, N), u1)
if SOLVE_DENSE:
    u_dense = sol_dense[NX+NX::NX+NX+NU+NG+NG]
    plt.step(np.linspace(0,T, N), u_dense, '--')
plt.step(np.linspace(0,T, N), pt_sol_u[0::NU], 'o')
plt.grid()
plt.xlabel(r"$t$")
plt.ylabel(r"$u$")
plt.show()
