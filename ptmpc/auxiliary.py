import numpy as np
import casadi as ca
from .integrator import Integrator
np_t = np.transpose

def compute_qp_res(ocp):
    """
    Compute residuals associated with current QP.
    """
    N = ocp.dims.N
    ngnN = ocp.dims.ngN
    ng = ocp.dims.ng

    # vectors
    i = 0
    A = ocp.A[i]
    B = ocp.B[i]
    Q = ocp.Hxx[i]
    R = ocp.Huu[i]
    C = ocp.C[i]
    D = ocp.D[i]

    dx   = ocp.dx[i]
    du   = ocp.du[i]
    dlam = ocp.dlam[i]
    dlam_next = ocp.dlam[i+1]
    dnu  = ocp.dnu[i]
    dt  = ocp.dt[i]

    x   = ocp.x[i]
    u   = ocp.u[i]
    lam = ocp.lam[i]
    lam_next = ocp.lam[i+1]
    nu  = ocp.nu[i]
    t  = ocp.t[i]

    ocp.r_lam_qp[i] = +ocp.r_lam[i] - dx

    if ng > 0:
        ocp.r_x_qp[i] = np.dot(Q,dx) + \
            np.dot(np_t(A), dlam_next) - dlam + \
            np.dot(np_t(C), dnu) + ocp.r_x[i]

        ocp.r_u_qp[i] = np.dot(R,du) + np.dot(np_t(B), dlam_next) + \
            np.dot(np_t(D), dnu) + ocp.r_u[i]

        ocp.r_nu_qp[i] = np.dot(C,dx) + np.dot(D,du) + dt + ocp.r_nu[i]  

        ocp.e_qp[i] = np.dot(np.diagflat(t), dnu) + \
            np.dot(np.diagflat(nu), dt) + ocp.e[i] 
    else:
        ocp.r_x_qp[i] = np.dot(Q,dx) + \
            np.dot(np_t(A), dlam_next) - dlam + \
            ocp.r_x[i]

        ocp.r_u_qp[i] = np.dot(R,du) + np.dot(np_t(B), dlam_next) + \
            ocp.r_u[i]

    for i in range(1,N):
        A = ocp.A[i]
        B = ocp.B[i]
        A_prev = ocp.A[i-1]
        B_prev = ocp.B[i-1]
        Q = ocp.Hxx[i]
        R = ocp.Huu[i]
        C = ocp.C[i]
        D = ocp.D[i]
        x   = ocp.x[i]
        u   = ocp.u[i]
        x_prev = ocp.x[i-1]
        u_prev = ocp.u[i-1]
        lam = ocp.lam[i]
        lam_next = ocp.lam[i+1]
        nu  = ocp.nu[i]
        t  = ocp.t[i]
        dx   = ocp.dx[i]
        du   = ocp.du[i]
        dx_prev   = ocp.dx[i-1]
        du_prev   = ocp.du[i-1]
        dlam = ocp.dlam[i]
        dlam_next = ocp.dlam[i+1]
        dnu  = ocp.dnu[i]
        dt  = ocp.dt[i]

        ocp.r_lam_qp[i] = -dx + np.dot(A_prev, dx_prev) + \
            np.dot(B_prev, du_prev) + ocp.r_lam[i]

        if ng > 0:
            ocp.r_x_qp[i] = np.dot(Q,dx) + \
                np.dot(np_t(A), dlam_next) - dlam + \
                np.dot(np_t(C), dnu) + ocp.r_x[i]

            r_u_qp_t = np.dot(ocp.Huu_t[i],du) + np.dot(np_t(B), dlam_next) + \
                + ocp.r_u_t[i]

            ocp.r_u_qp[i] = np.dot(R,du) + \
                np.dot(np_t(B), dlam_next)  + \
                np.dot(np_t(D), dnu) + ocp.r_u[i]

            ocp.r_nu_qp[i] = np.dot(C,dx) + np.dot(D,du) + dt + \
                ocp.r_nu[i]  

            ocp.e_qp[i] = np.dot(np.diagflat(t), dnu) + \
                np.dot(np.diagflat(nu), dt) + ocp.e[i] 
        else:
            ocp.r_x_qp[i] = np.dot(Q,dx) + \
                np.dot(np_t(A), dlam_next) - dlam + \
                ocp.r_x[i]

            ocp.r_u_qp[i] = np.dot(R,du) + \
                np.dot(np_t(B), dlam_next)  + \
                ocp.r_u[i]

    if ocp.dims.ngN > 0:
        i = N
        Q = ocp.Hxx_t[i]
        C = ocp.C[i]
        A_prev = ocp.A[i-1]
        B_prev = ocp.B[i-1]
        x = ocp.x[i]
        x_prev = ocp.x[i-1]
        u_prev = ocp.u[i-1]
        lam = ocp.lam[i]
        nu  = ocp.nu[i]
        t  = ocp.t[i]
        dx   = ocp.dx[i]
        dx_prev   = ocp.dx[i-1]
        du_prev   = ocp.du[i-1]
        dlam = ocp.dlam[i]
        dnu  = ocp.dnu[i]
        dt  = ocp.dt[i]

        ocp.r_lam_qp[i] = -dx + np.dot(A_prev,dx_prev) + \
            np.dot(B_prev,du_prev) + ocp.r_lam[i]

        ocp.r_x_qp[i] = np.dot(Q,dx) - dlam + np.dot(np_t(C), dnu) + \
            ocp.r_x[i]

        ocp.r_nu_qp[i] = np.dot(C,dx) + dt + ocp.r_nu[i]  
        ocp.e_qp[i] = np.dot(np.diagflat(t), dnu) + np.dot(np.diagflat(nu), dt) + ocp.e[i] 

    if ocp.print_level > 1:
        # compute and print residuals
        r_lam_qp = np.linalg.norm(np.vstack(ocp.r_lam_qp))
        r_x_qp = np.linalg.norm(np.vstack(ocp.r_x_qp))
        r_u_qp = np.linalg.norm(np.vstack(ocp.r_u_qp))
        r_nu_qp = np.linalg.norm(np.vstack(ocp.r_nu_qp))
        e_qp = np.linalg.norm(np.vstack(ocp.e_qp))
        print('r_lam_qp: {:.1e}, r_x_qp: {:.1e}, r_u_qp: {:.1e}, r_nu_qp: {:.1e}, e_qp: {:.1e}'.format(r_lam_qp, r_x_qp, r_u_qp, r_nu_qp, e_qp))

    return

def solve_dense_nonlinear_system(ocp, newton_iters=10, alpha=1.0):

    
    dims = ocp.dims
    NX = dims.nx
    NU = dims.nu
    NG = dims.ng
    NGN = dims.ngN
    N = dims.N
    M = dims.M

    if M != 0:
        raise Exception('Cannot build dense nonlinear system if M != 0')

    integrator = ocp.integrator
    g = ocp.g
    gN = ocp.gN
    jac_x_f = ocp.jac_x_f
    jac_u_f = ocp.jac_u_f
    jac_x_g = ocp.jac_x_g
    jac_u_g = ocp.jac_u_g
    jac_x_gN = ocp.jac_x_gN
    jac_x_l = ocp.jac_x_l
    jac_x_lN = ocp.jac_x_lN
    jac_u_l = ocp.jac_u_l

    # build dense linear system
    w=[]
    lam=[]
    x=[]
    u=[]
    nu=[]
    t=[]
    w0 =[]

    # define variables
    for i in range(N):
        Lamk = ca.MX.sym('Lam_' + str(i), NX, 1)
        w += [Lamk]
        w0 += [np.zeros((NX,1))]
        lam += [Lamk]

        Xk = ca.MX.sym('X_' + str(i), NX, 1)
        w += [Xk]
        x+=[Xk]
        w0 += [np.zeros((NX,1))]

        Uk = ca.MX.sym('U_' + str(i), NU, 1)
        w += [Uk]
        u+=[Uk]
        w0 += [np.zeros((NU,1))]

        Nuk = ca.MX.sym('Nu_' + str(i), NG, 1)
        w += [Nuk]
        nu+=[Nuk]
        w0 += [np.ones((NG,1))]

        Tk = ca.MX.sym('T_' + str(i), NG, 1)
        w += [Tk]
        t+=[Tk]
        w0 += [np.ones((NG,1))]

    i = N
    Lamk = ca.MX.sym('Lam_' + str(i), NX, 1)
    w += [Lamk]
    lam += [Lamk]
    w0 += [np.zeros((NX,1))]

    Xk = ca.MX.sym('X_' + str(i), NX, 1)
    w += [Xk]
    x+=[Xk]
    w0 += [np.zeros((NX,1))]

    sys = []
    sys +=[-x[0] + ocp.x0]

    # formulate the NLP
    for i in range(0, N):

        # new NLP variable for the control
        Lami = lam[i]
        Lam_next = lam[i+1]
        Xi = x[i]
        X_next = x[i+1]
        Ui = u[i]
        Nui = nu[i]
        Ti = t[i]

        sys+= [jac_x_l(Xi, Ui).T + ca.mtimes(jac_x_f(Xi, Ui).T, \
            Lam_next) - Lami + ca.mtimes(jac_x_g(Xi, Ui).T, Nui)]

        sys+= [jac_u_l(Xi, Ui).T + ca.mtimes(jac_u_f(Xi, Ui).T, \
            Lam_next)  + ca.mtimes(jac_u_g(Xi, Ui).T, Nui)]

        sys+= [g(Xi, Ui) + Ti]

        sys+= [ca.mtimes(ca.diag(Nui), Ti) - ocp.tau*np.ones((NG,1))]

        sys+= [-X_next + integrator.xplus(Xi,Ui)]

    i = N 
    Lami = lam[i]
    Xi = x[i]
    # Nui = nu[i]
    # Ti = t[i]

    sys+= [jac_x_lN(Xi).T - Lami]
    # sys+= [gN(Xi, Ui) + Ti]
    # sys+= [ca.mtimes(ca.diag(Nui), Ti) + ocp.tau*np.ones((NG,1))]
    # sys+= [jac_x_lN(Xi).T - Lami + ca.mtimes(jac_x_gN(Xi).T, Nui)]
    # sys+= [gN(Xi, Ui) + Ti]
    # sys+= [ca.mtimes(ca.diag(Nui), Ti) + ocp.tau*np.ones((NG,1))]

    sys = ca.vertcat(*sys)
    w = ca.vertcat(*w)
    w0 = ca.vertcat(*w0)

    J = ca.Function('J', [w], [ca.jacobian(sys, w)])
    rhs = ca.Function('J', [w], [sys])

    # compute Newton step
    for i in range(newton_iters):

        rhs_e = rhs(w0)
        delta_w = -np.dot(np.linalg.inv(J(w0)), rhs_e)
        w0 = w0 + alpha*delta_w
    
    return w0
    
