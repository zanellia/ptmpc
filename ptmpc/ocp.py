import numpy as np
import casadi as ca
from .integrator import Integrator

class OcpDims:
    def __init__(self, nx, nu, ng, ngN, N, M):
        """
        Define dimensions of an optimal control formulation

        Parameters
        ----------
        nx  : int
            number of states
        nu  : int
            number of inputs 
        ng  : int
            number of constraints
        ngN : int
            number of terminal constraints
        N  : int
            prediction horizon
        M  : int
            untightened prediction horizon
        """
        self.nx = nx
        self.nu = nu
        self.ng = ng
        self.ngN = ngN
        self.N = N
        self.M = M

class Ocp:
    def __init__(self, dims, x, u, lc_, lcN_, g_, gN_, fc_, T, tau):
        """
        Define an optimal control formulation

        Parameters
        ----------
        dims : OcpDims 
            dimensions of the optimal control problem
        x    : CasADi MX
            CasADi symbolic variables representing the states
        u    : CasADi MX
            CasADi symbolic variables representing the inputs
        lc_  : CasADi expression 
            lc: R^{nx} x R^{nu} -> R (continuous-time Lagrange term)
        lcN_ : CasADi expression 
            lcN: R^{nx} -> R (Mayer term)
        g_   : CasADi expression 
            g: R^{n_x} x R^{n_u} -> R^{ng} (constraint function)
        gN_  : CasADi expression 
            gN: R^{nx} -> R^{ngN} (constraint function at t=T)
        fc_  : CasADi expression 
            fc: R^{nx} x R^{nu} -> R^{nx} (continuous time dynamics)
        T    : float 
            prediction horizon
        tau  : float 
            tightening factor
        """

        # define CasADi functions
        lc = ca.Function('lc', [x,u], [lc_])
        lcN = ca.Function('lcN', [x], [lcN_])
        g = ca.Function('g', [x,u], [g_])
        gN = ca.Function('gN', [x], [gN_])
        fc = ca.Function('fc', [x, u], [fc_])

        self.lc = lc
        self.lcN = lcN
        self.g = g

        self.gN = gN
        self.fc = fc

        self.dims = dims
        NX = dims.nx
        NU = dims.nu
        NG = dims.ng
        NGN = dims.ngN
        N = dims.N
        M = dims.M

        Td = T/N

        # create integrator
        integrator = Integrator(x, u, fc_, Td)
        self.integrator = integrator

        # build OCP
        w=[]
        w0 = []
        lbw = []
        ubw = []
        c=[]
        lbc = []
        ubc = []
        Xk = ca.MX.sym('X0', NX, 1)
        w += [Xk]
        lbw += [0, 0]
        ubw += [0, 0]
        w0 += [0, 0]
        f = 0
    
        # formulate the NLP
        for k in range(M):

            # new NLP variable for the control
            Uk = ca.MX.sym('U_' + str(k), NU, 1)

            # update variable list
            w   += [Uk]
            lbw += [-np.inf*np.ones((NU, 1))]
            ubw += [np.inf*np.ones((NU, 1))]
            w0  += [np.zeros((NU, 1))]

            # add cost contribution
            f = f + Td*lc(Xk, Uk)

            # add constraints
            c += [g(Xk, Uk)]
            lbc += [-np.inf*np.ones((NG, 1))]
            ubc += [np.zeros((NG, 1))]

            # integrate till the end of the interval
            Xk_end = integrator.xplus(Xk, Uk)

            # new NLP variable for state at end of interval
            Xk = ca.MX.sym('X_' + str(k+1), NX, 1)
            w   += [Xk]
            lbw += [-np.inf*np.ones((NX, 1))]
            ubw += [np.inf*np.ones((NX, 1))]
            w0  += [np.zeros((NX, 1))]

            # add equality constraint
            c   += [Xk_end-Xk]
            lbc += [np.zeros((NX, 1))]
            ubc += [np.zeros((NX, 1))]

        for k in range(M, N):

            # new NLP variable for the control
            Uk = ca.MX.sym('U_' + str(k), NU, 1)

            # compute barrier term
            barr_term = 0
            for i in range(NG):
                barr_term = barr_term + -tau*np.log(-g(Xk, Uk)[i])

            f = f + Td*lc(Xk, Uk) + barr_term

            w   += [Uk]
            lbw += [-np.inf*np.ones((NU, 1))]
            ubw += [np.inf*np.ones((NU, 1))]
            w0  += [np.zeros((NU, 1))]

            # integrate till the end of the interval
            Xk_end = integrator.xplus(Xk, Uk)

            # new NLP variable for state at end of interval
            Xk = ca.MX.sym('X_' + str(k+1), NX, 1)
            w   += [Xk]
            lbw += [-np.inf*np.ones((NX, 1))]
            ubw += [np.inf*np.ones((NX, 1))]
            w0  += [np.zeros((NX, 1))]

            # add equality constraint
            c   += [Xk_end-Xk]
            lbc += [np.zeros((NX, 1))]
            ubc += [np.zeros((NX, 1))]

        if M == N:
            # compute barrier term
            barr_term = 0
            for i in range(NGN):
                barr_term = barr_term + -tau*np.log(-gN(Xk)[i])

            f = f + lcN(Xk) + barr_term
        else:

            f = f + lcN(Xk)

            # add constraints
            c += [gN(Xk)]
            lbc += [np.zeros((NGN, 1))]
            ubc += [np.zeros((NGN, 1))]

        c = ca.vertcat(*c)
        w = ca.vertcat(*w)
        
        # convert lists to numpy arrays
        lbw_a = np.vstack(lbw)
            
        self._lbw = np.vstack(lbw)
        self._ubw = np.vstack(ubw)

        self._lbc = np.vstack(lbc)
        self._ubc = np.vstack(ubc)

        self._w0 = np.vstack(w0)

        # create an NLP solver
        prob = {'f': f, 'x': w, 'g': c}
        # opts = {'ipopt': {'print_level': 2}}
        opts = {}
        self.nlp_solver = ca.nlpsol('solver', 'ipopt', prob, opts);

        #----------------------------------------------------------------------
        #                       partially tightened RTI 
        #----------------------------------------------------------------------

        # define CasADi functions for linearization

        nabla_x_f = ca.Function('nabla_x_f', [integrator.x, integrator.u], \
            [ca.jacobian(integrator.xplus_expr, x)])
        self.nabla_x_f = nabla_x_f

        nabla_u_f = ca.Function('nabla_u_f', [integrator.x, integrator.u], \
            [ca.jacobian(integrator.xplus_expr, u)])
        self.nabla_u_f = nabla_u_f

        nabla_xx_l = ca.Function('nabla_xx_l', [x, u], \
            [ca.hessian(lc_, x)[0]])
        self.nabla_xx_l = nabla_xx_l

        nabla_uu_l = ca.Function('nabla_uu_l', [x, u], \
            [ca.hessian(lc_, u)[0]])
        self.nabla_uu_l = nabla_uu_l

        nabla_xx_lN = ca.Function('nabla_xx_lN', [x], \
            [ca.hessian(lcN_, x)[0]])
        self.nabla_xx_lN = nabla_xx_lN

        # these are the primal-dual iterates of the partially tightened RTI
        self.x = []
        self.u = []
        self.lam = []
        self.s = []
        self.nu = []

        for i in range(N):
            self.x.append(np.zeros((NX,1)))
            self.u.append(np.zeros((NU,1)))
            self.lam.append(np.zeros((NX,1)))
            self.s.append(np.zeros((NG,1)))
            self.nu.append(np.zeros((NG,1)))

        self.x.append(np.zeros((NX,1)))
        self.lam.append(np.zeros((NX,1)))
        self.s.append(np.zeros((NGN,1)))
        self.nu.append(np.zeros((NGN,1)))

        # these are the variables associated with the linearized problem
        self.A = []
        self.B = []
        self.C = []
        self.D = []
        self.Hxx = []
        self.Huu = []
        self.Hxu = []
        self.Hxx_t = []
        self.Huu_t = []
        self.Hxu_t = []

        for i in range(N):
            self.A.append(np.zeros((NX,NX)))
            self.B.append(np.zeros((NX,NU)))
            self.C.append(np.zeros((NG,NX)))
            self.D.append(np.zeros((NG,NU)))
            self.Hxx.append(np.zeros((NX,NX)))
            self.Huu.append(np.zeros((NU,NU)))
            self.Hxu.append(np.zeros((NU,NX)))
            self.Hxx_t.append(np.zeros((NX,NX)))
            self.Huu_t.append(np.zeros((NU,NU)))
            self.Hxu_t.append(np.zeros((NU,NX)))

        self.C.append(np.zeros((NGN,NX)))
        self.D.append(np.zeros((NGN,NU)))
        self.Hxx.append(np.zeros((NX,NX)))
        self.Hxx_t.append(np.zeros((NX,NX)))

        # these are the variables associated with the Riccati recursion 
        self.P = []

        for i in range(N+1):
            self.P.append(np.zeros((NX,1)))

    def update_x0(self, x0):
        """
        Update the initial condition in the OCP

        Parameters:
        -----------
        x0 : numpy array
            new value of x0
        """
        for i in range(self.dims.nx):
            self._lbw[i] = x0[i]
            self._ubw[i] = x0[i]

    def eval(self):
        """
        Compute exact solution to OCP
        """
        sol = self.nlp_solver(x0=self._w0, lbx=self._lbw, ubx=self._ubw,\
            lbg=self._lbc, ubg=self._ubc)

        return sol

    def linearize(self):
        N = self.dims.N
        for i in range(N):
            x = self.x[i]
            u = self.u[i]
            self.A[i] = self.nabla_x_f(x,u).full()
            self.B[i] = self.nabla_u_f(x,u).full()
            # TODO(andrea): add Hessian contributions from dynamics and constraints?
            self.Hxx[i] = self.nabla_xx_l(x,u).full()
            self.Huu[i] = self.nabla_uu_l(x,u).full()

        x = self.x[N]
        self.Hxx[N] = self.nabla_xx_lN(x).full()

        return

    def eliminate_s_lam(self):
        # TODO(andrea): add actual Hessian update!
        N = self.dims.N
        for i in range(N):
            self.Hxx_t[i] = self.Hxx[i]
            self.Huu_t[i] = self.Huu[i]

        self.Hxx_t[N] = self.Hxx[N]

        return

    def backward_riccati(self):
        N = self.dims.N
        self.P[N] = self.Hxx_t[N]
        for i in range(N-1,0,-1):
            A = self.A[i]
            B = self.B[i]
            Q = self.Hxx_t[i]
            R = self.Huu_t[i]
            P = self.P[i+1]

            Sigma = -np.dot(np.dot(np.dot(np.transpose(A), P), B), \
                np.linalg.inv(R + np.dot(np.dot(np.transpose(B), P), B)))

            self.P[i] = Q + np.dot(np.dot(np.transpose(A), P), A) + \
                    np.dot(np.dot(np.dot(Sigma, np.transpose(B)), P), A)

            print(self.P[i])
            
        return

    def forward_riccati(self):
        return

