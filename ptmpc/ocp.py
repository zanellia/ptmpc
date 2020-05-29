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
        x    : CasADi SX
            CasADi symbolic variables representing the states
        u    : CasADi SX
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

            import pdb; pdb.set_trace()
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
        Solve OCP
        """
        sol = self.nlp_solver(x0=self._w0, lbx=self._lbw, ubx=self._ubw,\
            lbg=self._lbc, ubg=self._ubc)
        return sol
