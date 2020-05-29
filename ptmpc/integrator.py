import casadi as ca

class Integrator():

    def __init__(self, x, u, xdot, Td, M = 1):
        """
        Integrator for dynamics of a continuous-time system

        Parameters:
        x   : CasADi sym
            state of the system
        u   : CasADi sym
            input of the system
        xdot : CasADi expression
            function describing the system's dynamics
        Td  : float
            discretization time
        M  : int
            number of Runge-Kutta steps
        """

        self.x = x
        self.u = u
        self.xdot = xdot

        DT = Td/M
        f  = ca.Function('f', [x, u], [xdot])
        X0 = ca.MX.sym('X0', 2, 1)
        U  = ca.MX.sym('U', 1, 1)
        X  = X0

        for j in range(M):
            k1 = f(X, U)
            k2 = f(X + DT/2 * k1, U)
            k3 = f(X + DT/2 * k2, U)
            k4 = f(X + DT * k3, U)
            X=X+DT/6*(k1 + 2*k2 + 2*k3 +k4)

        self.xplus = ca.Function('x_plus', [X0, U], [X])

    def eval(self, x, u):
        """
        Integrate

        Parameters:
        x   : numpy array 
            state of the system
        u   : numpy array 
            input of the system
        """

        return self.xplus(x, u).full()
