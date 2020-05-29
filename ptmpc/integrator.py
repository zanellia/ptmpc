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
        X  = x

        for j in range(M):
            k1 = f(X, u)
            k2 = f(X + DT/2 * k1, u)
            k3 = f(X + DT/2 * k2, u)
            k4 = f(X + DT * k3, u)
            X=X+DT/6*(k1 + 2*k2 + 2*k3 +k4)

        self.xplus = ca.Function('x_plus', [x, u], [X])
        self.xplus_expr = X

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
