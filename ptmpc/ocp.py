import numpy as np
import casadi as ca

class OcpDims:
    def _init__(self, nx, nu, ng, ngN):
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
        """
        self.nx = nx
        self.nu = nu
        self.ng = ng
        self.ngN = ngN
        self.N = N

class Ocp:
    def __init__(self, lc, lcN, g, gN, fc, T):
        """
        Define an optimal control formulation

        Parameters
        ----------
        lc  : CasADi function 
            lc: R^{nx} x R^{nu} -> R (continuous-time Lagrange term)
        lcN : CasADi function 
            lcN: R^{nx} -> R (Mayer term)
        g   : CasADi function 
            g: R^{n_x} x R^{n_u} -> R^{ng} (constraint function)
        gN  : CasADi function 
            gN: R^{nx} -> R^{ngN} (constraint function at t=T)
        fc  : CasADi function 
            fc: R^{nx} x R^{nu} -> R^{nx} (continuous time dynamics)
        T  : float 
            T: prediction horizon
        """
