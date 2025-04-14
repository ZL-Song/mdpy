r"""The potential functions."""
# Authors: Zilin Song.


import abc
import numpy as np

import mdpy.box


class Potential(abc.ABC):
  r"""The abstract potential."""

  @abc.abstractmethod
  def compute_energy(self, coordinates: np.ndarray, box: mdpy.box.PBCBox):
    r"""Compute the potential energy."""
    raise NotImplementedError(r"To be implemented by sub-classes.")
  
  @abc.abstractmethod
  def compute_forces(self, coordinates: np.ndarray, box: mdpy.box.PBCBox):
    r"""Compute the potential forces."""
    raise NotImplementedError(r"To be implemented by sub-classes.")


class LJ126(Potential):
  r"""The Lennard-Jones (LJ) 12-6 potential.
    .. math:`4\varepsilon \left[ \left(\frac{\sigma}{r_{ij}}\right)^{12} - 
                                 \left(\frac{\sigma}{r_{ij}}\right)^{6} \right]`.
  """

  def __init__(self, sigma: float, epsilon: float):
    r"""Create a Lennard-Jones (LJ) 12-6 potential.
    
      Args:
        sigma   (float): The distance between two particles when the LJ 12-6 potential is zero.
        epsilon (float): The depth of the LJ 12-6 potential well.
    """
    assert isinstance(sigma, float),   r"`sigma` should be a float number."
    assert            sigma>0.,        r"`sigma` should be positive."
    assert isinstance(epsilon, float), r"`epsilon` should be a float number."
    assert            epsilon>0.,      r"`epsilon` should be positive."
    self.sigma   = float(sigma)
    self.epsilon = float(epsilon)
  
  def compute_energy(self, coordinates: np.ndarray, box: mdpy.box.PBCBox) -> float:
    r"""Compute the potential energy.
    
      Args:
        coordinates (np.ndarray): The particle coordinates [N, 3].
        box (mdpy.box.PBCBox): The periodic boundary conditioned box.
      
      Returns:
        energy (float): The total energy.
    """
    # distances: set the lower diagonals (k<1) to np.inf to prevent double counting.
    d_ij = box.compute_distances(coordinates=coordinates, return_grad=False)  # [N, N]
    d_ij[np.triu(d_ij, k=1)==0.] = np.inf
    # energy.
    sig_r_6  = np.power(self.sigma/d_ij, 6)
    ener_lj = 4. * self.epsilon * (np.power(sig_r_6, 2.) - sig_r_6)
    return np.sum(ener_lj)

  def compute_forces(self, coordinates: np.ndarray, box: mdpy.box.PBCBox) -> np.ndarray:
    r"""Compute the potential forces.
    
      Args:
        coordinates (np.ndarray): The particle coordinates [N, 3].
        box (mdpy.box.PBCBox): The periodic boundary conditioned box.
      
      Returns:
        forces (float): The forces [N, 3].
    """
    # distances [N, N]: set the on-diagonals (k=0) to np.inf.
    d_ij, g_d_ij = box.compute_distances(coordinates=coordinates, return_grad=True)
    d_ij[np.eye(d_ij.shape[0])==1.] = np.inf
    # gradients.
    eps_r_1 =          self.epsilon / d_ij
    sig_r_6 = np.power(self.sigma   / d_ij, 6)
    g_lj = -24. * eps_r_1 * (2.*np.power(sig_r_6, 2) - sig_r_6)   # [N, N]
    return -np.sum(np.expand_dims(g_lj, axis=2) * g_d_ij, axis=1) # [N, Ni, 1]*[N, Ni, 3] -> [N, 3]