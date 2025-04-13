"""The potential functions."""
# Authors: Zilin Song.


import abc
import numpy as np
import mdpy.pbc as _pbc


class Force(abc.ABC):
  r"""The abstract Force."""

  @abc.abstractmethod
  def get_energy(self, **kwargs):
    r"""Compute the potential energy."""
    raise NotImplementedError(r"To be implemented by sub-classes.")
  
  @abc.abstractmethod
  def get_forces(self, **kwargs):
    r"""Compute the potential forces."""
    raise NotImplementedError(r"To be implemented by sub-classes.")


class LJ126(Force):
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
    assert isinstance(sigma, float), r"`sigma` should be a float number."
    assert            sigma>0.,      r"`sigma` should be positive."
    assert isinstance(epsilon, float), r"`epsilon` should be a float number."
    assert            epsilon>0.,      r"`epsilon` should be positive."
    self.sigma   = float(sigma)
    self.epsilon = float(epsilon)
  
  def get_energy(self, coords: np.ndarray, box_dims: np.ndarray) -> float:
    r"""Compute the potential energy.
    
      Args:
        coords (np.ndarray): The particle coordinates [N, 3].
        box_dims (np.ndarray): The periodic box dimensions [1, 3].
      
      Returns:
        energy (float): The energy.
    """
    r_ij = _pbc.compute_minimum_image_distances(coords=coords, box_dims=box_dims) # [N, N]
    r_ij[np.triu(r_ij, k=1)==0.] = np.inf # set the lower diagonals (k<1) to np.inf.
    # energy.
    ener_6  = np.power(self.sigma/r_ij, 6.)
    ener_lj = 4. * self.epsilon * (np.power(ener_6, 2.) - ener_6)
    return np.sum(ener_lj)


  def get_forces(self, coords: np.ndarray, box_dims: np.ndarray) -> np.ndarray:
    r"""Compute the potential forces.
    
      Args: 
        coords (np.ndarray): The particle coordinates [N, 3].
        box_dims (np.ndarray): The periodic box dimensions [1, 3].
      
      Returns:
        forces (float): The forces [N, 3].
    """
    r_ij, dx_ij = _pbc.compute_minimum_image_distances(coords=coords, box_dims=box_dims, return_dx_ij=True)
    r_ij[np.eye(r_ij.shape[0])==1.] = np.inf  # set the diagonal (k=0) to np.inf.
    # grads.
    sigma_div_r = self.sigma / r_ij
    grads = 24*(self.epsilon / self.sigma) * (2*np.power(sigma_div_r, 13.) - np.power(sigma_div_r, 6.)) / r_ij # [N, N]
    grads[grads==np.inf] = 0. # set the inf (from the sigma_div_r=inf dividing r_ij=0.) elements to 0.
    grads = np.sum(np.expand_dims(grads, axis=-1) * dx_ij, axis=1)
    return grads