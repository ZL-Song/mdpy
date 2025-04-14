r"""The simulation box of homogeneous particles."""
# Authors: Zilin Song.


import typing
import numpy as np


class PBCBox:
  r"""The periodic boundary conditioned simulation box centered at the origin [0., 0., 0.])."""

  def __init__(self, xdim: float, ydim: float, zdim: float, coordinates: np.ndarray):
    r"""Create a periodic boundary conditioned simulation box centered at the origin [0., 0., 0.].
    
      Args:
        xdim (float): The length of x dimension.
        ydim (float): The length of y dimension.
        zdim (float): The length of z dimension.
        coordinates (np.ndarray): The particle coordinates [N, 3].
    """
    assert isinstance(coordinates, np.ndarray), r"`coordinates` should be a NumPy array."
    assert len(coordinates.shape)  ==2, r"`coordinates` should be a NumPy array of shape [N, 3]."
    assert     coordinates.shape[1]==3, r"`coordinates` should be a NumPy array of shape [N, 3]."
    self._coordinates = np.copy(coordinates)
    assert isinstance(xdim, float),         r"`xdim` should be a float number."
    assert isinstance(ydim, float),         r"`ydim` should be a float number."
    assert isinstance(zdim, float),         r"`zdim` should be a float number."
    assert xdim>0. and ydim>0. and zdim>0., r"`xdim`, `ydim`, `zdim` should be positive."
    self._dims = np.expand_dims(np.asarray([float(xdim), float(ydim), float(zdim)]), axis=0)# [1, 3]
    self.coordinates = self.wrap(coordinates=self.coordinates)

  @property
  def dims(self) -> np.ndarray:
    r"""Get the box dimensions."""
    return np.copy(self._dims)

  @property
  def volume(self) -> float:
    r"""Get the box volume."""
    return np.prod(self.dims)

  @property
  def num_particles(self): 
    r"""The number of particles."""
    return int(self._coordinates.shape[0])
  
  @property
  def coordinates(self) -> np.ndarray:
    r"""The particle coordinates."""
    return np.copy(self._coordinates)
  
  @coordinates.setter
  def coordinates(self, coordinates: np.ndarray) -> None:
    r"""Set the particle coordinates."""
    assert isinstance(coordinates, np.ndarray),        r"`coordinates` should be a NumPy array."
    assert coordinates.shape==self._coordinates.shape, r"Inconsistent `coordinates` array shape."
    self._coordinates = np.copy(coordinates)

  def wrap(self, coordinates: np.ndarray) -> np.ndarray:
    r"""Wraps the coordinates within the box.
    
      Arg:
        coordinates (np.ndarray): The particle coordinates [N, 3].

      Returns:
        coordinates (np.ndarray): The wrapped particle coordinates [N, 3].
    """
    coordinates -= self._dims * np.round(coordinates / self._dims)
    return coordinates

  def compute_distances(self, 
                        coordinates: np.ndarray, 
                        return_grad: bool = False, 
                        ) -> typing.Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    r"""Compute the minimum image interatomic distances of the particle coordinates under the PBC,
      optionally returns the gradients to the particle coordinates:
      
        .. math:: 
          \nabla_{\mathbf{x}_{i}} d(\mathbf{x}_{i}, \mathbf{x}_{j}) 
          = \frac{1}{d(\mathbf{x}_{i}, \mathbf{x}_{j})} (\mathbf{x}_{i} - \mathbf{x}_{i})
    
      Args:
        coordinates (np.ndarray):     The particle coordinates [N, 3].
        return_grad (bool, optional): If returns the particle coordinates gradients [N, N, 3].

      Returns:
        distances (np.ndarray): The minimum image interatomic distances [N, N].
        gradients (np.ndarray): The particle coordinates gradients [N, N, 3].
    """
    dx_ij  = np.expand_dims(coordinates, axis=1) - np.expand_dims(coordinates, axis=0)  # [N, N, 3]
    dx_ij -= self.dims[np.newaxis, :] * np.round(dx_ij / self.dims[np.newaxis, :])      # [N, N, 3]
    d_ij: np.ndarray = np.linalg.norm(dx_ij, ord=2, axis=-1)                            # [N, N]
    if return_grad:
      d_ij_ = (1.-np.eye(d_ij.shape[0])) / (d_ij+np.eye(d_ij.shape[0])) # 1/dij w/ diag=0, [N, N].
      g_ij  = np.expand_dims(d_ij_, axis=-1) * dx_ij                    # dd_ij/dx_i, [N, N, 3].
      return d_ij, g_ij
    return d_ij