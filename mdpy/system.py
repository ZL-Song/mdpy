"""The MD system."""
# Authors: Zilin Song.


import numpy as np

import mdpy.box
import mdpy.potentials


class System:
  """The MD system that holds the box of particles."""

  def __init__(self, box: mdpy.box.PBCBox, masses: np.ndarray, coordinates: np.ndarray):
    r"""Create an MD system that holds the box of particles.
    
      Args:
        box         (mdpy.box.PBCBox): The periodic boundary conditioned simulation box.
        masses      (np.ndarray):      The particle masses [N, ].
        coordinates (np.ndarray):      The particle coordinates [N, 3].
    """
    # box
    assert isinstance(box, mdpy.box.PBCBox), r"`box` should be an `mdpy.box.PBCBox` instance."
    self._box = box
    # masses.
    assert isinstance(masses, np.ndarray), r"`masses` should be an array."
    assert len(masses.shape)==1,           r"`masses` should be an array of shape [N, ]."
    self._masses = np.expand_dims(np.copy(masses), axis=1)
    # n_particles.
    self._n = int(masses.shape[0])
    # coordinates
    assert isinstance(coordinates, np.ndarray), r"`coordinates` should be an array."
    assert coordinates.shape==(self._n, 3),     r"`coordinates` should be an array of shape [N, 3]."
    self._coordinates = np.copy(coordinates)
    # velocities.
    self._velocities  = np.zeros(self.coordinates.shape)
    # forces.
    self.potentials: list[mdpy.potentials.Potential] = []

  @property
  def box(self) -> mdpy.box.PBCBox:
    r"""The periodic boundary conditioned simulation box."""
    return self._box

  @property
  def num_particles(self) -> int:
    r"""The number of particles."""
    return self._n

  @property
  def masses(self) -> np.ndarray:
    r"""The particle masses [N, 1]."""
    return np.copy(self._masses)

  @property
  def coordinates(self) -> np.ndarray:
    r"""The particle coordinates [N, 3]."""
    return np.copy(self._coordinates)
  @coordinates.setter
  def coordinates(self, val: np.ndarray) -> None:
    r"""Set the particle coordinates."""
    assert isinstance(val, np.ndarray),        r"`coordinates` is a NumPy array."
    assert val.shape==(self.num_particles, 3), r"`coordinates` is a array of shape [N, 3]."
    self._coordinates = np.copy(val)

  @property
  def velocities(self) -> np.ndarray:
    r"""The particle velocities [N, 3]."""
    return np.copy(self._velocities)
  @velocities.setter
  def velocities(self, val: np.ndarray) -> None:
    assert isinstance(val, np.ndarray),        r"`coordinates` is a NumPy array."
    assert val.shape==(self.num_particles, 3), r"Inconsistent `coordinates` array shape."
    self._velocities = np.copy(val)
  
  def add_potential(self, potential: mdpy.potentials.Potential) -> None:
    r"""Add a potential function to the System."""
    self.potentials.append(potential)
  
  def compute_potential_energy(self) -> float:
    r"""Compute the potential energy of the system."""
    energy = 0.
    for p in self.potentials:
      energy += p.compute_energy(coordinates=self.coordinates, box=self.box)
    return energy
  
  def compute_forces(self) -> np.ndarray:
    r"""COmpute the forces of the system."""
    forces = np.zeros((self.num_particles, 3))
    for p in self.potentials:
      forces += p.compute_forces(coordinates=self.coordinates, box=self.box)
    return forces
  
