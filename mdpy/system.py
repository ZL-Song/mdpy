r"""The MD system."""
# Authors: Zilin Song.


import numpy as np

import mdpy.box
import mdpy.potentials
import mdpy.utils


class System:
  r"""The MD system that holds the box of particles."""

  def __init__(self, box: mdpy.box.PBCBox, masses: np.ndarray, coordinates: np.ndarray):
    r"""Create an MD system that holds the box of particles.
    
      Args:
        box         (mdpy.box.PBCBox): The periodic boundary conditioned simulation box.
        masses      (np.ndarray):      The particle masses (in amu) [N, ].
        coordinates (np.ndarray):      The particle coordinates (in Å) [N, 3].
    """
    # box
    assert isinstance(box, mdpy.box.PBCBox), r"`box` should be an instance of `mdpy.box.PBCBox`."
    self._box = box
    # masses.
    assert isinstance(masses, np.ndarray), r"`masses` should be an array."
    assert len(masses.shape)==1,           r"`masses` should be an array of shape [N, ]."
    self._masses = np.expand_dims(np.copy(masses), axis=1)
    # n_particles.
    self._natoms = int(masses.shape[0])
    # n_dofs.
    self._ndofs = int(self._natoms * 3.)
    # coordinates
    assert isinstance(coordinates, np.ndarray), r"`coordinates` should be an array."
    assert coordinates.shape==(self.natoms, 3), r"`coordinates` should be an array of shape [N, 3]."
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
    return self._natoms
  
  @property
  def num_dofs(self) -> int:
    r"""The number of degrees of freedom."""
    return self._ndofs

  @property
  def masses(self) -> np.ndarray:
    r"""The particle masses (in amu) [N, 1]."""
    return np.copy(self._masses)

  @property
  def coordinates(self) -> np.ndarray:
    r"""The particle coordinates (in Å) [N, 3]."""
    return np.copy(self._coordinates)
  @coordinates.setter
  def coordinates(self, val: np.ndarray) -> None:
    r"""Set the particle coordinates (in Å)."""
    assert isinstance(val, np.ndarray),        r"`coordinates` should be an array."
    assert val.shape==(self.num_particles, 3), r"`coordinates` should be an array of shape [N, 3]."
    self._coordinates = np.copy(val)

  @property
  def velocities(self) -> np.ndarray:
    r"""The particle velocities [N, 3] (in Å/ps)."""
    return np.copy(self._velocities)
  @velocities.setter
  def velocities(self, val: np.ndarray) -> None:
    assert isinstance(val, np.ndarray),        r"`velocities` should be an array."
    assert val.shape==(self.num_particles, 3), r"`velocities` should be an array of shape [N, 3]."
    self._velocities = np.copy(val)
  
  def add_potential(self, potential: mdpy.potentials.Potential) -> None:
    r"""Add a potential function to the System."""
    self.potentials.append(potential)
  
  def compute_forces(self) -> np.ndarray:
    r"""Compute the forces (in kcal/mol/Å) of the system."""
    forces = np.zeros((self.num_particles, 3))
    for p in self.potentials:
      forces += p.compute_forces(coordinates=self.coordinates, box=self.box)
    return forces
  
  def compute_potential_energy(self) -> float:
    r"""Compute the potential energy (in kcal/mol) of the system using the state coordinates x(dt).
    """
    potential_energy = 0.
    for p in self.potentials:
      potential_energy += p.compute_energy(coordinates=self.coordinates, box=self.box)
    return potential_energy