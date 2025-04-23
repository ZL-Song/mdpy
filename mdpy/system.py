r"""The MD system."""
# Authors: Zilin Song.


import numpy as np

import mdpy.box
import mdpy.topology
import mdpy.potentials


class System:
  r"""The MD system that holds the box of particles."""

  def __init__(self, 
               box:         mdpy.box.PBCBox, 
               topology:    mdpy.topology.Topology, 
               coordinates: np.ndarray):
    r"""Create an MD system that holds the box of particles.
    
      Args:
        box         (mdpy.box.PBCBox):        The periodic boundary conditioned simulation box.
        topology    (mdpy.topology.Topology): The topology of the system.
        coordinates (np.ndarray):             The particle coordinates (in Å) [N, 3].
    """
    # box
    assert isinstance(box, mdpy.box.PBCBox), r"`box` should be an instance of `mdpy.box.PBCBox`."
    self._box = box
    # topology.
    assert isinstance(topology, mdpy.topology.Topology), r"`topology` should be an `mdpy.topology.Topology` object."
    self._topology = topology
    # coordinates
    assert isinstance(coordinates, np.ndarray),                 r"`coordinates` should be an array."
    assert coordinates.shape==(self.topology.num_particles, 3), r"`coordinates` should be an array of shape [N, 3]."
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
  def topology(self) -> mdpy.topology.Topology:
    r"""The number of particles."""
    return self._topology
  
  @property
  def coordinates(self) -> np.ndarray:
    r"""The particle coordinates (in Å) [N, 3]."""
    return np.copy(self._coordinates)
  @coordinates.setter
  def coordinates(self, val: np.ndarray) -> None:
    r"""Set the particle coordinates (in Å)."""
    assert isinstance(val, np.ndarray),                 r"`coordinates` should be an array."
    assert val.shape==(self.topology.num_particles, 3), r"`coordinates` should be an array of shape [N, 3]."
    self._coordinates = np.copy(val)

  @property
  def velocities(self) -> np.ndarray:
    r"""The particle velocities [N, 3] (in Å/ps)."""
    return np.copy(self._velocities)
  @velocities.setter
  def velocities(self, val: np.ndarray) -> None:
    assert isinstance(val, np.ndarray),                 r"`velocities` should be an array."
    assert val.shape==(self.topology.num_particles, 3), r"`velocities` should be an array of shape [N, 3]."
    self._velocities = np.copy(val)
  
  def add_potential(self, potential: mdpy.potentials.Potential) -> None:
    r"""Add a potential function to the System."""
    self.potentials.append(potential)
  
  def compute_forces(self) -> np.ndarray:
    r"""Compute the forces (in kcal/mol/Å) of the system."""
    forces = np.zeros((self.topology.num_particles, 3))
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