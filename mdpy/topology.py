r"""The MD topology."""
# Authors: Zilin Song.


import abc
import mdtraj
import numpy as np


# TODO (ZS 20250423): 
#   The topology implementation is purposed for homogeneous ideal gas, i.e., no connectivity b/w the
#   particles, although extensions to bonded systems with are possible.

class Topology(abc.ABC):
  """The abstract MD topology."""

  @property
  @abc.abstractmethod
  def num_particles(self) -> int:
    r"""The number of particles."""
  
  @property
  @abc.abstractmethod
  def num_dofs(self) -> int:
    r"""The number of degrees of freedom."""
  
  @property
  @abc.abstractmethod
  def masses(self) -> np.ndarray:
    r"""The particle masses (in amu) [N, 1]."""

  @abc.abstractmethod
  def as_mdtraj(self) -> mdtraj.Topology:
    r"""Convert the topology to a `mdtraj.Topology()` object."""


class HomogeneousIdealGas(Topology):
  r"""The MD topology for homogeneous ideal gas."""

  def __init__(self, num_particles: int, mass: float, ):
    r"""Create and MD topology for homogeneous ideal gas.
    
      Args:
        num_particles (int):   The number of homogeneous ideal gas particles.
        mass          (float): The atomic mass (in amu) of the homogeneous ideal gas particles.
    """
    assert int(num_particles)>0, f"`num_particles` should be a positive integer."
    self._natoms = int(num_particles)
    self._ndofs  = int(self._natoms * 3)
    assert float(mass)       >0, f"`mass` should be a positive float."
    self._masses = np.ones((self._natoms, 1)) * float(mass)

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

  def as_mdtraj(self) -> mdtraj.Topology:
    r"""Convert the topology to a `mdtraj.Topology()` object."""
    top = mdtraj.Topology()
    # chain -> residue -> atom.
    c = top.add_chain(chain_id='A')
    for idx in range(self.num_particles):
      r = top.add_residue(name='DUM', chain=c, resSeq=idx, segment_id='A')
      a = top.add_atom   (name='DUM', element=mdtraj.element.argon, residue=r, serial=idx, formal_charge=0)
    return top