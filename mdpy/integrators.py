r"""The MD integrators."""
# Authors: Zilin Song.


import abc
import numpy as np

import mdpy.utils
import mdpy.system


class Integrator(abc.ABC):
  """The abstract integrator."""

  @abc.abstractmethod
  def step(self, system: mdpy.system.System) -> None:
    r"""Advance the dynamical integration by one step."""
    raise NotImplementedError(r"To be implemented by sub-classes.")
  
  @abc.abstractmethod
  def initialize_velocites(self, system: mdpy.system.System) -> None:
    r"""Initialize the particle velocities per Maxwellian at the thermostat temperature."""
    raise NotImplementedError(r"To be implemented by sub-classes.")

  @abc.abstractmethod
  def compute_kinetic_energy(self, system: mdpy.system.System) -> float:
    r"""Compute the total kinetic energy of the particles."""
    raise NotImplementedError(r"To be implemented by sub-classes.")


class LangevinIntegrator(Integrator):
  r"""The Langevin integrator.
    In all cases, the velocities are always half-dt behind the the coordinates in this integrator.
  """

  def __init__(self, 
               timestep: float = .001, 
               friction: float = 5., 
               temperature: float = 300., ):
    r"""Create a Langevin integrator.
    
      Args:
        timestep    (float): The timestep size (in ps), default=.001.
        friction    (float): The friction coefficient (in 1/ps), default=5.
        temperature (float): The temperature (in K), default=300.
    """
    assert isinstance(timestep, float), r"`timestep` should be a float number."
    assert            timestep>0.,      r"`timestep` should be positive."
    self.dt = timestep
    assert isinstance(friction, float), r"`friction` should be a float number."
    assert            friction>=0.,     r"`friction` should be non-negative."
    self.gamma = friction
    assert isinstance(temperature, float), r"`temperature` should be a float number."
    assert            temperature>=0.,     r"`temperature` should be non-negative."
    self.T = temperature
    # scaling pre-factors: a, b, c.
    # v(t') = a * v(t) - b * gradient / m + c * rand_normal * sqrt(RT/m)
    self._a = np.exp(-self.gamma*self.dt)
    self._b = self.dt if self.gamma==0. else (1.-np.exp(-self.gamma*self.dt)) / self.gamma
    self._c = np.sqrt(1.-np.exp(-2.*self.gamma*self.dt))
    self.RT = mdpy.utils.IDEAL_GAS_CONSTANT_R * temperature

  def step(self, system: mdpy.system.System) -> None:
    r"""Advance the integration by one timestep.
    
      Args:
        system (mdpy.system.System): The MD system.
    """
    # extract system states.
    m = system.masses           # [N, 1]
    x = system.coordinates      # [N, 3]
    v = system.velocities       # [N, 3]
    f = system.compute_forces() # [N, 3]
    r = np.random.randn(*f.shape)
    # integration.
    ## integrate v(.5dt) to v(1.5dt).
    v_half = self._a*v + self._b*f/m + self._c*r*np.sqrt(self.RT/m)
    ## integrate x(1.dt) to x(2.dt).
    x_full_next = x + v_half*self.dt
    ## correct v(1.5dt).
    v_half_next = (x_full_next - x) / self.dt
    # update system states.
    system.velocities  = v_half_next
    system.coordinates = x_full_next
  
  def initialize_velocites(self, system: mdpy.system.System) -> None:
    r"""Initialize the particle velocities per Maxwellian at the thermostat temperature.
    
      Args:
        system (mdpy.system.System): The MD system.
    """
    # extract system states.
    m = system.masses
    # initialize v(dt).
    v = np.random.normal(loc=0., scale=self.RT/m, size=(system.num_particles, 3))
    # shift v(dt) to v(.5dt).
    v -= .5*self.dt*system.compute_forces()/m
    # update system states.
    system.velocities = v
  
  def compute_kinetic_energy(self, system: mdpy.system.System) -> float:
    r"""Compute the total kinetic energy of the particles.
    
      Args:
        system (mdpy.system.System): The MD system.
    
      Returns:
        kinetic_energy (float): The total kinetic energy.
    """
    # extract system states.
    m = system.masses
    v = system.velocities
    # shift from v(.5dt) to v(dt).
    v += .5*self.dt*system.compute_forces()/m
    # total K.
    kinetic_energy = np.sum(.5*m*v**2)
    return kinetic_energy