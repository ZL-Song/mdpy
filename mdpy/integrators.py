r"""The MD integrators."""
# Authors: Zilin Song.


import abc
import numpy as np

import mdpy.utils
import mdpy.system


class Integrator(abc.ABC):
  r"""The abstract integrator."""

  def __init__(self, system: mdpy.system.System):
    r"""Create an abstract integrator.
    
      Args:
        system (mdpy.system.System): The MD system.
    """
    assert isinstance(system, mdpy.system.System), r"`system` should be `mdpy.system.System()`."
    self.system = system

  @abc.abstractmethod
  def step(self) -> None:
    r"""Advance the dynamical integration by one step."""
    raise NotImplementedError(r"To be implemented by sub-classes.")
  
  @abc.abstractmethod
  def initialize_velocities(self, temperature: float) -> None:
    r"""Initialize the particle velocities per Maxwellian at the thermostat temperature."""
    raise NotImplementedError(r"To be implemented by sub-classes.")

  @abc.abstractmethod
  def compute_kinetic_energy(self) -> float:
    r"""Compute the kinetic energy (in kcal/mol) of the system using the velocities at v(dt)."""
    
  def compute_temperature(self) -> float:
    r"""Compute the temperature (in K) of the system using the velocities at v(dt).

      Returns:
        temperature (float): The temperature (in K).
    """
    # K.
    kinetic_energy = self.compute_kinetic_energy()
    # T.
    temperature = 2.*kinetic_energy / self.system.num_dofs / mdpy.utils.IDEAL_GAS_CONSTANT_R
    return temperature


class LangevinIntegrator(Integrator):
  r"""The Langevin leap-frog integrator. 
    The velocities are always half-dt behind the coordinates in this integrator. 
    The `compute_kinetic_energy()` function in the `mdpy.system.System()` instance is automatically 
    replaced by the function with the same name implemented in this class.
    The `compute_temperature()` function in the `mdpy.system.System()` instance is automatically 
    replaced by the function with the same name implemented in this class.
  """

  def __init__(self, 
               system: mdpy.system.System, 
               timestep:    float = .001, 
               friction:    float = 5., 
               temperature: float = 300., ):
    r"""Create a Langevin leap-frog integrator.
    
      Args:
        system (mdpy.system.System): The MD system.
        timestep    (float): The timestep size (in ps), default=.001.
        friction    (float): The friction coefficient (in 1/ps), default=5.
        temperature (float): The temperature (in K), default=300.
    """
    Integrator.__init__(self, system=system)
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

  def step(self) -> None:
    r"""Advance the integration by one timestep."""
    # extract system states.
    m = self.system.masses           # [N, 1]
    x = self.system.coordinates      # [N, 3]
    v = self.system.velocities       # [N, 3]
    f = self.system.compute_forces() # [N, 3]
    r = np.random.randn(*f.shape)
    # integration.
    ## integrate v(.5dt) to v(1.5dt).
    v_half = self._a*v + self._b*f/m + self._c*r*np.sqrt(self.RT/m)
    ## integrate x(1.dt) to x(2.dt).
    x_full_next = x + v_half*self.dt
    ## correct v(1.5dt).
    v_half_next = (x_full_next - x) / self.dt
    # update system states.
    self.system.velocities  = v_half_next
    self.system.coordinates = x_full_next
  
  def initialize_velocities(self, temperature: float) -> None:
    r"""Initialize the particle velocities per Maxwellian at the designated temperature.
    
      Args:
        temperature (float): The temperature (in K).
    """
    RT = mdpy.utils.IDEAL_GAS_CONSTANT_R * temperature
    # extract system states.
    m = self.system.masses
    f = self.system.compute_forces()
    # initialize v(dt).
    v = np.random.normal(loc=0., scale=RT/m, size=(self.system.num_particles, 3))
    # shift v(dt) to v(.5dt).
    v -= .5*self.dt*f/m
    # update system states.
    self.system.velocities = v

  def compute_kinetic_energy(self) -> float:
    r"""Compute the kinetic energy (in kcal/mol) of the system using the shifted velocities v(dt).
    
      Returns:
        kinetic_energy (float): The kinetic energy (in kcal/mol).
    """
    # extract system states.
    m = self.system.masses
    v = self.system.velocities
    f = self.system.compute_forces()
    # shift v(.5dt) to v(dt).
    v += .5*self.dt*f/m
    # K.
    kinetic_energy = np.sum(.5*m*v**2)
    return kinetic_energy