r"""Unit tests for the `mdpy.integrator` module."""
# Authors: Zilin Song.


import unittest
import numpy as np

import mdpy.box
import mdpy.topology
import mdpy.potentials
import mdpy.integrators
import mdpy.system


class HarmonicPotential3D(mdpy.potentials.Potential):
  r"""A 3D harmonic oscillator potential."""

  def __init__(self, kx: float, ky: float, kz: float, x0: float, y0: float, z0: float):
    r"""Create a 3D harmonic oscillator potential centered at [0., 0., 0.].
    
      Args:
        kx (float): Force constant on the x dimension (in kcal/mol/Å**2).
        ky (float): Force constant on the y dimension (in kcal/mol/Å**2).
        kz (float): Force constant on the z dimension (in kcal/mol/Å**2).
        x0 (float): Equilibrium position on the x dimension.
        y0 (float): Equilibrium position on the y dimension.
        z0 (float): Equilibrium position on the z dimension.
    """
    assert isinstance(kx, float), r"`kx` should be a float number."
    assert isinstance(ky, float), r"`ky` should be a float number."
    assert isinstance(kz, float), r"`kz` should be a float number."
    assert            kx>=0.,     r"`kx` should be non-negative."
    assert            ky>=0.,     r"`ky` should be non-negative."
    assert            kz>=0.,     r"`kz` should be non-negative."
    self.ks = np.asarray([[kx, ky, kz]])
    assert isinstance(x0, float), r"`x0` should be a float number."
    assert isinstance(y0, float), r"`y0` should be a float number."
    assert isinstance(z0, float), r"`z0` should be a float number."
    self.x0 = np.asarray([[x0, y0, z0]])

  def compute_energy(self, coordinates: np.ndarray, box: mdpy.box.PBCBox) -> float:
    r"""Compute the potential energy.
    
      Args:
        coordinates (np.ndarray): The particle coordinates (in Å) [N, 3].
        box (mdpy.box.PBCBox): unused.
      
      Returns:
        energy (float): The total energy (in kcal/mol).
    """
    return np.sum(self.ks * (coordinates-self.x0)**2)
  
  def compute_forces(self, coordinates: np.ndarray, box: mdpy.box.PBCBox) -> np.ndarray:
    r"""Compute the potential forces.
    
      Args:
        coordinates (np.ndarray): The particle coordinates (in Å) [N, 3].
        box (mdpy.box.PBCBox): unused.
        
      Returns:
        forces (float): The forces (in kcal/mol/Å) [N, 3].
    """
    grad = 2. * self.ks * (coordinates-self.x0)
    return -grad


class LangevinIntegratorTest(unittest.TestCase):
  r"""Test cases for `mdpy.integrators.LangevinIntegrator.`."""
  
  def setUp(self):
    self.lj126_system = mdpy.system.System(box        =mdpy.box.PBCBox(xdim=10., ydim=10., zdim=10.), 
                                           topology   =mdpy.topology.HomogeneousIdealGas(num_particles=20, mass=2.), 
                                           coordinates=np.random.randn(20, 3), )
    self.lj126_system.add_potential(potential=mdpy.potentials.LJ126(sigma=1., epsilon=1.))

    self.harmo_system = mdpy.system.System(box        =mdpy.box.PBCBox(xdim=1e5, ydim=1e5, zdim=1e5), 
                                           topology   =mdpy.topology.HomogeneousIdealGas(num_particles=1, mass=2.), 
                                           coordinates=np.asarray([[1., -1., 0.]]), )
    self.harmo_system.add_potential(potential=HarmonicPotential3D(kx=1., ky=1., kz=0., x0=.75, y0=-.75, z0=.75))
    self.harmo_omega     = np.sqrt(1. - .05**2) # sqrt(2k/m - \gamma**2))
    self.harmo_get_ref_coords = lambda t: np.asarray([[ .75+.25*np.exp(-.05*t)*np.cos(self.harmo_omega*t), 
                                                       -.75-.25*np.exp(-.05*t)*np.cos(self.harmo_omega*t), 
                                                       0., ]])
    self.harmo_get_ref_velocs = lambda t: np.asarray([[-.25*np.exp(-.05*t)*(.05*np.cos(self.harmo_omega*t) + self.harmo_omega*np.sin(self.harmo_omega*t)), 
                                                        .25*np.exp(-.05*t)*(.05*np.cos(self.harmo_omega*t) + self.harmo_omega*np.sin(self.harmo_omega*t)), 
                                                       0., ]])

  def tearDown(self):
    del self.lj126_system
    del self.harmo_system
    del self.harmo_omega
    del self.harmo_get_ref_coords
    del self.harmo_get_ref_velocs

  def test_initialize_velocity(self):
    integrator = mdpy.integrators.LangevinIntegrator(system=self.lj126_system, timestep=.01, friction=5., temperature=0.)
    for _ in range(100):
      # change coordinates.
      self.lj126_system.coordinates = np.random.randn(self.lj126_system.topology.num_particles, 3) * np.random.randint(1, 1000)
      # init velocities.
      integrator.initialize_velocities(temperature=0.)
      # forward shift: v(0.dt) = v(-.5dt) + .5 dt f / m
      v = self.lj126_system.velocities + .5 * integrator.dt * self.lj126_system.compute_forces() / self.lj126_system.topology.masses
      assert (v == np.zeros(v.shape)).all()

  def test_energy_conservation(self):
    integrator = mdpy.integrators.LangevinIntegrator(system=self.harmo_system, timestep=.01, friction=0., temperature=0.)
    # initialize md.
    for _ in range(100):
      integrator.step()
    # test total energy conserved.
    ener_init = integrator.compute_kinetic_energy() + self.harmo_system.compute_potential_energy()
    for _ in range(2000):
      ener = integrator.compute_kinetic_energy() + self.harmo_system.compute_potential_energy()
      assert np.abs(ener-ener_init)<5e-6, ener-ener_init
      integrator.step()

  def test_harmonic_oscillator_solution(self):
    integrator = mdpy.integrators.LangevinIntegrator(system=self.harmo_system, timestep=.01, friction=.1, temperature=0.)
    for _ in range(2000):
      t = _ * integrator.dt
      ref_coords = self.harmo_get_ref_coords(t=t)
      ref_velocs = self.harmo_get_ref_velocs(t=t)
      # test.
      coords = self.harmo_system.coordinates
      velocs = self.harmo_system.velocities + .5 * integrator.dt * self.harmo_system.compute_forces() / self.harmo_system.topology.masses
      # test atol=.02: https://github.com/openmm/openmm/blob/master/tests/TestLangevinIntegrator.h.
      assert np.allclose(coords, ref_coords, rtol=0., atol=.02)
      assert np.allclose(velocs, ref_velocs, rtol=0., atol=.02)
      integrator.step()