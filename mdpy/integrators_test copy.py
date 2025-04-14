"""Unit tests for the `mdpy.particles` module."""
# Authors: Zilin Song


import unittest
import numpy as np

import mdpy.box
import mdpy.potentials
import mdpy.integrators


class HarmonicPotential3D(mdpy.potentials.Potential):
  r"""A 3D harmonic oscillator potential."""

  def __init__(self, kx: float, ky: float, kz: float, x0: float, y0: float, z0: float):
    r"""Create a 3D harmonic oscillator potential centered at [0., 0., 0.].
    
      Args:
        kx (float): Force constant on x dim.
        ky (float): Force constant on y dim.
        kz (float): Force constant on z dim.
        x0 (float): Equilibrium position on x dim.
        y0 (float): Equilibrium position on y dim.
        z0 (float): Equilibrium position on z dim.
    """
    assert isinstance(kx, float), r"`kx` should be a float number."
    assert            kx>=0.,     r"`kx` should be non-negative."
    assert isinstance(ky, float), r"`ky` should be a float number."
    assert            ky>=0.,     r"`ky` should be non-negative."
    assert isinstance(kz, float), r"`kz` should be a float number."
    assert            kz>=0.,     r"`kz` should be non-negative."
    self.ks = np.asarray([[kx, ky, kz]])
    assert isinstance(x0, float), r"`x0` should be a float number."
    assert isinstance(y0, float), r"`y0` should be a float number."
    assert isinstance(z0, float), r"`z0` should be a float number."
    self.x0 = np.asarray([[x0, y0, z0]])

  def compute_energy(self, coordinates: np.ndarray) -> float:
    r"""Compute the potential energy.
    
      Args:
        coordinates (np.ndarray): The particle coordinates [N, 3].
      
      Returns:
        energy (float): The total energy.
    """
    return np.sum(self.ks * (coordinates-self.x0)**2)
  
  def compute_forces(self, coordinates: np.ndarray) -> np.ndarray:
    r"""Compute the potential forces.
    
      Args:
        coordinates (np.ndarray): The particle coordinates [N, 3].
        
      Returns:
        forces (float): The forces [N, 3].
    """
    grad = 2. * self.ks * (coordinates-self.x0)
    return -grad


class LangevinIntegratorTest(unittest.TestCase):
  """Test cases for `mdpy.integrators.LangevinIntegrator.`."""

  def setUp(self):
    # for test_initialize_velocity.
    self.initv_particles  = mdpy.particles.Particles(masses=np.ones((100, ))*2., coordinates=np.random.randn(100, 3))
    self.initv_potential  = mdpy.potentials.LJ126(sigma=1., epsilon=1., box=mdpy.box.PBCBox(xdim=2., ydim=2., zdim=2.))
    self.initv_integrator = mdpy.integrators.LangevinIntegrator(timestep=.01, friction=10., temperature=0.)
    # for test_harmonic_oscillator_solution.
    self.harmo_particles  = mdpy.particles.Particles(masses=np.ones((1, ))*2., coordinates=np.asarray([[1., -1., 0.]]))
    self.harmo_potential  = HarmonicPotential3D(kx=1., ky=1., kz=0., x0=.75, y0=-.75, z0=.75)
    self.harmo_integrator = mdpy.integrators.LangevinIntegrator(timestep=.01, friction=.1, temperature=0.)
    self.harmo_omega = np.sqrt(1. - .05**2) # sqrt(2k/m - \gamma**2))
    self.harmo_get_true_coords = lambda t: np.asarray([[ .75+.25*np.exp(-.05*t)*np.cos(self.harmo_omega*t), 
                                                        -.75-.25*np.exp(-.05*t)*np.cos(self.harmo_omega*t), 
                                                        0., ]])
    self.harmo_get_true_velocs = lambda t: np.asarray([[-.25*np.exp(-.05*t)*(.05*np.cos(self.harmo_omega*t) + self.harmo_omega*np.sin(self.harmo_omega*t)), 
                                                         .25*np.exp(-.05*t)*(.05*np.cos(self.harmo_omega*t) + self.harmo_omega*np.sin(self.harmo_omega*t)), 
                                                         0., ]])
    # for test_energy_conservation.
    self.econs_particles  = mdpy.particles.Particles(masses=np.ones((1, ))*2., coordinates=np.asarray([[1., 1., 0.]]))
    self.econs_potential  = HarmonicPotential3D(kx=1., ky=0., kz=0., x0=.75, y0=.75, z0=.75)
    self.econs_integrator = mdpy.integrators.LangevinIntegrator(timestep=.01, friction=0., temperature=0.)
    
  def tearDown(self):
    # for test_initialize_velocity.
    del self.initv_particles
    del self.initv_potential
    del self.initv_integrator
    # for test_harmonic_oscillator_solution.
    del self.harmo_particles
    del self.harmo_potential
    del self.harmo_integrator
    del self.harmo_omega
    del self.harmo_get_true_coords
    del self.harmo_get_true_velocs
    # for test_energy_conservation.
    del self.econs_particles
    del self.econs_potential
    del self.econs_integrator

  def test_initialize_velocity(self):
    for _ in range(100):
      # change coordinates and compute forces.
      self.initv_particles.coordinates = np.random.randn(self.initv_particles.num_particles, 3) * np.random.randint(1, 1000)
      forces = self.initv_potential.compute_forces(coordinates=self.initv_particles.coordinates)
      # init velocities.
      self.initv_integrator.initialize_velocites(particles=self.initv_particles, forces=forces)
      # forward shift: v(0.dt) = v(-.5dt) + .5 dt f / m
      v_full = self.initv_particles.velocities + .5 * self.initv_integrator.dt * forces / self.initv_particles.masses
      assert np.allclose(v_full, np.zeros(forces.shape))

  def test_harmonic_oscillator_solution(self):
    for _ in range(2000):
      # ref coordinates and velocities.
      t = self.harmo_integrator.dt * _
      ref_coordinates = self.harmo_get_true_coords(t=t)
      ref_velocities  = self.harmo_get_true_velocs(t=t)
      # test.
      forces = self.harmo_potential.compute_forces(coordinates=self.harmo_particles.coordinates)
      coordinates = self.harmo_particles.coordinates
      velocities  = self.harmo_particles.velocities + .5 * self.harmo_integrator.dt * forces / self.harmo_particles.masses
      # test atol=.02: https://github.com/openmm/openmm/blob/master/tests/TestLangevinIntegrator.h
      assert np.allclose(coordinates, ref_coordinates, rtol=0., atol=.02)
      assert np.allclose(velocities , ref_velocities , rtol=0., atol=.02)
      self.harmo_integrator.step(particles=self.harmo_particles, forces=forces)

  def test_energy_conservation(self):
    forces = self.econs_potential.compute_forces(coordinates=self.econs_particles.coordinates)
    # for _ in range(1000):
    #   forces = self.econs_potential.compute_forces(coordinates=self.econs_particles.coordinates)
    #   self.econs_integrator.step(particles=self.econs_particles, forces=forces)

    forces = self.econs_potential.compute_forces(coordinates=self.econs_particles.coordinates)
    energy_init = (  self.econs_integrator.compute_kinetic_energy(particles=self.econs_particles, forces=forces) 
                   + self.econs_potential.compute_energy(coordinates=self.econs_particles.coordinates) )
    
    for _ in range(5000):
      forces = self.econs_potential.compute_forces(coordinates=self.econs_particles.coordinates)
      energy = (  (k:=self.econs_integrator.compute_kinetic_energy(particles=self.econs_particles, forces=forces))
                + (u:=self.econs_potential.compute_energy(coordinates=self.econs_particles.coordinates)) )
      assert np.allclose(energy, energy_init, rtol=0., atol=5e-6), energy-energy_init
      self.econs_integrator.step(self.econs_particles, forces=forces)