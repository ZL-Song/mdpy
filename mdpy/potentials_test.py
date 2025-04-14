"""Unit tests for the `mdpy.potentials` module."""
# Authors: Zilin Song.


import unittest
import numpy as np
import torch

import mdpy.box
import mdpy.potentials


class LJ126Test(unittest.TestCase):
  """Test cases for `mdpy.potentials.LJ126()`."""

  def setUp(self):
    nparticles = 4
    xdim = float(np.random.randint(1, 10))
    ydim = float(np.random.randint(1, 10))
    zdim = float(np.random.randint(1, 10))
    dims = np.expand_dims(np.asarray([xdim, ydim, zdim]), axis=0)
    # reusable objects.
    self.cor = (np.random.rand (nparticles, 3)-.5) * dims
    self.box = mdpy.box.PBCBox(xdim=xdim, ydim=ydim, zdim=zdim)
    self.lj_sig = np.abs(np.random.randint(1, 10) * np.random.randn())
    self.lj_eps = np.abs(np.random.randint(1, 10) * np.random.randn())
    self.lj_126 = mdpy.potentials.LJ126(sigma=self.lj_sig, epsilon=self.lj_eps)

  def tearDown(self):
    del self.cor
    del self.box
    del self.lj_sig
    del self.lj_eps
    del self.lj_126

  def test_compute_energy(self):
    # ref ener.
    N = self.cor.shape[0]
    ref_ener = 0.
    for i in range(N):
      ref_xi_j = np.copy(self.cor[i:, :]) # [N-i, 3]
      # ref: shift xi_j coordinates such that all xi is centered at the origin.
      ref_xi_j -= ref_xi_j[0, :].reshape(1, 3)
      ref_xi_j  = self.box.wrap(coordinates=ref_xi_j)
      ref_di_j  = np.linalg.norm(ref_xi_j[1:, :], ord=2, axis=-1)
      ref_sig_d = self.lj_sig / ref_di_j
      ref_ener += np.sum(4. * self.lj_eps * (np.power(ref_sig_d, 12) - np.power(ref_sig_d, 6)))
    # test.
    ener = self.lj_126.compute_energy(coordinates=self.cor, box=self.box)
    assert np.allclose(ener, ref_ener, rtol=1e-8, atol=0.)

  def test_compute_forces(self):
    # ref grad.
    N = self.cor.shape[0]
    ref_x_ij = torch.tensor(self.box.wrap(coordinates=np.copy(self.cor)))
    ref_x_ij.requires_grad = True
    # ref: shift across boundary if dx_ij is larger than boxdim/2 (shifted as `ref_dx_ij_neg`).
    ref_dims = torch.tensor(self.box.dims).unsqueeze(0)           # [1, 1, 3]
    ref_dx_ij_pos = ref_x_ij.unsqueeze(1) - ref_x_ij.unsqueeze(0) # [N, N, 3], NOTE: x_j on graph.
    ref_dx_ij_neg = (ref_dims-torch.abs(ref_dx_ij_pos)) * -torch.sign(ref_dx_ij_pos)  # dx>0: dx=-dx
    ref_dx_ij = torch.where(torch.abs(ref_dx_ij_pos)<(ref_dims/2.), ref_dx_ij_pos, ref_dx_ij_neg)
    ref_d_ij = torch.norm(ref_dx_ij, p=2, dim=-1)                 # [N, N]
    # ref: mask-out k<1 distances to prevent double counting the distances.
    ref_d_ij = torch.flatten(ref_d_ij*torch.triu(torch.ones((N, N)), diagonal=1))
    ref_sig_d = self.lj_sig / torch.take(ref_d_ij, torch.nonzero(ref_d_ij))
    ref_ener  = torch.sum(4. * self.lj_eps * (torch.pow(ref_sig_d, 12) - torch.pow(ref_sig_d, 6)))
    ref_ener.backward()
    # test.
    ener = self.lj_126.compute_energy(coordinates=self.cor, box=self.box)
    forc = self.lj_126.compute_forces(coordinates=self.cor, box=self.box)
    assert np.allclose(ener,  ref_ener.detach().numpy(), rtol=1e-8, atol=0.)
    assert np.allclose(forc, -ref_x_ij.grad    .numpy(), rtol=1e-8, atol=0.)