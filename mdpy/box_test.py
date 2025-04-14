"""Unit tests for the `mdpy.box` module."""
# Authors: Zilin Song.


import unittest
import numpy as np
import torch

import mdpy.box


class PBCBox(unittest.TestCase):
  r"""Test cases for `mdpy.box.PBCBox()`."""
  
  def setUp(self):
    nparticles = 250
    xdim = float(np.random.randint(1, 10))
    ydim = float(np.random.randint(1, 10))
    zdim = float(np.random.randint(1, 10))
    dims = np.expand_dims(np.asarray([xdim, ydim, zdim]), axis=0)
    # reusable objects.
    self.box = mdpy.box.PBCBox(xdim=xdim, ydim=ydim, zdim=zdim)
    assert (dims==self.box.dims).all(), f"Illegal inequal dims: {dims} != {self.box.dims}"
    self.cor_in_box  = (np.random.rand (nparticles, 3)-.5) * dims
    assert (self.cor_in_box<= self.box.dims/2).all(), f"Illegal coordinates out box."
    assert (self.cor_in_box>=-self.box.dims/2).all(), f"Illegal coordinates out box."
    self.cor_out_box = np.random.randn(nparticles, 3)*np.random.randint(0, 10, (nparticles, 3))*dims
    assert (self.cor_out_box>= self.box.dims/2).any(), f"Illegal coordinates in box."
    assert (self.cor_out_box<=-self.box.dims/2).any(), f"Illegal coordinates in box."

  def tearDown(self):
    del self.box
    del self.cor_in_box
    del self.cor_out_box

  def test_wrap(self):
    # unwrap in box coordinates by shifting across periodic boundaries multiple times.
    coords_to_wrap  = np.copy(self.cor_in_box)
    coords_to_wrap += np.random.randint(-99, 100, coords_to_wrap.shape)*self.box.dims
    assert (coords_to_wrap>= self.box.dims/2).any()
    assert (coords_to_wrap<=-self.box.dims/2).any()
    # test.
    coords_wrapped = self.box.wrap(coordinates=coords_to_wrap)
    assert np.allclose(coords_wrapped, self.cor_in_box, rtol=0., atol=1e-8)

  def test_compute_distances(self):
    # ref x_ij.
    N = self.cor_out_box.shape[0]
    ref_x_i  = np.copy(self.cor_out_box)    # [N, 3]
    ref_x_ij = np.tile(ref_x_i, [N, 1, 1])  # [N, N, 3]
    # ref: shift all coordinates such that all x[i, i, :] is centered at the origin.
    ref_x_ij -= np.repeat(np.expand_dims(ref_x_i, axis=1), repeats=N, axis=1)
    ref_x_ij  = self.box.wrap(coordinates=ref_x_ij.reshape(-1, 3)).reshape(N, N, 3)
    ref_d_ij  = np.linalg.norm(ref_x_ij, ord=2, axis=-1)
    # test.
    d_ij = self.box.compute_distances(coordinates=np.copy(self.cor_out_box), return_grad=False)
    assert np.allclose(d_ij, ref_d_ij, rtol=1e-8, atol=0.)


  def test_compute_distances_grad(self):
    # ref x_ij
    ref_x_ij = torch.tensor(self.box.wrap(coordinates=np.copy(self.cor_out_box)))
    ref_x_ij.requires_grad = True
    # ref: shift across boundary if dx_ij is larger than boxdim/2 (shifted is `ref_dx_ij_neg`).
    ref_dims = torch.tensor(self.box.dims).unsqueeze(0)                             # [1, 1, 3]
    ref_dx_ij_pos = ref_x_ij.unsqueeze(1) - ref_x_ij.unsqueeze(0).clone().detach()  # [N, N, 3]
    ref_dx_ij_neg = (ref_dims-torch.abs(ref_dx_ij_pos)) * -torch.sign(ref_dx_ij_pos)# dx>0: dx=-dx
    ref_dx_ij = torch.where(torch.abs(ref_dx_ij_pos)<(ref_dims/2.), ref_dx_ij_pos, ref_dx_ij_neg)
    ref_d_ij = torch.norm(ref_dx_ij, p=2, dim=-1)
    torch.sum(ref_d_ij).backward()
    # test.
    d_ij, g_ij = self.box.compute_distances(coordinates=self.cor_out_box, return_grad=True)
    g_i = np.sum(g_ij, axis=1)
    assert np.allclose(d_ij, ref_d_ij.detach().numpy(), rtol=1e-8, atol=0.)
    assert np.allclose(g_i , ref_x_ij.grad    .numpy(), rtol=1e-8, atol=0.)