API reference
=============

Interpolant classes
-------------------

.. autoclass:: chebyshevnd.ChebyshevInterpolant1D
   :members:
   :special-members: __init__

.. autoclass:: chebyshevnd.ChebyshevInterpolant2D
   :members:
   :special-members: __init__

.. autoclass:: chebyshevnd.ChebyshevInterpolant3D
   :members:
   :special-members: __init__

.. autoclass:: chebyshevnd.ChebyshevInterpolant4D
   :members:
   :special-members: __init__

.. autoclass:: chebyshevnd.ChebyshevInterpolantND
   :members:
   :special-members: __init__

.. autoclass:: chebyshevnd.ChebyshevBatch
   :members:
   :special-members: __init__

Physical coordinates: domain maps
---------------------------------

A :class:`~chebyshevnd.MappedInterpolant` evaluates a table in physical
coordinates, through one :class:`~chebyshevnd.AxisMap` per axis, so a table
can cover a region that is not a box (e.g. above a separatrix, or around a
reference value that depends on another coordinate).

.. autoclass:: chebyshevnd.AxisMap
   :members:
   :special-members: __init__

.. autoclass:: chebyshevnd.MappedInterpolant
   :members:
   :special-members: __init__

.. autoclass:: chebyshevnd.MappedBatch
   :members:

Saving and loading
------------------

Tables and grids are stored in HDF5 (needs ``h5py``:
``pip install chebyshevnd[io]``); the layout is described in
:doc:`file_format`.

.. autofunction:: chebyshevnd.save

.. autofunction:: chebyshevnd.load

.. autofunction:: chebyshevnd.read_metadata

Grids and sampling
------------------

.. autoclass:: chebyshevnd.Axis

.. autoclass:: chebyshevnd.ChebyshevGrid
   :members:

.. autoexception:: chebyshevnd.OutOfDomainError

Evaluation from compiled code (Numba)
-------------------------------------

For an ODE right-hand side written with Numba: get the table's arrays with
:meth:`ChebyshevInterpolantND.raw <chebyshevnd.ChebyshevInterpolantND.raw>`,
a workspace with :meth:`RawTable.workspace <chebyshevnd.RawTable.workspace>`,
and evaluate inside ``@njit`` code with these functions.

.. autofunction:: chebyshevnd.eval_into

.. autofunction:: chebyshevnd.eval_grad_into

.. autoclass:: chebyshevnd.RawTable
   :members: ndim, n_outputs, dtype, workspace

.. autoclass:: chebyshevnd.Workspace

.. autofunction:: chebyshevnd.eval_physical_into

.. autofunction:: chebyshevnd.eval_grad_physical_into

.. autoclass:: chebyshevnd.RawMappedTable
   :members: ndim, n_outputs, dtype, workspace

Batched evaluation (NumPy and CuPy)
-----------------------------------

.. autofunction:: chebyshevnd.evaluate_batch

.. autofunction:: chebyshevnd.value_and_grad_batch

.. autofunction:: chebyshevnd.batched.chebyshev_basis_batch

.. autofunction:: chebyshevnd.batched.precontract_tables

Node generation
---------------

.. autofunction:: chebyshevnd.chebyshev_nodes

.. autofunction:: chebyshevnd.chebyshev_gauss_lobatto_nodes

Coefficient computation
-----------------------

.. autofunction:: chebyshevnd.compute_chebyshev_coefficients_nd

.. autofunction:: chebyshevnd.chebyshev_values_at_nodes_nd

.. autofunction:: chebyshevnd.renormalize_coefficients_Chebyshev

.. autofunction:: chebyshevnd.renormalize_coefficients_GL

Derivatives
-----------

.. autofunction:: chebyshevnd.differentiate_coefficients

Clenshaw evaluation kernels
---------------------------

.. autofunction:: chebyshevnd.clenshaw_1d

.. autofunction:: chebyshevnd.clenshaw_2d

.. autofunction:: chebyshevnd.clenshaw_3d

.. autofunction:: chebyshevnd.clenshaw_4d

.. autofunction:: chebyshevnd.clenshaw_nd

Coefficient pruning
-------------------

.. autofunction:: chebyshevnd.prune_trailing_slices_greedy_nd

.. autofunction:: chebyshevnd.find_significant_coeffs_1d

.. autofunction:: chebyshevnd.find_significant_coeffs_2d
