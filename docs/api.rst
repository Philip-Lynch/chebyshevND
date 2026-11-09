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

.. autoexception:: chebyshevnd.OutOfDomainError

Batched evaluation (NumPy and CuPy)
-----------------------------------

.. autofunction:: chebyshevnd.evaluate_batch

.. autofunction:: chebyshevnd.batched.chebyshev_basis_batch

.. autofunction:: chebyshevnd.batched.precontract_tables

Node generation
---------------

.. autofunction:: chebyshevnd.chebyshev_nodes

.. autofunction:: chebyshevnd.chebyshev_gauss_lobatto_nodes

Coefficient computation
-----------------------

.. autofunction:: chebyshevnd.compute_chebyshev_coefficients_nd

.. autofunction:: chebyshevnd.renormalize_coefficients_Chebyshev

.. autofunction:: chebyshevnd.renormalize_coefficients_GL

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
