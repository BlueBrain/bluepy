.. include:: _common.rst


Migrations
----------

During its history, |name| has been in constant evolution and the application programming interface (API)
evolved with the new needs. This page is here to simplify the different migrations of the user codes
with the different versions of |name|.


Moving from v2.0.0 to superior to v2.3.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the newest versions of bluepy, we decided to drop the ``.v2`` API entirely. Dropping the ``.v2``
is a major milestone but needed to be done at some point. Hopefully, since ``bluepy==2.0.0``, the
``.v2`` is just a rerouting of the normal classes. So you can just remove all the ``.v2`` seamlessly
and use the normal classes.


.. code-block:: python

    >>> from bluepy.v2 import Circuit
    >>> circuit = Circuit('CircuitConfig')

becomes :

.. code-block:: python

    >>> from bluepy import Circuit
    >>> circuit = Circuit('CircuitConfig')

We also introduced in ``v2.3.0`` morphio as the official morphology reader. If this change breaks your
codes, you can convert morphio morphologies to ``neuroM`` morphologies using :

.. code-block:: python

    >>> from bluepy import Circuit
    >>> import neurom

    >>> circuit = Circuit("CircuitConfig")
    >>> gid = 1
    >>> morph = neurom.load_neuron(circuit.morph.get(gid))   # returns a NeuroM object

or :
.. code-block:: python

    >>> from bluepy import Circuit
    >>> import neurom

    >>> circuit = Circuit("CircuitConfig")
    >>> gid = 1
    >>> filepath = circuit.morph.get_filepath(gid)
    >>> morph = neurom.load_neuron(filepath)   # returns a NeuroM object


Moving from v0.16.0 to v2.0.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first breaking change is the migration from the *v0.16.0* to the *v2.0.0*. The *v2.0.0* brings
two major breaking features : the Python2 support drop and the API v1 removal. Plus, the effective
removal of many deprecated features of bluepy. We will try to indicate what are all these changes here.

.. note::

    Summary for the migration:

    - Can't use python2 anymore
    - Can't use the v1 API anymore
    - Still able to use the ``.v2`` but will throw warnings
    - Strongly recommend to simply remove the ``.v2`` from your codes
    - Need to migrate the regex queries from ``{"<property>": '@<regex>'}`` to ``{"<property>": {'$regex': '<regex>'}}`` (<property> and <regex> being a random property and regex respectively).
    - Need to migrate ``_PRE/POST_SEGMENT_ID`` and ``_PRE/POST_SEGMENT_OFFSET`` to ``Synapse.PRE/POST_SEGMENT_ID`` and ``Synapse.PRE/POST_SEGMENT_OFFSET``


Below you will find some more in depth explanations.

Removal of Python 2
^^^^^^^^^^^^^^^^^^^

You will not be able to update bluepy to a version superior to *v0.16.0* using Python 2. The support
for python has been over since the 1st January 2020 and a lot of our main backends stopped supporting
python2 : h5py, pandas, matplotlib.


Removal of the v1 API
^^^^^^^^^^^^^^^^^^^^^

You will not be able to use the old bluepy API. This includes :

- bluepy/api.py
- bluepy/extrator.py
- bluepy/morphology sub-package
- bluepy/parsers sub-package
- bluepy/synapses sub-package
- bluepy/targets sub-package
- bluepy/geometry/mosaic.py
- bluepy/geometry/hexagons.py
- bluepy/geometry/pickled.py

They were Python 2 only old codes, working only for the older versions of the circuits. If you
really need them you can still use the *v0.16.0* version of |name| on devpi or in Spack.

v2 API becomes the main API
^^^^^^^^^^^^^^^^^^^^^^^^^^^

It was confusing to keep both v1 and v2 in |name|. The v1 api is deprecated for a long time and was still
here only for the oldest circuits and analyses. Hopefully, `spack` , `devpi` or even
`git` can be used to retrieve the old versions of |name| and rerun the historical circuits/analyses.

So from now on:

- The v2 API becomes main |name| API. It means you can now do :

.. code-block:: python

    >>> from bluepy import Circuit
    >>> circuit = Circuit('CircuitConfig')

and the new API will be used.

- You can still use :

.. code-block:: python

    >>> from bluepy.v2 import Circuit
    >>> circuit = Circuit('CircuitConfig')

But this will be strictly identical to :

.. code-block:: python

    >>> from bluepy import Circuit
    >>> circuit = Circuit('CircuitConfig')

Except you will have warning messages saying to use the version without the ``v2``.

So, to operate the migration, the only thing to do is to remove all ``v2`` from your codes.

Cell queries with regex
^^^^^^^^^^^^^^^^^^^^^^^

In the versions prior to *v2.0.0*, multiple syntax for the regexp queries coexisted. This
included :

    >>> from bluepy import Circuit
    >>> cells = Circuit('CircuitConfig').cells
    >>> cells.ids({Cell.MTYPE: '@.*BP'}})
    >>> cells.ids({Cell.MTYPE: 'regex:.*BP'}})
    >>> cells.ids({Cell.MTYPE: {'$regex': '.*BP'}})

All queries were equivalent: select the gids which mtypes match the regex ".*BP".

Having multiple ways of doing the same thing is confusing and can lead to problems. Also using values
included directly inside the "string query" can be error prone. So we decided to separate clearly the
query action (using a regex) and the argument. So the only remaining syntax is now :

.. code-block:: python

    >>> from bluepy import Circuit
    >>> cells = Circuit('CircuitConfig').cells
    >>> cells.ids({Cell.MTYPE: {'$regex': '.*BP'}})

This is also more future ready and we can envision a "à la mongo" type of complex queries.

.. code-block:: python

    >>> from bluepy import Circuit
    >>> cells = Circuit('CircuitConfig').cells
    >>> cells.ids({Cell.MTYPE: {'$complex_query': [arg1, arg2, ...]}})


Synapse _PRE/POST_SEGMENT_ID and _PRE/POST_SEGMENT_OFFSET
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``_PRE/POST_SEGMENT_ID`` and ``_PRE/POST_SEGMENT_OFFSET`` were some kind of hidden synapse variables and
were a special case. There is no reason to keep these variables hidden so we added them to the
Synapse enum. To use them properly in the *v2.0.0* you need to :

.. code-block:: python

    >>> from bluepy import Circuit
    >>> from bluepy import Synapse
    >>> connectome = Circuit('CircuitConfig').connectome
    >>> connectome.synapse_properties(edge_ids, [Synapse.POST_SEGMENT_ID, Synapse.POST_SEGMENT_OFFSET])
