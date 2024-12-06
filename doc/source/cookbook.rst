.. include:: _common.rst

Cookbook
========

Some common usage patterns; as well as tips & tricks for effective usage of |name|.

Get unique me-types
-------------------

.. code-block:: python

    >>> metypes = circuit.cells.get(properties=[Cell.MTYPE, Cell.ETYPE]).drop_duplicates()

If you'd like to convert the result DataFrame to a list of tuples:

    >>> list(metypes.itertuples(index=False, name=None))


Joining Synapse properties table with Cell properties
-----------------------------------------------------

Suppose we'd like to know presynaptic cell mtype for some synapses.


Given a DataFrame with synapse properties, one of which is ``Synapse.PRE_GID``:

.. code-block:: python

    >>> synapses = circuit.connectome.afferent_synapses(42, [Synapse.PRE_GID])


one of the ways to add 'mtype' column with presynaptic mtype would be:

.. code-block:: python

    >>> mtypes = circuit.cells.get(properties=Cell.MTYPE)
    >>> synapses = synapses.join(mtypes, on=Synapse.PRE_GID)
