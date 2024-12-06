Simulation
----------

We use term `simulation` to address the result of a dynamic simulation of the network, i.e. spikes, voltage traces and other binary reports recording neural activity.

Each simulation is defined by *BlueConfig*, a file defining static structure of network, configuring how to run the simulation (simulation duration, conditions, etc); and what to observe and record (spike report, binary reports).

Please refer to `this <https://sonata-extension.readthedocs.io/en/latest/blueconfig.html>`_ for the details how BlueConfig is organized.

First we must obtain a ``Simulation`` object.
We will use some simulation stored at BBP GPFS as an example:

.. code-block:: python

    >>> from bluepy import Simulation
    >>> sim = Simulation('/gpfs/bbp.cscs.ch/project/proj64/circuits/test/simulations/001/BlueConfig')
    >>> type(sim)
    bluepy.simulation.Simulation

As every simulation is based upon some circuit we can use the ``circuit`` attribute to obtain the underlying ``Circuit`` object:

.. code-block:: python

    >>> sim.circuit
    <bluepy.circuit.Circuit at 0x1da72d0>

The most important aspect of a simulation is Reports API which gives access to underlying reports that contain the simulation data acquired.

Simulation report location
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One needs to define carefully the simulation output location inside the BlueConfig. The
location is determined by the fields OutputRoot/CurrentDir inside the BlueConfig. The rationale in BluePy to
find the simulation reports is:

* uses OutputRoot directly if it's an abspath
* if OutputRoot is a relative path (which is deprecated) then use CurrentDir as parent directory for OutputRoot.
* if OutputRoot is a relative path and CurrentDir is not defined then the directory containing the BlueConfig is used as parent directory
* if the CurrentDir is not defined and if the BlueConfig has not been created from a file it raises
* if the output directory cannot be found it raises.

If all the requirements are ticked then you can access your report via the BluePy API.

Spike report
~~~~~~~~~~~~

``SpikeReport`` is a special type of report which allows you to obtain spike times for all GIDs:

.. code-block:: python

    >>> sim.spikes.get()
    # return result is pandas Series with GIDs indexed by spike times
    t
    0.0 30
    1.0 20
    1.1 20
    1.9 10
    ...

or for some specific GIDs:

.. code-block:: python

    >>> sim.spikes.get(gids=[10, 20])
    >>> data.head(3)
    t
    1.0 20
    1.1 20
    1.9 10
    ...

or for a single GID:

.. code-block:: python

    >>> sim.spikes.get_gid(100)
    # return result is an array with GID spike times (empty if no spikes)
    array([ 26.925, 54.75, 398.175, 481.875, 534.35 ])

If you are interested in spikes within given time range:

.. code-block:: python

    >>> sim.spikes.get(t_end=100)
    >>> sim.spikes.get_gid(t_start=1, t_end=100)

.. note::

    For all query methods spike times returned are sorted in increasing order.

Finally, to get sorted unique list of all spiking GIDs:

.. code-block:: python

    >>> sim.spikes.gids
    array([ 10, 20, 30,... ])


Binary reports
~~~~~~~~~~~~~~

``sim.report(name)`` method would return an object for binary report access:

.. code-block:: python

  >>> sim.report('soma')
  bluepy.impl.compartment_report.SomaReport


SomaReport
^^^^^^^^^^

.. code-block:: python

  >>> report = sim.report('soma')
  >>> report.gids
  >>> array([1, 2, ...], dtype=uint32)
  >>> report.meta
  >>> {'start_time': 0.0, 'end_time': 10.0, 'time_step': 0.1, ...}
  >>> data = report.get(t_start=0, t_end=3.0, gids=[62693, 62694])
  >>> data.shape
  >>> (30, 2)
  >>> data.head(3)
  >>> t/gid    62693      62694
      0.0 -64.966301 -64.830650
      0.1 -64.945824 -64.756416
      0.2 -64.679489 -64.611526

The return result is ``pandas.DataFrame`` where rows correspond to timesteps and columns -- to GIDs.

Downsampling
^^^^^^^^^^^^

To reduce the size of the fetched data, it could be downsampled using ``t_step`` optional argument (it should be a multiple of time step defined in the report).
Only the frames corresponding to the specified time slices would be loaded from binary report file.

.. code-block:: python

  >>> data = report.get(t_start=0, t_end=3.0, t_step=1.0, gids=[62693, 62694])
  >>> data.shape
  >>> (3, 16232)
  >>> data.head(3)
  >>> t/gid    62693      62694
      0.0 -64.966301 -64.830650
      1.0 -64.945824 -64.756416
      2.0 -64.679489 -64.611526

Slicing the result by time range and/or GID
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    >>> data[t0:t1]
    # pandas DataFrame with all measurements within (t_start, t_end)
    >>> data[gid]
    # pandas Series with all `gid` measurements
    >>> data[t0:t1][gid]
    # pandas Series with `gid` measurements within (t_start, t_end)
    >>> data[gid][t0:t1]
    # same as above

.. note::
    While indexing by simulation time provides a convenient way to slice the data, it comes at the cost of floating point precision gotchas.
    When slicing by ``(t0, t1)`` time range, ``t1`` frame is sometimes included and other times not.
    Please do not rely on the exact number of time frames in a slice or add some margin.

Querying single GID
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    >>> data = report.get_gid(62693)
    >>> data.head(3)
    >>> t
      0.0 -64.966301
      0.1 -64.945824
      0.2 -64.679489

Report representation
^^^^^^^^^^^^^^^^^^^^^

The report data can be produced in different formats. Currently three formats are
supported: `Bin` (`.bbp` in-house binary format), `HDF5` with a specific structure and `SONATA`.
Today the preferred format is the `SONATA` format.

Since ``bluepy>=2.4.0`` the format is read directly from the BlueConfig using the `Format` key in the Report sections.

.. note::
    The `source` option from the ``simulation.report()`` function has been removed since ``bluepy==2.4.0``.
    Using `source` will not break your code but a warning will be thrown and the source value ignored.


CompartmentReport
^^^^^^^^^^^^^^^^^

Similar to ``SomaReport``, but columns in ``get()`` result are ``(Cell.ID, Section.ID)`` pairs.


SynapseReport
^^^^^^^^^^^^^

Similar to ``SomaReport``, but columns in ``get()`` result are synapse IDs.

In addition to the methods described for `SomaReport`, there is one for querying only synapses of interest:

.. code-block:: python

    >>> data = report.get_synapses(<synapse-ids>)

Depending on your needs, synapse properties could be joined with synapse report data one way

.. code-block:: python

    >>> data = report.get_gid(post_gid)
    >>> properties = sim.circuit.connectome.synapse_properties(measurements.columns.values, [Synapse.G_SYNX])

or another:

.. code-block:: python

    >>> properties = sim.circuit.connectome.afferent_synapses(post_gid, [Synapse.G_SYNX])
    >>> data = report.get_synapses(properties.index)

or even:

.. code-block:: python

    >>> synapse_ids = sim.circuit.connectome.pair_synapses(pre_gid, post_gid)
    >>> properties = sim.circuit.connectome.synapse_properties(synapse_ids, [Synapse.G_SYNX])
    >>> data = report.get_synapses(synapse_ids)
