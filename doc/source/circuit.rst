.. include:: _common.rst

Circuit
-------

The term `circuit` is used to address the static structure of a neural network.

Each circuit is defined by *CircuitConfig*, a file defining paths to different files constituting a circuit.

For a brief overview of those files, please refer to the following pages:

- `Circuit Files <https://sonata-extension.readthedocs.io/en/latest/>`_
- `Circuit Building <https://bbpteam.epfl.ch/documentation/projects/circuit-build/latest/index.html>`_

In the versions prior to *v2.0.0*, two different APIs lived altogether inside the library.
They were the so called v1 and v2 APIs. To simplify |name|, the historical *v1* has been
dropped entirely in the *v2.0.0*. Nowadays, the v2 api becomes the main and only API for |name|.

.. warning::
    The v2 API has been hard deprecated in the v2.0.0 and has been removed in the v2.3.0 version.

First we'll need to instantiate ``Circuit`` object passing it a path to CircuitConfig.
We will use a test circuit stored on BBP GPFS as an example:

.. code-block:: python

    >>> from bluepy import Circuit
    >>> c = Circuit('/gpfs/bbp.cscs.ch/project/proj64/circuits/test/CircuitConfig')

``Circuit`` methods are grouped into "namespaces":

- *cells*, for querying cell properties
- *connectome*, for querying circuit connectivity and synapse properties
- *projection*, for querying circuit connectivity and synapse properties related to projections
- *morph*, for access to detailed cell morphologies
- *emodels*, for access to cell electrical models
- *subcellular*, for access to subcellular data
- *stats*, miscellaneous circuit statistics

Here we briefly describe the methods provided in each namespace and give usage examples.
Please refer to method docstrings for the details.

cells
~~~~~

Querying cell properties.

cells.available_properties
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Returns a python set() containing all the properties present/queryable from the cell file.
The meaning of these properties is documented in :ref:`Constants`

.. code-block:: python

    >>> c.cells.available_properties
    {'y', 'minicolumn', 'rotation_angle_xaxis', '@dynamics:propA', ... }


cells.get()
^^^^^^^^^^^

Returns a Pandas DataFrame with cell properties for GIDs matching given *cell group*.

Cell group could be:

- None (i.e., all circuit GIDs)
- single GID
- list of GIDs
- target name (string)
- a dict defining cell properties filter

The last one contains key-value pairs, where key is one of cell properties stored in MVD file, and value could be:

- 2-elements tuple defining range for floating point properties
- a single value or a list of values (OR query) for all the rest
- a dictionary specifying *complex filter*

At the moment we support single type of complex filter: using regular expression.

More specifically,

.. code-block:: python

    >>> {<property>: {'$regex': <pattern>}}

will try to match ``<property>`` values using ``<pattern>`` as a regular expression.

Full match is used, i.e this filter will not match all L5 mtypes:

.. code-block:: python

    >>> {'mtype': {'$regex': 'L5'}}

while this one will:

.. code-block:: python

    >>> {'mtype': {'$regex': 'L5.*'}}


Typical cell properties names are defined as :ref:`Constants` within `bluepy.enums.Cell`:

.. code-block:: python

    >>> from bluepy import Cell
    >>> Cell.X, Cell.MTYPE

If you'd like to combine cell properties filter with target selection, you could use *meta property* ``$target``.

Restrictions imposed by key-values pairs are joined with AND.

Examples:

.. code-block:: python

    # All properties of cell GID=42
    >>> c.cells.get(42)
    x                                                          366.655
    y                                                          559.569
    z                                                           583.47
    orientation      [[0.8746336471305276, 0.0, 0.4847844709839124]...
    etype                                                         bNAC
    hypercolumn                                                      2
    ...

.. code-block:: python

    # GID=42 mtype
    >>> c.cells.get(42, Cell.MTYPE)
    'L6_SBC'

.. code-block:: python

    # mtype and etype for cells from 'mc2_Column' target AND L6 region
    >>> c.cells.get({'$target': 'mc2_Column', Cell.REGION: 'L6'}, properties=[Cell.MTYPE, Cell.ETYPE]).head(3)
        mtype  etype
    3   L6_BP    bAC
    4   L6_BP  dSTUT
    17  L6_BP    bAC

.. code-block:: python

    # All properties for cells with 'L23_MC' mtype AND X ranging from 100 to 200
    >>> c.cells.get({Cell.MTYPE: 'L23_MC', Cell.X: (100, 200)})
    ...


cells.ids()
^^^^^^^^^^^

GIDs matching given cell group.

*Cell groups* are defined analogous to ``cells.get``.

Optional argument supported:

- `limit`: return no more than N gids from match result
- `sample`: randomly sample N gids from match result

.. code-block:: python

    >>> gids = c.cells.ids('L23_MC', limit=4)
    >>> gids
    [919, 921, 1058, 1067]
    >>> properties = c.cells.get(gids)


cells.count()
^^^^^^^^^^^^^

Total number of cells for a given cell group.

*Cell groups* are defined analogous to ``cells.get``.

.. code-block:: python

    >>> c.cells.count('L23_MC')
    101


cells.targets
^^^^^^^^^^^^^

Set of predefined cell group names (as specified in .target files belonging to the circuit.)

.. code-block:: python

    >>> sorted(c.cells.targets)
    ['All', 'Excitatory', 'Inhibitory', 'L1_DAC', 'L1_HAC',...]


The possible types for the targets are now : Cell, Compartment and Section. Only cell ids are
returned when using ``c.cells.targets`` for Compartment and Section targets.

The Section targets must follow the rules:

- having two section targets (ex : ``my_target soma axon``) is not allowed.
- not having a target associated to a section target is prohibited.
- the supported section targets are : ``axon``, ``soma`` only.

cells.mtypes
^^^^^^^^^^^^

Set of cell mtypes in the circuit.

.. code-block:: python

    >>> c.cells.mtypes
    {'L1_DAC', 'L1_HAC', ...}


cells.etypes
^^^^^^^^^^^^

Set of cell etypes in the circuit.

.. code-block:: python

    >>> c.cells.etypes
    {'bAC', 'bIR', ...}


cells.spatial_index
^^^^^^^^^^^^^^^^^^^

Access to FLATIndex spatial index for somas.


connectome
~~~~~~~~~~

Querying circuit connectivity and synapse properties.


connectome.available_properties
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Returns a python set() containing all the properties present/queryable from the connectome file.
Typical cell properties names are defined as :ref:`Constants` within `bluepy.enums.Synapses`:

.. code-block:: python

    >>> c.connectome.available_properties
    {<Synapse.F_SYN: 'f_syn'>, <Synapse.TYPE: 'type'>, <Synapse.POST_Z_CONTOUR: 'post_z_contour'>, ... }


connectome.afferent_gids()
^^^^^^^^^^^^^^^^^^^^^^^^^^

For given GID, get sorted unique array of afferent GIDs.

.. code-block:: python

    >>> c.connectome.afferent_gids(10)
    array([41, 196, 448])


connectome.afferent_synapses()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For given GID, get all afferent synapses.
If `properties` parameter is provided, returns Pandas Series/DataFrame with requested synapse properties,
indexed by synapse IDs.  Otherwise, returns a list of synapse IDs.

.. code-block:: python

    >>> # with a nrn connectome file
    >>> c.connectome.afferent_synapses(10)
    [(10, 0), (10, 1), (10, 2), (10, 3),...]

    >>> # with a sonata connectome file
    >>> c.connectome.afferent_synapses(10)
    [1585, 1586, 1587, 1588,...]

The representation of the synapse ID changed from the nrn to the sonata format. A synapse ID is a
tuple (gid, connection id) within the nrn representation when a single int is used for the sonata format.
We recommend to never use hardcoded synapse IDs and to use bluepy functions to fetch them instead.

connectome.efferent_gids()
^^^^^^^^^^^^^^^^^^^^^^^^^^

For given GID, get sorted unique array of efferent GIDs.

.. code-block:: python

    >>> c.connectome.efferent_gids(10)
    array([34, 56, 90,...])


connectome.efferent_synapses()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For given GID, get all efferent synapses.
If `properties` parameter is provided, returns Pandas Series/DataFrame with requested synapse properties,
indexed by synapse IDs.
Otherwise, returns just a list of synapse IDs.

.. code-block:: python

    >>> # with a nrn connectome file
    >>> c.connectome.efferent_synapses(10)
    [(34, 0), (34, 1), (34, 2),...]

    >>> # with a sonata connectome file
    >>> c.connectome.efferent_synapses(10)
    [5660, 5789, 48442, 89484,...]

connectome.pair_synapses()
^^^^^^^^^^^^^^^^^^^^^^^^^^

Find all the synapses connecting two GIDs (directional).
If `properties` parameter is provided, returns Pandas Series/DataFrame with requested synapse properties,
indexed by synapse IDs.
Otherwise, returns just a list of synapse IDs.

.. code-block:: python

    >>> # with a nrn connectome file
    >>> c.connectome.pair_synapses(10, 90)
    [(90, 0), (90, 1), (90, 2),...]

    >>> # with a sonata connectome file
    >>> c.connectome.pair_synapses(10, 90)
    [1561, 1688, 9944]


connectome.pathway_synapses()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Find all the synapses connecting two *cell groups* (directional).
If `properties` parameter is provided, returns Pandas Series/DataFrame with requested synapse properties,
indexed by synapse IDs.
Otherwise, returns just a list of synapse IDs.

.. code-block:: python

    >>> c.connectome.pathway_synapses('L23_MC', 570)
    [(570, 199), (570, 200), (570, 201),...]


connectome.synapse_properties()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Returns Pandas Series/DataFrame with synapse properties for given synapse IDs.
Available properties are defined in `bluepy.enums.Synapse`.
Depending on the circuit, some of them would be taken directly from synapse properties files, while others calculated "on the fly".

.. code-block:: python

    >>> from bluepy.enums import Synapse
    >>> synapse_ids = c.connectome.efferent_synapses(10)
    >>> c.connectome.synapse_properties(synapse_ids, [Synapse.G_SYNX, Synapse.TOUCH_DISTANCE])
          Synapse.G_SYNX  Synapse.TOUCH_DISTANCE
    10 0        1.377060                1.808825
       1        1.778944                1.361132

The representation of the synapse ID changed from the nrn to the sonata format. A synapse ID is a
tuple (gid, connection id) within the nrn representation when a single int is used for the sonata format.
We recommend to never use hardcoded synapse IDs and to use bluepy functions to fetch them instead.

Using a sonata connectome, indices of the DataFrame will not provide any information about the
pre and/or post gids. If this is an information you want to retrieve, you can use the
``[Synapse.POST_GID, Synapse.PRE_GID]`` properties.

    >>> from bluepy.enums import Synapse
    >>> synapse_ids = c.connectome.efferent_synapses(10)
    >>> c.connectome.synapse_properties(synapse_ids, [Synapse.TOUCH_DISTANCE, Synapse.POST_GID, Synapse.PRE_GID])
          Synapse.TOUCH_DISTANCE   Synapse.POST_GID   Synapse.PRE_GID
    8448                1.808825                540                10
    5454                1.361132               8787                10


connectome.synapse_positions()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Returns Pandas DataFrame with synapse positions for given synapse IDs.

Arguments supported:

- `side`: "pre" for presynaptic, "post" for postsynaptic
- `kind`: "center" for position in center of the segment, "contour" for position on segment surface

.. code-block:: python

    >>> synapse_ids = c.connectome.efferent_synapses(10)
    >>> c.connectome.synapse_positions(synapse_ids, 'pre', 'center')
                   x           y           z
    10 0  293.446259  532.554077  625.940247
       1  281.184174  603.646057  631.733337

connectome.iter_connections()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Iterate through pre -> post connections.
Yields `(pre_gid, post_gid)` tuples.

Optional arguments:

- `unique_gids`: if set to True, no GID would be used more than once
- `shuffle`: if set to True, result order would be (somewhat) randomized
- `return_synapse_count`: add connection synapse count to yield result
- `return_synapse_ids`: add list of connection synapse IDs to yield result

`return_synapse_count` and `return_synapse_ids` are mutually exclusive.

Combining it with ``itertools`` Python module, one could filter the connected pairs, for example:

.. code-block:: python

    >>> from bluepy.utils import take_n
    >>> from itertools import ifilter
    >>> it = c.connectome.iter_connections(
          {Cell.MTYPE: 'L23_MC'}, {Cell.LAYER: 5}, unique_gids=True, return_synapse_count=True
        )
    >>> take_n(ifilter(lambda p: p[2] > 10, it), 5)

will give a list with at most five connections with more than ten synapses, as well as their synapse count.


connectome.spatial_index
^^^^^^^^^^^^^^^^^^^^^^^^

Access to FLATIndex spatial index for synapses.

.. code-block:: python

    >>> c.connectome.spatial_index.q_window((350, 1180, 570), (400, 1190, 580))
       Synapse.PRE_X_CENTER  Synapse.PRE_Y_CENTER  Synapse.PRE_Z_CENTER  ...
    0            368.990143           1182.229858            572.878540
    1            360.406555           1183.802734            575.062744
    2            360.365540           1184.016235            575.084106



projection
~~~~~~~~~~~

A projection is a special type of connectome that corresponds to connections from an external
source. This is usually used to connect two brain regions together.

You can access the projections using:

.. code-block:: pycon

    >>> from bluepy import Circuit
    >>> circuit = Circuit("BlueConfig")
    >>> projection = circuit.projection("projection_name")

The ``projection`` object is exactly the same as a ``connectome`` object and they share the exact same
api.

morph
~~~~~

Access to detailed cell morphologies.


morph.get_filepath()
^^^^^^^^^^^^^^^^^^^^

Get path to file storing detailed morphology for given GID.

By default returns path to H5v1 morphology representation, use `source` argument for alternative representations (at the moment only ``h5v1`` and ``ascii`` are available).

.. code-block:: python

    >>> c.morph.get_filepath(42)
    '/gpfs/bbp.cscs.ch/project/proj64/circuits/test/morphologies/v1/rp100428-12_idK_-_Clone_2.h5'

    >>> c.morph.get_filepath(42, source="ascii")
    '/gpfs/bbp.cscs.ch/project/proj64/circuits/test/morphologies/ascii/rp100428-12_idK_-_Clone_2.asc'


morph.get()
^^^^^^^^^^^

Get `Morphio <http://morphio.readthedocs.io/en/latest/>`_ morphology object for given GID.

.. code-block:: python

    >>> c.morph.get(42)
    <morphio._morphio.Morphology at 0x7f396d66f9d0>

Passing `transform=True` to ``morph.get()`` will rotate and transform morphology according to cell position and orientation in the circuit.

morph.section_features()
^^^^^^^^^^^^^^^^^^^^^^^^

Get section features for given GID.
Available features are defined in ``bluepy.enums.Section`` enum.

.. code-block:: python

    >>> from bluepy import Section
    >>> c.morph.section_features(1, [Section.NEURITE_TYPE, Section.BRANCH_ORDER]).head()
                Section.NEURITE_TYPE  Section.BRANCH_ORDER
    Section.ID
    0                              2                     0
    1                              3                     0
    2                              3                     1
    3                              3                     2
    4                              3                     3


morph.segment_features()
^^^^^^^^^^^^^^^^^^^^^^^^

Get segment features for given GID.
Available features are defined in ``bluepy.enums.Segment`` enum.

.. code-block:: python

    >>> from bluepy import Segment
    >>> c.morph.segment_features(1, [Segment.LENGTH]).head()
                           Segment.LENGTH
    Section.ID Segment.ID
    0          0                 1.296203
               1                 1.728381
               2                 1.113463
               3                 1.244227
               4                 0.982497


morph.spatial_index
^^^^^^^^^^^^^^^^^^^^^^

Access to FLATIndex spatial index for morphology segments.

.. code-block:: python

    >>> c.morph.spatial_index.q_window((350, 1180, 570), (400, 1190, 580)).head()
       Segment.X1   Segment.Y1  Segment.Z1  ...
    0  374.040833  1187.340210  570.442993
    1  368.899963  1182.200317  570.277161
    2  368.614441  1185.508789  570.026611


stats
~~~~~

Miscellaneous circuit statistics.

Work in progress.


emodels
~~~~~~~

Access to cell electrical morphologies.

emodels.get_filepath()
^^^^^^^^^^^^^^^^^^^^^^

Path to HOC emodel file corresponding to given GID.

.. code-block:: python

    >>> c.emodels.get_filepath(1)
    '/gpfs/bbp.cscs.ch/project/proj64/entities/emodels/2017.11.03/hoc/bNAC_L23SBC.hoc'


emodels.get_properties()
^^^^^^^^^^^^^^^^^^^^^^^^

Dictionary with me_combo properties corresponding to given GID.

.. code-block:: python

    >>> c.emodels.get_properties(1)
    {'holding_current': -0.079053, 'threshold_current': 0.168666}

Returns `None` for old-style emodel releases with separate HOC for each `me_combo`.


subcellular
~~~~~~~~~~~

Access to subcellular data (gene expressions, protein concentrations etc).

.. note::

  | Methods in this namespace rely on some optional |name| dependencies.
  | If you install |name| with *pip*, please make sure to specify either ``[ngv]`` or ``[all]`` suffix:

  .. code-block:: console

      $ pip install -i https://bbpteam.epfl.ch/repository/devpi/simple/ bluepy[all]


subcellular.gene_mapping
^^^^^^^^^^^^^^^^^^^^^^^^

Gene to protein correspondence.

Pandas DataFrame / Series indexed by gene names, with three columns:

- ``lead_protein``,  name of the main protein associated with the gene
- ``maj_protein``, semicolon-separated list of other proteins associated with the gene
- ``comment``, free-form optional comment

Optional ``genes`` argument can be used to limit the result to some gene(s); by default all the genes are queried.

.. code-block:: python

    >>> c.subcellular.gene_mapping('0610011F06Rik')
    lead_protein                              Q9DCS2
    maj_protein                 Q9DCS2;E9Q7K5;G5E8X1
    comment         UPF0585 protein C16orf13 homolog


subcellular.gene_expressions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Gene expressions for given GID.

Pandas Series indexed by gene names, with gene expression levels.

Optional ``genes`` argument can be used to limit the result to some gene(s); by default all the genes are queried.

.. code-block:: python

    >>> c.subcellular.gene_expressions(1)
    gene
    Tshz1       0.0
    Fnbp1l      1.0
    Adamts15    0.0
    ...



subcellular.cell_proteins
^^^^^^^^^^^^^^^^^^^^^^^^^

Protein concentration in organelles for given GID.

Pandas DataFrame / Series indexed by gene names, with nine columns corresponding to protein concentration in each of cell organelles; plus ``total`` for protein concentration across all the cell. Concentrations are measured in nM (nanomoles / litre); missing values encoded with ``NaN``.

Optional ``organelles`` argument can be used to limit the result to some organelle(s); by default all the organelles are queried (+ total). For organelle names, please use of ``Organelle`` enum provided in ``bluepy.subcellular`` module.

Optional ``genes`` argument can be used to limit the result to some gene(s); by default all the genes are queried.

.. code-block:: python

    >>> from bluepy.subcellular import Organelle
    >>> c.subcellular.cell_proteins(1, [Organelle.NUCLEUS, Organelle.TOTAL]).head()
                                                         nucleus       total
    gene
    0610009B22Rik                                       1.729418   37.076374
    0610011F06Rik                                       4.510578  191.242493
    0610012G03Rik                                       0.000000   14.134702
    ...


subcellular.synapse_proteins
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Protein concentrations for a given synapse.

Pandas Series indexed by gene names, with either:

- protein *concentrations* [nM] (if presynaptic side is queried)
- protein molecule *counts* (if postsynaptic side is queried)

First positional (required) is synapse ID (a pair); one should also specify which side of the synapse to query (`pre` or `post`).

Optional ``genes`` argument can be used to limit the result to some gene(s); by default all the genes are queried.

To obtain protein molecule counts for postsynaptic side, we estimate synapse area [um^2] from its conductance, multiplying conductance [nS] by 0.12 for excitatory synapses; and 0.071 for inhibitory ones. An optional argument, ``area_per_nS`` can be used to the default scaling factor.

.. code-block:: python

    >>> c.subcellular.synapse_proteins((1, 0), 'post', area_per_nS=0.1).head()
    gene
    Plekhg2    0.025047
    Plekhg3    0.095636
    Cers4      0.020641
