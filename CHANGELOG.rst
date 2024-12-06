Changelog
=========

Version 2.5.7
-------------

Bug Fixes
~~~~~~~~~
- Fix lint and functional tests.
- Pin numpy<2 to avoid incompatibilities with brion 3.3.10.

Version 2.5.6
-------------

Bug Fixes
~~~~~~~~~
- Fix functional tests.


Version 2.5.5
-------------

Bug Fixes
~~~~~~~~~
- Fix internal HDF reader for compartment reports.

Version 2.5.4
-------------

Bug Fixes
~~~~~~~~~
- Fix functional tests with snap 2.0.2.

Version 2.5.3
-------------

Improvements
~~~~~~~~~~~~
- Improve performance of cells.get() when selecting few ids.

Bug Fixes
~~~~~~~~~
- Fix DeprecationWarnings and FutureWarnings with pandas 2.1 and numpy 1.26.

Version 2.5.2
-------------

Improvements
~~~~~~~~~~~~
- Relax Pandas requirements and support Pandas 2.0.
- Make functional tests faster with nbclient.

Bug Fixes
~~~~~~~~~
- Fix for python 3.7 and 3.8 and old SSL libraries on BB5.

Version 2.5.1
-------------

Improvements
~~~~~~~~~~~~
- Avoid duplicated warnings when a Simulation is passed to subprocesses [BLPY-287]

Version 2.5.0
-------------

Removed
~~~~~~~
- dependency to bluepysnap
- functionality to perform complex queries (sonata cells) with ``$or`` and ``$and`` keywords

Version 2.4.6
-------------

Improvements
~~~~~~~~~~~~
- Update Circuit Documentation links
- Update functional tests
- Fix warnings with Pandas 1.5.0, snap, imageio [BLPY-285]
- Make target parsing faster [BLPY-286]

Version 2.4.5
-------------

Bug Fixes
~~~~~~~~~
- Raise a clearer error when trying to load cells from a nonexistent target.
- Fix mis-indexing morphology sections

Improvements
~~~~~~~~~~~~
- make Circuit & Simulation pickle-able

Version 2.4.4
-------------

Bug Fixes
~~~~~~~~~
- Enforce dtype int64 in spike and compartment reports.

Improvements
~~~~~~~~~~~~
- Parallelize functional tests.

Version 2.4.3
-------------

Improvements
~~~~~~~~~~~~
- Use 'tstride' to read only the desired samples from libsonata.ElementReportReader
- Use bluepysnap that supports BBP Sonata format

Bug Fixes
~~~~~~~~~
- fix documentation links

Version 2.4.2
-------------

Improvements
~~~~~~~~~~~~
- Enable functional CI

Bug Fixes
~~~~~~~~~
- BLPY-267: pandas 1.3.0 doesn't auto-convert big-endian arrays, some of which
  exist due to the BG/Q: work around this

Version 2.4.1
-------------

Improvements
~~~~~~~~~~~~
- Remove flatindex dependency from [all] extras.
- Reduce the memory needed to store targets (note: the Target class is not backward compatible).

Bug Fixes
~~~~~~~~~
- Fix ``circuit.cells.ids`` for projections.
- Fix errors from the new pylint version 2.10.2.

Version 2.4.0
-------------

New Features
~~~~~~~~~~~~
- Add ``sim.report_names`` function that returns all report names.
- Format of the reports is now read from the config.
- Add a lookup priority for the spike reports. First look at a `out.dat` file and if it does not
  exist look at a `out.h5` file. The previous versions implemented the sonata spike report reader
  but the simulation always pointed to the `out.dat` file.

Removed
~~~~~~~
- Removed the colon target feature. This has not been implemented in neurodamus and can cause
  problem with legacy targets.
- The 'source' argument from the ``sim.report`` function has been removed. The source is now read
  directly from the config.

Version 2.3.0
-------------

New Features
~~~~~~~~~~~~
- Use morphio>3 for morphology reading.

Removed
~~~~~~~
- Remove the v2 module entirely.
- Remove the v2 functions from the Circuit and Simulation classes.
- Remove the maths module.

Improvements
~~~~~~~~~~~~
- Remove the stat module which was 100% redundant with the circuit_stats and was using old api.

Bug Fixes
~~~~~~~~~
- Fixes the ``circuit.stats.segment_region_distribution`` function.
- Removed the MORPH_CACHE causing problems with the morphIO instances.


Version 2.2.0
-------------

New Features
~~~~~~~~~~~~
- If the orientation fields are not valid in the sonata node file it raises at cell init.
- Can access the full EModel information using `circuit.emodels.get_mecombo_info`.
- Implement the colon format for the targets. If the circuit uses a sonata file as a cell file,
  the available targets are restricted to the targets with the name schema
  `population_name:target` and normal `target`. If the file is a mvd file, a warning is shown.
- Implement the colon format for the connectomes. If the circuit uses a sonata file as a connectome
  file, the user can now define the desired population using:
  `filepath:population_name`. If the user does not provide a population name and if the file is a
  multipopulation file then an error is thrown. If the file is a mvd file and the user defines a
  population name then an error is thrown.
- Add a filter function to the TargetContext to filter the targets based on their names.

Improvements
~~~~~~~~~~~~
- Targets are now cached properties
- Improved data accessor for the sonata node files. It now uses direct accesses on a custom _data
  for the ProxyNodePopulation class.
- Standardized errors for the SonataCellCollection when bad gid or bad property (only BluePyError).
- Rework of the ``emodels.py`` module: factorization for the file and circuit based emodels, better
  use of the internal dataframe, cached extra properties instead of looping when using the
  get_properties.
- Use pytest instead of nose for the unit/functional testing.
- Pinned major versions of dependencies.


Bug Fixes
~~~~~~~~~
- Pinned major versions of neuroM to <2.0.0.


Version 2.1.0
-------------

New Features
~~~~~~~~~~~~
- Added Sonata compartment/spikes report support using libsonata.
- Added the Section target support.
- Bump brain to its new name, brion.

Improvements
~~~~~~~~~~~~
- Refactoring of the compartment_report.py module. The module was built to emulate the
  brion API at first but since we will not use it for the sonata reports, keeping the View/Mapping
  classes did not make much sense anymore. So they have been removed and the internal API changed.
- Add the ``orientation`` field to the available_properties and by default to the returned
  cells.get() when using sonata files.

Bug Fixes
~~~~~~~~~~
- The test sonata report file should have been a conversion of the h5 to the sonata format but
  it did not contain 0-based node ids. This has been fixed.
- Due to the corrupted test file the conversion from 1 to 0 based has been forgotten in the sonata
  implementation of the report. This has been fixed.

Removed
~~~~~~~
- Removed the old h5 report support.
- Drop spatial index for Sonata connectome because it is built with NRN tools and inconsistent with
  Sonata. See [BLPY-238] for details.


Version 2.0.0
-------------

Improvements
~~~~~~~~~~~~
- Removed python2 support
- Bluepy v2 API is now the official API.
- Redirects 'from bluepy.v2 import' to 'from bluepy import'
- Adds v2 property to Circuit and Simulation for backward compatibility
- For MVDCell {"regex": "reg_value"} is the only way of querying regex
- Moved spatial indices to index subpackage alongside dias index
- Fixed roi module and remove deprecated functions from ABC
- Use cached_property instead of lazy
- Add 37 tox
- Bumped the h5py and voxcell versions
- removed six
- Add a to_node_sets function to the targets. This is used to simplify a lot the SonataCellCollection
- Better factorization of the SonataCellCollection implementation
    - Use a proxy class to bypass the check on the gids
    - Use the node sets instead of a workaround of the targets
    - Change the query to catch and change the $target before using snap
- Factorization with snap of the SonataConnectome
    - Use the snap implementation for the connectome and a proxy class to redirect the _nodes function
      to the bluepy circuit.cells instead
- Better handling of the morphological properties for the sonata implementations
    - The morphological properties are now displayed when using : available_properties if the we
      can compute them from the fields inside the sonata file
- Add a check for the sonata format checking the "node_population" attribute.
- Add the PRE/POST_SEGMENT_ID and PRE/POST_SEGMENT_OFFSET to the synapse enum.

Deprecated
~~~~~~~~~~~

- Hard deprecate bluepy/api.py
- Deprecate bluepy/v2 sub-package
- Hard deprecate bluepy/v2/impl sub-package

Removed
~~~~~~~

- Removed bluepy/extrator.py
- Removed bluepy/morphology sub-package
- Removed bluepy/parsers sub-package
- Removed bluepy/synapses sub-package
- Removed bluepy/targets sub-package
- Removed talks (!?) directory
- Removed bluepy/geometry/mosaic.py
- Removed bluepy/geometry/hexagons.py
- Removed bluepy/geometry/pickled.py
- Removed unused functions from bluepy/utils
- Removed the bluepy/compat.py module
- Removed v1 tests
- Removed BluePyException (shadow code)
- Removed unused yaml config
- Removed entity_management dependency
- Replaced _PRE/POST_SEGMENT_ID and _PRE/POST_DISTANCE connectome variables by enums
- Removed BLUEPY_USE_SYN2, BLUEPY_SYN2_NRN_INDEX, BLUEPY_USE_BRION unused setting variables

Bug Fix
~~~~~~~~

- Fixed the functional tests


Version 0.16.0
---------------

Notes
~~~~~
- Will be the last v0.x.x version. The v1 API will be removed in the 2.0.0 version.
- It fixes the deployment problems from the 0.15.0


Version 0.15.0
---------------

Notes
~~~~~
- This is a shadow version and has not been released.

Bug Fix
~~~~~~~

- Fix the version of bluepysnap and voxcell to match the h5py<3.0.0 requirement.


Version 0.14.16
---------------

Removed
~~~~~~~

- Removed the sonata tag for pip. Sonata being the main implementation now, keeping the tag does
  not make sense anymore.
- Deprecate the functions from bluepy.v2.impl.math.

Improvements
~~~~~~~~~~~~

- Allow relative paths for the Circuit Targets using CurrentDir as default dir and the
  BlueConfigDir if CurrentDir is not present.
- Use voxcell.CellCollection as mvd2/3 backends.
- Voxcell is now a main requirement for BluePy.


Version 0.14.15
---------------

New Features
~~~~~~~~~~~~

- Add experimental internal sonata report reader.

Bug Fixes
~~~~~~~~~

- Force the version of h5py to h5py<3.0.0. h5py==3.0.0 dropped the python2 support and changed the
  string behaviors.
- Allow relative paths for the Simulation OutputRoot using CurrentDir as default dir and the
  BlueConfigDir if CurrentDir is not present.

Improvements
~~~~~~~~~~~~

- Use importlib to access the version number inside the setup.py instead of the deprecated imp.
- Use the user time range in the simulation firing animation instead of the first/last spiking times


Version 0.14.14
---------------

Bug Fixes
~~~~~~~~~

- Fix properly python2 dependencies of `pylint` and `shapely`.

Version 0.14.13
---------------

New Features
~~~~~~~~~~~~

- Can access the dynamics params of the edge populations for the sonata connectome impl.


Improvements
~~~~~~~~~~~~

- When calling proj.pathway_synapses or iter_connections we had problems with the offset ids
  if the cells was a sonata node file. We now bypass the node_ids existence checks from snap.

Version 0.14.12
---------------

Bug Fixes
~~~~~~~~~

- Apparently, according to neurodamus, the version number for the nrn is not mandatory in the nrn
  files. Instead of raising if not present we fix the version to 0.

Version 0.14.11
---------------

New Features
~~~~~~~~~~~~

- Adding the MorphologyType field to the config file. This allows the user to choose the morphology
  format from the blueconfig.
- Adding the new synapse enum values : U_HILL_COEFFICIENT and CONDUCTANCE_RATIO.

Improvements
~~~~~~~~~~~~

- The ``MorphHelper`` now uses the MorphologyType as default and "h5v1" as default if no
  format is provided (keep the historical default format).
- Allows to query directly the sonata property names for the sonata implementation of the
  connectome instead of the enum ones only.

Bug Fixes
~~~~~~~~~

- Fix the errors due to pandas==1.1.0. In pandas==1.1.0 using a list with missing value(s) in
  the ``.loc`` or ``[]`` is not correct anymore.


Version 0.14.10
---------------

Improvements
~~~~~~~~~~~~

- The h5 reports have better format compliance checks at instantiation.
- Possibility to choose the format with ``MorphHelper.get``.


Bug Fixes
~~~~~~~~~

- Fix unwanted raise when calling ``cells.ids`` with a group resulting in an empty array and calling
  sample.(ex: ``cells.ids(group=[], sample=42)``)


Version 0.14.9
--------------

New Features
~~~~~~~~~~~~

- Add a config property to the circuit to access the paths directly from the circuit object.

Improvements
~~~~~~~~~~~~

- Add the missing POST/PRE_GID to available_properties to the sonata connectome.
- Better documentation for installation.

Bug Fixes
~~~~~~~~~

- For sonata connectome object, fix the afferent_gids and efferent_gids functions when using a
  negative gid. The behavior is the same as the nrn one --> not existing gid returns an empty list.

- The custom h5 reader for the report could read corrupted h5 files with bad pointer_index. It
  now raises if the length of the gids is different than the length of the pointer_index.

- A dispatch has been added to the TimeSeriesReport class in case Brain does not support h5 report
  file format anymore (which should be the case).

- One dependency dropped the pylru dependency and it was not included in the setup.py


Version 0.14.8
--------------

Improvements
~~~~~~~~~~~~

- Use the new sphinx-bluebrain-theme for the documentation.

Bug Fixes
~~~~~~~~~

- Fixed the unique_gids behavior for the sonata implementation of the connectome.
- Fixed the difference in behavior between mvd3 and sonata nodes when querying a list with single
  gid in the 'cells.get()' function.


Version 0.14.7
--------------

New Features
~~~~~~~~~~~~
- Add the morphological features:

  * pre/post neurite distances
  * pre/post section distances
  * touch distance (using pre/post contour positions. The contour fields has been re-added to the
    sonata files by HPC. This feature can fail for the old sonata files.)

to the sonata connectome. These features were already included for the historical nrn connectome.

Improvements
~~~~~~~~~~~~

- Changed the raster plots so they now represent the empty spiking cells by empty spaces. This
  gives a constant representation for a given circuit no matter which cells are spiking.
- Bumped the bluepysnap's version. Bluepy can be installed alongside bluepysnap without
  version conflicts.
- Bumped the neurom version and removed the deprecated fst in profit of the new 'features' version.
  Bluepy should be ready for the future NeuroM==2.0.1 release.

Bug Fixes
~~~~~~~~~

- Pinned the imageio's version for the python 2.7 functional tests.


Version 0.14.6
--------------

Bug Fixes
~~~~~~~~~
- Force the snap version to the 0.1.2. This is done to prevent the problems coming from the
  version 0.2.0 that includes the multipopulation support. This is temporary.
- Bug in plot legend for "query like" groups (dictionaries).
- Bumped the blueconfig_file version to 0.1.11 to prevent the commented projection [BLPY-179]

Improvements
~~~~~~~~~~~~
- Following BlueConfig specs for connectome's lookup when using nrnPath. It was problematic
  when providing a directory path with multiple connectome files inside. The lookup is now:

  * if nrnPath is not a directory return the path.
  * if nrnPath is a directory try to open `edges.sonata`/`edges.h5` first.
  * if edges.sonata/h5 does not exists try nrn.


Version 0.14.5
--------------

Improvements
~~~~~~~~~~~~
- Use the new definition for CellLibraryFile. The new rule is:

  * if the CellLibraryFile is in the config try to use it first
  * if the CellLibraryFile is circuit.mvd3 or 2 join the CircuitPath to it
  * if it is something else use this as an absolute path
- relaxing the file extension for cells accepting .sonata and .h5


Version 0.14.4
--------------

New Features
~~~~~~~~~~~~
- Add sonata nodes implementation for cells. You can now use sonata node files
  in bluepy. Some problems can arise when used with projections if you
  did not add the virtual gids inside the sonata node file (which is the
  correct "sonata way" to deal with projections.)

Bug Fixes
~~~~~~~~~
- Bug fix on TimeSeriesReport when gids are unsigned int.
  (Fixed in bluepy but a ticket to brain has been made)
- Fix sampling in spikeraster_and_psth.

Improvements
~~~~~~~~~~~~
- Don't load unnecessary report data if plot_type is 'raster' in spikeraster_and_psth.


Version 0.14.3
--------------

Bug Fixes
~~~~~~~~~
- Fix functional tests due to pandas' table display change.
- Bug fix for the python2 pytable version not been supported anymore.
  This was blocking for python2 users. Bluepy could not be installed anymore
  with python2.


Version 0.14.2
--------------

New Features
~~~~~~~~~~~~
- Old/new sonata synapse mapping compatibility.
  We have two kinds of sonata formats with different attr names for the
  morphology part. Here, we suppose that all files are either a new format
  or an old one. i.e.: hybrids cannot exist.

Improvements
~~~~~~~~~~~~
- add a compatibility test for the old and new versions.


Version 0.14.1
--------------

New Features
~~~~~~~~~~~~
- Add metadata public member to the circuit.connectome object.
  This is mainly useful for the projections to retrieve the PopulationID value.

Bug Fixes
~~~~~~~~~
- Fix the x/y labels for spikeraster_and_psth plots.
- Fix errors from the new pylint version.
- Fix functional tests plots.

Improvements
~~~~~~~~~~~~
- add vmin, vmax args to spikeraster_and_psth plot.


(Old untracked changes from git logs)
-------------------------------------

- Fix sampling in spikeraster_and_psth [BLPY-174] [tomdele]

  * sampling was done multiple times resulting in different used gids

- BLPY-174: make spikeraster_and_psth faster. [Mike Gevaert]

  * don't load unnecessary report data if plot_type is 'raster'

- Tag version to 0.14.4.dev1. [tomdele]

- Merge "Add sonata nodes implementation for cells [BLPY-171]" [Thomas
  Delemontex]

- Add sonata nodes implementation for cells [BLPY-171] [tomdele]

- Bug fix on TimeSeriesReport when gids are unusigned int. [tomdele]

  * brain.CompartmentReport is not compliant with uint64 gids (raises error)

- Release bluepy==0.14.3. [tomdele]

- Quick fix of the python2 pytable version problem. [tomdele]

- Fix functional tests due to pandas table display change. [tomdele]

- Release bluepy==0.14.2. [tomdele]

- Old/new sonata synapse mapping compatibility [BLPY-172] [tomdele]

  We have two kinds of sonata formats with different attr names for the
  morphology part. Here, we suppose that all files are either a new format
  or an old one. i.e.: hybrids cannot exist.

  * detects if one attribute name in the sonata file corresponds to an old
    sonata version (cannot check all because having all attributes is not
    a requirement of sonata format).
  * set the correct morphology mapping in sonata connectome
  * changed the default test file for the sonata connectome to the new format
  * add a compatibility test for the old and new version

- Release bluepy==0.14.1. [tomdele]

- Add metadata to connectome/projection [BLPY-169] [tomdele]

  * add a metadata public member to connectome
  * for projections, it reads the blueconfig file and add all keys/
    values from the projection section to metadata
    (including the projection Path)

  Notes: To keep compatibility with old circuits instantiate
  with a Mapping object, a 'projections_metadata' has been added to
  the _parse_blueconfig instead of extending the
  'projections' key.

- Fix labels/improve spikeraster_and_psth [BLPY-165]/[BLPY-167]
  [tomdele]

  * add vmin, vmax args to spikeraster_and_psth
  * Homogeneous x/y labels
  * fix some 'new version pylint' errors
  * fix examples using plots

- Release bluepy==0.14.0. [tomdele]

- Fix functional tests. [tomdele]

  * pandas==0.25.1 tables display in jupyter changed from <th> to <td>

- Raise when syn2 file is provided as connectome input. [tomdele]

  * raise a bluepy error when a syn2 is provided
  * add tests for the fail

- Fix problem with check_times [BLPY-164] [tomdele]

  * add a report name argument to the checktime function
  * use the correct t_start/t_end if using a report name
  * reset t_start/t_end if not in report time range
  * using index directly in matplotlib sometimes leads to error so
    use array instead

- Collections.{Mapping,Sequence,Iterable} deprecated. [Mike Gevaert]

  * they are starting to raise warnings as they will be removed in
    py3.8 or py3.9

- Better functional tests. [tomdele]

  * Removed the hierarchical errors
  * Better diff for text outputs when an error is thrown
  * Retrieve stacktrace when an error occurs during cell execution
  * Difference between tested and ref png files are saved
  * Removed dependence to pypng and use imageio instead
  * Rerun notebooks to match the new comportment from pandas=0.25.0 (lexicographical ordering of columns)

- Private plotting functions are now public. [tomdele]

  * add default values in docstrings
  * moved all private function from plotting to public

- Removed 'default.nix' [Arseny V. Povolotsky]

- Removed 'bluepy.sonata' sub-package. [Arseny V. Povolotsky]

  * it has been open-sourced as:
    https://github.com/bluebrain/snap

- Release bluepy==0.13.6. [Arseny V. Povolotsky]

- Fixed Synapse.POST_BRANCH_TYPE attribute name. [Arseny V. Povolotsky]

- Ignore dtype differences in unit tests. [Arseny V. Povolotsky]

- Update examples for matplotlib 3.1.0. [tomdele]

  * change linestyle to drawstyle in plotting
  * rerun exemples

- Release bluepy==0.13.5. [Arseny V. Povolotsky]

- Switched to open-sourced 'libsonata' [Arseny V. Povolotsky]

- [SONATA] Synapse positions access. [Arseny V. Povolotsky]

- Migrated 'bluepy.sonata' tests to 'py.test' [Arseny V. Povolotsky]

- Merge "Miscellaneous changes for 'bluepy.sonata'" [Arseniy
  Povolotskiy]

- Miscellaneous changes for 'bluepy.sonata' [Arseny V. Povolotsky]

  * replaced 'pylru' with 'functools32' (less restrictive licence)
  * removed some imports from "main" 'bluepy'

- Merge "Added SONATA tests" [Arseniy Povolotskiy]

- Added SONATA tests. [Arseny V. Povolotsky]

- Release bluepy==0.13.4. [genrich]

- Change names for plots [BLPY-136] [tomdele]

  * Change plotting name functions (Elisabetta inputs)
  * Add possible titles to all plots

- Fixed synapse index auto-detection. [Arseny V. Povolotsky]

  * when nrnPath points to a file

- Proposal for adding new visualizations [BLPY-136] [tomdele]

  * Proposal for a new version of the plotting module in bluepy
  * Change of bluepy/v2/plotting.py. Contains only free functions now.
  * Move SimulationPlotHelper to simulation.py
  * Keep the lazy import in Simulation.plot and the return of SimulationPlotHelper
  * Dynamic import of functions in the SimulationPlotHelper
  * Adding the thalamus plots
  * Some names need to change in the thalamus module
  * Tests for plots are removed (almost impossible to test with different
    version of matplotlib for python3 and 2)

- Hotfix for sonata tests. [tomdele]

  * sonata=0.0.1 changed the behavior of connectome_sonnata.e/afferent_gids
    functions for "out-of-range" gids.
    It used to raise but now returns an empty list.
  * This behavior should be temporary. For this reason, old asserts are not
    removed but just commented.

- Reduce memory usage in cell dataframe by not using categoricals. [Mike
  Gevaert]

  ================ ============ =============== ===========
    Old               Values       New             values
  ================ ============ =============== ===========
    Index          74.553424     Index            74.553424
    x              74.553424     x                74.553424
    y              74.553424     y                74.553424
    z              74.553424     z                74.553424
    orientation    74.553424     orientation      74.553424
    etype          9.319586      etype             9.319586
    hemisphere     9.319274      hemisphere        9.319274
    layer          74.553424     layer            74.553424
    me_combo       447.374456    me_combo         74.553424
    morph_class    9.319274      morph_class       9.319274
    morphology     447.374456    morphology       74.553424
    mtype          9.322218      mtype             9.322218
    region         9.322426      region            9.322426
    synapse_class  9.319274      synapse_class     9.319274
    Total:         1397.991508                   652.349444
  ================ ============ =============== ===========

  All numbers in MB, cell count 9319178

- Introduced firing_animation [BLPY-152] [tomdele]

  * Creation of a simple plot animation displaying the firing neurons in 2d
    space.
  * Correct also some lyint error

- Release bluepy==0.13.3. [Arseny V. Povolotsky]

- Fixed a performance issue in iter_connections() [Arseny V. Povolotsky]

- Release bluepy==0.13.2. [Arseny V. Povolotsky]

- Fixed connectome.pathway_synapses() [Arseny V. Povolotsky]

  * it didn't like asking synapse properties
    for empty synapse ID list

- Introduced 'connectome.available_properties' method. [Arseny V.
  Povolotsky]

  * fixed Synapse.NRRP attribute name for SONATA
  * added 'sonata' to '[bbp]' extras

- Release bluepy==0.13.1. [Arseny V. Povolotsky]

- Fixed 'SonataConnectome._estimate_range_size()' [Arseny V. Povolotsky]

- Merge "Optimized MVDCellCollection._check_ids()" [Arseniy Povolotskiy]

- Optimized MVDCellCollection._check_ids() [Arseny V. Povolotsky]

- Optimized 'iter_connections' for SONATA. [Arseny V. Povolotsky]

  * quick-n-dirty heuristic for choosing
    "optimal" side for querying connectivity

- Enable 'hidden' Synapse properties for SONATA. [Arseny V. Povolotsky]

- Lookup 'start.target' in CircuitPath. [Arseny V. Povolotsky]

  * if it's not found in nrnPath

- Release bluepy==0.13.0. [Arseny V. Povolotsky]

- Revised string matching in `bluepy.sonata` [Arseny V. Povolotsky]

  * added tentative support for 'regex:' prefix in v2
  * see also:
    https://bbpcode.epfl.ch/code/#/c/43670/

- Fixed '[all]' extra. [Arseny V. Povolotsky]

- 0.13 documentation updates. [Arseny V. Povolotsky]

  * do not encourage Nix module usage
  * `pip install bluepy[all]` as first install command
  * regex support in cell queries

- Revised string property matching. [Arseny V. Povolotsky]

  * seeking compatibility with SONATA node sets:
    https://github.com/AllenInstitute/sonata/blob/master/docs/SONATA_DEVELOPER_GUIDE.md#node-sets-file
  * no check for categorical values
  * Mongo-like syntax for regex match
  * see also:
    https://github.com/AllenInstitute/sonata/issues/82

- Support full path to 'nrn.h5' for projections. [Arseny V. Povolotsky]

- Revived SONATA edges support in v2 API. [Arseny V. Povolotsky]

  * bringing back:
    https://bbpcode.epfl.ch/code/#/c/43028/
  * without BLUEPY_USE_SONATA setting this time

- Removed some 'deprecate.fail()' warnings. [Arseny V. Povolotsky]

- Migrated  check to Python 3. [Arseny V. Povolotsky]

- Update the return value from circuit.connectome.pathway_synapses
  functional tests. [tomdele]

  The optimization in presynaptic NRN access changes the synapse index order

- Optimized presynaptic NRN access [BLPY-92] [Arseny V. Povolotsky]

  * read properties / positions from '_efferent' files
    when presynaptic access is preferable

- Fixed 'morph.get_filepath()' [Arseny V. Povolotsky]

  * look for morphology files in several subfolders

- Remove pandas.clip deprecation warning. [tomdele]

- `sonata.Circuit` as a drop-in replacement for `v2.Circuit` (temporary)
  [Arseny V. Povolotsky]

  * single-population `cells`, `connectome`
  * `morph` as `cells.morph` alias
  * v2.Synapse supported in `sonata.connectome`
  * `cells.mtypes`, `cells.etypes` preserved

- Revised SONATA support. [Arseny V. Povolotsky]

- Add new h5 compartment report format [BLPY-150] [tomdele]

- Fixed 'h5py' deprecation warning. [Arseny V. Povolotsky]

- Fixed pylint warnings. [Arseny V. Povolotsky]

- Hotfix for 'virtual' GIDs in projections [BLPY-149] [tomdele]

- Merge "Support Connectome stored in SONATA Edges [NSETM-641]" [Arseniy
  Povolotskiy]

- Support Connectome stored in SONATA Edges [NSETM-641] [Arseny V.
  Povolotsky]

- Merge "Support CellCollection stored in SONATA Nodes [NSETM-641]"
  [Arseniy Povolotskiy]

- Support CellCollection stored in SONATA Nodes [NSETM-641] [Arseny V.
  Povolotsky]

- Rerun of example with matplotlib 3.0.1. [tomdele]

- Release bluepy==0.12.7. [Arseny V. Povolotsky]

- Make sure plot.ISI works when there are no spikes. [Mike Gevaert]

  - returns an empty plot
  - BLPY-147

- Add seed for sampling in example. [tomdele]

- Functional tests compatibility with matplotlib3 and dry run for
  python2. [tomdele]

- JSON circuit config with relative paths. [Arseny V. Povolotsky]

- Unpin 'pylint' [Arseny V. Povolotsky]

  * 'devpi' now handles Python 2 / 3 package versions [HELP-9318]

- Fix crash empty target [BLPY-145] [tomdele]

- Remove old api from functional tests, python2-3 compatibility.
  [tomdele]

- Updated tox.ini. [Arseny V. Povolotsky]

  * 'pycodestyle' + 'pylint' -> 'lint'

- Updated tox.ini. [Arseny V. Povolotsky]

  * added 'py36' to envlist
  * merged 'testenv' for unit tests
  * installing 'bbp-nse-ci' as a dependency

- Add new functional tests [BLPY-134] [tomdele]

- Don't show empty groups in spike raster plot. [Arseny V. Povolotsky]

- Release bluepy==0.12.6. [Arseny V. Povolotsky]

- Require 'enum34' only for Python<3.4. [Arseny V. Povolotsky]

- Stop pin of shapely: BLPY-144. [Mike Gevaert]

- Bump dev version. [Arseny V. Povolotsky]

- Warn about ignored GIDs [BLPY-141] [Arseny V. Povolotsky]

- Minor documentation fix. [Arseny V. Povolotsky]

- Merge "BluePyError for out-of-range GIDs [BLPY-141]" [Arseniy
  Povolotskiy]

- BluePyError for out-of-range GIDs [BLPY-141] [Arseny V. Povolotsky]

- Fixed 'v2.plot.voltage_collage()' [BLPY-141] [Arseny V. Povolotsky]

- Fixed 'v2.stats.fibre_density()' method. [Arseny V. Povolotsky]

- Added short description of what a group is. [Mike Gevaert]

  * as suggested in BLPY-138

- Release bluepy==0.12.5. [Arseny V. Povolotsky]

- Fixed incompatibility with tables==3.4.2. [Arseny V. Povolotsky]

- Release bluepy==0.12.4. [Arseny V. Povolotsky]

- Region filtering for bouton density calculation [BRBLD-57] [Arseny V.
  Povolotsky]

- Introduced v2.morph.segment_points() method. [Arseny V. Povolotsky]

- Access to circuit source atlas. [Arseny V. Povolotsky]

- Release bluepy==0.12.3. [Arseny V. Povolotsky]

- Added lost '_common.rst' file. [Arseny V. Povolotsky]

- Release bluepy==0.12.2. [Arseny V. Povolotsky]

- Merge "Get rid of v1 back-pointer in v2.Simulation" [Arseniy
  Povolotskiy]

- Get rid of v1 back-pointer in v2.Simulation. [Arseny V. Povolotsky]

- Fix-docs. [Arseny V. Povolotsky]

- Merge "Revised documentation (BLPY-133)" [Alexander Dietz]

- Revised documentation (BLPY-133) [Arseny V. Povolotsky]

  * revised documentation structure
  * updated installation instructions
  * removed deprecated or non-informative parts
  * added 'settings' section
  * got rid of 'v2' suffix in examples

- Merge "Expose v2.Circuit, v2.Simulation imports" [Arseniy Povolotskiy]

- Expose v2.Circuit, v2.Simulation imports. [Arseny V. Povolotsky]

- Documentation for 'subcellular' namespace. [Arseny V. Povolotsky]

- Introduced 'subcellular' namespace (NGV-65) [Arseny V. Povolotsky]

- Improving test coverage. [Arseny V. Povolotsky]

- Hard-deprecated some methods. [Arseny V. Povolotsky]

- Release bluepy==0.12.1. [Arseny V. Povolotsky]

- Support .swc extension in `v2.morph.get_filepath` [Arseny V.
  Povolotsky]

- Introduced 'bluepy.v2.transcriptome' namespace (BLPY-74) [Arseny V.
  Povolotsky]

  * v2.transcriptome.genes()
  * v2.transcriptome.get()
  * v2.transcriptome.which()


Old Change Logs
----------------

v0.12.0 - 2018-03-20
--------------------

* **Dropped pybinreports support**
* **Python 3 compatibility for v2 API**
* Deprecated ``raw``, ``source`` arguments for ``TimeSeriesReport.get()`` methods
* Introduced ``v2.emodels`` namespace
* Strict mode for turning warnings into exceptions
* Migrated to ``nse/bluepy``, ``nse/bluepy-configfile`` repos
* tox-based continuous integration
* Enum updates:

  - Removed ``Connection`` enum
  - Removed ``Synapse.PRE_MTYPE``, ``Synapse.POST_MTYPE``
  - Replaced ``Synapse.ASE`` with ``Synapse.NRRP``
  - Added ``Cell.REGION``


v0.11.0 - 2017-07-24
--------------------

* **Using Brion for compartment report access**
* Introduced ``Connectome.pathway_synapses()`` method
* Introduced ``Connectome.synapse_positions()`` method
* Introduced ``SimulationPlotHelper`` class
* Introduced ``bluepy.geometry.roi`` module
* Introduced ``CircuitStats`` methods:

  - ``cell_density()``
  - ``fibre_density()``
  - ``synapse_density()``
  - ``sample_convergence()``
  - ``sample_divergence()``
* Revised example notebooks
* Removed:

  - ``bbp2h5``
  - ``bluepy.geometry.quatern``
  - ``bluepy.serializers.*``

v0.10.0 - 2017-06-29
--------------------

* **Dropped support for v1 report access API**
* Revised ``Circuit`` class, introduced ``cells``, ``connectome`` namespaces
* Experimental support for SYN2 files
* Introduced ``TimeSeriesReport.get_gid()`` method
* Introduced ``SynapseReport.get_synapses()`` method
* Introduced additional ``Synapse`` properties:

  - ``Synapse.TOUCH_DISTANCE``
  - ``Synapse.[PRE|POST]_[X|Y|Z]_SURFACE``
* Introduced ``CircuitStats`` methods:

  - ``sample_bouton_density()``
  - ``sample_pathway_synapse_count()``
* Fixed MVD2 parsing (layer number)
* Deprecated:

  - ``bluepy.reports``
  - ``bluepy.serializers.*``
  - ``bbp2h5`` tool in favor of ``compartmentConverter``
  - ``bluepy.utils.lazy_property`` in favor of ``lazy`` package

v0.9.0 - 2017-05-18
-------------------

* Revised ``SpikeReport`` class
* Revised ``TimeSeriesReport`` class
* Revised synapse indexing
* Introduced ``v2.Circuit.morph`` namespace
* Introduced ``v2.experimental.NrnWriter`` class
* Various improvements to ``v2.Circuit.synapses()``
* BluePy configuration variables
* ``bbp2h5`` utility
* Removed:

  - ``bluepy.bbpsdk``
  - ``bluepy.interact``
* Deprecated:

  - ``bluepy.geometry.quatern``
  - ``bluepy.geometry.vector``
  - ``bluepy.report``
  - ``bluepy.serializers.*``
  - ``bluepy.targets.target_generation``

v0.8.0 - 2017-05-03
-------------------

* New cell query syntax
* Support projections in ``v2.Circuit.synapses()``
* Changed ``v2.Circuit.synapses()`` indexing scheme
* Merged ``v2.Circuit.*_spatial_index()`` methods
* Introduced ``v2.Circuit.stats`` methods:

  - ``mtype_divergence()``
  - ``segment_region_distribution()``
  - ``synapse_region_distribution()``
* Using dict-like access for BlueConfig access in ``v2.Circuit``
* Migrated to ``pybinreports>=0.4``

v0.7.0 - 2017-04-05
-------------------

* **Dropped support for Python 2.6**

* Pandas-like access to:

  - cell properties [``bluepy.Circuit.v2.cells``]
  - synapse properties [``bluepy.Circuit.v2.synapses``]
  - DIASIndex query results [``bluepy.Circuit.v2.segment_spatial_index``]
  - Simulation reports [``bluepy.Simulation.v2.reports``]

* Migrated to ``libFLATIndex >= 1.8.0`` (please refer to `SPIND-63 <https://bbpteam.epfl.ch/project/issues/browse/SPIND-63>`_ for the details)

* Deprecated BBPSDK-based functionality:

  - ``bluepy.api.Simulation.bbp_sdk_experiment``
  - ``bluepy.bbpsdk.*``
  - ``bluepy.interact.*``
  - ``bluepy.morphology.Morph``
  - ``bluepy.morphology.MorphDB``
  - ``bluepy.morphology.Sections``

* Relaxed library dependencies

  - not using virtual modules anymore
  - using latest versions of ``numpy`` / ``pandas`` / ``h5py`` / ``matplotlib``

v0.6.0 - 2014-04-02
-------------------

* Thorough overhaul of the code
* Implementation of continuous integration
* Automatic pep8 and pylint checking
* Split viz specific code to bluepyviz
* Fix tests and ensure they pass

v0.4.0 - 2013-09-30
-------------------

* Adding sub-population functionality to plot_psth, outdat now has a _spikes_hash
* Added API level support for projections
* First version of projection support. No longer relies on nrn_summary. Support only in synapse_property_from_gids for now
  - Added translation to synapse positions, works now without merged nrn nor merge script
* Fix to psth plotting to intersect plotted gids with circuit target
* Added Not, Neuron MType, EType to default namespace of mvddb query expression DSL
* Returning now values for all gids instead of nonempty ones only
* added ability to add morphology rotations to synapse positions
* function for synapse locations relavitve to the soma
* Added to docstring documentation on the meaning of the 19 parameters of a synapse for get_postsynaptic_data, get_presynaptic_data.
* Initial implementation of plot_raster
* Added plotting module; first implementation of plot_psth function
* Cleaned up examples dir to be more useful for new users
* Added plot_mosaic and plot_layers convenience methods to Circuit for quick viz
* Added mvddb eval_query_expr and select_query_expr which allows rudimentary query string expression syntax.  Added mvddb.gids_query first try
* Added functionality to remove all the unused targets from an extracted circuit (disable by default) Small syntactic changes in extractor.py
* _get_data_single in report.py now doesn't crash when asked for data of a gid that doesn't exist. It now return None Small syntactic fixes in report.py
* Added a little test to see if get_presynaptic_gids returns None in case a gid is given of a cell that doesn't have presynaptic cells
* Fix for case of getting synapse data from h5 when it does not exist
* Fixed a bug during the creation of start.ncs, where the old gids were used instead of the gids for the new circuit. Also added build.sh to .gitignore
* Fixed a bug during the generation of the start.ncs by the extractor, where a '}' symbol was added after every cell
* Optimization to Circuit.get_mtypes_in_layer
* Added circuit extractor functionality and tests.  Added test for parsing empty target, some fixes to mvddb, added more general select function, of which select_gids is a special case, some perf improvements for methods of mvddb
* Added select function to mvddb, select_gids now calls it with a performance improvement using q.values.  Added test cases, and Circuit.get_pathway_pairs now also will accept a Query object from mvddb.select in lieu of pre_mtype or post_mtype
* Fix to CellReport introduced in recent commit
* Added mvddb mtype_query, etype_query
* Added support for synaptic weights reports
* Added local CircuitConfigs for each test.  This way they can be updated to point at data, much of which was moved after the /bgscratch reorganization.  Most tests now run.

v0.3.0 - 2012-11-22
-------------------

* Valentin Haenel's last release before leaving the BBP
* Complete overhaul of the interactive API: 'bluepy.api'
* New 'Simulation' and 'Circuit' classes
* New documentation (Tutorial, Developer docs and API docs)
* New Report API: 'bluepy.report'
* Many, many minor bugfixes, docstring upgrades and pep8 fixes

v0.2.0 - 2012-07-16
-------------------

* Second release by Valentin Haenel
* Includes Eilif's enhancements from the last two years
* Tag before undertaking major refactorings (potentially causing breakage)
  in BluePy

v0.1.0 - 2010-08-27
-------------------

* First release by Miha Pelko
