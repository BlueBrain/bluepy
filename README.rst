.. include:: _common.rst

|name| documentation
====================

Introduction
------------

|name| is a Python library for accessing BlueBrain circuit models.
The main two interface classes exposed are ``Circuit`` and ``Simulation``.

``Circuit`` corresponds to the *static* structure of a neural network, i.e.:

- cell positions / properties
- synapse positions / properties
- detailed morphologies
- electrical models
- subcellular data (gene expressions, protein concentrations)
- spatial index of morphology segments
- spatial index of synapses

``Simulation`` corresponds to the result of a *dynamic* simulation of the network, i.e. spikes, voltage traces and other binary reports recording neural activity.

Most of |name| methods return `pandas <https://pandas.pydata.org/>`_ Series or DataFrames, indexed in a way to facilitate combining data from different sources (for instance, synapse reports with corresponding synapse positions).

Among other dependencies, |name| relies on BBP-provided libraries:

- `MorphIO <http://morphio.readthedocs.io/en/latest/>`_ for access to detailed morphologies
- `Brion <https://bbpteam.epfl.ch/documentation/projects/Brion/latest/index.html>`_ for access to binary simulation reports
- `bluepy_configfile <https://bbpteam.epfl.ch/documentation/projects/bluepy-configfile/latest/index.html>`_ for BlueConfig parsing
- `libFLATIndex <https://bbpteam.epfl.ch/project/spaces/display/BBPDIAS/Overview>`_ for access to segment / synapse spatial indices

.. toctree::
   :hidden:
   :titlesonly:

   Home <self>
   install
   usage
   migration
   settings
   cookbook
   devguide
   changelog


Acknowledgments
---------------

Authors and Contributors
~~~~~~~~~~~~~~~~~~~~~~~~

* Eilif Muller
* Miha Pelko
* Jeff Muller
* Ronny Hattelan
* Pierson Fleischer
* John Kenyon
* Valentin Haenel
* Mike Gevaert


Reporting issues
----------------

|name| is currently maintained by the BlueBrain NSE team.

Should you face any issue with using it, please submit a ticket to the `issue tracker <https://bbpteam.epfl.ch/project/issues/browse/BLPY>`_; or drop us an `email <mailto: bbp-ou-nse@groupes.epfl.ch>`_.
