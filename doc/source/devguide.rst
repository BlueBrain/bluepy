.. include:: _common.rst

Developer guide
===============

|name| follows usual NSE conventions for repo layout and continuous integration.
Please refer to `this <https://bbpteam.epfl.ch/project/spaces/display/BBPNSE/Python>`_ Confluence page for the details.

Design decisions
----------------

- breaking methods into namespaces
- Pandas Series / DataFrames for return results
- lazy caching
- in-memory cell collection


Deprecation policy
------------------

In order to gracefully deprecate methods, please use ``bluepy.utils.deprecate`` module:

.. code-block:: python

    from bluepy.utils import deprecate

    def deprecated_but_still_available(self, name):
        deprecate.warn("Would be removed from in the next release")
        return 42

    def deprecated_and_unavailable()
        deprecate.fail("Not supported starting from version X")

Usually we follow the following procedure:

- mark deprecated method with ``deprecate.warn`` during one minor release cycle (``0.<X>.*``)
- switch to ``deprecate.fail`` in the following one (``0.<X+1>.0``)
- finally, remove the deprecated method in ``0.<X+2>.0``
