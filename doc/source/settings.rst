.. include:: _common.rst

Settings
========

Certain aspects of |name| can be configured via environment variables.

BLUEPY_STRICT_MODE
------------------

|name| strict mode (default: ``False``).

If enabled, all warnings (including deprecation warnings) would lead to exceptions.

.. code:: console

    export BLUEPY_STRICT_MODE=1

----

.. note::

    Each of ``BLUEPY_<X>`` variables can be as well set programmatically, for instance:

    .. code:: python

        >>> from bluepy import settings
        >>> settings.MORPH_CACHE_SIZE = 10

    If you'd like to use this option, we recommend doing it prior to executing any other |name| code.