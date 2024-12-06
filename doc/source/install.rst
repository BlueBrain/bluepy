.. include:: _common.rst

Installation
============

|name| is distributed as a Python package available at BBP devpi server:

.. code-block:: console

    $ pip install -i https://bbpteam.epfl.ch/repository/devpi/simple/ bluepy[all]

According to the global Python policy, the Python 2.7 support has been dropped in the ``v2.0.0`` version.


Prior to running ``pip install``, we strongly recommend upgrading ``pip`` in your virtual environment
unless you have a compelling reason not to do it:

.. code:: console

    $ pip install -U pip setuptools

.. warning::

    This upgrade of pip is really important and skipping this part can lead to problems installing
    ``libsonata``.

The tag ``[all]`` will bring |name| with binary of ``brion``.
In case you don't need it (i.e. do not access binary simulation report), you can omit ``[all]`` specifier:

.. code-block:: console

    $ pip install -i https://bbpteam.epfl.ch/repository/devpi/simple/ bluepy

If you are using python 3.6 or python 3.7, you may also install the ``flatindex``:

.. code-block:: console

    $ pip install -i https://bbpteam.epfl.ch/repository/devpi/simple/ bluepy[flatindex]


.. DANGER::

    You need to be careful not to install bluepy from the global pypi.

    That is, you shoud **NEVER** :

    .. code-block:: console

        $ pip install bluepy

    This does not fail but will provide you with an unintended bluetooth package.

    Spotting this common error is easy. If you try to use the wrong version of bluepy within your
    codes, you should have this kind of error :

    .. code-block:: python

        >>> from bluepy import Circuit
            Traceback (most recent call last):
            File "<stdin>", line 1, in <module>
            ImportError: cannot import name 'Circuit'

On BB5
~~~~~~

On BB5, you have two main ways of working with bluepy.

Spack package
-------------
For convenience sake, we provide a |name| spack package available on BB5 :

.. code-block:: console

  $ module purge
  $ module load unstable
  $ module load py-bluepy

This will come along with all the dependencies except ``libFLATIndex`` which is not available on spack.
The versions available on spack usually follow the pip release versions. Unfortunately, releasing on spack
is a manual process that can take time and some versions may be skipped. So if you want
to use the latest version please use the second option described below.

Virtual environments
---------------------
This way of setting up |name| usually gives you access to the latest versions but it is less straightforward.

You can create a dedicated virtual environment using this sequence of commands :

.. code-block:: console

    $ module load archive/2020-12 python/3.7.4
    $ python -m venv bluepy-env

The name ``bluepy-env`` is just a suggestion and can be changed. Once the virtual environment is
created you need to activate it using :

.. code-block:: console

    $ . bluepy-env/bin/activate

This provides you with a virtual env using the latest supported version of python on BB5. You
can now install bluepy in this virtual environment :

.. code-block:: console

    $ pip install -U pip setuptools
    $ pip install -i https://bbpteam.epfl.ch/repository/devpi/simple/ bluepy[all]

Don't forget to install ipython if needed :

.. code-block:: console

    $ pip install ipython

Only if you are part of a development process or eager for a new features, you can use the pre
released version using the ``--pre`` tag. Be careful though, this is at your own risks.

.. code-block:: console

    $ pip install -i https://bbpteam.epfl.ch/repository/devpi/simple/ --pre bluepy[all]
