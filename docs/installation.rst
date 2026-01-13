.. highlight:: shell

============
Installation
============


Stable release
--------------

To install OpenAI Gym Environments for Donkey Car using uv (recommended):

.. code-block:: console

    $ uv add gym_donkeycar

Or with pip:

.. code-block:: console

    $ pip install gym_donkeycar

If you don't have `uv`_ installed, see the `uv installation guide`_.

.. _uv: https://docs.astral.sh/uv/
.. _uv installation guide: https://docs.astral.sh/uv/getting-started/installation/


From sources
------------

The sources for OpenAI Gym Environments for Donkey Car can be downloaded from the `Github repo`_.

You can clone the public repository:

.. code-block:: console

    $ git clone git://github.com/tawnkramer/gym-donkeycar

Or download the `tarball`_:

.. code-block:: console

    $ curl  -OL https://github.com/tawnkramer/gym-donkeycar/tarball/master

Once you have a copy of the source, install it with uv:

.. code-block:: console

    $ cd gym-donkeycar
    $ uv sync

To install with development dependencies:

.. code-block:: console

    $ uv sync --extra tests --extra docs


.. _Github repo: https://github.com/tawnkramer/gym-donkeycar
.. _tarball: https://github.com/tawnkramer/gym-donkeycar/tarball/master
