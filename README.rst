BOHR-runtime (engine)
------------------------
For the overview of BOHR, please see https://github.com/giganticode/bohr

This repository contains CLI and engine for running the BOHR pipeline:
 - reading the BOHR config
 - fetching the heuristics from remote BOHR repository needed for a specific task
 - fetching the datasets
 - applying heuristics to artifacts and combining their outputs
 - preparing new dataset with the trained model;
 
Moreover, BOHR-runtime provides utilities for debugging heuristics and evaluating their effectiveness.

|GitHub license| |Maintainability| |GitHub make-a-pull-requests|

.. |GitHub license| image:: https://img.shields.io/github/license/giganticode/bohr-framework.svg
   :target: https://github.com/giganticode/bohr-framework/blob/master/LICENSE
   
.. |GitHub make-a-pull-requests| image:: https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square
   :target: http://makeapullrequest.com

.. |Maintainability| image:: https://codeclimate.com/github/giganticode/bohr-framework/badges/gpa.svg
   :target: https://codeclimate.com/github/giganticode/bohr-framework
   :alt: Code Climate
   
Getting started with development
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Clone the repository.
#. Inside the repository, run ``poetry install``. This will create a virtual environment and install the dependencies.
#. To run python interpreter within the virtual environment, use ``poetry run ...``
#. For example, to run the tests, execute: ``poetry run pytest --doctest-modules``
