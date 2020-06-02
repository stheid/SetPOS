================================
Setvalued Part of Speech Tagging
================================

|license| |thesis|

.. |license| image:: https://img.shields.io/github/license/stheid/SetPOS
    :target: LICENSE

.. |thesis| image:: https://img.shields.io/badge/thesis-10.17619%2FUNIPB%2F1--957-informational
    :target: https://digital.ub.upb.de/hs/download/pdf/3395154

This package provide `set-valued`_ POS-taggers.
The code relies on existing probabilistic taggers like CoreNLP_ and the TreeTagger_.
Additionally the code also provides two simple taggers.
Information about the Baseline can be found in my thesis.

.. _`CoreNLP`: https://stanfordnlp.github.io/CoreNLP/pos.html
.. _TreeTagger: https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger
.. _`set-valued`: https://arxiv.org/pdf/1906.08129v1.pdf

Overview
--------
- :code:`data` contains data
- :code:`examples` and :code:`scripts` contain usage files
- :code:`setpos` contains the implementation


Installation
------------
**Disclaimer**: The code probably doesn't run without modifications on Windows.
It should work on any standard Linux distribution.

Simple
^^^^^^
* Install Python package::

  $ pip install .



Complete
^^^^^^^^

* Download TreeTagger_ and place the binaries :code:`tree-tagger` and :code:`train-tree-tagger` in the :code:`setpos/tagger/treetagger` folder.
  Make sure the executable flag is set.
  This code is tested with version 3.2.2.
* Install :code:`java` version 11 (for CoreNLP)
* Install :code:`swig-3` (for hyperopt)
* Install Python package::

  $ pip install .[extra]


Corenlp
^^^^^^^

The CoreNLP tagger is provided as a patched version.
The `patch`_ and packed jar_ is in :code:`setpos/tagger/corenlp`, the patch is applied to this `version`_.

.. _patch: setpos/tagger/corenlp/read_expansions—export_proba.patch
.. _jar: setpos/tagger/corenlp/stanford-corenlp.jar
.. _version: https://github.com/stanfordnlp/CoreNLP/commit/0d4cfd4209feec7ddbda9eab3fa9c9791caa3e36

The Patch changes the following:
    - CoreNLP will write the posterior probability into debug files (needed for pos tagging)
    - Additional command line option for modifying the deterministic tag expansion [`thesis`_, 5.5.3]

.. _`thesis`: https://digital.ub.upb.de/hs/download/pdf/3395154

Data
----
.. image:: https://img.shields.io/badge/license-CC--BY%204.0-informational
    :target: https://creativecommons.org/licenses/by/4.0/

Data stems from the Intergramm_ and ReN_ project.
The corpus consists of historic Middle Lower German texts.
The provided versions here have slight modifications like orthographic unification.

.. _Intergramm: https://www.uni-paderborn.de/forschungsprojekte/Intergramm
.. _ReN: https://corpora.uni-hamburg.de/hzsk/de/islandora/object/text-corpus:ren-1.0

Usage
-----

.. code-block:: python

    import logging

    from sklearn.model_selection import LeaveOneGroupOut
    import pandas as pd

    from setpos.tagger import MostFrequentTag, CoreNLPTagger, TreeTagger
    from setpos.data.split import load

    if __name__ == '__main__':
        logging.basicConfig(level=logging.INFO)

        toks, tags, groups = load()
        train, test = next(LeaveOneGroupOut().split(toks, tags, groups))

        clf = TreeTagger()
        clf.fit(toks[train], tags[train])
        result = pd.DataFrame([toks[test][:20, 1].tolist(), clf.setpredict(toks[test][:20])], index=['token', 'tag']).T

        print(result)

.. code-block::

               token                                                tag
    0     stadtrecht                                        {"FM": 1.0}
    1   braunschweig  {"NE": 0.946357, "NA": 0.025582, "ADJD": 0.011...
    2           1227                     {"OA": 0.5348, "XY": 0.458823}
    3      blankline                                   {"$.": 0.995565}
    4        SWelich  {"OA": 0.839456, "DIA": 0.087804, "ADJA": 0.03...
    5       vo+eghet  {"NA": 0.636112, "VVFIN.*": 0.182379, "NE": 0....
    6           enen             {"DIART": 0.934728, "CARDA": 0.062113}
    7         richte                                        {"NA": 1.0}
    ...



Citation
--------
::

    @mastersthesis{heid2019setpos,
        author = {Heid, Stefan},
        title = {Set-Valued Prediction for Part-of-Speech Tagging},
        institution={Paderborn University},
        date = {02/12/2019},
        url = {https://digital.ub.upb.de/hs/download/pdf/3395154},
        doi = {10.17619/UNIPB/1-957},
    }


Acknowledgement
---------------

I want to thank my supervisors and co-authors Marcel Wewer and Prof. Eyke Hüllermeier
for the helpful feedback during the thesis


