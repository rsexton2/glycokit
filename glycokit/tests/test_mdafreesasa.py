import pytest

from glycokit.mdafreesasa import FreesasaAnalysis


def test_freesasa_threads_clamped(small_universe):
    from glycokit.mdafreesasa import FreesasaAnalysis
    with pytest.warns(UserWarning, match="single-threaded"):
        fsa = FreesasaAnalysis(small_universe.atoms, 1.4, n_threads=8)
    assert fsa.params.nThreads() == 1
    fsa.run()  # must not crash


def test_freesasa_threads_clamped(small_universe):
    with pytest.warns(UserWarning, match="single-threaded"):
        fsa = FreesasaAnalysis(small_universe.atoms, 1.4, n_threads=8)
    assert fsa.params.nThreads() == 1
    fsa.run()  # must not segfault
