def test_runtime_imports():
    """Smoke test: import runtime and assert APIs are present."""
    from w2vbert_speaker_scripted import load_scripted, load_feature_extractor, compute_input_features_from_wave

    assert callable(load_scripted)
    assert callable(load_feature_extractor)
    assert callable(compute_input_features_from_wave)
