import pandas as pd
import pytest

from api.pipelines.training.pipeline import (
    AUDIO_DESCRIPTOR_COLUMNS,
    extract_training_features,
    load_dataset,
    run,
)


def _make_row(**overrides):
    row = {
        "id": "abc123",
        "name": "Test Track",
        "album_name": "Test Album",
        "artists": "['Someone']",
        "tempo": 120.0,
        "loudness": -8.0,
        "key": 1,
        "mode": 1,
        "energy": 0.7,
        "danceability": 0.6,
        "speechiness": 0.05,
        "acousticness": 0.2,
        "instrumentalness": 0.0,
        "liveness": 0.1,
        "valence": 0.582,
        "duration_ms": 200000,
        "lyrics": "i am so happy today everything feels wonderful and bright",
        "year": 2023,
        "genre": "Pop",
        "popularity": 50,
    }
    row.update(overrides)
    return row


def test_extract_training_features_adds_expected_columns():
    df = pd.DataFrame([_make_row()])
    result = extract_training_features(df)

    assert "lyric_sentiment" in result.columns
    assert "valence_normalized" in result.columns
    assert "alignment_gap" in result.columns

    assert result["valence_normalized"].iloc[0] == pytest.approx(0.164)
    assert result["alignment_gap"].iloc[0] == pytest.approx(
        result["lyric_sentiment"].iloc[0] - result["valence_normalized"].iloc[0]
    )


def test_audio_descriptor_columns_pass_through_unchanged():
    df = pd.DataFrame([_make_row()])
    result = extract_training_features(df)
    for col in AUDIO_DESCRIPTOR_COLUMNS:
        assert result[col].iloc[0] == df[col].iloc[0]


def test_extract_training_features_does_not_mutate_input():
    df = pd.DataFrame([_make_row()])
    original_columns = list(df.columns)
    extract_training_features(df)
    assert list(df.columns) == original_columns


def test_missing_lyrics_row_does_not_crash_and_is_neutral():
    df = pd.DataFrame([_make_row(lyrics=None), _make_row(lyrics="")])
    result = extract_training_features(df)
    assert (result["lyric_sentiment"] == 0.0).all()
    assert result["alignment_gap"].notna().all()


def test_load_dataset_raises_clear_error_on_missing_columns(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame([{"id": "x", "valence": 0.5}]).to_csv(bad_csv, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_dataset(bad_csv)


def test_run_writes_output_csv_with_new_columns(tmp_path):
    input_csv = tmp_path / "input.csv"
    output_csv = tmp_path / "nested" / "output.csv"
    pd.DataFrame([_make_row(), _make_row(lyrics=None)]).to_csv(input_csv, index=False)

    result = run(input_csv, output_csv)

    assert output_csv.exists()
    on_disk = pd.read_csv(output_csv)
    assert len(on_disk) == len(result) == 2
    assert "alignment_gap" in on_disk.columns
