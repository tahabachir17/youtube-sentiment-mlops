"""
Unit tests for data preprocessing, emoji mapping, and text cleaning utilities.
"""

import pytest
from src.utils.text_cleaning import clean_text, label_to_sentiment_name, sentiment_name_to_label


def test_url_removal():
    raw = "Check out our site https://example.com/test?a=1&b=2 for more info!"
    cleaned = clean_text(raw)
    assert "https" not in cleaned
    assert "example.com" not in cleaned
    assert "check out our site for more info!" in cleaned


def test_mention_removal():
    raw = "Hey @johndoe and u/reddituser checkout r/MachineLearning"
    cleaned = clean_text(raw)
    assert "@" not in cleaned
    assert "u/reddituser" not in cleaned
    assert "r/machinelearning" not in cleaned


def test_emoji_normalization():
    positive_raw = "Great work! 🔥❤️"
    cleaned_pos = clean_text(positive_raw)
    assert "awesome" in cleaned_pos
    assert "love" in cleaned_pos

    negative_raw = "Awful video 👎😡"
    cleaned_neg = clean_text(negative_raw)
    assert "bad" in cleaned_neg
    assert "angry" in cleaned_neg


def test_repeated_characters():
    raw = "Sooooo cooool and goooood"
    cleaned = clean_text(raw)
    assert cleaned == "soo cool and good"


def test_html_entities_and_tags():
    raw = "<b>Wow &amp; amazing</b>"
    cleaned = clean_text(raw)
    assert "<b>" not in cleaned
    assert "&amp;" not in cleaned
    assert "wow" in cleaned and "amazing" in cleaned


def test_empty_and_none_inputs():
    assert clean_text(None) == ""
    assert clean_text("") == ""
    assert clean_text("   \n\t  ") == ""


def test_label_mappings():
    assert label_to_sentiment_name(1) == "positive"
    assert label_to_sentiment_name(0) == "neutral"
    assert label_to_sentiment_name(-1) == "negative"

    assert sentiment_name_to_label("positive") == 1
    assert sentiment_name_to_label("neutral") == 0
    assert sentiment_name_to_label("negative") == -1
