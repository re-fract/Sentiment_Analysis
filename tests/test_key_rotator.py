"""
tests/test_key_rotator.py — Unit test verifying key rotation, daily limit detection, and state tracking.
"""

import os
import unittest
from unittest.mock import MagicMock, patch
from src.augmentation.key_rotator import (
    GroqKeyRotator,
    AllKeysExhaustedError,
    is_daily_limit_error,
    load_api_keys_from_env,
)


class TestGroqKeyRotator(unittest.TestCase):

    def test_daily_limit_detection(self):
        # Daily limit messages
        self.assertTrue(is_daily_limit_error("Rate limit reached for model on requests per day (RPD): Limit 1000")[0])
        self.assertTrue(is_daily_limit_error("Rate limit reached on tokens per day (TPD)")[0])
        self.assertTrue(is_daily_limit_error("Error 429: daily limit exceeded")[0])
        self.assertTrue(is_daily_limit_error("Rate limit: try again in 14h30m")[0])
        self.assertTrue(is_daily_limit_error("Rate limit: try again in 450s")[0])

        # Short term per-minute messages
        is_daily, _ = is_daily_limit_error("Rate limit on tokens per minute (TPM). Please try again in 4.5s")
        self.assertFalse(is_daily)

    def test_key_rotation_across_5_keys(self):
        keys = ["key_1", "key_2", "key_3", "key_4", "key_5"]
        rotator = GroqKeyRotator(api_keys=keys, model="llama-3.3-70b-versatile")

        self.assertEqual(rotator.current_index, 0)
        self.assertEqual(len(rotator.exhausted_keys), 0)

        # Exhaust Key 1
        rotator.mark_current_key_exhausted("RPD reached")
        self.assertEqual(rotator.current_index, 1)
        self.assertIn(0, rotator.exhausted_keys)

        # Exhaust Key 2, 3, 4
        rotator.mark_current_key_exhausted("RPD reached")
        self.assertEqual(rotator.current_index, 2)
        rotator.mark_current_key_exhausted("TPD reached")
        self.assertEqual(rotator.current_index, 3)
        rotator.mark_current_key_exhausted("RPD reached")
        self.assertEqual(rotator.current_index, 4)

        # Exhaust Key 5 -> All exhausted
        self.assertFalse(rotator.is_all_exhausted())
        rotator.mark_current_key_exhausted("TPD reached")
        self.assertTrue(rotator.is_all_exhausted())

        # Attempting generate after all exhausted raises AllKeysExhaustedError
        with self.assertRaises(AllKeysExhaustedError):
            rotator.generate("test prompt")

    def test_load_keys_from_env(self):
        with patch.dict(os.environ, {
            "GROQ_API_KEYS": "k1, k2, k3",
            "GROQ_API_KEY_4": "k4",
            "GROQ_API_KEY_5": "k5",
        }, clear=True):
            keys = load_api_keys_from_env()
            self.assertEqual(keys, ["k1", "k2", "k3", "k4", "k5"])

    def test_label_sanitization_and_validation(self):
        from scripts.run_daily_labeling import sanitize_and_validate_labels

        review_text = "The pizza was delicious, but service was terribly slow."
        raw_labels = [
            {"category": "FOOD#QUALITY", "sentiment": "pos", "aspect_term": "pizza", "is_implicit": False},
            {"category": "service", "sentiment": "negative", "aspect_term": "service", "is_implicit": False},
            # Hallucinated term not in text -> should convert to is_implicit=True
            {"category": "ambience", "sentiment": "neutral", "aspect_term": "dining room", "is_implicit": False},
            # Duplicate
            {"category": "Food", "sentiment": "positive", "aspect_term": "pizza", "is_implicit": False},
        ]

        ok, cleaned = sanitize_and_validate_labels(raw_labels, review_text)
        self.assertTrue(ok)
        self.assertEqual(len(cleaned), 3)  # Duplicate removed

        # Pizza -> Food, positive
        self.assertEqual(cleaned[0]["category"], "Food")
        self.assertEqual(cleaned[0]["sentiment"], "positive")
        self.assertEqual(cleaned[0]["aspect_term"], "pizza")
        self.assertFalse(cleaned[0]["is_implicit"])

        # Service -> Service, negative
        self.assertEqual(cleaned[1]["category"], "Service")
        self.assertEqual(cleaned[1]["sentiment"], "negative")

        # Dining room -> not in text -> converted to implicit
        self.assertEqual(cleaned[2]["category"], "Ambience")
        self.assertIsNone(cleaned[2]["aspect_term"])
        self.assertTrue(cleaned[2]["is_implicit"])


if __name__ == "__main__":
    unittest.main()
