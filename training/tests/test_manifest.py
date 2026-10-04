import unittest

from data.manifest import validate

BASE = {
    "path": "cv/en/1.wav", "kind": "negative", "split": "train", "source": "Common Voice",
    "url": "https://commonvoice.mozilla.org", "license": "CC0-1.0",
    "license_url": "https://creativecommons.org/publicdomain/zero/1.0/", "retrieved": "2026-10-03",
    "duration_seconds": "4.2", "speaker_id": "", "release_id": "",
}


def row(**overrides):
    return {**BASE, **overrides}


class ManifestTests(unittest.TestCase):
    def test_clean_rows_pass(self):
        report = validate([row(), row(path="cv/en/2.wav", license="CC-BY-4.0")])
        self.assertTrue(report.ok, report.errors)
        self.assertAlmostEqual(report.hours[("negative", "train")], 8.4 / 3600)

    def test_rejects_non_commercial_and_unknown_licenses(self):
        report = validate([
            row(path="a", license="CC-BY-NC-SA-4.0"),
            row(path="b", license="CC-BY-ND-4.0"),
            row(path="c", license="Unknown"),
        ])
        self.assertEqual(3, len(report.errors))
        self.assertIn("non-commercial", report.errors[0])

    def test_share_alike_needs_explicit_sign_off(self):
        self.assertFalse(validate([row(license="CC-BY-SA-4.0")]).ok)
        self.assertTrue(validate([row(license="CC-BY-SA-4.0")], allow_share_alike=True).ok)

    def test_own_recordings_need_a_release_and_held_out_speakers(self):
        report = validate([
            row(path="p1", kind="positive", split="train", license="Contributor-Release", speaker_id="s1", release_id="R-001"),
            row(path="p2", kind="positive", split="test", license="Contributor-Release", speaker_id="s1", release_id="R-001"),
            row(path="p3", kind="positive", split="test", license="Contributor-Release", speaker_id="s2", release_id=""),
        ])
        self.assertTrue(any("signed release_id" in e for e in report.errors))
        self.assertTrue(any("speaker s1" in e for e in report.errors))

    def test_duplicates_and_bad_provenance(self):
        report = validate([row(), row(url="", retrieved="last week", duration_seconds="0")])
        self.assertEqual(4, len(report.errors))


if __name__ == "__main__":
    unittest.main()
