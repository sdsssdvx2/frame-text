import unittest

from hardsub_extractor.sources.player_probe import jwplayer_media_candidates
from hardsub_extractor.sources.jwplayer_public_adapter import (
    JWPlayerPublicAdapterError,
    parse_public_jwplayer_setup,
)


BASE = "https://player.example/video/example"


class JWPlayerMediaDedupeTests(unittest.TestCase):
    def test_duplicate_url_merges_stricter_with_credentials_signal(self):
        item = {
            "sources": [
                {
                    "file": "https://media.example/master.m3u8",
                    "type": "hls",
                    "label": "Auto",
                    "withCredentials": False,
                }
            ],
            "allSources": [
                {
                    "file": "https://media.example/master.m3u8",
                    "withCredentials": True,
                    "default": True,
                }
            ],
        }

        found = jwplayer_media_candidates(item, BASE)

        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].source, "jwplayer-sources")
        self.assertEqual(found[0].media_type, "hls")
        self.assertEqual(found[0].label, "Auto")
        self.assertTrue(found[0].default)
        self.assertTrue(found[0].with_credentials)

    def test_duplicate_protection_signal_makes_public_adapter_fail_closed(self):
        html = '''<script>
        jwplayer("player").setup({
          "playlist": [{
            "sources": [{
              "file": "https://media.example/master.m3u8",
              "type": "hls",
              "withCredentials": false
            }],
            "allSources": [{
              "file": "https://media.example/master.m3u8",
              "withCredentials": true
            }]
          }]
        });
        </script>'''

        with self.assertRaisesRegex(JWPlayerPublicAdapterError, "withCredentials"):
            parse_public_jwplayer_setup(html, BASE)

    def test_duplicate_unprotected_metadata_keeps_single_candidate(self):
        item = {
            "file": "https://media.example/master.m3u8",
            "sources": [
                {
                    "file": "https://media.example/master.m3u8",
                    "type": "application/vnd.apple.mpegurl",
                    "label": "Auto",
                }
            ],
        }

        found = jwplayer_media_candidates(item, BASE)

        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].media_type, "hls")
        self.assertEqual(found[0].label, "Auto")
        self.assertFalse(found[0].with_credentials)


if __name__ == "__main__":
    unittest.main()
