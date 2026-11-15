import unittest

from hardsub_extractor.sources.jwplayer_public_adapter import (
    JWPlayerPublicAdapterError,
    parse_public_jwplayer_setup,
)


PLAYER = "https://player.example/video/example"


class JWPlayerItemBindingTests(unittest.TestCase):
    def test_multiple_hls_playlist_items_fail_closed(self):
        html = '''<script>
        jwplayer("player").setup({
          "playlist": [
            {
              "sources": [{"file": "https://media.example/ad.m3u8", "type": "hls"}],
              "tracks": [{"kind": "captions", "file": "/subs/ad.vtt"}]
            },
            {
              "sources": [{"file": "https://media.example/main.m3u8", "type": "hls", "default": true}],
              "tracks": [{"kind": "captions", "file": "/subs/main-ko.vtt"}]
            }
          ]
        });
        </script>'''

        with self.assertRaisesRegex(JWPlayerPublicAdapterError, "ambiguous"):
            parse_public_jwplayer_setup(html, PLAYER)

    def test_multiple_hls_setups_fail_closed(self):
        html = '''<script>
        jwplayer("ad").setup({
          "playlist": [{
            "sources": [{"file": "https://media.example/ad.m3u8", "type": "hls"}],
            "tracks": [{"kind": "captions", "file": "/subs/ad.vtt"}]
          }]
        });
        jwplayer("main").setup({
          "playlist": [{
            "sources": [{"file": "https://media.example/main.m3u8", "type": "hls", "default": true}],
            "tracks": [{"kind": "subtitles", "file": "/subs/main.vtt"}]
          }]
        });
        </script>'''

        with self.assertRaisesRegex(JWPlayerPublicAdapterError, "ambiguous"):
            parse_public_jwplayer_setup(html, PLAYER)

    def test_only_hls_item_supplies_tracks_even_if_other_item_is_progressive(self):
        html = '''<script>
        jwplayer("player").setup({
          "playlist": [
            {
              "sources": [{"file": "https://media.example/preview.mp4", "type": "video/mp4"}],
              "tracks": [{"kind": "captions", "file": "/subs/preview.vtt"}]
            },
            {
              "sources": [{"file": "https://media.example/main.m3u8", "type": "hls"}],
              "tracks": [{"kind": "captions", "file": "/subs/main-ko.vtt"}]
            }
          ]
        });
        </script>'''

        result = parse_public_jwplayer_setup(html, PLAYER)

        self.assertEqual(result.hls_url, "https://media.example/main.m3u8")
        self.assertEqual(
            [track.url for track in result.subtitle_candidates],
            ["https://player.example/subs/main-ko.vtt"],
        )
        self.assertEqual(result.media_candidate_count, 2)

    def test_loose_html_caption_object_is_not_attached_to_strict_hls(self):
        html = '''<script>
        var unrelated = {kind: "captions", file: "/subs/unrelated.vtt"};
        jwplayer("main").setup({
          "playlist": [{
            "sources": [{"file": "https://media.example/main.m3u8", "type": "hls"}]
          }]
        });
        </script>'''

        result = parse_public_jwplayer_setup(html, PLAYER)

        self.assertEqual(result.hls_url, "https://media.example/main.m3u8")
        self.assertEqual(result.subtitle_candidates, [])


if __name__ == "__main__":
    unittest.main()
