import unittest

from hardsub_extractor.sources.jwplayer_public_adapter import (
    JWPlayerPublicAdapterError,
    parse_public_jwplayer_setup,
)


class JWPlayerPublicAdapterTests(unittest.TestCase):
    PLAYER = "https://player.example/video/example"

    def test_strict_json_hls_and_tracks_are_exposed(self):
        html = '''<script>
        jwplayer("player").setup({
          "playlist": [{
            "sources": [
              {"file": "https://media.example/master.m3u8", "type": "hls", "default": true},
              {"file": "https://media.example/backup.mp4", "type": "video/mp4"}
            ],
            "tracks": [
              {"kind": "captions", "file": "/subs/ko.vtt", "label": "Korean"}
            ]
          }]
        });
        </script>'''

        result = parse_public_jwplayer_setup(html, self.PLAYER)

        self.assertEqual(result.hls_url, "https://media.example/master.m3u8")
        self.assertEqual(result.strict_setup_count, 1)
        self.assertEqual(result.media_candidate_count, 2)
        self.assertEqual(len(result.subtitle_candidates), 1)
        self.assertEqual(
            result.subtitle_candidates[0].url,
            "https://player.example/subs/ko.vtt",
        )

    def test_default_hls_is_preferred(self):
        html = '''<script>
        jwplayer("player").setup({
          "playlist": [{
            "sources": [
              {"file": "https://a.example/first.m3u8", "type": "hls"},
              {"file": "https://b.example/default.m3u8", "type": "hls", "default": true}
            ]
          }]
        });
        </script>'''

        result = parse_public_jwplayer_setup(html, self.PLAYER)
        self.assertEqual(result.hls_url, "https://b.example/default.m3u8")

    def test_string_default_false_is_not_treated_as_true(self):
        html = '''<script>
        jwplayer("player").setup({
          "playlist": [{
            "sources": [
              {"file": "https://a.example/first.m3u8", "type": "hls", "default": "false"},
              {"file": "https://b.example/default.m3u8", "type": "hls", "default": "true"}
            ]
          }]
        });
        </script>'''

        result = parse_public_jwplayer_setup(html, self.PLAYER)
        self.assertEqual(result.hls_url, "https://b.example/default.m3u8")

    def test_javascript_object_literal_is_not_evaluated(self):
        html = '''<script>
        jwplayer("player").setup({playlist:[{sources:[{file:"/master.m3u8",type:"hls"}]}]});
        </script>'''
        with self.assertRaisesRegex(JWPlayerPublicAdapterError, "strict-JSON"):
            parse_public_jwplayer_setup(html, self.PLAYER)

    def test_mp4_only_setup_remains_unsupported(self):
        html = '''<script>
        jwplayer("player").setup({
          "playlist": [{"file": "https://media.example/video.mp4"}]
        });
        </script>'''
        with self.assertRaisesRegex(JWPlayerPublicAdapterError, "HLS"):
            parse_public_jwplayer_setup(html, self.PLAYER)

    def test_with_credentials_hls_remains_unsupported(self):
        html = '''<script>
        jwplayer("player").setup({
          "playlist": [{
            "withCredentials": true,
            "sources": [{"file": "https://media.example/master.m3u8", "type": "hls"}]
          }]
        });
        </script>'''
        with self.assertRaisesRegex(JWPlayerPublicAdapterError, "withCredentials"):
            parse_public_jwplayer_setup(html, self.PLAYER)

    def test_string_with_credentials_false_is_accepted(self):
        html = '''<script>
        jwplayer("player").setup({
          "playlist": [{
            "withCredentials": "false",
            "sources": [{"file": "https://media.example/master.m3u8", "type": "hls"}]
          }]
        });
        </script>'''
        result = parse_public_jwplayer_setup(html, self.PLAYER)
        self.assertEqual(result.hls_url, "https://media.example/master.m3u8")

    def test_unknown_with_credentials_value_fails_closed(self):
        html = '''<script>
        jwplayer("player").setup({
          "playlist": [{
            "withCredentials": "maybe",
            "sources": [{"file": "https://media.example/master.m3u8", "type": "hls"}]
          }]
        });
        </script>'''
        with self.assertRaisesRegex(JWPlayerPublicAdapterError, "withCredentials"):
            parse_public_jwplayer_setup(html, self.PLAYER)

    def test_credential_bearing_hls_is_not_accepted(self):
        html = '''<script>
        jwplayer("player").setup({
          "playlist": [{
            "sources": [{"file": "https://user:secret@media.example/master.m3u8", "type": "hls"}]
          }]
        });
        </script>'''
        with self.assertRaisesRegex(JWPlayerPublicAdapterError, "HLS"):
            parse_public_jwplayer_setup(html, self.PLAYER)

    def test_insecure_remote_http_hls_is_not_accepted(self):
        html = '''<script>
        jwplayer("player").setup({
          "playlist": [{
            "sources": [{"file": "http://media.example/master.m3u8", "type": "hls"}]
          }]
        });
        </script>'''
        with self.assertRaisesRegex(JWPlayerPublicAdapterError, "HLS"):
            parse_public_jwplayer_setup(html, self.PLAYER)

    def test_explicit_localhost_http_fixture_is_allowed(self):
        html = '''<script>
        jwplayer("player").setup({
          "playlist": [{
            "sources": [{"file": "http://127.0.0.1:8765/master.m3u8", "type": "hls"}],
            "tracks": [{"kind": "captions", "file": "http://localhost:8765/ko.vtt"}]
          }]
        });
        </script>'''
        result = parse_public_jwplayer_setup(
            html,
            self.PLAYER,
            allow_localhost_http=True,
        )
        self.assertEqual(result.hls_url, "http://127.0.0.1:8765/master.m3u8")
        self.assertEqual(
            [item.url for item in result.subtitle_candidates],
            ["http://localhost:8765/ko.vtt"],
        )

    def test_non_web_subtitle_is_filtered(self):
        html = '''<script>
        jwplayer("player").setup({
          "playlist": [{
            "sources": [{"file": "https://media.example/master.m3u8", "type": "hls"}],
            "tracks": [
              {"kind": "captions", "file": "javascript:alert(1)"},
              {"kind": "captions", "file": "https://sub.example/ko.vtt"}
            ]
          }]
        });
        </script>'''

        result = parse_public_jwplayer_setup(html, self.PLAYER)
        self.assertEqual(
            [item.url for item in result.subtitle_candidates],
            ["https://sub.example/ko.vtt"],
        )


if __name__ == "__main__":
    unittest.main()
