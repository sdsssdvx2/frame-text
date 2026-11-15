import unittest
from pathlib import Path

from hardsub_extractor import site_adapter


class SiteAdapterCoreTests(unittest.TestCase):
    def test_source_timeline_never_collides_with_target(self):
        base = Path("out") / "episode.srt"
        source = site_adapter.source_subtitle_base(base)
        self.assertNotEqual(source.with_suffix(".srt"), base.with_suffix(".srt"))
        self.assertEqual(source.parent, base.parent)

    def test_non_site_urls_are_never_site_episodes(self):
        for value in ("https://www.youtube.com/watch?v=x", "C:/video.mp4", "not a url"):
            self.assertFalse(site_adapter.is_site_episode_url(value))

    def test_public_build_has_no_adapter_or_matches_private_contract(self):
        adapter = site_adapter.SITE_ADAPTER
        if adapter is None:
            self.assertFalse(site_adapter.site_adapter_available())
            return
        for name in ("SITE_LABEL", "URL_LABEL", "extract_epsite",
                     "extract_and_sync_epsite", "AutoEpSiteClient",
                     "CurrentAutoEpSiteClient", "is_episode_url"):
            self.assertTrue(hasattr(adapter, name), name)


if __name__ == "__main__":
    unittest.main()
