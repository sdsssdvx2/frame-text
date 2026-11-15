"""No-network contract tests for independent public subtitle proxy routing."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hardsub_extractor.gui import ExtractorGUI, _user_error_summary
from hardsub_extractor.subtitle_acquisition import (
    CaptionAccessError, PublicCaptionCollector,
)
from hardsub_extractor.subtitle_batch_cli import main as cli_main
from hardsub_extractor.subtitle_proxy import (
    CaptionProxyRoute, parse_site_proxy_specs, route_for_site,
    validate_proxy_url,
)

PROXY = "socks5h://127.0.0.1:17920"
YOUTUBE = "https://www.youtube.com/watch?v=sample"
WETV = "https://wetv.vip/en/play/sample"
TEXT = "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n한국어 자막\n".encode("utf-8")


class Var:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


class ProxyFakeRunner:
    def __init__(self, meta=None):
        self.metas = {"": meta or {"id": "proxy_test", "subtitles": {"ko": [{"ext": "vtt"}]}}}
        self.commands = []
        self.fail_metadata = set()
        self.fail_download = set()

    def __call__(self, command, *, timeout_seconds, max_stdout_bytes):
        self.commands.append(list(command))
        route = command[command.index("--proxy") + 1]
        if "--dump-single-json" in command:
            if route in self.fail_metadata:
                raise CaptionAccessError("provider not reachable")
            return json.dumps(self.metas[route]).encode("utf-8")
        if route in self.fail_download:
            raise CaptionAccessError("subtitle not reachable")
        track = command[command.index("--sub-langs") + 1][1:-1]
        extension = command[command.index("--sub-format") + 1]
        target = command[command.index("--output") + 1]
        Path(target.replace("%(ext)s", track + "." + extension)).write_bytes(TEXT)
        return b""


class ProxyPolicyTests(unittest.TestCase):
    def test_explicit_standard_schemes_and_remote_dns_supported(self):
        for address in (
            PROXY, "socks5://127.0.0.1:17920",
            "http://proxy.example:8080", "socks5h://[::1]:17920",
        ):
            with self.subTest(address=address):
                self.assertEqual(validate_proxy_url(address), address)

    def test_reject_bad_or_private_proxy_parameters_before_running(self):
        for address in (
            "", "ftp://127.0.0.1:17920", "file:///secret",
            "socks5h://user:pass@127.0.0.1:17920",
            "http://127.0.0.1", "socks5://127.0.0.1:99999",
            "socks5://127.0.0.1:0",
            "socks5://127.0.0.1:17920?x=y",
            "socks5://127.0.0.1:17920/#fragment",
            "socks5://127.0.0.1:17920/extra",
            "socks5://127.0.0.1:17920\n--no-proxy",
        ):
            with self.subTest(address=address), self.assertRaises(ValueError):
                validate_proxy_url(address)

    def test_default_is_direct_and_rejects_ambiguous_routes(self):
        self.assertEqual(CaptionProxyRoute().mode, "direct")
        for mode, url in (
            ("direct", PROXY), ("unknown", ""),
            ("required", ""), ("fallback", ""),
        ):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                CaptionProxyRoute(mode, url)

    def test_site_routes_are_explicit_independent_and_not_wetv_hardcoded(self):
        mapping = parse_site_proxy_specs(
            required=[f"wetv={PROXY}"],
            fallback=["youtube=http://127.0.0.1:8080"],
        )
        self.assertEqual(mapping["wetv"].mode, "required")
        self.assertEqual(mapping["youtube"].mode, "fallback")
        self.assertEqual(route_for_site("wetv", CaptionProxyRoute(), mapping), mapping["wetv"])
        self.assertEqual(route_for_site("youtube", CaptionProxyRoute(), mapping), mapping["youtube"])
        self.assertEqual(route_for_site("other-site", CaptionProxyRoute(), mapping).mode, "direct")

    def test_reject_duplicate_or_unsupported_site_policy(self):
        for required, fallback in (
            ([f"wetv={PROXY}", f"wetv={PROXY}"], []),
            ([f"wetv={PROXY}"], [f"wetv={PROXY}"]),
            ([f"other={PROXY}"], []),
            (["wetv"], []),
        ):
            with self.assertRaises(ValueError):
                parse_site_proxy_specs(required, fallback)


class ProxyCollectorTests(unittest.TestCase):
    def collect(self, runner, route, *, url=WETV, languages=("ko",), by_site=None):
        collector = PublicCaptionCollector(
            "yt-dlp-test", command_runner=runner,
            proxy_route=route, site_proxy_routes=by_site,
        )
        with tempfile.TemporaryDirectory() as td:
            results = collector.collect(url, td, languages=languages)
            if results[0].status == "downloaded":
                self.assertEqual(results[0].original.read_bytes(), TEXT)
                self.assertTrue(results[0].srt.exists())
                self.assertTrue(results[0].ass.exists())
                manifest = json.loads(results[0].manifest.read_text(encoding="utf-8"))
                self.assertNotIn(PROXY, str(manifest))
            return results

    def test_default_direct_is_explicit_even_if_environment_has_a_proxy(self):
        runner = ProxyFakeRunner()
        result = self.collect(runner, CaptionProxyRoute())[0]
        self.assertEqual(result.status, "downloaded")
        self.assertEqual(result.connection_route, "direct")
        self.assertEqual(len(runner.commands), 2)
        for command in runner.commands:
            self.assertEqual(command[command.index("--proxy") + 1], "")
            self.assertIn("--ignore-config", command)
            self.assertIn("--skip-download", command)
            self.assertNotIn("--cookies-from-browser", command)

    def test_required_proxy_applies_to_metadata_and_download(self):
        runner = ProxyFakeRunner()
        runner.metas[PROXY] = runner.metas[""]
        result = self.collect(runner, CaptionProxyRoute("required", PROXY))[0]
        self.assertEqual(result.status, "downloaded")
        self.assertEqual(result.connection_route, "proxy")
        self.assertEqual(
            [command[command.index("--proxy") + 1] for command in runner.commands],
            [PROXY, PROXY],
        )

    def test_required_proxy_failure_never_direct_falls_back(self):
        runner = ProxyFakeRunner()
        runner.fail_metadata.add(PROXY)
        collector = PublicCaptionCollector(
            "yt-dlp-test", command_runner=runner,
            proxy_route=CaptionProxyRoute("required", PROXY),
        )
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(CaptionAccessError):
                collector.collect(WETV, td)
            self.assertFalse(list(Path(td).iterdir()))
        self.assertEqual(len(runner.commands), 1)
        self.assertEqual(runner.commands[0][runner.commands[0].index("--proxy") + 1], PROXY)

    def test_direct_metadata_failure_retries_proxy_as_one_complete_route(self):
        runner = ProxyFakeRunner()
        runner.metas[PROXY] = runner.metas[""]
        runner.fail_metadata.add("")
        result = self.collect(runner, CaptionProxyRoute("fallback", PROXY))[0]
        self.assertEqual(result.status, "downloaded")
        self.assertEqual(result.connection_route, "proxy")
        self.assertEqual(
            [c[c.index("--proxy") + 1] for c in runner.commands],
            ["", PROXY, PROXY],
        )

    def test_missing_direct_language_retries_proxied_metadata_and_bytes(self):
        runner = ProxyFakeRunner({"id": "proxy_test", "subtitles": {}})
        runner.metas[PROXY] = {
            "id": "proxy_test", "subtitles": {"ko": [{"ext": "vtt"}]},
        }
        result = self.collect(runner, CaptionProxyRoute("fallback", PROXY))[0]
        self.assertEqual(result.status, "downloaded")
        self.assertEqual(result.connection_route, "proxy")
        self.assertEqual([c[c.index("--proxy") + 1] for c in runner.commands], ["", PROXY, PROXY])

    def test_partial_direct_success_kept_and_only_missing_language_retried(self):
        runner = ProxyFakeRunner({
            "id": "proxy_test", "subtitles": {"ko": [{"ext": "vtt"}]},
        })
        runner.metas[PROXY] = {
            "id": "proxy_test",
            "subtitles": {"ko": [{"ext": "vtt"}], "ja": [{"ext": "vtt"}]},
        }
        collector = PublicCaptionCollector(
            "yt-dlp-test", command_runner=runner,
            proxy_route=CaptionProxyRoute("fallback", PROXY),
        )
        with tempfile.TemporaryDirectory() as td:
            results = collector.collect(YOUTUBE, td, languages=("ko", "ja"))
            self.assertEqual([r.status for r in results], ["downloaded", "downloaded"])
            self.assertEqual([r.connection_route for r in results], ["direct", "proxy"])
            self.assertEqual(results[0].original.read_bytes(), TEXT)
            self.assertEqual(results[1].original.read_bytes(), TEXT)
            self.assertEqual(len(list(Path(td).glob("*.original.vtt"))), 2)
        self.assertEqual(
            [c[c.index("--proxy") + 1] for c in runner.commands],
            ["", "", PROXY, PROXY],
        )
        self.assertEqual(
            runner.commands[-1][runner.commands[-1].index("--sub-langs") + 1],
            "^ja$",
        )

    def test_partial_direct_success_is_reported_even_if_proxy_retry_fails(self):
        runner = ProxyFakeRunner({
            "id": "partial", "subtitles": {"ko": [{"ext": "vtt"}]},
        })
        runner.fail_metadata.add(PROXY)
        collector = PublicCaptionCollector(
            "yt-dlp-test", command_runner=runner,
            proxy_route=CaptionProxyRoute("fallback", PROXY),
        )
        with tempfile.TemporaryDirectory() as td:
            result = collector.collect(WETV, td, languages=("ko", "ja"))
            self.assertEqual(
                [item.status for item in result],
                ["downloaded", "proxy_unavailable"],
            )
            self.assertEqual(
                [item.connection_route for item in result],
                ["direct", "proxy"],
            )
            self.assertEqual(result[0].original.read_bytes(), TEXT)
            self.assertTrue(result[0].srt.is_file())
            self.assertFalse(list(Path(td).glob("*-ja-*")))
        self.assertEqual(
            [c[c.index("--proxy") + 1] for c in runner.commands],
            ["", "", PROXY],
        )

    def test_proxy_required_missing_subtitle_does_not_claim_direct(self):
        runner = ProxyFakeRunner()
        runner.metas[PROXY] = {"id": "only_proxy", "subtitles": {}}
        result = self.collect(
            runner, CaptionProxyRoute("required", PROXY),
        )[0]
        self.assertEqual(result.status, "needs_ocr")
        self.assertEqual(result.connection_route, "proxy")
        self.assertEqual(len(runner.commands), 1)
        self.assertEqual(runner.commands[0][runner.commands[0].index("--proxy") + 1], PROXY)

    def test_proxy_retry_failure_is_reported_not_hidden(self):
        runner = ProxyFakeRunner({"id": "proxy_test", "subtitles": {}})
        runner.fail_metadata.add(PROXY)
        collector = PublicCaptionCollector(
            "yt-dlp-test", command_runner=runner,
            proxy_route=CaptionProxyRoute("fallback", PROXY),
        )
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(CaptionAccessError, "proxy retry"):
                collector.collect(WETV, td)
        self.assertEqual(
            [c[c.index("--proxy") + 1] for c in runner.commands],
            ["", PROXY],
        )

    def test_per_site_required_does_not_change_other_sites(self):
        runner = ProxyFakeRunner()
        runner.metas[PROXY] = runner.metas[""]
        route = CaptionProxyRoute()
        overrides = {"wetv": CaptionProxyRoute("required", PROXY)}
        self.collect(runner, route, by_site=overrides, url=WETV)
        self.collect(runner, route, by_site=overrides, url=YOUTUBE)
        self.assertEqual(
            [c[c.index("--proxy") + 1] for c in runner.commands],
            [PROXY, PROXY, "", ""],
        )


class ProxyUiCliTests(unittest.TestCase):
    def test_gui_validates_proxy_before_network_and_preserves_default(self):
        with tempfile.TemporaryDirectory() as td:
            gui = ExtractorGUI.__new__(ExtractorGUI)
            gui.source_mode_var = Var("captions")
            gui.url_var = Var(YOUTUBE)
            gui.target_var = Var("")
            gui.out_var = Var(td)
            gui.languages_var = Var("ko")
            gui.proxy_mode_var = Var("required")
            gui.proxy_url_var = Var(PROXY)
            gui._validate()
            self.assertEqual(gui._caption_proxy_route, CaptionProxyRoute("required", PROXY))
            gui.proxy_url_var = Var("http://user:secret@localhost:17920")
            with self.assertRaises(ValueError):
                gui._validate()
            gui.proxy_mode_var = Var("direct")
            gui._validate()
            self.assertEqual(gui._caption_proxy_route, CaptionProxyRoute())

    def test_gui_error_redacts_socks_proxy_endpoint(self):
        summary = _user_error_summary(
            CaptionAccessError("failed via " + PROXY + " and https://private.example/x")
        )
        self.assertNotIn(PROXY, summary)
        self.assertNotIn("private.example", summary)

    def test_cli_site_required_and_default_direct_pass_structured_options(self):
        output = io.StringIO()
        with mock.patch(
            "hardsub_extractor.subtitle_batch_cli.PublicCaptionCollector"
        ) as adapter, contextlib.redirect_stdout(output):
            adapter.return_value.collect.return_value = []
            cli_main([
                "--url", WETV,
                "--site-proxy", f"wetv={PROXY}",
            ])
        kwargs = adapter.call_args.kwargs
        self.assertEqual(kwargs["proxy_route"], CaptionProxyRoute())
        self.assertEqual(kwargs["site_proxy_routes"]["wetv"], CaptionProxyRoute("required", PROXY))
        self.assertNotIn(PROXY, output.getvalue())

    def test_cli_rejects_malformed_proxy_before_adapter_is_created(self):
        with mock.patch(
            "hardsub_extractor.subtitle_batch_cli.PublicCaptionCollector"
        ) as adapter, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as exit_info:
                cli_main([
                    "--url", WETV,
                    "--proxy-mode", "required",
                    "--proxy", "socks5://user:pass@127.0.0.1:17920",
                ])
        self.assertEqual(exit_info.exception.code, 2)
        adapter.assert_not_called()


if __name__ == "__main__":
    unittest.main()
