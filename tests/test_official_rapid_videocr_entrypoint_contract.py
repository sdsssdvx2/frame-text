"""Regression contract for the isolated official RapidVideOCR research entrypoint.

Does not download models, invoke OCR, or touch existing subtitle files.
"""
import ast
from pathlib import Path
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "tools" / "compare_official_rapid_videocr.py"


class ExplicitPathArgumentsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        cls.main = next(
            node for node in cls.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )

    def test_paths_are_required_keyword_only(self):
        arguments = self.main.args
        self.assertEqual([arg.arg for arg in arguments.kwonlyargs], ["model_root", "manifest"])
        self.assertEqual(arguments.kw_defaults, [None, None])

    def test_no_module_global_path_assignment(self):
        assigned = set()
        for node in self.tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    assigned.update(
                        child.id for child in ast.walk(target)
                        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)
                    )
        self.assertFalse({"model_root", "manifest"} & assigned)

    def test_cli_forwards_both_explicit_paths(self):
        entrypoints = [
            node for node in self.tree.body
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
        ]
        self.assertEqual(len(entrypoints), 1)
        calls = [
            node for node in ast.walk(entrypoints[0])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name) and node.func.id == "main"
        ]
        self.assertEqual(len(calls), 1)
        keywords = {item.arg: item.value for item in calls[0].keywords}
        self.assertEqual(set(keywords), {"model_root", "manifest"})
        self.assertEqual(ast.unparse(keywords["model_root"]), "options.model_dir")
        self.assertEqual(ast.unparse(keywords["manifest"]), "options.manifest")

    def test_main_compiles_without_optional_dependencies(self):
        compile(self.tree, str(SOURCE), "exec")


if __name__ == "__main__":
    unittest.main()
