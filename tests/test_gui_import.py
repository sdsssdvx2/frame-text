import unittest

class GUIImportTests(unittest.TestCase):
    def test_gui_module_imports_without_creating_window(self):
        import hardsub_extractor.gui as gui
        self.assertTrue(callable(gui.main))
        self.assertTrue(hasattr(gui,"ExtractorGUI"))

if __name__=="__main__":
    unittest.main()
