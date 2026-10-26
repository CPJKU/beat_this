import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from beat_this import desktop_i18n as i18n


class LanguageTests(unittest.TestCase):
    def test_all_desktop_messages_are_localized_and_covered(self):
        import ast
        root = Path(i18n.__file__).parent
        for path in [*root.glob('desktop*.py'),root/'tempo_analysis.py']:
            if path.name == 'desktop_i18n.py':
                continue
            tree=ast.parse(path.read_text(encoding='utf-8'))
            parents={child:parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
            for node in ast.walk(tree):
                if not (isinstance(node,ast.Constant) and isinstance(node.value,str)):
                    continue
                if node.value=='עברית' or not any('\u0590'<=c<='\u05ff' for c in node.value):
                    continue
                with self.subTest(file=path.name,line=node.lineno):
                    parent=parents.get(node)
                    self.assertIsInstance(parent,ast.Call)
                    self.assertEqual(getattr(parent.func,'id',None),'tr')
                    self.assertIn(node.value,i18n.ENGLISH)

    def test_english_is_default_and_invalid_preferences_fall_back(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {'LOCALAPPDATA':folder}):
            with patch.dict(os.environ, {'BPM_FINDER_LANGUAGE':''}):
                self.assertEqual(i18n.read_language(), 'en')
                i18n.preference_file().parent.mkdir(parents=True)
                i18n.preference_file().write_text('broken',encoding='utf-8')
                self.assertEqual(i18n.read_language(), 'en')

    def test_preference_survives_and_only_supported_languages_are_saved(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {'LOCALAPPDATA':folder, 'BPM_FINDER_LANGUAGE':''}):
            i18n.save_language('he')
            self.assertEqual(i18n.read_language(), 'he')
            i18n.save_language('en')
            self.assertEqual(i18n.read_language(), 'en')
            with self.assertRaises(ValueError):
                i18n.save_language('invalid')

    def test_both_languages_and_formatted_values(self):
        with patch.object(i18n, 'LANGUAGE', 'en'):
            self.assertEqual(i18n.tr('התחלת ניתוח'), 'Start analysis')
            self.assertEqual(i18n.tr('מנתח {0} מתוך {1} · {2}', 2, 5, 'שיר.wav'), 'Analyzing 2 of 5 · שיר.wav')
            self.assertEqual(i18n.tr(' מתוך {0:.1f} MB', 12.345), ' of 12.3 MB')
        with patch.object(i18n, 'LANGUAGE', 'he'):
            self.assertEqual(i18n.tr('התחלת ניתוח'), 'התחלת ניתוח')
            self.assertEqual(i18n.tr('עמוד {0}',3), 'עמוד 3')

    def test_translation_fields_match_originals(self):
        from string import Formatter
        formatter = Formatter()
        for original, translated in i18n.ENGLISH.items():
            with self.subTest(message=original):
                fields = lambda text: sorted((name, spec, conversion) for _, name, spec, conversion in formatter.parse(text) if name is not None)
                self.assertEqual(fields(original), fields(translated))


if __name__ == '__main__':
    unittest.main()
