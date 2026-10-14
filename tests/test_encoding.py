"""Exercise API text with the default file encodings used by Windows locales."""
import subprocess
import sys
import textwrap

import pytest

from analytics import ROOT


@pytest.mark.parametrize('default_encoding', ['cp1251', 'cp1252'])
def test_kazakh_labels_and_metadata_ignore_system_encoding(default_encoding):
    # A fresh interpreter matters: region names are loaded once at module import.
    script = textwrap.dedent('''
        import io
        import sys
        import tempfile
        from pathlib import Path

        original_read_text = Path.read_text
        def windows_read_text(self, encoding=None, errors=None, **kwargs):
            return original_read_text(
                self, encoding=encoding or sys.argv[1], errors=errors, **kwargs
            )
        Path.read_text = windows_read_text

        import app as application
        from analytics import ROOT

        with application.app.test_client() as client:
            response = client.get('/api/analysis')
            assert response.status_code == 200
            names = {r['id']: r['name'] for r in response.json['regions']}
            assert names['astana'] == 'Астана қаласы', ascii(names['astana'])
            assert names['almaty'] == 'Алматы қаласы', ascii(names['almaty'])
            assert names['north_kz'] == 'Солтүстік Қазақстан облысы'
            assert response.json['dataset']['title'] == 'Ұлттық статистика бюросының ресми деректері'
            assert response.json['dataset']['sources'][0]['title'].startswith('Барлық ағындар')

            # Uploaded source descriptions must survive UTF-8 write/read too.
            with tempfile.TemporaryDirectory() as folder:
                application.INSTANCE = Path(folder)
                source = 'Ұлттық статистика бюросы — қазақша дереккөз'
                uploaded = client.post('/api/import', data={
                    'file': (io.BytesIO((ROOT/'data/demo.csv').read_bytes()), 'data.csv'),
                    'source': source,
                    'boundaries': 'confirmed',
                })
                assert uploaded.status_code == 200
                dataset = uploaded.json['dataset_id']
                restored = client.get('/api/analysis', query_string={'dataset': dataset})
                assert restored.status_code == 200
                assert restored.json['dataset']['source'] == source
    ''')
    result = subprocess.run(
        [sys.executable, '-c', script, default_encoding], cwd=ROOT,
        capture_output=True, text=True, encoding='utf-8', timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
