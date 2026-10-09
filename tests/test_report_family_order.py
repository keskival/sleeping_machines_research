"""A multi-page public benchmark record precedes the complete model chapter."""
import pytest
import pymupdf
from report import family_report


@pytest.mark.parametrize('prior_family', [False, True])
def test_benchmark_continuation_stays_before_family(tmp_path, prior_family):
    source, chapter, output = [tmp_path / name for name in ('source.pdf', 'family.pdf', 'output.pdf')]
    with pymupdf.open() as doc:
        labels = ['Cover and public benchmark record']
        if prior_family:
            labels.append(family_report.MARKER)
        labels.extend(['Public benchmark record continued: Stack Overflow', 'Appendix A. Deep event recognition'])
        for label in labels:
            doc.new_page().insert_text((50, 50), label)
        doc.save(source)
    with pymupdf.open() as doc:
        for _ in family_report.sections():
            doc.new_page().insert_text((50, 50), family_report.MARKER)
        doc.save(chapter)
    result = family_report.integrate(source, output, chapter)
    assert result['front_pages'] == 2
    with pymupdf.open(output) as doc:
        assert 'Cover and public benchmark record' in doc[0].get_text()
        assert 'record continued: Stack Overflow' in doc[1].get_text()
        assert family_report.MARKER in doc[2].get_text()
        assert 'Appendix A. Deep event recognition' in doc[-1].get_text()
