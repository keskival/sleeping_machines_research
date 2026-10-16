from scripts.public_headline_consistency import text_problem

def test_stale_generated_headline_is_rejected_even_when_other_numbers_are_correct():
    assert text_problem('Nine public leaderboard wins;53 of61 seed/split scores ahead.',13)
    assert text_problem('13 public benchmark wins. Nine public leaderboard wins.',13)

def test_wrapped_and_markdown_headlines_are_accepted():
    assert text_problem('**13 public benchmark wins**;53 of61 seed/split scores ahead.',13) is None
    assert text_problem('13 public\nbenchmark wins. Best of nine classical detectors.',13) is None
