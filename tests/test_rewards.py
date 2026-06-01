from cleaner.rewards import badges_for, level_from_xp, xp_progress


def test_level_from_xp_starts_at_one():
    assert level_from_xp(0) == 1
    assert level_from_xp(29) == 1
    assert level_from_xp(30) == 2


def test_xp_progress_bounds():
    _, _, progress = xp_progress(10)
    assert 0.0 <= progress <= 1.0


def test_badges():
    badges = badges_for(total_labeled=120, discard_mb=1200, train_count=1)
    assert "100 decisions" in badges
    assert "1GB saved" in badges
    assert "First AI training" in badges
