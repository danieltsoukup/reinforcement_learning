from src.learning import FirstTimeMarker


def test_marker_pop_true():
    marker = FirstTimeMarker()
    is_first = marker.pop(0)

    assert is_first is True


def test_marker_second_false():
    marker = FirstTimeMarker()
    _ = marker.pop(0)

    assert marker[0] is False


def test_marker_get_true():
    marker = FirstTimeMarker()

    assert marker[0] is True
