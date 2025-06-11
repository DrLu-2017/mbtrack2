import numpy as np

from mbtrack2 import Beam, Bunch


def assert_attr_changed(element,
                        bunch,
                        attrs_changed=["xp", "yp", "delta"],
                        change=True):

    if isinstance(bunch, Bunch):
        assert_attr_changed_bunch(element, bunch, attrs_changed, change=change)
    elif isinstance(bunch, Beam):
        assert_attr_changed_beam(element, bunch, attrs_changed, change=change)
    else:
        raise TypeError


def assert_attr_changed_bunch(element,
                              bunch,
                              attrs_changed=["xp", "yp", "delta"],
                              change=True):

    attrs = ["x", "xp", "y", "yp", "tau", "delta"]
    attrs_unchanged = [attr for attr in attrs if attr not in attrs_changed]

    initial_values_changed = {
        attr: bunch[attr].copy()
        for attr in attrs_changed
    }

    initial_values_unchanged = {
        attr: bunch[attr].copy()
        for attr in attrs_unchanged
    }

    element.track(bunch)

    for attr in attrs_changed:
        if change:
            assert not np.array_equal(initial_values_changed[attr],
                                      bunch[attr]), f"{attr}"
        else:
            assert np.array_equal(initial_values_changed[attr],
                                  bunch[attr]), f"{attr}"

    for attr in attrs_unchanged:
        assert np.array_equal(initial_values_unchanged[attr],
                              bunch[attr]), f"{attr}"


def assert_attr_changed_beam(element,
                             beam,
                             attrs_changed=["xp", "yp", "delta"],
                             change=True):

    attrs = ["x", "xp", "y", "yp", "tau", "delta"]
    attrs_unchanged = [attr for attr in attrs if attr not in attrs_changed]

    initial_values_changed_b = [{
        attr: bunch[attr].copy()
        for attr in attrs_changed
    } for bunch in beam]

    initial_values_unchanged_b = [{
        attr: bunch[attr].copy()
        for attr in attrs_unchanged
    } for bunch in beam]

    element.track(beam)

    for i, bunch in enumerate(beam):
        initial_values_changed = initial_values_changed_b[i]
        initial_values_unchanged = initial_values_unchanged_b[i]
        for attr in attrs_changed:
            if change and (bunch.charge != 0):
                assert not np.array_equal(initial_values_changed[attr],
                                          bunch[attr]), f"{attr}"
            else:
                assert np.array_equal(initial_values_changed[attr],
                                      bunch[attr]), f"{attr}"

        for attr in attrs_unchanged:
            assert np.array_equal(initial_values_unchanged[attr],
                                  bunch[attr]), f"{attr}"
