# Add modules here to access fixtures from other files
pytest_plugins = [
    "tests.unit.utilities.test_optics",
    "tests.unit.tracking.test_synchrotron",
    "tests.unit.tracking.test_parallel",
    "tests.unit.tracking.test_particle",
    "tests.unit.impedance.test_wakefield"
]
