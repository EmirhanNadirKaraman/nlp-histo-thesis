

def pytest_runtest_logstart(nodeid, location):
    """Print the test name to the terminal before each test runs."""
    print(f"\n>>> {nodeid}", flush=True)
