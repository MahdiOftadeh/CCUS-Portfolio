from vpsa_core_lite.core import run_lite_simulation

def test_run_lite_simulation():
    res = run_lite_simulation({})
    assert res["status"] == "success"
