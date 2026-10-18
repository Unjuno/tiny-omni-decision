from scripts.run_with_hotgpu_guard import (
    build_child_environment,
    launch_conditions_met,
    reservation_size,
)


def test_reservation_closes_supervisor_free_memory_gate():
    amount = reservation_size(total_mib=16384, used_mib=1, reserve_mib=3200, block_free_mib=14000)

    assert amount == 3200
    assert 16384 - amount < 14000
    assert 16384 - amount >= 8000


def test_reservation_grows_when_free_memory_gate_requires_it():
    amount = reservation_size(total_mib=32768, used_mib=100, reserve_mib=3200, block_free_mib=14000)

    assert amount == 18924


def test_launch_gate_requires_quiet_gpu_no_external_pid_and_safe_free_memory():
    own = {100}

    assert launch_conditions_met({100}, own, 0, 12000, 14000, 8000)
    assert not launch_conditions_met({100, 200}, own, 0, 12000, 14000, 8000)
    assert not launch_conditions_met({100}, own, 5, 12000, 14000, 8000)
    assert not launch_conditions_met({100}, own, 0, 14000, 14000, 8000)
    assert not launch_conditions_met({100}, own, 0, 7000, 14000, 8000)


def test_child_environment_identifies_the_reserving_parent_process():
    environment = build_child_environment(
        {"PATH": "python"},
        parent_pid=123,
        slot={
            "reserved_mib": 3200,
            "free_mib": 13184,
            "total_mib": 16384,
            "utilization_pct": 0,
        },
        block_if_free_mib=14000,
        min_child_free_mib=8000,
    )

    assert environment["TINY_OMNI_GPU_GATE_PARENT_PID"] == "123"
    assert environment["TINY_OMNI_GPU_GATE_RESERVED_MIB"] == "3200"
    assert environment["TINY_OMNI_GPU_GATE_FREE_MIB"] == "13184"
