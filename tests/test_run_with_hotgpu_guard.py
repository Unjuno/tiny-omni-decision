"""CPU-only checks for the cooperative GPU memory reservation guard."""

from scripts.run_with_hotgpu_guard import launch_conditions_met, reservation_size


def test_reservation_targets_block_threshold_with_child_headroom():
    assert reservation_size(16384, 53, 1, 14000) == 2587


def test_guard_accepts_only_guard_and_verified_ui_pids():
    assert launch_conditions_met(
        active_pids={11, 22},
        own_pids={11},
        safe_ui_processes={22},
        utilization=0,
        free_mib=13744,
        block_free_mib=14000,
        min_child_free_mib=12288,
    )


def test_guard_rejects_unknown_gpu_process_or_insufficient_headroom():
    args = dict(
        own_pids={11},
        safe_ui_processes={22},
        utilization=0,
        free_mib=13744,
        block_free_mib=14000,
        min_child_free_mib=12288,
    )
    assert not launch_conditions_met(active_pids={11, 22, 33}, **args)
    assert not launch_conditions_met(active_pids={11, 22}, **{**args, "free_mib": 12000})
