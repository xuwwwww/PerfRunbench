from __future__ import annotations

import unittest

from autotune.profiler.hardware_info import generate_notes


class HardwareInfoTest(unittest.TestCase):
    def test_generate_notes_uses_input_facts(self) -> None:
        info = {
            "is_wsl": True,
            "total_memory_mb": 1024.0,
            "cpu_affinity_supported": False,
            "packages": {"torch": None, "onnxruntime": "1.0"},
            "runtime": {"torch_cuda_available": False, "onnxruntime_providers": ["CPUExecutionProvider"]},
            "limits": {"cgroup_memory_max_mb": None},
        }
        notes = generate_notes(info)
        self.assertTrue(any("WSL environment detected" in note for note in notes))
        self.assertTrue(any("PyTorch is not installed" in note for note in notes))
        self.assertTrue(any("CPU affinity is not supported" in note for note in notes))

    def test_generate_notes_empty_when_no_rules_match(self) -> None:
        info = {
            "is_wsl": False,
            "total_memory_mb": 1024.0,
            "cpu_affinity_supported": True,
            "packages": {"torch": "1.0", "onnxruntime": "1.0"},
            "runtime": {"torch_cuda_available": True, "onnxruntime_providers": ["CPUExecutionProvider"]},
            "limits": {"cgroup_memory_max_mb": None},
        }
        self.assertEqual(generate_notes(info), [])


if __name__ == "__main__":
    unittest.main()

