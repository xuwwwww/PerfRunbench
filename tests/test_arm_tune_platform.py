from __future__ import annotations

import unittest

from autotune.arm_tune.platform.architecture import collect_architecture_info
from autotune.arm_tune.platform.cpu_features import merge_features
from autotune.arm_tune.platform.cpu_identity import merge_cpu_identity, parse_cpuinfo_identity, parse_lscpu_identity
from autotune.arm_tune.platform.topology import parse_lscpu_topology


LSCPU_JSON = """{
  "lscpu": [
    {"field": "Architecture:", "data": "aarch64"},
    {"field": "Vendor ID:", "data": "ARM"},
    {"field": "Model name:", "data": "Neoverse-V2"},
    {"field": "CPU(s):", "data": "72"},
    {"field": "On-line CPU(s) list:", "data": "0-71"},
    {"field": "Thread(s) per core:", "data": "1"},
    {"field": "Core(s) per socket:", "data": "72"},
    {"field": "Socket(s):", "data": "1"},
    {"field": "NUMA node(s):", "data": "4"},
    {"field": "Flags:", "data": "fp asimd asimddp sve sve2 i8mm bf16"}
  ]
}"""


CPUINFO = """processor   : 0
BogoMIPS    : 2000.00
Features    : fp asimd asimddp sve sve2 i8mm bf16
CPU implementer : 0x41
CPU architecture: 8
CPU variant : 0x0
CPU part    : 0xd49
CPU revision    : 0
"""


class ArmTunePlatformTest(unittest.TestCase):
    def test_collect_architecture_identifies_linux_arm64(self) -> None:
        info = collect_architecture_info(system="Linux", machine="arm64", kernel_release="6.8.0")

        self.assertTrue(info.is_linux)
        self.assertTrue(info.is_arm64)
        self.assertEqual(info.normalized_machine, "aarch64")

    def test_cpu_identity_merges_cpuinfo_and_lscpu_without_guessing(self) -> None:
        identity = merge_cpu_identity(parse_cpuinfo_identity(CPUINFO), parse_lscpu_identity(LSCPU_JSON))

        self.assertEqual(identity.model_name, "Neoverse-V2")
        self.assertEqual(identity.implementer, "0x41")
        self.assertEqual(identity.part, "0xd49")

    def test_unknown_identity_remains_unknown(self) -> None:
        identity = parse_cpuinfo_identity("")

        self.assertEqual(identity.model_name, "unknown")
        self.assertEqual(identity.part, "unknown")

    def test_capabilities_are_feature_evidence_only(self) -> None:
        capabilities = merge_features(CPUINFO, LSCPU_JSON)

        self.assertTrue(capabilities.neon)
        self.assertTrue(capabilities.dotprod)
        self.assertTrue(capabilities.sve)
        self.assertTrue(capabilities.sve2)
        self.assertTrue(capabilities.i8mm)
        self.assertTrue(capabilities.bf16)
        self.assertIn("i8mm", capabilities.features)

    def test_topology_parses_lscpu_json(self) -> None:
        topology = parse_lscpu_topology(LSCPU_JSON)

        self.assertEqual(topology.logical_cpus, 72)
        self.assertEqual(topology.cores_per_socket, 72)
        self.assertEqual(topology.sockets, 1)
        self.assertEqual(topology.numa_nodes, 4)


if __name__ == "__main__":
    unittest.main()
