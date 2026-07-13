# ArmTune Devpost Draft

This is an early draft skeleton. It must not be submitted until real Arm
validation is complete and all result claims are backed by artifacts.

## Project Name

ArmTune

## Tagline

Reproducible optimization, profiling, and performance passports for AI
inference on Arm64 CPUs.

## Track

Cloud AI

## Overview

ArmTune is an Arm-first AI inference autotuning subsystem inside PerfRunbench.
It helps developers run controlled llama.cpp experiments on real Linux Arm64
targets, compare portable and optimized configurations, validate output
quality, and generate a versioned Arm Performance Passport that captures the
evidence needed to reproduce and audit the result.

## Problem

Developers can run AI inference on Arm64 servers, but it is hard to prove what
CPU and features were present, what runtime path was actually used, whether a
speedup is larger than noise, whether output quality was preserved, and whether
another developer can reproduce the same experiment.

## Planned Functionality

- Controller/target workflow for local or SSH Linux Arm64 targets.
- Arm host identity, feature, and topology inspection.
- llama.cpp build and runtime evidence.
- Repeated baseline and candidate measurements.
- Quality validation using a deterministic reference workload.
- Arm Performance Passport JSON plus HTML and Markdown reports.
- Passport comparison and CI regression gate.
- Optional Arm Performix profiling evidence.

## Result Claims

No measured Arm result is available yet. This section must be filled only after
Phase 6 real Arm validation.

## Setup

To be completed after Phase 1 and Phase 2 implementation.

## Demo

To be completed after the reference BootTrace workflow is implemented.
