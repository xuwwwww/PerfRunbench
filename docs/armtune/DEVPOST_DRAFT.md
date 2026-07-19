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

## Implemented Evidence Pipeline

- Controller/target workflow for local or SSH Linux Arm64 targets.
- Arm host identity, feature, and topology inspection.
- llama.cpp build and runtime evidence.
- Repeated baseline and candidate measurements.
- Deterministic quality validation using a public synthetic BootTrace
  reference workload.
- Quality calibration with strict raw-output mode and optional audited
  fenced-JSON normalization.
- Declarative build-profile evidence for `portable-baseline`, `arm-native`,
  and `arm-optimized`.
- Quality-aware profile comparison and Performance Passport JSON/Markdown
  generation with digest verification.

## Planned Functionality

- Real quality-preserving build-profile comparison on an existing Linux Arm64
  target.
- Optional Arm Performix profiling evidence.
- Passport comparison and CI regression gate.

## Result Claims and Boundaries

ArmTune has bounded real Arm pipeline evidence for `llama-bench`,
`llama-server`, streaming request timing, quality-aware rejection, cleanup, and
artifact digest validation. These runs are not final competition benchmark
results and do not prove KleidiAI usage or an optimized profile.

Phase 5 fixture passports are pipeline-validation artifacts only. They must not
be presented as measured Arm performance.

## Setup

Use private target and workload configuration outside the repository for real
Arm runs. Public examples are available under `configs/armtune/` and are
templates only.

## Demo

The current demo flow shows `arm doctor`, `arm inspect-host`,
`arm quality-calibrate`, `arm build-plan`, `arm build-profile`,
`arm profile-compare`, `arm passport`, and `arm verify-passport`. Fixture
passport output is explicitly labeled `fixture_only`.
