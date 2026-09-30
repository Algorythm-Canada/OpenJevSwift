# Developer entry points. docs/development.md explains each one.

SWIFT ?= swift
FORMAT_PATHS := Package.swift Sources Tests

.PHONY: format lint test

# Rewrite the Swift sources in place according to .swift-format.
format:
	$(SWIFT) format format --in-place --recursive --parallel $(FORMAT_PATHS)

# Fail on any formatting or lint violation.
lint:
	$(SWIFT) format lint --strict --recursive --parallel $(FORMAT_PATHS)

# Build and run every test target that exists on this platform.
test:
	$(SWIFT) test
