# Developer entry points. docs/development.md explains each one.

SWIFT ?= swift
FORMAT_PATHS := Package.swift Sources Tests
# The upstream OpenJev commit this project is compatible with. Keep in step with THIRD_PARTY.md.
UPSTREAM_OPENJEV_COMMIT := dcd2094

.PHONY: format lint test upstream

# Rewrite the Swift sources in place according to .swift-format.
format:
	$(SWIFT) format format --in-place --recursive --parallel $(FORMAT_PATHS)

# Fail on any formatting or lint violation.
lint:
	$(SWIFT) format lint --strict --recursive --parallel $(FORMAT_PATHS)

# Build and run every test target that exists on this platform.
test:
	$(SWIFT) test

# Check out the pinned upstream OpenJev source under Upstream/openjev for reference. The folder is
# ignored by git; issues cite its files by path and line.
upstream:
	@test -d Upstream/openjev/.git || git clone -q https://github.com/razorback16/openjev.git Upstream/openjev
	@git -C Upstream/openjev fetch -q origin
	@git -C Upstream/openjev checkout -q $(UPSTREAM_OPENJEV_COMMIT)
	@git -C Upstream/openjev log -1 --format='Upstream/openjev at %h (%ad): %s' --date=short
