# Developer entry points. docs/development.md explains each one.

SWIFT ?= swift
FORMAT_PATHS := Package.swift Sources Tests
# The upstream OpenJev commit this project is compatible with. Keep in step with THIRD_PARTY.md.
UPSTREAM_OPENJEV_COMMIT := dcd2094
# The interpreter Tools/fixtures/.venv is made from. The committed fixtures were written by
# CPython 3.14.7; another version changes the "python" pin recorded in every file.
PYTHON ?= python3.14
FIXTURES_PYTHON := Tools/fixtures/.venv/bin/python
# The interpreter Tools/sdk-compat/.venv is made from; CI runs the suite with CPython 3.12.
SDK_COMPAT_PYTHON ?= python3.12
SDK_COMPAT_VENV_PYTHON := Tools/sdk-compat/.venv/bin/python

.PHONY: format lint test upstream fixtures fixtures-venv sdk-compat sdk-compat-venv docs docs-preview

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

# Create Tools/fixtures/.venv with the pinned packages the fixture scripts import.
fixtures-venv:
	$(PYTHON) -m venv Tools/fixtures/.venv
	$(FIXTURES_PYTHON) -m pip install --quiet --requirement Tools/fixtures/requirements.txt

# Regenerate every file under Fixtures/ from the pinned upstream checkout and the pinned tokenizer.
# The first run downloads the tokenizer (about 32 MB) and the checkpoint's small JSON files into
# the Hugging Face cache. Running it twice gives no diff.
fixtures:
	@test -x $(FIXTURES_PYTHON) || { echo "$(FIXTURES_PYTHON) is missing; run make fixtures-venv"; exit 1; }
	@test -d Upstream/openjev/.git || { echo "Upstream/openjev is missing; run make upstream"; exit 1; }
	PYTHONHASHSEED=0 $(FIXTURES_PYTHON) Tools/fixtures/python_json_tables.py
	PYTHONHASHSEED=0 $(FIXTURES_PYTHON) Tools/fixtures/wire_tables.py
	PYTHONHASHSEED=0 $(FIXTURES_PYTHON) Tools/fixtures/upstream_tables.py
	PYTHONHASHSEED=0 $(FIXTURES_PYTHON) Tools/fixtures/chat_tables.py
	PYTHONHASHSEED=0 $(FIXTURES_PYTHON) Tools/fixtures/checkpoint_tables.py

# Create Tools/sdk-compat/.venv with the pinned Python SDK, and install the pinned TypeScript SDK
# into Tools/sdk-compat/typescript/node_modules (Node.js 20 or later).
sdk-compat-venv:
	$(SDK_COMPAT_PYTHON) -m venv Tools/sdk-compat/.venv
	$(SDK_COMPAT_VENV_PYTHON) -m pip install --quiet --requirement Tools/sdk-compat/requirements.txt
	npm ci --prefix Tools/sdk-compat/typescript --no-audit --no-fund

# Run TypeSafe's official SDKs against openjev-stub-server, which this builds first. Pass
# SDK_COMPAT_ARGS=--swift-sdk to also run NSStudent's JevSwiftSDK.
sdk-compat:
	@test -x $(SDK_COMPAT_VENV_PYTHON) || { echo "$(SDK_COMPAT_VENV_PYTHON) is missing; run make sdk-compat-venv"; exit 1; }
	$(SWIFT) build --product openjev-stub-server
	$(SDK_COMPAT_VENV_PYTHON) Tools/sdk-compat/run.py --server "$$($(SWIFT) build --show-bin-path)/openjev-stub-server" $(SDK_COMPAT_ARGS)

# Build the DocC documentation of the four library modules into one static site at
# .build/docs-site, as the Documentation workflow publishes it (macOS only). Any DocC warning fails.
docs:
	Tools/docs/build-site.sh

# Serve the site make docs built at http://localhost:8000/OpenJevSwift/, its path on GitHub Pages.
docs-preview:
	Tools/docs/serve-site.sh
