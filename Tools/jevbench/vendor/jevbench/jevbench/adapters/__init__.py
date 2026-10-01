"""Package marker written for OpenJevSwift, not JevBench's own file.

JevBench's adapters/__init__.py imports every adapter it ships, most of which need packages this
harness does not install. Only base.py and typesafe.py are vendored, unchanged, so this file
imports nothing.
"""
