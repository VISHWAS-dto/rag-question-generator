"""Application orchestration: turns HTTP requests into engine calls + DB writes.

This is the only layer that knows about both the database and the engines. It
holds the interview state machine (seed questions -> answer -> follow-up or
next -> complete -> report) and the deterministic report assembly. The engines
and clients are injected, so this layer is testable with fakes and has no
import-time side effects.
"""
