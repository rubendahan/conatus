"""Telemetry of a run, read by a monitor (any process reading the spool): protocol v1, non-blocking emitter, spool reader.

See The computation creates a :class:`Telemetry` explicitly and
passes its callbacks; a monitor reads the spool.
"""
from .film import FilmIndex
from .protocol import (COMMANDS, CONFIRM_COMMANDS, CONTROL_KINDS, FILM_ENCODING, FRAME_KINDS, KINDS,
                       PROTOCOL_VERSION, SCALES, STAT_KINDS, VIEW_KINDS, check_controls, check_set,
                       decode_curve, decode_frame, decode_frame3d, decode_points, decode_vec2d,
                       encode_curve, encode_frame, encode_frame3d, encode_points, encode_vec2d,
                       film_frame, film_record, record)
from .reader import RunState, SpoolReader
from .sink import FILM_POLICY, StopRun, Telemetry, default_spool

__all__ = ["COMMANDS", "CONFIRM_COMMANDS", "CONTROL_KINDS", "FILM_ENCODING", "FILM_POLICY",
           "FRAME_KINDS", "KINDS", "PROTOCOL_VERSION", "SCALES", "STAT_KINDS", "VIEW_KINDS",
           "check_controls", "check_set", "decode_curve", "decode_frame",
           "decode_frame3d", "decode_points", "decode_vec2d", "encode_curve", "encode_frame",
           "encode_frame3d", "encode_points", "encode_vec2d", "film_frame", "film_record", "record",
           "FilmIndex", "RunState", "SpoolReader", "StopRun", "Telemetry", "default_spool"]
