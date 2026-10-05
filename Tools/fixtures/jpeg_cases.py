"""Record what upstream's Pillow makes of JPEGs that exercise libjpeg-turbo's handling of damaged
and unusual data, for LibjpegTurboDecoder's regression tests.

The JPEG port (Sources/OpenJevDiffusionGemma/Vision/LibjpegTurboDecoder.swift) must decode every
JPEG to the bytes upstream's `ImagePrompt.pil` gives (`PIL.Image.open(...).convert("RGB")`, with
the libjpeg-turbo 3.1.4.1 that Pillow 12.3.0 bundles) and refuse every JPEG on which it raises.
This script builds a set of small cases that pin the behaviour a fuzzing review of the port found it
lacked: scan data that runs out, the refusals of libjpeg-turbo and of Pillow's own header reading,
the standard Huffman tables, codes longer than 16 bits, restart markers out of sequence or missing,
blocks per MCU counted per scan, block smoothing, the Arm Neon inverse DCT's 16-bit arithmetic, and
libjpeg-turbo's fast Huffman path and Pillow's 65,536-byte reads. Six more pin the end of a
single-scan JPEG, where Pillow reads no further than the 65,536-byte reads it has made, and seven
the checks of a lossless JPEG's first scan. Twenty-eight (D-057) pin arithmetic-coded and lossless
JPEGs past their first scan, which the port reads without decoding their scans' data: faults in
later scans, restart markers, arithmetic-coded scans at the 65,536-byte reads (jdarith.c cannot
suspend), single scans cut short or followed by a fault, and a lossless component no scan reaches.
It runs each case through upstream's `ImagePrompt.pil` and writes Fixtures/vision/jpeg_cases.json.

Each case is built from one of the committed JPEGs (Fixtures/vision/baseline.jpg and
progressive.jpg, which Tools/fixtures/vision_oracle.py draws) or from bytes this script writes out
in full, by a list of edits applied in order:

- ["cut", n]: keep the first n bytes.
- ["set", at, hex]: overwrite bytes from `at`.
- ["insert", at, hex]: insert bytes before `at`.
- ["delete", at, n]: delete n bytes from `at`.
- ["append", hex]: append bytes.
- ["copy", at, start, end]: insert a copy of bytes[start:end] before `at`.
- ["repeat", hex, n]: append the bytes n times.
- ["entropy", seed, n]: append n pseudo-random bytes, the low byte of each output of splitmix64
  seeded with `seed`, each 0xFF followed by a stuffed 0x00, as entropy-coded data.

The Swift test (Tests/OpenJevDiffusionGemmaTests/Vision/JPEGParityTests.swift) applies the same
edits, checks the bytes' SHA-256, and compares its decode with the record: `decoded` (the size and
the SHA-256 of the RGB bytes) or `error` (the exception upstream raises, which the port must answer
with a refusal). A case with `port` set is one where the port knowingly departs: `unsupported`
means it hands the JPEG to ImageIO (arithmetic coding, lossless, 4 components) after the checks
it shares with libjpeg-turbo for those, either because Pillow decodes it or because only decoding
its scans would tell whether Pillow raises (a `note` saying "undecided").

With `--check` nothing is written: the run is compared with the committed file.

Usage, from the repository root (Tools/oracle/requirements.txt pins Pillow 12.3.0):

    make upstream
    Tools/oracle/.venv/bin/python Tools/fixtures/jpeg_cases.py [--check] [--upstream PATH]

`--upstream` names another checkout of upstream at the pinned commit (default Upstream/openjev).
"""
import argparse
import base64
import hashlib
import io
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VISION = ROOT / "Fixtures" / "vision"
OUT = VISION / "jpeg_cases.json"
SCRIPT = "Tools/fixtures/jpeg_cases.py"
GENERATOR_VERSION = 1
UPSTREAM_COMMIT = "dcd2094"
SOURCES = ("baseline.jpg", "progressive.jpg")


# Building cases ------------------------------------------------------------------------------

def be16(v):
    return bytes([v >> 8 & 255, v & 255])


def seg(marker, body):
    return bytes([0xFF, marker]) + be16(len(body) + 2) + bytes(body)


SOI, EOI = b"\xff\xd8", b"\xff\xd9"


def dqt(table=0, value=1, precise=False):
    if precise:
        return seg(0xDB, [0x10 | table] + [value >> 8, value & 255] * 64)
    return seg(0xDB, [table] + [value] * 64)


def sof(marker, width, height, components, precision=8):
    """components: (id, h, v, table) tuples."""
    body = [precision] + list(be16(height)) + list(be16(width)) + [len(components)]
    for cid, h, v, table in components:
        body += [cid, h << 4 | v, table]
    return seg(marker, body)


def dht(table_class, table, counts, values):
    return seg(0xC4, [table_class << 4 | table] + list(counts) + list(values))


def sos(components, ss=0, se=63, ah=0, al=0):
    """components: (id, dc, ac) tuples."""
    body = [len(components)]
    for cid, dc, ac in components:
        body += [cid, dc << 4 | ac]
    return seg(0xDA, body + [ss, se, ah << 4 | al])


def dri(interval):
    return seg(0xDD, list(be16(interval)))


def com(size):
    """A COM segment of `size` bytes in all (4 to 65,537), to move what follows it."""
    return seg(0xFE, [(i * 7 + 3) & 0x7F for i in range(size - 4)])


ONE_CODE = [1] + [0] * 15  # one 1-bit code, 0


def counts(*lengths):
    """The 16 code counts with one code of each given length."""
    out = [0] * 16
    for length in lengths:
        out[length - 1] += 1
    return out


def splitmix64(seed):
    state = seed & 0xFFFFFFFFFFFFFFFF
    while True:
        state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        yield z ^ (z >> 31)


def entropy(seed, count):
    out = bytearray()
    stream = splitmix64(seed)
    for _ in range(count):
        byte = next(stream) & 0xFF
        out.append(byte)
        if byte == 0xFF:
            out.append(0)
    return bytes(out)


def apply(data, edits):
    b = bytearray(data)
    for edit in edits:
        op = edit[0]
        if op == "cut":
            del b[edit[1]:]
        elif op == "set":
            value = bytes.fromhex(edit[2])
            b[edit[1]:edit[1] + len(value)] = value
        elif op == "insert":
            b[edit[1]:edit[1]] = bytes.fromhex(edit[2])
        elif op == "delete":
            del b[edit[1]:edit[1] + edit[2]]
        elif op == "append":
            b += bytes.fromhex(edit[1])
        elif op == "copy":
            b[edit[1]:edit[1]] = b[edit[2]:edit[3]]
        elif op == "repeat":
            b += bytes.fromhex(edit[1]) * edit[2]
        elif op == "entropy":
            b += entropy(edit[1], edit[2])
        else:
            raise ValueError(f"unknown edit {op}")
    return bytes(b)


def segments(b):
    """(start, marker, end of header, end of entropy data) for each segment after SOI."""
    out, i = [], 2
    while i < len(b):
        if b[i] != 0xFF:
            i += 1
            continue
        while i < len(b) and b[i] == 0xFF:
            i += 1
        m = b[i]
        i += 1
        if m == 0xD9:
            out.append((i - 2, m, i, i))
            break
        if 0xD0 <= m <= 0xD7 or m == 0x01:
            continue
        end = i + (b[i] << 8 | b[i + 1])
        data_end = end
        if m == 0xDA:
            j = end
            while j + 1 < len(b):
                if b[j] == 0xFF and b[j + 1] != 0 and not 0xD0 <= b[j + 1] <= 0xD7:
                    break
                j += 1
            data_end = j
        out.append((i - 2, m, end, data_end))
        i = data_end
    return out


class Cases:
    def __init__(self, sources):
        self.sources = sources
        self.cases = {}

    def edit(self, name, source, edits, port=None, note=None):
        assert name not in self.cases, name
        case = {"from": source, "edits": edits}
        if port:
            case["port"] = port
        if note:
            case["note"] = note
        self.cases[name] = case

    def raw(self, name, data, edits=(), port=None, note=None):
        assert name not in self.cases, name
        case = {"from": "bytes", "base64": base64.b64encode(data).decode(), "edits": list(edits)}
        if port:
            case["port"] = port
        if note:
            case["note"] = note
        self.cases[name] = case

    def build(self, case):
        data = base64.b64decode(case["base64"]) if case["from"] == "bytes" else self.sources[case["from"]]
        return apply(data, case["edits"])


def define(cases, base, prog):
    """Every case, by name. `base` and `prog` are the committed fixture JPEGs."""
    bsegs, psegs = segments(base), segments(prog)
    b_scan = [s for s in bsegs if s[1] == 0xDA][0]
    p_scans = [s for s in psegs if s[1] == 0xDA]
    data_start, data_end = b_scan[2], b_scan[3]

    # The review's reproductions (PR #121's fuzzing review).
    tiny = (SOI + dqt(0, 16) + sof(0xC0, 8, 16, [(1, 1, 1, 0)]) + dht(0, 0, ONE_CODE, [0x01])
            + dht(1, 0, ONE_CODE, [0x00]) + sos([(1, 0, 0)]) + EOI)
    cases.raw("R0_tiny_empty_scan", tiny, note="no entropy-coded data: one MCU of zero bits, then the rest left zero")
    cases.edit("R0_baseline_cut_3234_plus_EOI", "baseline.jpg", [["cut", 3234], ["append", "ffd9"]])
    cases.edit("R1_baseline_Tq4_byte170", "baseline.jpg", [["set", 170, "04"]])
    cases.edit("R2_baseline_Ta4_byte621", "baseline.jpg", [["set", 621, "04"]])
    cases.edit("R3_prog_dup_selector_byte244", "progressive.jpg", [["set", 244, "02"]])
    cases.edit("R4_baseline_scan_twice", "baseline.jpg", [["copy", 3277, 615, 3277]])
    cases.edit("R5_baseline_reserved_marker_FF80", "baseline.jpg", [["insert", 2, "ff800002"]])
    cases.edit("R6_prog_height_65534", "progressive.jpg", [["set", 163, "fffe"]])
    cases.edit("R7_baseline_FFC0_at_629", "baseline.jpg", [["insert", 629, "ffc0"]])
    cases.edit("R8_prog_first_9_scans", "progressive.jpg", [["cut", 2291], ["append", "ffd9"]])

    # Scan data that runs out before a marker: zero bits for the MCU, the rest of the segment left.
    for i in range(8):
        cut = data_start + 1 + (data_end - data_start - 2) * i // 7
        cases.edit(f"exhausted_baseline_{cut}", "baseline.jpg", [["cut", cut], ["append", "ffd9"]])
    for scan in p_scans[1:]:
        for fraction in (3, 2):
            cut = scan[2] + (scan[3] - scan[2]) // fraction
            cases.edit(f"exhausted_progressive_{cut}", "progressive.jpg", [["cut", cut], ["append", "ffd9"]])
    # Data that ends without a marker: Pillow raises unless libjpeg-turbo has stopped reading.
    for cut in (2, 1, 3, 4, 5, 8):
        cases.edit(f"eof_baseline_minus_{cut}", "baseline.jpg", [["cut", len(base) - cut]])
    for cut in (2, 1, 6):
        cases.edit(f"eof_progressive_minus_{cut}", "progressive.jpg", [["cut", len(prog) - cut]])
    cases.edit("eof_baseline_in_header", "baseline.jpg", [["cut", 300]])

    # libjpeg-turbo's refusals.
    cases.edit("quant_selector_255", "baseline.jpg", [["set", 173, "ff"]])
    cases.edit("dqt_index_4", "baseline.jpg", [["set", 93, "04"]])
    cases.edit("dqt_length_short", "baseline.jpg", [["set", 23, "42"], ["delete", 88, 1]])
    cases.edit("dc_selector_4", "baseline.jpg", [["set", 623, "41"]])
    cases.edit("ac_selector_5", "baseline.jpg", [["set", 625, "15"]])
    cases.edit("dht_index_4", "baseline.jpg", [["set", 181, "04"]])
    cases.edit("dht_class_2", "baseline.jpg", [["set", 397, "21"]])
    cases.edit("components_out_of_order", "baseline.jpg", [["set", 620, "02"], ["set", 622, "01"]])
    cases.edit("component_unknown", "baseline.jpg", [["set", 624, "09"]])
    cases.edit("second_soi_before_scan", "baseline.jpg", [["insert", 158, "ffd8"]])
    cases.edit("second_soi_between_scans", "progressive.jpg", [["insert", p_scans[1][0], "ffd8"]])
    cases.edit("reserved_marker_between_scans", "progressive.jpg", [["insert", p_scans[1][0], "ff020002"]])
    cases.edit("tem_between_scans", "progressive.jpg", [["insert", p_scans[1][0], "ff01"]])
    cases.edit("width_65501", "baseline.jpg", [["set", 165, "ffdd"]])
    cases.edit("sos_in_single_scan_data", "baseline.jpg", [["insert", 1200, "ffda"]])
    cases.edit("eoi_in_single_scan_data", "baseline.jpg", [["insert", 1200, "ffd9"]])
    cases.edit("dqt_marker_in_data", "baseline.jpg", [["insert", 1200, "ffdb"]])
    cases.edit("sof5", "baseline.jpg", [["set", 159, "c5"]])
    cases.edit("sof_jpg", "baseline.jpg", [["set", 159, "c8"]])
    cases.edit("sof10_progression", "baseline.jpg", [["set", 159, "ca"]])
    cases.edit("sof11_lossless_arithmetic", "baseline.jpg", [["set", 159, "cb"]])
    cases.edit("dri_length_5", "baseline.jpg", [["set", 611, "0005"], ["insert", 615, "00"]])
    cases.edit("dac_index_32", "baseline.jpg", [["insert", 158, "ffcc00042010"]])
    cases.edit("exp_marker", "baseline.jpg", [["insert", 158, "ffdf000311"]])
    cases.edit("dhp_marker", "baseline.jpg", [["insert", 158, "ffde" + base[160:177].hex()]])
    cases.edit("dht_after_single_scan_bad", "baseline.jpg", [["insert", len(base) - 2, "ffc40013" + "00" + "ff" * 16]])
    cases.edit("dht_after_single_scan_good", "baseline.jpg", [["insert", len(base) - 2, "ffc40014" + "00" + "01" + "00" * 15 + "00"]])
    cases.edit("tables_only_with_frame", "baseline.jpg", [["insert", 0, base[:177].hex() + "ffd9"]])
    cases.edit("tables_only_then_junk", "baseline.jpg", [["insert", 0, base[:158].hex() + "ffd900"]])
    cases.edit("tables_only_then_image", "baseline.jpg", [["insert", 0, base[:158].hex() + base[177:615].hex() + "ffd9"]])
    # Progressive scan parameters jdphuff.c refuses.
    first_ac = p_scans[1][0]
    cases.edit("prog_ss_above_se", "progressive.jpg", [["set", first_ac + 7, "06"]])
    cases.edit("prog_se_64", "progressive.jpg", [["set", first_ac + 8, "40"]])
    cases.edit("prog_al_14", "progressive.jpg", [["set", first_ac + 9, "0e"]])
    cases.edit("prog_ah_not_al_plus_1", "progressive.jpg", [["set", p_scans[7][0] + 9, "31"]])
    # A progressive DC coefficient past 32 bits (JERR_BAD_DCT_COEF): 65,792 blocks of +32,767.
    overflow = (SOI + dqt() + sof(0xC2, 2056, 2048, [(1, 1, 1, 0)]) + dht(0, 0, ONE_CODE, [15])
                + sos([(1, 0, 0)], 0, 0, 0, 0))
    cases.raw("prog_dc_overflow", overflow, [["repeat", "7fff00", 65792], ["append", "ffd9"]])

    # Pillow's own header reading.
    for code in ("01", "02", "bf"):
        cases.edit(f"pillow_marker_{code}_before_scan", "baseline.jpg", [["insert", 158, "ff" + code]])
    cases.edit("pillow_jfif_short", "baseline.jpg", [["insert", 20, "ffe000084a4649460001"]])
    cases.edit("pillow_jfif_seven", "baseline.jpg", [["insert", 20, "ffe000094a464946000102"]])
    cases.edit("pillow_adobe_short", "baseline.jpg", [["insert", 20, "ffee000841646f626500"]])
    cases.edit("pillow_icc_short_before_frame", "baseline.jpg", [["insert", 20, "ffe2000e4943435f50524f46494c4500"]])
    cases.edit("pillow_icc_short_after_frame", "baseline.jpg", [["insert", 177, "ffe2000e4943435f50524f46494c4500"]])
    cases.edit("pillow_photoshop_no_name", "baseline.jpg", [["insert", 20, "ffed0016" + b"Photoshop 3.0\x008BIM\x04\x04".hex()]])
    resolution = (b"Photoshop 3.0\x00" + b"8BIM\x03\xed\x00\x00" + be16(0) + be16(4) + bytes(4)
                  + b"8BIM\x04\x04")
    cases.edit("pillow_photoshop_short_resolution", "baseline.jpg",
               [["insert", 20, seg(0xED, resolution).hex()]],
               note="ResolutionInfo shorter than it reads stops Pillow's loop before the cut resource")
    cases.edit("pillow_frame_extra_byte", "baseline.jpg", [["set", 160, "0012"], ["insert", 177, "00"]])
    cases.edit("pillow_dqt_trailing_byte", "baseline.jpg", [["set", 91, "0044"], ["insert", 158, "00"]])
    cases.edit("pillow_12_bit", "baseline.jpg", [["set", 162, "0c"]])
    two = (SOI + dqt() + sof(0xC0, 8, 8, [(1, 1, 1, 0), (2, 1, 1, 0)]) + dht(0, 0, ONE_CODE, [0])
           + dht(1, 0, ONE_CODE, [0]) + sos([(1, 0, 0), (2, 0, 0)]) + b"\x00\x00" + EOI)
    cases.raw("pillow_two_components", two)
    cases.edit("pillow_segment_past_end", "baseline.jpg", [["insert", 20, "fffe7000"], ["cut", 60]])
    cases.edit("pillow_com_length_1", "baseline.jpg", [["insert", 20, "fffe0001"]])

    # Inputs libjpeg-turbo decodes that the port used to hand to ImageIO.
    no_dht = [["delete", s[0], s[2] - s[0]] for s in reversed(bsegs) if s[1] == 0xC4]
    cases.edit("standard_tables", "baseline.jpg", no_dht, note="no DHT: the standard tables")
    removed = sum(s[2] - s[0] for s in bsegs if s[1] == 0xC4 and s[0] < b_scan[0])
    cases.edit("standard_tables_slot_2", "baseline.jpg", no_dht + [["set", b_scan[0] + 6 - removed, "22"]],
               note="the standard tables fill slots 0 and 1 only")
    cases.edit("progressive_without_tables", "progressive.jpg",
               [["delete", s[0], s[2] - s[0]] for s in reversed(psegs) if s[1] == 0xC4])
    long_codes = (SOI + dqt(0, 3) + sof(0xC0, 64, 64, [(1, 1, 1, 0)]) + dht(0, 0, counts(1, 3), [3, 5])
                  + dht(1, 0, counts(2, 3, 16), [0x01, 0x00, 0x11]) + sos([(1, 0, 0)]))
    cases.raw("codes_over_16_bits", long_codes, [["entropy", 3, 900], ["append", "ffd9"]],
              note="most bit strings are no code: libjpeg-turbo reads 17 bits and takes symbol 0")
    rst = [i for i in range(data_start, data_end - 1) if base[i] == 0xFF and 0xD0 <= base[i + 1] <= 0xD7]
    for k in (0, len(rst) // 2, len(rst) - 1):
        at = rst[k]
        cases.edit(f"restart_{k}_deleted", "baseline.jpg", [["delete", at, 2]])
        cases.edit(f"restart_{k}_doubled", "baseline.jpg", [["insert", at, base[at:at + 2].hex()]])
        for delta in (1, 2, 3, 6, 7):
            number = 0xD0 + (base[at + 1] - 0xD0 + delta) % 8
            cases.edit(f"restart_{k}_plus_{delta}", "baseline.jpg", [["set", at + 1, f"{number:02x}"]])
        for code in ("d9", "fe", "01", "02", "c4"):
            cases.edit(f"restart_{k}_as_{code}", "baseline.jpg", [["set", at + 1, code]])
        cases.edit(f"restart_{k}_after_junk", "baseline.jpg", [["insert", at, "123400"]])
        cases.edit(f"restart_{k}_then_eoi", "baseline.jpg", [["cut", at + 2], ["append", "ffd9"]])
    # 18 blocks per MCU over the frame, but every scan non-interleaved: libjpeg-turbo decodes it.
    big_mcu = sof(0xC0, 32, 32, [(1, 4, 4, 0), (2, 1, 1, 0), (3, 1, 1, 0)])
    tables = dqt() + dht(0, 0, counts(2, 2, 2), [0, 1, 2]) + dht(1, 0, counts(2, 2, 2), [0x00, 0x01, 0x11])
    noninterleaved = SOI + tables + big_mcu
    edits = []
    for cid in (1, 2, 3):
        edits += [["append", sos([(cid, 0, 0)]).hex()], ["entropy", 40 + cid, 300 if cid == 1 else 40]]
    cases.raw("mcu_18_blocks_noninterleaved", noninterleaved, edits + [["append", "ffd9"]])
    cases.raw("mcu_18_blocks_interleaved", noninterleaved,
              [["append", sos([(1, 0, 0), (2, 0, 0), (3, 0, 0)]).hex()], ["entropy", 44, 400], ["append", "ffd9"]])
    # A component no scan reaches comes out as 128.
    unscanned = SOI + dqt() + dht(0, 0, counts(2, 2, 2), [0, 1, 2]) + dht(1, 0, counts(2, 2, 2), [0x00, 0x01, 0x11])
    unscanned += sof(0xC0, 16, 16, [(1, 1, 1, 0), (2, 1, 1, 0), (3, 1, 1, 0)])
    cases.raw("component_never_scanned", unscanned,
              [["append", sos([(1, 0, 0), (2, 0, 0)]).hex()], ["entropy", 50, 200], ["append", "ffd9"]])

    # Block smoothing: progressive scans that stop short of full precision.
    for k in range(1, len(p_scans)):
        cases.edit(f"smoothing_first_{k}_scans", "progressive.jpg", [["cut", p_scans[k][0]], ["append", "ffd9"]])
    chroma_ac = [s for s in p_scans[1:] if prog[s[0] + 5] != 1]
    cases.edit("smoothing_no_chroma_ac", "progressive.jpg",
               [["delete", s[0], s[3] - s[0]] for s in reversed(chroma_ac)])

    # The Arm Neon inverse DCT: 16-bit quantization values above 32,767 and products past 16 bits.
    dq = [s for s in bsegs if s[1] == 0xDB]
    for i, value in enumerate((32768, 40000, 65535, 300)):
        replace = []
        for s in reversed(dq):
            table = base[s[0] + 4] & 15
            values = list(base[s[0] + 5:s[2]])
            body = [0x10 | table]
            for k, v in enumerate(values):
                w = value if k % 7 == i % 7 else v * (1 + i)
                body += [w >> 8 & 255, w & 255]
            replace += [["delete", s[0], s[2] - s[0]], ["insert", s[0], seg(0xDB, body).hex()]]
        cases.edit(f"neon_quant_{value}", "baseline.jpg", replace)

    # libjpeg-turbo's fast Huffman path: without restart markers and with plenty of data left, an
    # MCU that meets FF FF is decoded again on the slow path, over the coefficients the fast path
    # wrote from zero bits (its AC table's all-zero code is a coefficient, not an end of block).
    # The seeds and offsets are ones where those coefficients show in the image.
    def huffman_grey(side):
        return (SOI + dqt() + sof(0xC0, side, side, [(1, 1, 1, 0)])
                + dht(0, 0, counts(1, 2, 3, 4), [0, 1, 2, 3])
                + dht(1, 0, counts(2, 2, 3, 3, 4, 4), [0x01, 0x00, 0x11, 0x02, 0x21, 0x31])
                + sos([(1, 0, 0)]))
    fast = huffman_grey(128)
    for seed, offset in ((3, 300), (6, 120)):
        cases.raw(f"fast_path_ffff_seed_{seed}_at_{offset}", fast,
                  [["entropy", seed, 4000], ["insert", len(fast) + offset, "ffff00"], ["append", "ffd9"]])
    cases.raw("fast_path_plain", fast, [["entropy", 3, 4000], ["append", "ffd9"]])
    # Pillow reads 65,536 bytes at a time, and libjpeg-turbo takes its fast path only with 512
    # bytes per block at hand, so near the boundary these MCUs are slow ones in Pillow.
    chunk = huffman_grey(1400)
    for seed, offset in ((9, 65415), (11, 65124)):
        cases.raw(f"chunk_ffff_seed_{seed}_at_{offset}", chunk,
                  [["entropy", seed, 80000], ["insert", offset, "ffff00"], ["append", "ffd9"]])
    # Once a single scan's rows are out, Pillow stops where the 65,536 bytes it has read end
    # (jpeg_finish_decompress suspends there): a second scan, a second frame or a reserved marker
    # after the scan raises only when libjpeg-turbo reaches it within them.
    end = len(base) - 2
    frame = [s for s in bsegs if s[1] == 0xC0][0]
    trailers = (("scan", 65000, base[b_scan[0]:b_scan[2]]), ("scan", 65534, base[b_scan[0]:b_scan[2]]),
                ("scan", 65536, base[b_scan[0]:b_scan[2]]), ("frame", 65534, base[frame[0]:frame[2]]),
                ("reserved", 65534, b"\xff\x02"), ("reserved", 65537, b"\xff\x02"))
    for kind, offset, marker in trailers:
        cases.edit(f"finish_{kind}_at_{offset}", "baseline.jpg",
                   [["cut", end], ["repeat", "00", offset - end], ["append", marker.hex() + "ffd9"]])

    # The review's slow cases: large frames from a few hundred bytes (work bound).
    w = h = 13376
    grey = SOI + dqt() + sof(0xC2, w, h, [(1, 1, 1, 0)]) + dht(0, 0, ONE_CODE, [0]) + dht(1, 0, ONE_CODE, [0x01])
    dc, ac = sos([(1, 0, 0)], 0, 0, 0, 0), sos([(1, 0, 0)], 1, 63, 0, 0)
    cases.raw("dos_prog_grey_dc_plus_99_ac", grey + dc + ac * 99 + EOI)
    grey_eob = SOI + dqt() + sof(0xC2, w, h, [(1, 1, 1, 0)]) + dht(0, 0, ONE_CODE, [0]) + dht(1, 0, ONE_CODE, [0xE0])
    cases.raw("dos_prog_grey_dc_plus_99_ac_eobrun", grey_eob + dc + ac * 99 + EOI)
    seq3 = SOI + dqt() + sof(0xC0, w, h, [(1, 1, 1, 0), (2, 1, 1, 0), (3, 1, 1, 0)]) + dht(0, 0, ONE_CODE, [0]) + dht(1, 0, ONE_CODE, [0x01])
    cases.raw("dos_seq_ycc444_1scan_empty", seq3 + sos([(1, 0, 0), (2, 0, 0), (3, 0, 0)]) + EOI)
    cases.raw("dos_seq_ycc444_3scans_nonint_empty", seq3 + sos([(1, 0, 0)]) + sos([(2, 0, 0)]) + sos([(3, 0, 0)]) + EOI)

    # Departures: Pillow decodes these; the port hands them to ImageIO.
    cases.raw("unsupported_arithmetic", SOI + dqt() + sof(0xC9, 8, 8, [(1, 1, 1, 0)]) + sos([(1, 0, 0)]) + bytes(range(1, 40)) + EOI,
              port="unsupported")
    cases.raw("unsupported_lossless", SOI + sof(0xC3, 8, 8, [(1, 1, 1, 0)]) + dht(0, 0, ONE_CODE, [0]) + sos([(1, 0, 0)], 1, 0, 0, 0) + b"\x00" * 8 + EOI,
              port="unsupported")
    cmyk = SOI + dqt() + sof(0xC0, 8, 8, [(1, 1, 1, 0), (2, 1, 1, 0), (3, 1, 1, 0), (4, 1, 1, 0)])
    cmyk += dht(0, 0, ONE_CODE, [0]) + dht(1, 0, ONE_CODE, [0]) + sos([(c, 0, 0) for c in (1, 2, 3, 4)]) + b"\x00" * 4 + EOI
    cases.raw("unsupported_cmyk", cmyk, port="unsupported")
    # A lossless JPEG goes to ImageIO only after the checks libjpeg-turbo makes before its first
    # scan's data: jdlhuff.c's DC tables (no standard ones, symbols up to 16), jdlossls.c's
    # predictor and point transform, and jddiffct.c's restart interval. No quantization table is
    # latched for it.
    lossless = SOI + sof(0xC3, 8, 8, [(1, 1, 1, 0)])
    one, body = dht(0, 0, ONE_CODE, [0]), b"\x00" * 16 + EOI
    cases.raw("lossless_predictor_0", lossless + one + sos([(1, 0, 0)], 0, 0, 0, 0) + body)
    cases.raw("lossless_point_transform_8", lossless + one + sos([(1, 0, 0)], 1, 0, 0, 8) + body)
    cases.raw("lossless_restart_part_row", lossless + dri(3) + one + sos([(1, 0, 0)], 1, 0, 0, 0) + body)
    cases.raw("lossless_no_tables", lossless + sos([(1, 0, 0)], 1, 0, 0, 0) + body)
    cases.raw("lossless_dc_symbol_17", lossless + dht(0, 0, counts(1, 2), [0, 17]) + sos([(1, 0, 0)], 1, 0, 0, 0) + body)
    cases.raw("lossless_dc_symbol_16", lossless + dht(0, 0, counts(1, 2), [0, 16]) + sos([(1, 0, 0)], 1, 0, 0, 0) + body,
              port="unsupported")
    cases.raw("lossless_quant_table_3", SOI + sof(0xC3, 8, 8, [(1, 1, 1, 3)]) + one + sos([(1, 0, 0)], 1, 0, 0, 0) + body,
              port="unsupported")

    # Arithmetic-coded and lossless JPEGs past their first scan (D-057). The port reads them as
    # libjpeg-turbo does without decoding their scans' data: each scan's checks, restart markers,
    # the markers between and after the scans, and the end of the file. jdarith.c cannot suspend,
    # so a read past the 65,536-byte reads Pillow has made raises ("broken data stream"); where
    # only decoding tells whether it reads that far, or whether jdlhuff.c reads past the end
    # before the last row, the port hands the JPEG to ImageIO ("undecided" in the note).
    rgb = [(1, 1, 1, 0), (2, 1, 1, 0), (3, 1, 1, 0)]
    arith3 = SOI + dqt() + sof(0xC9, 16, 16, rgb)
    scans3 = [sos([(c, 0, 0)]) + entropy(70 + c, 60) for c in (1, 2, 3)]
    cases.raw("arith_multiscan", arith3 + b"".join(scans3) + EOI, port="unsupported")
    cases.raw("arith_multiscan_cut", arith3 + scans3[0] + scans3[1][:30])
    cases.raw("arith_multiscan_quant_table_1", SOI + dqt() + sof(0xC9, 16, 16, rgb[:2] + [(3, 1, 1, 1)])
              + b"".join(scans3) + EOI, note="the third scan latches table 1, which is not defined")
    dac_bad = seg(0xCC, [0x01, 0x2F])
    cases.raw("arith_dac_between_scans", arith3 + scans3[0] + dac_bad + scans3[1] + scans3[2] + EOI)
    prog = SOI + dqt() + sof(0xCA, 16, 16, [(1, 1, 1, 0)])
    prog_scans = [sos([(1, 0, 0)], 0, 0, 0, 1) + entropy(80, 20), sos([(1, 0, 0)], 1, 63, 0, 1) + entropy(81, 40)]
    cases.raw("arith_progressive_refine_bad", prog + b"".join(prog_scans)
              + sos([(1, 0, 0)], 1, 63, 2, 0) + entropy(82, 20) + EOI, note="Ah 2 needs Al 1")
    prog3 = SOI + dqt() + sof(0xCA, 16, 16, rgb)
    cases.raw("arith_progressive_ac_two_components", prog3 + sos([(1, 0, 0), (2, 0, 0), (3, 0, 0)], 0, 0, 0, 0)
              + entropy(83, 30) + sos([(1, 0, 0), (2, 0, 0)], 1, 63, 0, 0) + entropy(84, 30) + EOI)
    grey_arith = SOI + dqt() + sof(0xC9, 8, 8, [(1, 1, 1, 0)]) + sos([(1, 0, 0)])
    cases.raw("arith_second_scan", grey_arith + entropy(85, 20) + sos([(1, 0, 0)]) + entropy(86, 20) + EOI)
    # Restart markers: misnumbered ones resynchronise as in a Huffman-coded scan.
    eight = SOI + dqt() + sof(0xC9, 64, 8, [(1, 1, 1, 0)]) + dri(1) + sos([(1, 0, 0)])
    order = (0, 2, 1, 3, 4, 6, 7)
    cases.raw("arith_restart_resync", eight + b"".join(entropy(90 + i, 9) + bytes([0xFF, 0xD0 + r])
                                                       for i, r in enumerate(order)) + entropy(99, 9) + EOI,
              port="unsupported")
    # The 65,536-byte reads. A restart marker past them is read by process_restart, which cannot
    # suspend; a scan whose data starts at them reads its first byte there.
    two = SOI + dqt() + sof(0xC9, 16, 8, [(1, 1, 1, 0)]) + dri(1) + sos([(1, 0, 0)])
    cases.raw("arith_restart_marker_past_read", two, [["entropy", 87, 70000], ["append", "ffd0"],
                                                      ["entropy", 88, 40], ["append", "ffd9"]])
    head = SOI + dqt() + sof(0xC9, 8, 8, [(1, 1, 1, 0)])
    start = sos([(1, 0, 0)])
    cases.raw("arith_scan_data_at_read", head + com(65536 - len(head) - len(start)) + start,
              [["entropy", 89, 50], ["append", "ffd9"]])
    # A scan whose data runs on past the reads, with no restart marker: how far jdarith.c reads
    # depends on decoding it, here on how many blocks there are to decode.
    for side, outcome in ((64, "decodes"), (1024, "raises")):
        head = SOI + dqt() + sof(0xC9, side, side, [(1, 1, 1, 0)])
        cases.raw(f"arith_scan_open_{side}", head + com(65536 - 4000 - len(head) - len(start)) + start,
                  [["entropy", 1, 30000], ["append", "ffd9"]], port="unsupported",
                  note=f"undecided: Pillow {outcome}; whether jdarith.c reads past 65,536 bytes depends on decoding")
    head = SOI + dqt() + sof(0xCA, 1024, 1024, [(1, 1, 1, 0)])
    dc = sos([(1, 0, 0)], 0, 0, 0, 0)
    cases.raw("arith_open_scan_then_bad_scan", head + com(65536 - 4000 - len(head) - len(dc)) + dc,
              [["entropy", 1, 30000], ["append", sos([(1, 0, 0)], 1, 63, 3, 0).hex()], ["entropy", 2, 20],
               ["append", "ffd9"]], note="the second scan raises whatever the first one reads")
    # Work in proportion to the input: a restart every MCU of a 2,000 by 2,000 frame, and EOI after
    # ten bytes, which every later segment leaves where it is.
    cases.raw("arith_restart_every_mcu_large", SOI + dqt() + sof(0xC9, 2000, 2000, [(1, 1, 1, 0)]) + dri(1)
              + sos([(1, 0, 0)]) + entropy(91, 10) + EOI, port="unsupported")

    ll = dht(0, 0, counts(1, 2, 3, 4), [0, 1, 2, 3])
    ll3 = SOI + sof(0xC3, 8, 8, rgb) + ll
    ll_scans = [sos([(c, 0, 0)], 1, 0, 0, 0) + entropy(100 + c, 40) for c in (1, 2, 3)]
    cases.raw("lossless_multiscan", ll3 + b"".join(ll_scans) + EOI, port="unsupported")
    cases.raw("lossless_second_scan_predictor_0", ll3 + ll_scans[0] + sos([(2, 0, 0)], 0, 0, 0, 0)
              + entropy(102, 40) + ll_scans[2] + EOI)
    cases.raw("lossless_multiscan_cut", ll3 + ll_scans[0] + ll_scans[1][:30],
              note="no EOI: libjpeg-turbo reads every scan before the first row")
    cases.raw("lossless_component_never_scanned", ll3 + ll_scans[0] + ll_scans[1] + EOI,
              note="jddiffct.c's arrays are not zeroed, so reading the third component raises")
    cases.raw("lossless_second_scan_no_table", ll3 + ll_scans[0] + sos([(2, 1, 0)], 1, 0, 0, 0)
              + entropy(102, 40) + ll_scans[2] + EOI)
    rows = SOI + sof(0xC3, 8, 8, rgb) + ll + dri(8)
    seg8 = b"".join(entropy(110 + r, 6) + bytes([0xFF, 0xD0 + r]) for r in range(7)) + entropy(117, 6)
    cases.raw("lossless_restart_changed_between_scans", rows + sos([(1, 0, 0)], 1, 0, 0, 0) + seg8 + dri(3)
              + sos([(2, 0, 0)], 1, 0, 0, 0) + seg8 + sos([(3, 0, 0)], 1, 0, 0, 0) + seg8 + EOI)
    resync = b"".join(entropy(120 + i, 6) + bytes([0xFF, 0xD0 + r]) for i, r in enumerate((0, 1, 3, 2, 4, 5, 6)))
    cases.raw("lossless_restart_resync", SOI + sof(0xC3, 8, 8, [(1, 1, 1, 0)]) + ll + dri(8)
              + sos([(1, 0, 0)], 1, 0, 0, 0) + resync + entropy(127, 6) + EOI, port="unsupported")
    # One scan, cut: the data is shorter than the samples take at the fewest bits each, so
    # libjpeg-turbo reads past the end.
    grey64 = SOI + sof(0xC3, 64, 64, [(1, 1, 1, 0)]) + ll + sos([(1, 0, 0)], 1, 0, 0, 0)
    cases.raw("lossless_single_scan_cut", grey64, [["entropy", 103, 100]])
    # One scan, cut, with more data than that: undecided.
    for side, extra, outcome in ((8, 40, "decodes"), (16, 0, "raises")):
        cases.raw(f"lossless_single_scan_open_{side}", SOI + sof(0xC3, side, side, [(1, 1, 1, 0)]) + ll
                  + sos([(1, 0, 0)], 1, 0, 0, 0), [["entropy", 1, side * side // 8 + 8 + extra]], port="unsupported",
                  note=f"undecided: Pillow {outcome}; whether libjpeg-turbo reads past the end depends on decoding")
    # After one scan, Pillow reads markers only within the 65,536-byte reads it has made.
    grey8 = SOI + sof(0xC3, 8, 8, [(1, 1, 1, 0)]) + ll + sos([(1, 0, 0)], 1, 0, 0, 0) + entropy(104, 50)
    cases.raw("lossless_single_scan_reserved_after", grey8 + b"\xff\x02" + EOI)
    cases.raw("lossless_single_scan_reserved_past_read", grey8 + com(65536 - len(grey8)) + b"\xff\x02" + EOI,
              port="unsupported")
    # A scan whose data runs on past the reads, then a fault: whether Pillow has read as far as the
    # fault when the last row is out depends on how far jdlhuff.c reads.
    grey512 = SOI + sof(0xC3, 512, 512, [(1, 1, 1, 0)]) + ll + sos([(1, 0, 0)], 1, 0, 0, 0)
    cases.raw("lossless_single_scan_open_fault", grey512, [["entropy", 105, 70000], ["append", "ff02ffd9"]],
              port="unsupported",
              note="undecided: Pillow raises; whether it has read as far as the fault depends on decoding")
    cases.raw("lossless_restart_every_row_large", SOI + sof(0xC3, 2000, 2000, [(1, 1, 1, 0)]) + ll + dri(2000)
              + sos([(1, 0, 0)], 1, 0, 0, 0) + entropy(106, 10) + EOI, port="unsupported")


# Recording -----------------------------------------------------------------------------------

def record(cases, image_prompt):
    out = {}
    for name, case in cases.cases.items():
        data = cases.build(case)
        entry = dict(case)
        entry["bytes"] = len(data)
        entry["sha256"] = hashlib.sha256(data).hexdigest()
        url = "data:image/jpeg;base64," + base64.b64encode(data).decode()
        try:
            image = image_prompt("", "", [url]).pil()[0]
            pixels = image.tobytes()
            entry["decoded"] = {"width": image.size[0], "height": image.size[1],
                                "sha256": hashlib.sha256(pixels).hexdigest()}
        except Exception as error:  # noqa: BLE001, upstream answers any of them with a 500
            entry["error"] = f"{type(error).__name__}: {re.sub(r' at 0x[0-9a-f]+', '', str(error))}"
        out[name] = entry
    return out


def upstream_head(upstream):
    try:
        return subprocess.run(["git", "-C", str(upstream), "rev-parse", "--short=7", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def render(payload):
    """One top-level key per line and one case per line, as the other fixtures."""
    lines = []
    for key, value in payload.items():
        if isinstance(value, dict) and value and all(isinstance(v, dict) for v in value.values()):
            items = ",\n".join(f" {json.dumps(k)}: {dumps(v)}" for k, v in value.items())
            lines.append(f"{json.dumps(key)}: {{\n{items}\n}}")
        else:
            lines.append(f"{json.dumps(key)}: {dumps(value)}")
    return "{\n" + ",\n".join(lines) + "\n}\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="compare with the committed file instead of writing it")
    ap.add_argument("--upstream", default=str(ROOT / "Upstream" / "openjev"),
                    help="a checkout of upstream at the pinned commit")
    args = ap.parse_args()
    upstream = Path(args.upstream)
    head = upstream_head(upstream)
    if head != UPSTREAM_COMMIT:
        raise SystemExit(f"{upstream} is at {head}, expected {UPSTREAM_COMMIT}; run make upstream")
    sys.path.insert(0, str(upstream))
    import PIL
    import PIL.features
    from openjev.mlx_backend import ImagePrompt

    sources = {name: (VISION / name).read_bytes() for name in SOURCES}
    cases = Cases(sources)
    define(cases, sources["baseline.jpg"], sources["progressive.jpg"])
    first, second = record(cases, ImagePrompt), record(cases, ImagePrompt)
    if first != second:
        raise SystemExit("two passes over the cases disagree")
    payload = {
        "generator": {
            "script": SCRIPT,
            "version": GENERATOR_VERSION,
            "upstream": "razorback16/openjev",
            "upstream_commit": UPSTREAM_COMMIT,
            "python": sys.version.split()[0],
            "pillow": PIL.__version__,
            "libjpeg_turbo": PIL.features.version("libjpeg_turbo"),
            "sources": {name: hashlib.sha256(data).hexdigest() for name, data in sources.items()},
        },
        "cases": first,
    }
    text = render(payload)
    if args.check:
        committed = OUT.read_text(encoding="utf-8")
        if committed != text:
            fresh, old = json.loads(text), json.loads(committed)
            changed = sorted(k for k in set(fresh["cases"]) | set(old["cases"])
                             if fresh["cases"].get(k) != old["cases"].get(k))
            raise SystemExit(f"{OUT.relative_to(ROOT)} differs: generator "
                             f"{'differs' if fresh['generator'] != old['generator'] else 'agrees'}, cases {changed}")
        print(f"{OUT.relative_to(ROOT)} agrees: {len(first)} cases")
        return
    OUT.write_text(text, encoding="utf-8")
    decoded = sum(1 for c in first.values() if "decoded" in c)
    print(f"wrote {OUT.relative_to(ROOT)}: {len(text.encode('utf-8'))} bytes, {len(first)} cases, "
          f"{decoded} decoded, {len(first) - decoded} raised", file=sys.stderr)


if __name__ == "__main__":
    main()
