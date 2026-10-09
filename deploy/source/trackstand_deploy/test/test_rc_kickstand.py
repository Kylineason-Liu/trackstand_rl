import random

from trackstand_deploy.crsf import (
    BoolDebouncer,
    CrsfStreamParser,
    crc8_dvb_s2,
    decode_crsf_frame,
    decode_three_position_switch,
    unpack_rc_channels,
)


def _pack_channels(channels):
    bits = 0
    bits_available = 0
    payload = bytearray()
    for channel in channels:
        bits |= int(channel) << bits_available
        bits_available += 11
        while bits_available >= 8:
            payload.append(bits & 0xFF)
            bits >>= 8
            bits_available -= 8
    return bytes(payload)


def _crsf_frame(channels):
    body_without_crc = bytes([0x16]) + _pack_channels(channels)
    body = body_without_crc + bytes([crc8_dvb_s2(body_without_crc)])
    return bytes([0xC8, len(body)]) + body


def test_crsf_parser_matches_sixteen_channel_payload():
    channels = (172, 992, 1811, 1200, 1300, 1400, 1500, 1600) + (992,) * 8
    frame = _crsf_frame(channels)
    assert decode_crsf_frame(frame, "strict") == channels
    parser = CrsfStreamParser("strict")
    assert parser.feed(b"noise" + frame[:9]) == []
    assert parser.feed(frame[9:]) == [channels]


def test_channel_unpacking_matches_independent_bitstream_for_random_payloads():
    randomizer = random.Random(42)
    for _ in range(500):
        channels = tuple(randomizer.randrange(2048) for _ in range(16))
        payload = _pack_channels(channels)
        assert unpack_rc_channels(payload) == channels
        assert unpack_rc_channels(payload + b"ignored") == channels
    for length in range(22):
        assert unpack_rc_channels(bytes(length)) is None


def test_soft_crc_mode_matches_ros2_ws_receiver_setting():
    channels = (992,) * 16
    frame = bytearray(_crsf_frame(channels))
    frame[-1] ^= 0xFF
    assert decode_crsf_frame(bytes(frame), "strict") is None
    assert decode_crsf_frame(bytes(frame), "soft") == channels


def test_channel_seven_three_position_mapping_and_debounce():
    channels = [992] * 16
    channels[6] = 1811
    assert decode_three_position_switch(channels, 172, 992, 1811, 7, True, 0.35) is True
    channels[6] = 172
    assert decode_three_position_switch(channels, 172, 992, 1811, 7, True, 0.35) is False
    channels[6] = 992
    assert decode_three_position_switch(channels, 172, 992, 1811, 7, True, 0.35) is None

    debouncer = BoolDebouncer(8)
    for _ in range(7):
        assert debouncer.observe(True) is None
    assert debouncer.observe(True) is True
    assert debouncer.observe(None) is True
    for _ in range(7):
        assert debouncer.observe(False) is True
    assert debouncer.observe(False) is False
