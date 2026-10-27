"""The ``advisor_model`` plugin setting selects native's ``--advisor``.

Hermes' auxiliary one-shots never get an advisor, the setting is trimmed and read per spawn, and
selecting an advisor while Claude Code's telemetry/feature-flag fetch is off is called out once.
"""
import json
import logging
import os
import sys

import pytest

from conftest import PLUGIN_NAME
from test_directsdk import FAKE

QUIET = ("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "DISABLE_TELEMETRY", "DISABLE_ERROR_REPORTING")
_ABSENT = object()
PROBE = {"type": "function", "function": {"name": "probe", "description": "TAIL",
                                           "parameters": {"type": "object", "properties": {"value": {"type": "string"}}}}}
CAPTURE_ARGV = "open(os.environ['ARGV_CAPTURE'], 'a').write(json.dumps(sys.argv) + '\\n')\n"
CAPTURE_ENV = "import json as _j, os as _o\nopen(_o.environ['ENV_CAPTURE'], 'a').write(_j.dumps(dict(_o.environ)) + '\\n')\n"
# A Hermes auxiliary one-shot sends no tools; the shared fake asserts the catalog probe's TAIL instead.
ONE_SHOT_FAKE = FAKE.replace("assert wire['tools'][0]['description'].endswith('TAIL')",
                             "assert wire['tools'] == []")


def _yaml(value):
    return json.dumps(value)


def _set_settings(*, telemetry=_ABSENT, advisor=_ABSENT):
    """Write the plugin settings where ``hermes config set plugins.entries.<id>.settings.<key>`` puts them."""
    from hermes_cli.config import get_config_path
    lines = ["plugins:", "  enabled: []"]
    fields = []
    if telemetry is not _ABSENT:
        fields.append(("claude_code_telemetry", telemetry))
    if advisor is not _ABSENT:
        fields.append(("advisor_model", advisor))
    if fields:
        lines += ["  entries:", "    %s:" % PLUGIN_NAME, "      settings:"]
        lines += ["        %s: %s" % (key, _yaml(value)) for key, value in fields]
    get_config_path().write_text("\n".join(lines) + "\n", encoding="utf-8")


def _client(profile, tmp_path, *, fake=FAKE):
    """A client whose fake native appends each spawn's argv (and environment) to the capture files."""
    native = tmp_path / "native.py"
    native.write_text(fake.replace("rows=[]", CAPTURE_ENV + CAPTURE_ARGV + "rows=[]"), encoding="utf-8")
    return profile.create_client(command=[sys.executable, str(native)], env={
        "PATH": os.defpath, "HOME": str(tmp_path),
        "ARGV_CAPTURE": str(tmp_path / "argv.jsonl"), "ENV_CAPTURE": str(tmp_path / "env.jsonl")})


def _turn(client, **kwargs):
    return client.create(model="sonnet", messages=[{"role": "user", "content": "fixture"}], tools=[PROBE], **kwargs)


def _argvs(tmp_path):
    return [json.loads(line) for line in (tmp_path / "argv.jsonl").read_text().splitlines()]


def _envs(tmp_path):
    return [json.loads(line) for line in (tmp_path / "env.jsonl").read_text().splitlines()]


def _advisor(argv):
    """The advisor native received as an adjacent pair, or None when the flag is absent."""
    assert argv.count("--advisor") <= 1, argv
    if "--advisor" not in argv:
        return None
    return argv[argv.index("--advisor") + 1]


@pytest.mark.parametrize("setting, expected", [("fable", "fable"), ("  opus  ", "opus")])
def test_advisor_model_reaches_native_argv_trimmed(profile, tmp_path, setting, expected):
    _set_settings(advisor=setting)
    client = _client(profile, tmp_path)
    try:
        _turn(client)
    finally:
        client.close()
    (argv,) = _argvs(tmp_path)
    assert _advisor(argv) == expected


@pytest.mark.parametrize("setting", [_ABSENT, "", "   "])
def test_no_advisor_flag_when_the_setting_is_absent_or_blank(profile, tmp_path, setting):
    _set_settings(advisor=setting)
    client = _client(profile, tmp_path)
    try:
        _turn(client)
    finally:
        client.close()
    (argv,) = _argvs(tmp_path)
    assert "--advisor" not in argv


def test_a_hermes_auxiliary_one_shot_never_gets_the_advisor(profile, tmp_path):
    _set_settings(advisor="fable")
    client = _client(profile, tmp_path, fake=ONE_SHOT_FAKE)
    try:
        client.create(model="sonnet", messages=[{"role": "user", "content": "summarize fixture"}],
                      hermes_auxiliary=True)
    finally:
        client.close()
    (argv,) = _argvs(tmp_path)
    assert "--advisor" not in argv


def test_the_setting_is_read_per_spawn_from_the_active_profile(profile, tmp_path):
    """No import-time caching: flipping the setting changes the very next spawn on the same client."""
    _set_settings(advisor="")
    client = _client(profile, tmp_path)
    try:
        _turn(client)
        assert "--advisor" not in _argvs(tmp_path)[0]
        _set_settings(advisor="fable")
        _turn(client)
    finally:
        client.close()
    first, second = _argvs(tmp_path)
    assert "--advisor" not in first
    assert _advisor(second) == "fable"


def test_a_hermes_auxiliary_one_shot_with_telemetry_off_never_warns_about_the_advisor(profile, tmp_path, caplog):
    """A one-shot never gets the advisor, so the telemetry caveat would be noise on titles and summaries."""
    _set_settings(telemetry=False, advisor="fable")
    caplog.set_level(logging.WARNING)
    client = _client(profile, tmp_path, fake=ONE_SHOT_FAKE)
    try:
        client.create(model="sonnet", messages=[{"role": "user", "content": "summarize fixture"}],
                      hermes_auxiliary=True)
    finally:
        client.close()
    messages = [record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING]
    assert not [message for message in messages
                if "advisor_model" in message and ("telemetry" in message.lower() or "feature" in message.lower())], messages
    (argv,) = _argvs(tmp_path)
    assert "--advisor" not in argv
    for env in _envs(tmp_path):
        assert {key: env.get(key) for key in QUIET} == dict.fromkeys(QUIET, "1")


def test_an_advisor_with_telemetry_off_warns_once_and_keeps_the_quiet_flags(profile, tmp_path, caplog):
    _set_settings(telemetry=False, advisor="fable")
    caplog.set_level(logging.WARNING)
    client = _client(profile, tmp_path)
    try:
        _turn(client)
        _turn(client)
    finally:
        client.close()
    messages = [record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING]
    warnings = [message for message in messages
                if "advisor_model" in message and ("telemetry" in message.lower() or "feature" in message.lower())]
    assert len(warnings) == 1, messages
    for env in _envs(tmp_path):
        assert {key: env.get(key) for key in QUIET} == dict.fromkeys(QUIET, "1")


def test_telemetry_on_makes_no_advisor_warning(profile, tmp_path, caplog):
    _set_settings(telemetry=True, advisor="fable")
    caplog.set_level(logging.WARNING)
    client = _client(profile, tmp_path)
    try:
        _turn(client)
        _turn(client)
    finally:
        client.close()
    messages = [record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING]
    assert not [message for message in messages
                if "advisor_model" in message and ("telemetry" in message.lower() or "feature" in message.lower())], messages
    for argv in _argvs(tmp_path):
        assert _advisor(argv) == "fable"
