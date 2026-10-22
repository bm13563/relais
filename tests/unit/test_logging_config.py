"""Unit tests for logging_config.py — relais logs through the standard `logging`
module: events with fields, under "relais", into whatever the host configured."""

import logging

from relais import pipeline, tools
from relais.engines import api
from relais.logging_config import get_logger


class Unprintable:
    """A field whose rendering fails — proves a disabled line never renders."""

    def __str__(self):
        raise AssertionError("rendered a disabled log line")

    __repr__ = __str__


class TestEventLogger:
    def test_an_event_is_a_standard_record_with_its_fields(self, caplog):
        caplog.set_level(logging.INFO, logger="relais")
        get_logger("relais.demo").info("step_done", run="r1", step="draft", turns=3)

        [record] = [r for r in caplog.records if r.name == "relais.demo"]
        assert record.levelno == logging.INFO
        assert record.getMessage() == "step_done run=r1 step=draft turns=3"
        assert record.relais == {"run": "r1", "step": "draft", "turns": 3}

    def test_the_record_names_relais_caller_not_the_adapter(self, caplog):
        caplog.set_level(logging.INFO, logger="relais")
        get_logger("relais.demo").info("here")

        [record] = [r for r in caplog.records if r.name == "relais.demo"]
        assert record.funcName == "test_the_record_names_relais_caller_not_the_adapter"

    def test_a_disabled_level_never_renders_its_fields(self, caplog):
        caplog.set_level(logging.INFO, logger="relais")
        get_logger("relais.demo").debug("step_context", context=Unprintable())

        assert not [r for r in caplog.records if r.name == "relais.demo"]

    def test_every_level_maps_to_its_standard_level(self, caplog):
        caplog.set_level(logging.DEBUG, logger="relais")
        log = get_logger("relais.demo")
        log.debug("a")
        log.info("b")
        log.warning("c")
        log.error("d")

        levels = [r.levelno for r in caplog.records if r.name == "relais.demo"]
        assert levels == [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR]


class TestTheLibraryConfiguresNothing:
    def test_relais_carries_only_a_null_handler(self):
        """A library never decides where lines go: the host's handlers do."""
        handlers = logging.getLogger("relais").handlers
        assert [type(h) for h in handlers] == [logging.NullHandler]

    def test_relais_loggers_are_children_of_relais(self):
        """So a host sets one level, on "relais", for all of them (the
        orchestrator's are "relais" and "relais.<pipeline>")."""
        for module in (pipeline, tools, api):
            assert module.log.logger.name.startswith("relais.")
