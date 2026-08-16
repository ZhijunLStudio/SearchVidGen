# VidHarness 健康门禁（一键跑齐：pytest/mypy/ruff/regress/doctor）
.PHONY: test coverage mypy lint check regress doctor

test:
	python -m pytest tests/ -q

coverage:
	python -m pytest tests/ -q --cov=vidharness.core --cov-report=term

mypy:
	python -m mypy vidharness/ --ignore-missing-imports

lint:
	python -m ruff check vidharness/

regress:
	python -m vidharness.cli regress --output experiments

doctor:
	python -m vidharness.cli doctor --all experiments

check: test coverage mypy lint
