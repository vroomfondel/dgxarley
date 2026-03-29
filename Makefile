.PHONY: gitleaks commit-checks help pypibuild pypipush
SHELL := /usr/bin/bash
.ONESHELL:

venv_activated=if [ -z $${VIRTUAL_ENV+x} ] && [ -z $${GITHUB_RUN_ID+x} ] ; then source .venv/bin/activate ; fi

help:
	@echo "gitleaks        — scan repo for leaked secrets"
	@echo "commit-checks   — run all pre-commit hooks on all files"
	@echo "pypibuild       — build package for pypi"
	@echo "pypipush        — push package to pypi"

.git/hooks/pre-commit:
	pre-commit install

gitleaks: .git/hooks/pre-commit
	pre-commit run gitleaks --all-files

commit-checks: .git/hooks/pre-commit
	pre-commit run --all-files

PKG_SOURCES := dgxarley/*
VERSION := $(shell $(venv_activated) > /dev/null 2>&1 && hatch version 2>/dev/null || echo HATCH_NOT_FOUND)

dist/dgxarley-$(VERSION).tar.gz dist/dgxarley-$(VERSION)-py3-none-any.whl dist/.touchfile: $(PKG_SOURCES) pyproject.toml
	@printf "VERSION: $(VERSION)\n"
	@$(venv_activated)
	hatch build --clean
	@touch dist/.touchfile

pypibuild: dist/dgxarley-$(VERSION).tar.gz dist/dgxarley-$(VERSION)-py3-none-any.whl

dist/.touchfile_push: dist/dgxarley-$(VERSION).tar.gz dist/dgxarley-$(VERSION)-py3-none-any.whl
	@$(venv_activated)
	hatch publish -r main
	@touch dist/.touchfile_push

pypipush: dist/.touchfile_push