.PHONY: tests help install lint isort tcheck commit-checks prepare gitleaks distcheck pypibuild pypipush update-dockerhub-readmes
SHELL := /usr/bin/bash
.ONESHELL:

# Every tool is invoked through .venv/bin/* explicitly. The former
# `source .venv/bin/activate` guard keyed on $$VIRTUAL_ENV, which a parent
# shell can export without putting .venv/bin on PATH. The activation was
# then skipped and black/mypy/pytest/hatch resolved to ~/.local/bin instead.
VENV := .venv
VENV_BIN := $(VENV)/bin
PYTHON := $(VENV_BIN)/python

help:
	@printf "\ninstall\n\tinstall requirements\n"
	@printf "\nisort\n\tmake isort import corrections\n"
	@printf "\nlint\n\tmake linter check with black\n"
	@printf "\ntcheck\n\tmake static type checks with mypy\n"
	@printf "\ntests\n\tLaunch tests\n"
	@printf "\nprepare\n\tLaunch tests and commit-checks\n"
	@printf "\ncommit-checks\n\trun pre-commit checks on all files\n"
	@printf "\ngitleaks\n\tscan repo for leaked secrets\n"
	@printf "\ndistcheck\n\tverify dist artifacts carry no vault/secret files\n"
	@printf "\npypibuild\n\tbuild package for pypi\n"
	@printf "\npypipush\n\tpush package to pypi\n"
	@printf "\nupdate-dockerhub-readmes\n\tpush DOCKERHUB_OVERVIEW_*.md to the matching Docker Hub repo descriptions\n"

install: .venv

.venv: .venv/touchfile

# requirements-build.txt pulls requirements-dev.txt (and that requirements.txt),
# so the venv gets the tools every target below invokes as .venv/bin/<tool>:
# black, isort, mypy, pytest, pre-commit, hatch. A venv built from
# requirements.txt alone left all of them missing.
# TORCH_REQUIREMENTS picks the torch build mypy resolves its stubs from
# (roles/k8s_dgx/files/*.py and scripts/debughelper/*.py import torch, and torch
# is not in the ignore_missing_imports list). Override on a CPU-only box:
#   make install TORCH_REQUIREMENTS=requirements-dev-cpu.txt
TORCH_REQUIREMENTS ?= requirements-dev-gpu.txt

.venv/touchfile: requirements.txt requirements-dev.txt requirements-build.txt $(TORCH_REQUIREMENTS)
	test -d .venv || python3.14 -m venv .venv
	$(VENV_BIN)/pip install -r requirements-build.txt -r $(TORCH_REQUIREMENTS)
	touch .venv/touchfile

# `pytest` takes no path argument on purpose: an explicit target overrides
# the testpaths = ["tests"] set in pyproject.toml, which made `pytest .`
# also walk gitignored working dirs in the repo root (pytest does not read
# .gitignore). An unpacked sdist there carries its own copy of
# tests/test_base.py, and collecting both is an "import file mismatch"
# error under the same module name.
tests: .venv
	$(VENV_BIN)/pytest

lint: .venv
	$(VENV_BIN)/black .

isort: .venv
	$(VENV_BIN)/isort .

tcheck: .venv
	$(VENV_BIN)/mypy .

gitleaks: .venv .git/hooks/pre-commit
	$(VENV_BIN)/pre-commit run gitleaks --all-files

.git/hooks/pre-commit: .venv
	$(VENV_BIN)/pre-commit install

commit-checks: .git/hooks/pre-commit
	$(VENV_BIN)/pre-commit run --all-files

prepare: tests commit-checks

PKG_SOURCES := dgxarley/*
VERSION := $(shell $(VENV_BIN)/hatch version 2>/dev/null || echo HATCH_NOT_FOUND)

dist/dgxarley-$(VERSION).tar.gz dist/dgxarley-$(VERSION)-py3-none-any.whl dist/.touchfile: $(PKG_SOURCES) pyproject.toml
	@printf "VERSION: $(VERSION)\n"
	$(VENV_BIN)/hatch build --clean
	@touch dist/.touchfile

# Refuses to let a build artifact leave the machine if it contains git-crypt
# protected paths or any file outside the published allowlist.
# Gates both `pypibuild` and `pypipush`.
distcheck: dist/dgxarley-$(VERSION).tar.gz dist/dgxarley-$(VERSION)-py3-none-any.whl
	$(PYTHON) repo_scripts/check_dist_secrets.py \
	  dist/dgxarley-$(VERSION).tar.gz \
	  dist/dgxarley-$(VERSION)-py3-none-any.whl

pypibuild: distcheck

dist/.touchfile_push: dist/dgxarley-$(VERSION).tar.gz dist/dgxarley-$(VERSION)-py3-none-any.whl
	$(MAKE) distcheck
	$(VENV_BIN)/hatch publish -r main
	@touch dist/.touchfile_push

pypipush: dist/.touchfile_push

# DOCKERHUB_OVERVIEW_<image>.md → xomoxcc/<image>
# Short description (`description`) is taken from a `<!-- short: ... -->` HTML
# comment on the first line of the file (capped at 100 chars by Docker Hub).
# Long description (`full_description`) is the file content with table padding stripped (Docker Hub caps it at 25000 bytes).
DOCKERHUB_NAMESPACE := xomoxcc
DOCKERHUB_OVERVIEW_FILES := $(wildcard DOCKERHUB_OVERVIEW_*.md)

update-dockerhub-readmes:
	@if [ -z "$(DOCKERHUB_OVERVIEW_FILES)" ]; then \
	  echo "No DOCKERHUB_OVERVIEW_*.md files found at repo root"; exit 1; \
	fi
	@AUTH=$$(jq -r '.auths["https://index.docker.io/v1/"].auth' ~/.docker/config.json | base64 -d) && \
	USERNAME=$$(echo "$$AUTH" | cut -d: -f1) && \
	PASSWORD=$$(echo "$$AUTH" | cut -d: -f2-) && \
	echo "Login as: $$USERNAME" && \
	TOKEN=$$(curl -sS -X POST https://hub.docker.com/v2/users/login/ \
	  -H "Content-Type: application/json" \
	  -d '{"username":"'"$$USERNAME"'","password":"'"$$PASSWORD"'"}' \
	  | jq -r .token) && \
	if [ -z "$$TOKEN" ] || [ "$$TOKEN" = "null" ]; then \
	  echo "Login failed"; exit 1; \
	fi && \
	for FILE in $(DOCKERHUB_OVERVIEW_FILES); do \
	  IMAGE=$${FILE#DOCKERHUB_OVERVIEW_}; \
	  IMAGE=$${IMAGE%.md}; \
	  REPO="$(DOCKERHUB_NAMESPACE)/$$IMAGE"; \
	  SHORT=$$(sed -n 's/^<!--[[:space:]]*short:[[:space:]]*\(.*\)[[:space:]]*-->.*/\1/p' "$$FILE" | head -1 | sed 's/[[:space:]]*$$//'); \
	  if [ -z "$$SHORT" ]; then \
	    echo "  -> ERROR: no '<!-- short: ... -->' line on first line of $$FILE"; continue; \
	  fi; \
	  if [ $${#SHORT} -gt 100 ]; then \
	    echo "  -> ERROR: short description is $${#SHORT} chars (Docker Hub limit is 100) — skipping $$REPO"; continue; \
	  fi; \
	  echo "Updating $$REPO from $$FILE (short=$${#SHORT} chars, long=$$(sed -E '/^\|/{s/ {2,}\|/ |/g; s/-{3,}/---/g}' "$$FILE" | wc -c) bytes after table-padding strip, limit 25000)..."; \
	  RESP=$$(curl -sS -X PATCH "https://hub.docker.com/v2/repositories/$$REPO/" \
	    -H "Authorization: Bearer $$TOKEN" \
	    -H "Content-Type: application/json" \
	    -d "$$(jq -n --arg desc "$$SHORT" --rawfile full <(sed -E '/^\|/{s/ {2,}\|/ |/g; s/-{3,}/---/g}' "$$FILE") '{description: $$desc, full_description: $$full}')"); \
	  if echo "$$RESP" | jq -e '.last_updated' >/dev/null 2>&1; then \
	    echo "$$RESP" | jq -r '"  -> short=\"\(.description)\"  long=\(.full_description|length) chars  updated=\(.last_updated)"'; \
	  else \
	    echo "  -> ERROR: Docker Hub rejected the update for $$REPO:"; \
	    echo "$$RESP" | jq -r 'if type=="object" then (.detail // .message // (to_entries|map("\(.key): \(.value)")|join("; "))) else . end' | sed 's/^/       /'; \
	  fi; \
	done
