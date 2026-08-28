.PHONY: tests help install lint isort tcheck commit-checks prepare gitleaks distcheck pypibuild pypipush update-dockerhub-readmes
SHELL := /usr/bin/bash
.ONESHELL:

venv_activated=if [ -z $${VIRTUAL_ENV+x} ]; then printf "activating venv...\n" ; source .venv/bin/activate ; else printf "venv already activated\n"; fi

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

.venv/touchfile: requirements.txt
	test -d .venv || python3.14 -m venv .venv
	source .venv/bin/activate
	pip install -r requirements.txt
	touch .venv/touchfile

tests: .venv
	@$(venv_activated)
	pytest .

lint: .venv
	@$(venv_activated)
	black .

isort: .venv
	@$(venv_activated)
	isort .

tcheck: .venv
	@$(venv_activated)
	mypy .

gitleaks: .venv .git/hooks/pre-commit
	@$(venv_activated)
	pre-commit run gitleaks --all-files

.git/hooks/pre-commit: .venv
	@$(venv_activated)
	pre-commit install

commit-checks: .git/hooks/pre-commit
	@$(venv_activated)
	pre-commit run --all-files

prepare: tests commit-checks

PKG_SOURCES := dgxarley/*
VERSION := $(shell $(venv_activated) > /dev/null 2>&1 && hatch version 2>/dev/null || echo HATCH_NOT_FOUND)

dist/dgxarley-$(VERSION).tar.gz dist/dgxarley-$(VERSION)-py3-none-any.whl dist/.touchfile: $(PKG_SOURCES) pyproject.toml
	@printf "VERSION: $(VERSION)\n"
	@$(venv_activated)
	hatch build --clean
	@touch dist/.touchfile

# Refuses to let a build artifact leave the machine if it contains git-crypt
# protected paths or any file outside the published allowlist.
# Gates both `pypibuild` and `pypipush`.
distcheck: dist/dgxarley-$(VERSION).tar.gz dist/dgxarley-$(VERSION)-py3-none-any.whl
	@$(venv_activated)
	python repo_scripts/check_dist_secrets.py \
	  dist/dgxarley-$(VERSION).tar.gz \
	  dist/dgxarley-$(VERSION)-py3-none-any.whl

pypibuild: distcheck

dist/.touchfile_push: dist/dgxarley-$(VERSION).tar.gz dist/dgxarley-$(VERSION)-py3-none-any.whl
	@$(venv_activated)
	$(MAKE) distcheck
	hatch publish -r main
	@touch dist/.touchfile_push

pypipush: dist/.touchfile_push

# DOCKERHUB_OVERVIEW_<image>.md → xomoxcc/<image>
# Short description (`description`) is taken from a `<!-- short: ... -->` HTML
# comment on the first line of the file (capped at 100 chars by Docker Hub).
# Long description (`full_description`) is the file content verbatim.
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
	  echo "Updating $$REPO from $$FILE (short=$${#SHORT} chars, long=$$(wc -c < $$FILE) chars)..."; \
	  RESP=$$(curl -sS -X PATCH "https://hub.docker.com/v2/repositories/$$REPO/" \
	    -H "Authorization: Bearer $$TOKEN" \
	    -H "Content-Type: application/json" \
	    -d "$$(jq -n --arg desc "$$SHORT" --rawfile full "$$FILE" '{description: $$desc, full_description: $$full}')"); \
	  if echo "$$RESP" | jq -e '.last_updated' >/dev/null 2>&1; then \
	    echo "$$RESP" | jq -r '"  -> short=\"\(.description)\"  long=\(.full_description|length) chars  updated=\(.last_updated)"'; \
	  else \
	    echo "  -> ERROR: Docker Hub rejected the update for $$REPO:"; \
	    echo "$$RESP" | jq -r 'if type=="object" then (.detail // .message // (to_entries|map("\(.key): \(.value)")|join("; "))) else . end' | sed 's/^/       /'; \
	  fi; \
	done
