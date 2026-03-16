.PHONY: format
format:
	uv run ruff format
	uv run ruff check --fix

.PHONY: format-unsafe
format-unsafe:
	uv run ruff format
	uv run ruff check --fix --unsafe-fixes

.PHONY: lint
lint:
	uv run ruff check

.PHONY: test
test:
	uv run pytest

.PHONY: zip-data
zip-data:
	cd data && zip -r ../data.zip .

.PHONY: unzip-data
unzip-data:
	@if [ -d data ]; then echo "Error: data/ directory already exists"; exit 1; fi
	mkdir -p data
	unzip -o data.zip -d data
