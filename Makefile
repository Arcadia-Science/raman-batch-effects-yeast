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

zip-data:
	cd data && zip -r ../data.zip .

unzip-data:
	-rm -rf data_unzipped
	mkdir -p data_unzipped
	unzip -o data.zip -d data
