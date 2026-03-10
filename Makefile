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

GDRIVE_REMOTE = gdrive
GDRIVE_FIGS_FOLDER_ID = 1TM-OurInS0BSn6DM404chWaMnpCSJgVX

# Manually set the ID of the Google Docs document to import.
GDRIVE_DOC_ID = 1uyDDqF-pRwffHfJm9eHOo19cOZmrJeOpp7eWDHQm1-Y

.PHONY: import-doc
import-doc:
	@echo "Downloading Google Doc as docx..."
	rclone backend copyid $(GDRIVE_REMOTE): $(GDRIVE_DOC_ID) pub/ --drive-export-formats docx
	@echo "Converting docx to markdown..."
	pandoc pub/*.docx -o pub/main-text.md --wrap=none --extract-media=pub/media
	rm -f pub/*.docx
	@echo "Done! Check pub/main-text.md"
