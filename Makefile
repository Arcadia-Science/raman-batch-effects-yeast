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
	unzip -o data.zip -d data_unzipped

# Google Drive sync commands
GDRIVE_REMOTE = gdrive
GDRIVE_FIGS_PATH = raman-batch-effects/figs
GDRIVE_DOC_NAME = raman-batch-effects-draft

.PHONY: sync-figs
sync-figs:
	rclone sync figs/ $(GDRIVE_REMOTE):$(GDRIVE_FIGS_PATH) -v

.PHONY: export-doc
export-doc:
	@echo "Converting markdown to docx..."
	pandoc pub/main-text.md -o pub/main-text.docx
	@echo "Uploading to Google Drive..."
	rclone copyto pub/main-text.docx $(GDRIVE_REMOTE):$(GDRIVE_DOC_NAME).docx
	@echo "Done! Open in Google Drive and it will convert to Google Docs format"

.PHONY: import-doc
import-doc:
	@echo "Downloading from Google Drive as docx..."
	rclone copyto $(GDRIVE_REMOTE):$(GDRIVE_DOC_NAME).docx pub/main-text-imported.docx --drive-export-formats docx
	@echo "Converting docx to markdown..."
	pandoc pub/main-text-imported.docx -o pub/main-text.md --wrap=none --extract-media=pub/media
	@echo "Done! Check pub/main-text.md"
