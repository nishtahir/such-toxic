.PHONY: lint
lint:
	poetry run ruff check .
	poetry run ruff check --select I .
	poetry run toml-sort --check pyproject.toml

.PHONY: lint-fix
lint-fix:
	poetry run ruff check --select I --fix .
	poetry run ruff format
	poetry run prisma format
	poetry run toml-sort -i pyproject.toml

.PHONY: typecheck
typecheck:
	poetry run mypy .

.PHONY: test
test:
	poetry run pytest .