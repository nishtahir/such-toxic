.PHONY: lint
lint:
	poetry run black --check .
	poetry run isort -c .

.PHONY: lint-fix
lint-fix:
	poetry run black .
	poetry run isort .
	poetry run nbqa isort .
	prisma format

.PHONY: typecheck
typecheck:
	poetry run mypy .

.PHONY: test
test:
	poetry run pytest .