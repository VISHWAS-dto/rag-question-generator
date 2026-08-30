# Developer entrypoints. `uv` handles the environment; no manual venv activation.

.DEFAULT_GOAL := help
COMPOSE := docker compose
COMPOSE_PROD := docker compose -f docker-compose.yml -f docker-compose.prod.yml

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# --- Local (uv) --------------------------------------------------------------

.PHONY: install
install: ## Create the venv and install all extras
	uv sync --extra app --extra rag --extra dev

.PHONY: ingest
ingest: ## Build the local vector index in-process
	uv run python scripts/ingest_knowledge_base.py --in-process

.PHONY: run-rag
run-rag: ## Run the RAG service locally on :8100
	uv run uvicorn rag_service.main:app --port 8100 --reload

.PHONY: run-app
run-app: ## Run the app service locally on :8000 (RAG in-process, echo LLM)
	APP_RAG_MODE=in_process APP_LLM_PROVIDER=echo \
		uv run uvicorn app.api.main:app --port 8000 --reload

.PHONY: questions
questions: ## One-shot question generation (uses .env / env)
	uv run python scripts/generate_questions.py "$(INFO)"

# --- Quality ---------------------------------------------------------------

.PHONY: test
test: ## Run unit + integration tests (no live LLM)
	uv run pytest -m "not live"

.PHONY: test-unit
test-unit: ## Run only the fast unit tests
	uv run pytest -m unit

.PHONY: lint
lint: ## ruff + mypy
	uv run ruff check .
	uv run mypy app shared rag_service

.PHONY: fmt
fmt: ## Auto-fix lint issues
	uv run ruff check --fix .

# --- Docker --------------------------------------------------------------

.PHONY: build
build: ## Build all images
	$(COMPOSE) build

.PHONY: up
up: ## Start the full local stack (app + rag + postgres + frontend, echo LLM)
	$(COMPOSE) up -d --build
	@echo "frontend:  http://localhost:3000"
	@echo "app docs:  http://localhost:8000/docs"

.PHONY: up-ingest
up-ingest: ## Trigger ingestion against the running rag container
	$(COMPOSE) exec rag python /app/scripts/ingest_knowledge_base.py --in-process

.PHONY: down
down: ## Stop the local stack
	$(COMPOSE) down

.PHONY: logs
logs: ## Tail stack logs
	$(COMPOSE) logs -f

.PHONY: prod-config
prod-config: ## Render the merged production compose config (validation)
	$(COMPOSE_PROD) config

.PHONY: prod-up
prod-up: ## Start the production stack (adds self-hosted vLLM; needs a GPU)
	$(COMPOSE_PROD) up -d --build
