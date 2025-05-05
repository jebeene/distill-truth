IMAGE_NAME := distill-truth
MAKEFLAGS += --no-print-directory
.PHONY: help clean build run chat ds14 ds32

help: ## [misc] Show this help message
	@printf "\n\033[1mUsage:\033[0m make <command>\n"

	@awk -F':.*?## \\[docker\\] ' \
		'/^[a-zA-Z0-9_-]+:.*## \[docker\]/ {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}' ${MAKEFILE_LIST} \
		| sed '1s/^/\nDocker Commands:\n/'

	@awk -F':.*?## \\[model\\] ' \
		'/^[a-zA-Z0-9_-]+:.*## \[model\]/ {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}' ${MAKEFILE_LIST} \
		| sed '1s/^/\nModel Shortcuts:\n/'

	@awk -F':.*?## \\[util\\] ' \
		'/^[a-zA-Z0-9_-]+:.*## \[util\]/ {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}' ${MAKEFILE_LIST} \
		| sed '1s/^/\nUtility Commands:\n/'

	@awk -F':.*?## \\[meta\\] ' \
		'/^[a-zA-Z0-9_-]+:.*## \[meta\]/ {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}' ${MAKEFILE_LIST} \
		| sed '1s/^/\nMisc:\n/'

	@printf "\n"

# DOCKER COMMANDS
build: ## [docker] Build the Docker image
	docker build -t $(IMAGE_NAME) .

run: ## [docker] Start a bash shell in the Docker container
	docker run --rm -it \
		-v $(PWD):/app \
		-w /app \
		$(IMAGE_NAME) \
		bash

# PROJECT COMMANDS
chat:
ifndef MODEL
	$(error $(shell echo "\033[31mMODEL is not set. Use: make ds32 PROMPT=\"Hello\"\033[0m"))
endif
ifndef PROMPT
	$(error $(shell echo "\033[31mPROMPT is not set. Use: make ds32 PROMPT=\"Hello\"\033[0m"))
endif
	python -m openrouter.scripts.run_chat --model $(MODEL) "$(PROMPT)"

evaluate:
ifndef MODEL
	$(error $(shell echo "\033[31mMODEL is not set. Use: make ds32 PROMPT=\"Hello\"\033[0m"))
endif
	python -m openrouter.scripts.run_eval --model $(MODEL) --split "$(SPLIT)" --limit $(LIMIT)

# CHAT SHORTCUTS
chat-ds14: ## [model] Query DEEPSEEK_R1_14B distilled with PROMPT. Usage: make ds14 PROMPT="your message"
	make chat MODEL=DEEPSEEK_R1_14B PROMPT="$(PROMPT)"

chat-ds32: ## [model] Query DEEPSEEK_R1_32B distilled with PROMPT. Usage: make ds32 PROMPT="your message"
	make chat MODEL=DEEPSEEK_R1_32B PROMPT="$(PROMPT)"

# EVAL SHORTCUTS
eval-ds14: ## [model] Evaluate DEEPSEEK_R1_14B distilled with PROMPT. Usage: make ds14 PROMPT="your message"
	make evaluate MODEL=DEEPSEEK_R1_14B SPLIT="$(SPLIT)" LIMIT=$(LIMIT)

eval-ds32: ## [model] Evaluate DEEPSEEK_R1_32B distilled with PROMPT. Usage: make ds32 PROMPT="your message"
	make evaluate MODEL=DEEPSEEK_R1_32B SPLIT="$(SPLIT)" LIMIT=$(LIMIT)

# UTILS
clean: ## [util] Remove Python caches and prune Docker system
	rm -rf __pycache__ */__pycache__ *.pyc *.pyo *.log *.out .pytest_cache .mypy_cache

docker-clean: ## [util] Remove all stopped containers, dangling images, and unused networks
	docker system prune -f

# Example usage:
# make chat MODEL=DEEPSEEK_R1_14B PROMPT="the sky is red"
# make ds32 PROMPT="the sky is red"