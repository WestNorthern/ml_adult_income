.PHONY: up rebuild

up:
	docker compose up --build

rebuild:
	docker compose build --no-cache
	docker compose up
