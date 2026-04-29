up-ops:
	docker compose --profile ops up -d

up-data:
	docker compose --profile data up -d

# Updated: ML always needs the Ops database/cache backend
up-ml:
	docker compose --profile ops --profile ml up -d

down:
	docker compose down

.PHONY: up-ops up-data up-ml down ai

ai:
	gemini