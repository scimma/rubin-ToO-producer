default:
	@echo "Targets:"
	@echo "docker - build docker image"
	@echo "test   - run tests"

docker:
	docker build .

test:
	@command -v pytest > /dev/null || (echo "pytest is required to run tests" 1>&2 && false)
	pytest -v tests.py