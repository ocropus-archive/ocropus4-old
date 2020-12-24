build:
	docker build - -t ocropus4-base < Dockerfile-base
	(cd .. && git archive --format=tar -o docker/ocropus4.tar HEAD)
	docker build --no-cache . -t ocropus4
