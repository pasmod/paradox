name = paradox

build:
	docker build -t $(name) $(BUILD_OPTS) .

stop:
	docker rm -f $(name) || true

run: stop start_redis
	docker run -it --rm=true -v $(shell pwd):/var/www \
		--link redis:db --name=$(name) $(name) bash -l

start_redis: stop_redis
	docker run --name redis -v $(shell pwd)/redis:/data -d redis redis-server --appendonly yes

stop_redis:
	docker rm -f redis || true

connect_redis:
	docker run -it --link redis:redis --rm redis redis-cli -h redis -p 8379:8379
