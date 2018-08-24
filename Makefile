# custom commands for working with this code
IMAGE=nbren12/samuwgh
gitbranch=$(shell git rev-parse --abbrev-ref HEAD)

clean:
	rm ./OBJ/*

build_image:
	docker build -t $(IMAGE):$(gitbranch) .

build_sam:
	docker run -it --privileged \
				-v $(pwd):/sam \
				-v $UWNET:/uwnet \
        -w /sam \
				$(IMAGE) ./Build

push-subtree:
	git subtree push --prefix ext/call_py_fort call_py_fort master


# open shell in docker
bash:
	docker run -it --privileged \
				-v $(pwd):/sam \
				-v $UWNET:/uwnet \
				$(IMAGE) bash


.PHONY: bash build
