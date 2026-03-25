all: experiment

kernel.o: kernel.mojo pyproject.toml
	uv run mojo build kernel.mojo -o kernel.o --emit object

experiment: main.swift kernel.o kernel.h
	swiftc main.swift kernel.o -import-objc-header kernel.h -o experiment

run: experiment
	./experiment

clean:
	rm -f experiment kernel.o
