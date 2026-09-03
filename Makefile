clean:
	rm -rf _docs/
	rm -rf _proc/_docs
deps:
	python -m pip install -e ".[dev]"
nbdev:
	nbdev-docs
	nbdev-readme
	nbdev-prepare
	nbdev-clean
	nbdev-export
