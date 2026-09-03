clean:
	rm -rf _docs/
	rm -rf _proc/_docs
deps:
	pipenv requirements --exclude-markers --dev > requirements.txt
	sed -i -e "/^-/d" -e "/^\./d" requirements.txt 
nbdev:
	nbdev_docs
	nbdev_readme
	nbdev_prepare
	nbdev_clean
	nbdev_export
