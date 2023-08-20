clean:
	rm -rf _docs/
	rm -rf _proc/_docs
deps:
	pipenv requirements --exclude-markers --dev > requirements.txt
	sed -i -e "/^-/d" -e "/^\./d" requirements.txt 
