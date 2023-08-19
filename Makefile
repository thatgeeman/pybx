clean:
	rm -rf _docs/
	rm -rf _proc/_docs
deps:
	pipenv lock -r > tmp.txt 
	tail -r tmp.txt | sed -e "/^-/d" -e "/^#/d" | cut -d';' -f1 > requirements.txt
	rm tmp.txt
