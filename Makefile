clean:
	rm -rf _docs/
	rm -rf _proc/_docs
deps:
	pipenv requirements --dev > tmp.txt 
	cat tmp.txt | sed -e "/^-/d" -e "/^#/d" | cut -d';' -f1 > requirements.txt
	rm tmp.txt
