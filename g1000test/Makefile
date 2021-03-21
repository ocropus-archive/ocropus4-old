quick: clean
	. ../venv/bin/activate; papermill quicktest.ipynb output.ipynb
	. ../venv/bin/activate; jupyter nbconvert output.ipynb output.html

clean:
	rm -rf *-execute
	rm -rf *.sqlite3 *.pth
	rm -rf *-seg.tar *-words.tar *-lines.tar
	rm -rf full.sos output.ipynb
	# we're not rerunning tess by default

full:
	rm -f full.sos
	. ../venv/bin/activate; sos convert quicktest.ipynb full.sos
	sed -i -e 's/^multiplier = .*/multiplier = 1000/' full.sos
	. ../venv/bin/activate; sos run full.sos train-segs
	. ../venv/bin/activate; sos run full.sos train-words
	. ../venv/bin/activate; sos run full.sos train-lines
	. ../venv/bin/activate; sos run full.sos page-recognition
	. ../venv/bin/activate; sos run full.sos train-rot
	. ../venv/bin/activate; sos run full.sos predict-rot
	. ../venv/bin/activate; sos run full.sos train-skew
	. ../venv/bin/activate; sos run full.sos predict-skew
	. ../venv/bin/activate; sos run full.sos train-scale
	. ../venv/bin/activate; sos run full.sos predict-scale
	. ../venv/bin/activate; sos run full.sos train-bin

