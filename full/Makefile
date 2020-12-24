hocr=/sdata/ocropus4/uw3/uw3-hocr.tar

all: linetrain

linetrain: uw3-lines.tar
	docropus rec train uw3-lines.tar --display --mdef linemodel.py --log-to linetrain.sqlite3.temp --epochs 50
	mv linetrain.sqlite3.temp linetrain.sqlite3

uw3-seg.tar:
	./docropus extract-seg hocr2seg $(hocr) --output uw3-seg.tar.temp --element ocr_line --invert True
	mv uw3-seg.tar.temp uw3-seg.tar

uw3-lines.tar:
	./docropus extract-lines hocr2rec $(hocr) --output uw3-lines.tar.temp --element ocr_line --bounds 10,2000,10,2000
	mv uw3-lines.tar.temp uw3-lines.tar

docker:
	(cd ../docker && make build)
