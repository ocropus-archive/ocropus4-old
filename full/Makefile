hocr=/sdata/ocropus4/uw3/uw3-hocr.tar

ocropus=vocropus
all: segtrain

segtrain: uw3-seg.tar
	$(ocropus) ocroseg train uw3-seg.tar --display --mdef segmodel.py --log-to segtrain.sqlite3.temp --epochs 3 # --weightmask 20
	mv segtrain.sqlite3.temp segtrain.sqlite3

linetrain: uw3-lines.tar
	$(ocropus) ocroline train uw3-lines.tar --display --mdef linemodel.py --log-to linetrain.sqlite3.temp --epochs 3
	mv linetrain.sqlite3.temp linetrain.sqlite3

uw3-seg.tar:
	$(ocropus) extract-seg hocr2seg $(hocr) --output uw3-seg.tar.temp --element ocr_line --invert True --show 10
	mv uw3-seg.tar.temp uw3-seg.tar

uw3-lines.tar:
	$(ocropus) extract-lines hocr2rec $(hocr) --output uw3-lines.tar.temp --element ocr_line --bounds 10,2000,10,2000
	mv uw3-lines.tar.temp uw3-lines.tar
