# hocr=/sdata/ocropus4/uw3/uw3-hocr.tar
hocr=./uw3-hocr.tar

# ocropus=env python3="coverage run -a" vocropus
ocropus=vocropus

OUTPUTS=\
	scaletrain.sqlite3 skewtrain.sqlite3 rottrain.sqlite3 \
	segtrain.sqlite3 linetrain.sqlite3 bintrain.sqlite3 \
	uw3-binarized.tar \
	deskew.out lines.out

# uw3-lines.tar uw3-seg.tar

all: $(OUTPUTS)

clean:
	rm -f $(OUTPUTS)
	rm -f *.pth

%.pth: %.sqlite3
	$(ocropus) slog getlast $< --output $@

uw3-binarized.tar: bintrain.pth
	$(ocropus) ocrobin binarize uw3-gray.tar --model bintrain.pth --output uw3-binarized.tar --limit 10

bintrain.sqlite3: uw3-gray.tar
	$(ocropus) ocrobin train uw3-gray.tar --num-workers 1 --num-epochs 3 --output bintrain.sqlite3

uw3-gray.tar: $(hocr)
	$(ocropus) ocrobin generate $(hocr) --output uw3-gray.tar --limit 20

lines.out: linetrain.pth
	$(ocropus) ocroline recognize --model linetrain.pth uw3-lines.tar --limit 10
	touch lines.out

deskew.out: rottrain.pth skewtrain.pth
	$(ocropus) ocrorot predict $(hocr) --limit 10 --rotmodel rottrain.pth --skewmodel skewtrain.pth | tee deskew.out

scaletrain.sqlite3:
	$(ocropus) ocrorot train-scale $(hocr) --output scaletrain.sqlite3.temp --nepochs 3 --replicate 8 --num-workers 8 --limit 50
	mv scaletrain.sqlite3.temp scaletrain.sqlite3

skewtrain.sqlite3:
	$(ocropus) ocrorot train-skew $(hocr) --output skewtrain.sqlite3.temp --nepochs 3 --replicate 8 --num-workers 8 --limit 50
	mv skewtrain.sqlite3.temp skewtrain.sqlite3

rottrain.sqlite3:
	$(ocropus) ocrorot train-rot $(hocr) --output rottrain.sqlite3.temp --nepochs 3 --replicate 8 --num-workers 8 --limit 50
	mv rottrain.sqlite3.temp rottrain.sqlite3

segtrain.sqlite3: uw3-seg.tar
	$(ocropus) ocroseg train uw3-seg.tar --display --mdef segmodel.py --log-to segtrain.sqlite3.temp --epochs 3 # --weightmask 20
	mv segtrain.sqlite3.temp segtrain.sqlite3

linetrain.sqlite3: uw3-lines.tar
	$(ocropus) ocroline train uw3-lines.tar --display --log-to linetrain.sqlite3.temp --epochs 5
	mv linetrain.sqlite3.temp linetrain.sqlite3

uw3-seg.tar:
	$(ocropus) extract-seg hocr2seg $(hocr) --output uw3-seg.tar.temp --element ocr_line --invert True --show 100 --maxcount 1000
	mv uw3-seg.tar.temp uw3-seg.tar

uw3-lines.tar:
	$(ocropus) extract-lines hocr2rec $(hocr) --output uw3-lines.tar.temp --element ocr_line --bounds 10,10,2000,2000 --maxcount 1000 --show 100
	mv uw3-lines.tar.temp uw3-lines.tar
