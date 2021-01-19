raw=./uw3-hocr.tar
hocr=./uw3-tess.tar

ocropus=vocropus

# all: scaletrain.pth skewtrain.pth rottrain.pth segtrain.pth linetrain.pth

%.pth: %.sqlite3
	$(ocropus) slog getlast $< --output $@

models: wordtrain.pth segtrain.pth

# wordtrain.sqlite3: uw3-words.tar
# 	$(ocropus) ocroline train uw3-words.tar --log-to wordtrain.sqlite3.temp --epochs 10
# 	mv wordtrain.sqlite3.temp wordtrain.sqlite3

segtrain.sqlite3: uw3-seg.tar
	$(ocropus) ocroseg train uw3-seg.tar --display --log-to segtrain.sqlite3.temp --epochs 10 # --weightmask 20
	mv segtrain.sqlite3.temp segtrain.sqlite3

uw3-words.tar:
	$(ocropus) extract-lines hocr2rec $(hocr) --output uw3-words.tar.temp --element ocrx_word --extensions 'page.jpg tess.hocr' --bounds 10,10,500,100 --show 100 --dictionary /usr/share/dict/american-english-huge
	mv uw3-words.tar.temp uw3-words.tar

uw3-seg.tar:
	$(ocropus) extract-seg hocr2seg $(hocr) --output uw3-seg.tar.temp --element ocrx_word --extensions 'page.jpg tess.hocr' --invert True --show 100 --labels='1, 2, 3' --acceptable "10, 10, 500, 200"
	mv uw3-seg.tar.temp uw3-seg.tar

uw3-tess.tar: uw3-framed.tar
	tarp proc -c $(shell pwd)/'dtess sample.page.jpg sample.tess -l eng hocr' uw3-framed.tar -o uw3-tess.tar.temp
	mv uw3-tess.tar.temp uw3-tess.tar

uw3-framed.tar: $(raw)
	$(ocropus) extract-lines hocr2framed $(raw) --output uw3-framed.tar.temp --element ocr_line --show 10
	mv uw3-framed.tar.temp uw3-framed.tar

#scaletrain.sqlite3:
#	$(ocropus) ocrorot train-scale $(hocr) --output scaletrain.sqlite3.temp --nepochs 3 --replicate 8 --num-workers 8
#	mv scaletrain.sqlite3.temp scaletrain.sqlite3
#
#skewtrain.sqlite3:
#	$(ocropus) ocrorot train-skew $(hocr) --output skewtrain.sqlite3.temp --nepochs 3 --replicate 8 --num-workers 8
#	mv skewtrain.sqlite3.temp skewtrain.sqlite3
#
#rottrain.sqlite3:
#	$(ocropus) ocrorot train-rot $(hocr) --output rottrain.sqlite3.temp --nepochs 3 --replicate 8 --num-workers 8
#	mv rottrain.sqlite3.temp rottrain.sqlite3
#
#linetrain.sqlite3: uw3-lines.tar
#	$(ocropus) ocroline train uw3-lines.tar --display --mdef linemodel.py --log-to linetrain.sqlite3.temp --epochs 3
#	mv linetrain.sqlite3.temp linetrain.sqlite3
#
#uw3-seg.tar:
#	$(ocropus) extract-seg hocr2seg $(hocr) --output uw3-seg.tar.temp --element ocr_line --invert True --show 10
#	mv uw3-seg.tar.temp uw3-seg.tar
#
