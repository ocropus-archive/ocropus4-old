hocr=/sdata/ocropus4/uw3/uw3-hocr.tar

all: uw3-seg.tar uw3-lines.tar

uw3-seg.tar: docker
	./docropus extract-seg hocr2seg $(hocr) --output uw3-seg.tar.temp --element ocr_line --invert True
	mv uw3-seg.tar.temp uw3-seg.tar

uw3-lines.tar: docker
	./docropus extract-lines hocr2rec $(hocr) --output uw3-lines.tar.temp --element ocr_line --bounds 10,2000,10,2000
	mv uw3-lines.tar.temp uw3-lines.tar

docker:
	(cd .. && git commit -a || true)
	(cd .. && git archive --format=tar -o docker/ocropus4.tar HEAD)
	docker build . -t ocropus4
