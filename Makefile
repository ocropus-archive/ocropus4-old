docstyle:
	pydocstyle ocropus/*.py > _errs || true
	/bin/echo -e 'g/^ocropus/.,.+1j\ng/^ocropus/s/ in /:/\nwq\n' | ed _errs
	cat _errs
vulture:
	vulture ocropus
