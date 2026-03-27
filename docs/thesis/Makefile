DOCKER_IMAGE := texlive/texlive:latest
DOCKER_RUN   := docker run --rm -v $(CURDIR):/work -w /work $(DOCKER_IMAGE)

.PHONY: all clean

all: main.pdf

main.pdf: main.tex references.bib
	$(DOCKER_RUN) sh -c "pdflatex main && bibtex main && pdflatex main && pdflatex main"

clean:
	rm -f main.pdf main.aux main.bbl main.blg main.log main.out main.toc
