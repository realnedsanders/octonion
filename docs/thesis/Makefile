DOCKER_IMAGE := texlive/texlive:latest
DOCKER_RUN   := docker run --rm -v $(CURDIR):/work -w /work $(DOCKER_IMAGE)

TARGETS := oct-neural-nets oct-trie

.PHONY: all clean $(TARGETS)

all: $(addsuffix .pdf,$(TARGETS))

%.pdf: %.tex references.bib
	$(DOCKER_RUN) sh -c "pdflatex $* && bibtex $* && pdflatex $* && pdflatex $*"

clean:
	rm -f $(foreach t,$(TARGETS),$(t).pdf $(t).aux $(t).bbl $(t).blg $(t).log $(t).out $(t).toc)
