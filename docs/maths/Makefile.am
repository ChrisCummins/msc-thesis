# Import common build script.
include $(top_srcdir)/Makefile.common

###############################################################################
# Build targets and configuration.
###############################################################################

# Our target document.
Document        = notes

# Derived variables.
DocumentPDF     = $(Document).pdf
DocumentTex     = $(Document).tex
DocumentSources = $(shell find . 			\
			\( ! -name '.*' \) 		\
			\( ! -name 'flycheck_*' \)	\
			-name '*.tex')

#------------------------------------------------------------------------
# Build targets.
#------------------------------------------------------------------------
doc_DATA = $(DocumentPDF)

#------------------------------------------------------------------------
# Clean list.
#------------------------------------------------------------------------
MOSTLYCLEANFILES +=		\
	$(DocumentPDF) 		\
	$(BuildLog) 		\
	*-blx.bib 		\
	*.acn 			\
	*.acr 			\
	*.alg 			\
	*.aux 			\
	*.bbl 			\
	*.bcf 			\
	*.blg 			\
	*.dvi 			\
	*.fdb_latexmk 		\
	*.glg 			\
	*.glo 			\
	*.gls 			\
	*.idx 			\
	*.ilg 			\
	*.ind 			\
	*.ist 			\
	*.lof 			\
	*.log 			\
	*.lol 			\
	*.lot 			\
	*.maf 			\
	*.mtc 			\
	*.mtc0 			\
	*.nav 			\
	*.nlo 			\
	*.out 			\
	*.pdfsync 		\
	*.ps 			\
	*.run.xml 		\
	*.snm 			\
	*.synctex.gz		\
	*.tdo 			\
	*.toc 			\
	*.vrb 			\
	*.xdy 			\
	.\#* 			\
	\#* 			\
	$(NULL)

# Import lib/*.sty libraries.
export TEXINPUTS := .:./lib:$(TEXINPUTS)

all: $(DocumentPDF)

$(DocumentPDF): $(DocumentSources)

#------------------------------------------------------------------------
# Local build rules.
#------------------------------------------------------------------------

# If we have xdg-open then we can open the PDF.
if HAVE_XDG_OPEN
.PHONY: open
open:
	$(AM_V_at)test -f $(DocumentPDF) || echo "File '$(DocumentPDF)' not found, is it compiled?"
	$(AM_V_at)test ! -f $(DocumentPDF) || $(XDG_OPEN) $(DocumentPDF) >/dev/null 2>&1 &
endif

# If we have texcount then we can perform word counts.
if HAVE_TEXCOUNT
.PHONY: wc
wc:
	$(AM_V_at)$(TEXCOUNT) $(DocumentSources) | $(SED) '1,/total/Id'
endif

# If we have checkcites then we can perform citation checks.
if HAVE_CHECKCITES
.PHONY: cite
cite:
	$(AM_V_at)$(CHECKCITES) $(Document).aux
endif

# Print make targets.
help-local:
	@echo "Build targets:"
	@echo
	@echo "  make all       Compile $(DocumentPDF)"
	@echo "  make clean     Remove all build files"
if HAVE_XDG_OPEN
	@echo "  make open      Open $(DocumentPDF) in default PDF viewer"
endif
if HAVE_TEXCOUNT
	@echo "  make wc        Show a word count"
endif
if HAVE_CHECKCITES
	@echo "  make cite      Check citations"
endif
