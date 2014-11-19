# Import LaTeX build script.
include $(top_srcdir)/Makefile.latex

###############################################################################
# Build targets and configuration.
###############################################################################

# Our target document.
Document        = summary
DocumentSources = $(shell find . 			\
			\( ! -name '.*' \) 		\
			\( ! -name 'flycheck_*' \)	\
			-name '*.tex')
