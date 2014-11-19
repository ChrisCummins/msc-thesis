# Import LaTeX build script.
include $(top_srcdir)/Makefile.latex

###############################################################################
# Build targets and configuration.
###############################################################################

# Our target document.
Document        = proposal
DocumentSources = $(shell find . 			\
			\( ! -name '.*' \) 		\
			\( ! -name 'flycheck_*' \)	\
			-name '*.tex')
