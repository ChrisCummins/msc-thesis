# Our target document.
Document = thesis

# Import lib/*.sty libraries.
export TEXINPUTS := .:./lib:$(TEXINPUTS)

# Only count chapters in wc.
export TEXCOUNT_MATCH := ^chap/|abstract

# Import LaTeX build script.
include $(top_srcdir)/Makefile.latex
