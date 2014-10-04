SUBDIRS = \
	exercises \
	$(NULL)

if HAVE_PDFLATEX
SUBDIRS += proposal
endif

export CXX_STD = -std=c++11

if HAVE_CLANG
export CXX_STDLIB = -stdlib=libstdc++
else
export CXX_STDLIB =
endif

export AM_CXXFLAGS = 			\
	$(CXX_STD)			\
	$(CXX_STDLIB)			\
	-pthread 			\
	-Wall 				\
	-Wextra 			\
	-Wno-unused-parameter 		\
	-O3\
	$(NULL)

export AM_LDFLAGS = 			\
	$(CXX_STD)			\
	$(CXX_STDLIB)			\
	-pthread			\
	$(NULL)
