AC_PREREQ(2.65)

dnl ================================================================
dnl     Autoconf script for msc-thesis.
dnl
dnl To rebuild the configure script from this, execute command
dnl     autoconf
dnl in the directory containing this script.
dnl
dnl Copyright 2014 Chris Cummins.
dnl
dnl ================================================================


dnl ================================================================
dnl The package version
dnl ================================================================
m4_define([report_major_version],[0])
m4_define([report_minor_version],[0])
m4_define([report_micro_version],[1])
m4_define([report_version],[report_major_version.report_minor_version.report_micro_version])


dnl ================================================================
dnl Project details (i.e. the "meta" stuff)
dnl ================================================================
m4_define([author],[chrisc.101@gmail.com])

AC_INIT([report],[report_version],[author])
AC_CONFIG_AUX_DIR([build])

dnl Support for --program-prefix, --program-suffix and
dnl --program-transform-name options
AC_ARG_PROGRAM

dnl Fairly arbitrary, older versions might work too.
AM_INIT_AUTOMAKE([1.13 foreign -Wno-portability])
AM_SILENT_RULES([yes])

AM_EXTRA_RECURSIVE_TARGETS([help run todo fixme])

dnl ================================================================
dnl Build requirements
dnl ================================================================
dnl Note that the CC and CXX compilers are configured later, since we
dnl have our own local version of clang.
AC_HEADER_STDC
AC_PROG_AWK
AC_PROG_GREP
AC_PROG_INSTALL
AC_PROG_MKDIR_P
AC_PROG_SED


dnl ================================================================
dnl Build configuration
dnl ================================================================
AC_CONFIG_SRCDIR([exercises/templates/dc.h])
AC_CONFIG_HEADERS([config.h])


dnl ================================================================
dnl Do we have nproc installed?
dnl ================================================================
AC_CHECK_PROG([NPROC],[nproc],[nproc])
AS_IF([test "x$NPROC" != x],
      [HAVE_NPROC=yes],
      [HAVE_NPROC=no
       AC_MSG_WARN([Unable to determine the number of processors available.])])
AC_SUBST([HAVE_NPROC],[$HAVE_NPROC])
AM_CONDITIONAL([HAVE_NPROC],[test "x$HAVE_NPROC" = xyes])


dnl ================================================================
dnl Export the number of available processors (if we have nproc).
dnl ================================================================
AS_IF([test "x$HAVE_NPROC" = xyes],
      [NUM_PROC=$($NPROC)],
      [NUM_PROC=1
       AC_MSG_WARN([Defaulting to sequential back-end.])])
AC_SUBST([NUM_PROC],[$NUM_PROC])
AC_DEFINE_UNQUOTED([NUM_PROC],[$NUM_PROC],
                   [Number of processing units available.])

AC_MSG_NOTICE([Number of processor available: $NUM_PROC])


dnl Export whether we have clang or not.
AS_IF([test "xclang" = "x$(basename $CC 2>/dev/null)"],
      [HAVE_CLANG=yes],
      [HANVE_CLANG=no])
AC_SUBST([HAVE_CLANG],[$HAVE_CLANG])
AM_CONDITIONAL([HAVE_CLANG],[test "x$HAVE_CLANG" = xyes])

AS_IF([test "xclang++" = "x$(basename $CXX 2>/dev/null)"],
      [HAVE_CLANGXX=yes],
      [HANVE_CLANGXX=no])
AC_SUBST([HAVE_CLANGXX],[$HAVE_CLANGXX])
AM_CONDITIONAL([HAVE_CLANGXX],[test "x$HAVE_CLANGXX" = xyes])

dnl Export opt and llvm-dis binary paths.
OPT="$LLVM_BUILD_PATH/bin/opt"
AC_SUBST([OPT],[$OPT])

LLVM_DIS="$LLVM_BUILD_PATH/bin/llvm-dis"
AC_SUBST([LLVM_DIS],[$LLVM_DIS])

AC_PROG_CXX
AC_PROG_CC

dnl ================================================================
dnl Do we have MPI tools?
dnl ================================================================
AC_CHECK_PROG([MPICPP],[mpic++],[mpic++])
AS_IF([test "x$MPICPP" != x],
      [HAVE_MPICPP=yes],
      [HAVE_MPICPP=no])
AC_SUBST([HAVE_MPICPP],[$HAVE_MPICPP])
AM_CONDITIONAL([HAVE_MPICPP],[test "x$HAVE_MPICPP" = xyes])


AC_CHECK_PROG([MPIRUN],[mpirun],[mpirun])
AS_IF([test "x$MPIRUN" != x],
      [HAVE_MPIRUN=yes],
      [HAVE_MPIRUN=no])
AC_SUBST([HAVE_MPIRUN],[$HAVE_MPIRUN])
AM_CONDITIONAL([HAVE_MPIRUN],[test "x$HAVE_MPIRUN" = xyes])


AS_IF([test "x$HAVE_MPIRUN" = xyes && \
       test "x$HAVE_MPICPP" = xyes],
      [HAVE_MPI_TOOLS=yes],
      [HAVE_MPI_TOOLS=no
       AC_MSG_WARN([Unable to build and execute MPI programs.])])
AC_SUBST([HAVE_MPI_TOOLS],[$HAVE_MPI_TOOLS])
AM_CONDITIONAL([HAVE_MPI_TOOLS],[test "x$HAVE_MPI_TOOLS" = xyes])


dnl ================================================================
dnl Do we have etags installed?
dnl ================================================================
AC_CHECK_PROG([ETAGS],[etags],[etags])
AS_IF([test "x$ETAGS" != x],
      [HAVE_ETAGS=yes],
      [HAVE_ETAGS=no
       AC_MSG_WARN([Unable to build tags files for source code.])])
AC_SUBST([HAVE_ETAGS],[$HAVE_ETAGS])
AM_CONDITIONAL([HAVE_ETAGS],[test "x$HAVE_ETAGS" = xyes])


dnl ================================================================
dnl Do we have pdflatex installed?
dnl ================================================================
AC_CHECK_PROG([PDFLATEX],[pdflatex],[pdflatex])
AS_IF([test "x$PDFLATEX" != x],
      [HAVE_PDFLATEX=yes],
      [HAVE_PDFLATEX=no
       AC_MSG_WARN([Unable to build LaTeX documentation.])])
AC_SUBST([HAVE_PDFLATEX],[$HAVE_PDFLATEX])
AM_CONDITIONAL([HAVE_PDFLATEX],[test "x$HAVE_PDFLATEX" = xyes])


dnl ================================================================
dnl Do we have biber installed?
dnl ================================================================
AC_CHECK_PROG([BIBER],[biber],[biber])
AS_IF([test "x$BIBER" != x],
      [HAVE_BIBER=yes],
      [HAVE_BIBER=no
       AC_MSG_WARN([Unable to build LaTeX documentation.])])
AC_SUBST([HAVE_BIBER],[$HAVE_BIBER])
AM_CONDITIONAL([HAVE_BIBER],[test "x$HAVE_BIBER" = xyes])


dnl ================================================================
dnl Do we have bibtool installed?
dnl ================================================================
AC_CHECK_PROG([BIBTOOL],[bibtool],[bibtool])
AS_IF([test "x$BIBTOOL" != x],
      [HAVE_BIBTOOL=yes],
      [HAVE_BIBTOOL=no
       AC_MSG_WARN([Unable to export local BibTeX files.])])
AC_SUBST([HAVE_BIBTOOL],[$HAVE_BIBTOOL])
AM_CONDITIONAL([HAVE_BIBTOOL],[test "x$HAVE_BIBTOOL" = xyes])


dnl ================================================================
dnl Should xdg-open be used to open PDFs?
dnl ================================================================
AC_CHECK_PROG([XDG_OPEN],[xdg-open],[xdg-open])
AS_IF([test "x$XDG_OPEN" != x],
      [HAVE_XDG_OPEN=yes],
      [HAVE_XDG_OPEN=no
       AC_MSG_WARN([PDF opening feature disabled.])])
AC_SUBST([HAVE_XDG_OPEN],[$HAVE_XDG_OPEN])
AM_CONDITIONAL([HAVE_XDG_OPEN],[test "x$HAVE_XDG_OPEN" = xyes])


dnl ================================================================
dnl Should emacsclient be used for editing files?
dnl ================================================================
AC_CHECK_PROG([EMACSCLIENT],[emacsclient],[emacsclient])
AS_IF([test "x$EMACSCLIENT" != x],
      [HAVE_EMACSCLIENT=yes],
      [HAVE_EMACSCLIENT=no
       AC_MSG_WARN([Text editor launching features disabled.])])
AC_SUBST([HAVE_EMACSCLIENT],[$HAVE_EMACSCLIENT])
AM_CONDITIONAL([HAVE_EMACSCLIENT],[test "x$HAVE_EMACSCLIENT" = xyes])


dnl ================================================================
dnl Do we have texcount installed?
dnl ================================================================
AC_CHECK_PROG([TEXCOUNT],[texcount],[texcount])
AS_IF([test "x$TEXCOUNT" != x],
      [HAVE_TEXCOUNT=yes],
      [HAVE_TEXCOUNT=no
       AC_MSG_WARN([LaTeX word counts disabled.])])
AC_SUBST([HAVE_TEXCOUNT],[$HAVE_TEXCOUNT])
AM_CONDITIONAL([HAVE_TEXCOUNT],[test "x$HAVE_TEXCOUNT" = xyes])


dnl ================================================================
dnl Do we have checkcites installed?
dnl ================================================================
AC_CHECK_PROG([CHECKCITES],[checkcites],[checkcites])
AS_IF([test "x$CHECKCITES" != x],
      [HAVE_CHECKCITES=yes],
      [HAVE_CHECKCITES=no
       AC_MSG_WARN([LaTeX citation checks disabled.])])
AC_SUBST([HAVE_CHECKCITES],[$HAVE_CHECKCITES])
AM_CONDITIONAL([HAVE_CHECKCITES],[test "x$HAVE_CHECKCITES" = xyes])


dnl ================================================================
dnl Do we have textlint installed?
dnl ================================================================
AC_CHECK_PROG([TEXTLINT],[textlint],[textlint])
AS_IF([test "x$TEXTLINT" != x],
      [HAVE_TEXTLINT=yes],
      [HAVE_TEXTLINT=no
       AC_MSG_WARN([LaTeX style checks disabled.])])
AC_SUBST([HAVE_TEXTLINT],[$HAVE_TEXTLINT])
AM_CONDITIONAL([HAVE_TEXTLINT],[test "x$HAVE_TEXTLINT" = xyes])


dnl ================================================================
dnl Do we have cpplint installed?
dnl ================================================================
AC_CHECK_PROG([CPPLINT],[cpplint],[cpplint])
AS_IF([test "x$CPPLINT" != x],
      [HAVE_CPPLINT=yes],
      [HAVE_CPPLINT=no])
AC_SUBST([HAVE_CPPLINT],[$HAVE_CPPLINT])
AM_CONDITIONAL([HAVE_CPPLINT],[test "x$HAVE_CPPLINT" = xyes])


dnl ================================================================
dnl Set build configuration for C++ compilation
dnl ================================================================

AC_ARG_ENABLE([memory_counter],
              [AS_HELP_STRING([--enable-memory-counter],
                              [Enable memory profiling (default: no)])],
              [enable_memory_counter=yes],)
AM_CONDITIONAL(ENABLE_MEMORY_COUNTER,[test "x$enable_memory_counter" = xyes])

# Configurable options:
CXX_STD=c++11
_CXX_STDLIB=libstdc++
CXX_OLEVEL="-O0"

# Flags for C++ compilation
cxx_flags="\
-Wall
-Wextra
-Wno-unused-parameter"

# Flags for linking
ld_flags=""

dnl Export debugging symbols if not building with optimisation.
AS_IF([test "x$CXX_OLEVEL" = x-O0],
      [CXX_OLEVEL="$CXX_OLEVEL -pg"])

dnl Export CXX_STD flag
AC_SUBST([CXX_STD],[-std=$CXX_STD])

AC_MSG_NOTICE([Using C compiler: $CC])
AC_MSG_NOTICE([Using C++ compiler: $CXX])

dnl Export CXX_STDLIB flag if we're using clang
AS_IF([test "x$HAVE_CLANGXX" = xyes],
      [AC_MSG_NOTICE([Using C++ standard library: $CXX_STDLIB])
       CXX_STDLIB=-stdlib=$_CXX_STDLIB
       AC_SUBST([CXX_STDLIB],[$CXX_STDLIB])])

dnl Set and export CXXFLAGS
CXXFLAGS="$CXXFLAGS $CXX_STD $CXX_STDLIB $CXX_OLEVEL"
while read -r flag; do
      CXXFLAGS="$CXXFLAGS $flag"
done <<< "$cxx_flags"
unset cxx_flags
AC_SUBST([CXXFLAGS],[$CXXFLAGS])
AC_MSG_NOTICE([Using C++ optimisation flags: $CXX_OLEVEL])

dnl Set and export LDFLAGS
LDFLAGS="$LDFLAGS $CXX_STD $CXX_STDLIB"
while read -r flag; do
      LDFLAGS="$LDFLAGS $flag"
done <<< "$ld_flags"
unset ld_flags
AC_SUBST([LDFLAGS],[$LDFLAGS])


dnl ================================================================
dnl Configure SkelCL build
dnl ================================================================
dnl Export skelcl paths.
SKELCL_DIR="$(pwd)/skelcl"
SKELCL_BUILD_DIR="$SKELCL_DIR/build"

AC_SUBST([SKELCL_DIR],[$SKELCL_DIR])
AC_SUBST([SKELCL_BUILD_DIR],[$SKELCL_BUILD_DIR])

dnl Create skelcl build directory if needed.
AS_IF([test ! -d "$SKELCL_BUILD_DIR"],
      [mkdir -v "$SKELCL_BUILD_DIR";])


dnl ================================================================
dnl Export OpenCL paths
dnl ================================================================
dnl libOpenCL.so
AC_ARG_ENABLE([exercises],
              [AS_HELP_STRING([--enable-exercises],
                              [Build exercises (default: yes)])],,
              [enable_exercises=yes])
AM_CONDITIONAL(ENABLE_EXERCISES,[test "x$enable_exercises" = xyes])

AS_IF([test "x$enable_exercises" = xyes],
      [AC_ARG_WITH(opencl-lib,
                   AS_HELP_STRING([--with-opencl-lib=DIR],
                                  [Location of libOpenCL.so]),
                   [opencl_lib="$withval"],
                   [AC_MSG_WARN([using default OpenCL library location.])
                    opencl_lib="/opt/intel/opencl-sdk/lib64"])

       OPENCL_LIB="$opencl_lib"
       AC_CHECK_FILE([$OPENCL_LIB/libOpenCL.so],[],[AC_MSG_ERROR([OpenCL library not found!])])
       AC_SUBST([OPENCL_LIB],[$OPENCL_LIB])

       dnl OpenCL headers
       AC_ARG_WITH(opencl-headers,
                   AS_HELP_STRING([--with-opencl-headers=DIR],
                                  [Location of OpenCL include files.]),
                   [opencl_headers="$withval"],
                   [AC_MSG_WARN([using default OpenCL header location.])
       opencl_headers="/opt/intel/opencl-sdk/include"])

       OPENCL_HEADERS="$opencl_headers"
       AC_CHECK_FILE([$OPENCL_HEADERS/CL/cl.h],[],[AC_MSG_ERROR([OpenCL headers not found!])])
       AC_SUBST([OPENCL_HEADERS],[$OPENCL_HEADERS])])

dnl ================================================================
dnl Generate files
dnl ================================================================
AC_CONFIG_FILES([
        Makefile
        docs/Makefile
        docs/ipp/Makefile
        docs/maths/Makefile
        docs/proposal/Makefile
        docs/r1/Makefile
        docs/r2/Makefile
        docs/summary/Makefile
        docs/thesis/Makefile
        exercises/Makefile
        exercises/algos/Makefile
        exercises/mpi/Makefile
        exercises/opencl/Makefile
        exercises/skelcl/Makefile
        exercises/tbb/Makefile
        exercises/templates/Makefile
        exercises/templates/dac/Makefile
        experiments/Makefile
        experiments/20141105-skel-opt-space/Makefile
        experiments/20141120-stable-sort/Makefile
        skepu/Makefile
        omnitune/Makefile
        omnitune/native/Makefile
        skepu/examples/Makefile
        scraps/Makefile
        scraps/05-20/Makefile
])

AC_OUTPUT

echo ""
dnl Use the "build/test" directory to determine whether we have a
dnl built version of SkelCL.
AS_IF([test -d "$SKELCL_BUILD_DIR/test"],
      [echo "You can now run \`make'."],
      [echo "You must now configure and build skelcl."])
