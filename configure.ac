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

AC_INIT([report],[report_version],[author]) : ${CXXFLAGS=""}
AC_CONFIG_AUX_DIR([build])

dnl Support for --program-prefix, --program-suffix and
dnl --program-transform-name options
AC_ARG_PROGRAM

dnl Fairly arbitrary, older versions might work too.
AM_INIT_AUTOMAKE([1.11 foreign -Wno-portability])
AM_SILENT_RULES([yes])

AM_EXTRA_RECURSIVE_TARGETS([help run])

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
dnl LLVM and Clang compiler configuration
dnl ================================================================
LLVM_TARGET="Release+Asserts"
AC_SUBST([LLVM_TARGET],[$LLVM_TARGET])

LLVM_BUILD_PATH="$(pwd)/llvm/$LLVM_TARGET"
AC_SUBST([LLVM_BUILD_PATH],[$LLVM_BUILD_PATH])

CC="$LLVM_BUILD_PATH/bin/clang"
dnl CC is exported by the AC_PROG_CC macro, called below.
CXX="$LLVM_BUILD_PATH/bin/clang++"
dnl CXX is exported by the AC_PROG_CXX macro, called below.

OPT="$LLVM_BUILD_PATH/bin/opt"
AC_SUBST([OPT],[$OPT])

LLVM_DIS="$LLVM_BUILD_PATH/bin/llvm-dis"
AC_SUBST([LLVM_DIS],[$LLVM_DIS])

test -f "$CXX" || {
     echo "Running one-time build of llvm...";
     cd llvm;
     ./configure --enable-optimisations;
     make;
     cd ..;
}

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
dnl Do we have bibtex installed?
dnl ================================================================
AC_CHECK_PROG([BIBTEX],[bibtex],[bibtex])
AS_IF([test "x$BIBTEX" != x],
      [HAVE_BIBTEX=yes],
      [HAVE_BIBTEX=no
       AC_MSG_WARN([Unable to build LaTeX documentation.])])
AC_SUBST([HAVE_BIBTEX],[$HAVE_BIBTEX])
AM_CONDITIONAL([HAVE_BIBTEX],[test "x$HAVE_BIBTEX" = xyes])


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
dnl Do we have cpplint installed?
dnl ================================================================
AC_CHECK_PROG([CPPLINT],[cpplint],[cpplint])
AS_IF([test "x$CPPLINT" != x],
      [HAVE_CPPLINT=yes],
      [HAVE_CPPLINT=no])
AC_SUBST([HAVE_CPPLINT],[$HAVE_CPPLINT])
AM_CONDITIONAL([HAVE_CPPLINT],[test "x$HAVE_CPPLINT" = xyes])


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
CXX_STDLIB=libstdc++
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

dnl Export CXX_STDLIB flag
AC_MSG_NOTICE([Using C++ standard library: $CXX_STDLIB])
CXX_STDLIB=-stdlib=$CXX_STDLIB
AC_SUBST([CXX_STDLIB],[$CXX_STDLIB])

dnl Set and export CXXFLAGS
CXXFLAGS="$CXX_STD $CXX_STDLIB $CXX_OLEVEL"
while read -r flag; do
      CXXFLAGS="$CXXFLAGS $flag"
done <<< "$cxx_flags"
unset cxx_flags
AC_SUBST([CXXFLAGS],[$CXXFLAGS])
AC_MSG_NOTICE([Using C++ optimisation flags: $CXX_OLEVEL])

dnl Set and export LDFLAGS
LDFLAGS="$CXX_STD $CXX_STDLIB"
while read -r flag; do
      LDFLAGS="$LDFLAGS $flag"
done <<< "$ld_flags"
unset ld_flags
AC_SUBST([LDFLAGS],[$LDFLAGS])


dnl ================================================================
dnl Generate files
dnl ================================================================
AC_CONFIG_FILES([
        Makefile
        docs/Makefile
        docs/maths/Makefile
        docs/proposal/Makefile
        docs/summary/Makefile
        exercises/Makefile
        exercises/algos/Makefile
        exercises/llvm/Makefile
        exercises/mpi/Makefile
        exercises/tbb/Makefile
        exercises/templates/Makefile
        exercises/templates/dac/Makefile
        experiments/Makefile
        experiments/20141105-skel-opt-space/Makefile
])

AC_OUTPUT

echo ""
echo "You can now run \`make'."
