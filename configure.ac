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
dnl Report version (i.e. the package version)
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
AM_INIT_AUTOMAKE([1.11 -Wall foreign no-define -Wno-portability])
AM_SILENT_RULES([yes])


dnl ================================================================
dnl Build requirements
dnl ================================================================
AC_HEADER_STDC
AC_PROG_AWK
AC_PROG_CC
AC_PROG_CXX([clang++ g++])
AC_PROG_INSTALL
AC_PROG_MKDIR_P
AC_PROG_SED

AM_CONDITIONAL([HAVE_CLANG], [test "x$CXX" = xclang++])


dnl ================================================================
dnl Build configuration
dnl ================================================================
AC_CONFIG_SRCDIR([exercises/templates/dc.h])
AC_CONFIG_HEADERS([config.h])


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
dnl Generate files
dnl ================================================================
AC_CONFIG_FILES([
        Makefile
        exercises/Makefile
        exercises/mpi/Makefile
        exercises/templates/Makefile
        exercises/templates/dac/Makefile
        proposal/Makefile
])

AC_OUTPUT

echo ""
echo "You can now run \`make'."
