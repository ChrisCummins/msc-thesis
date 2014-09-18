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

AC_INIT([report],[report_version],[author])
AC_CONFIG_AUX_DIR([build])

dnl Support for --program-prefix, --program-suffix and
dnl --program-transform-name options
AC_ARG_PROGRAM

dnl Fairly arbitrary, older versions might work too.
AM_INIT_AUTOMAKE([1.11 -Wall foreign no-define])
AM_SILENT_RULES([yes])


dnl ================================================================
dnl Build requirements
dnl ================================================================
AC_PROG_AWK
AC_PROG_INSTALL
AC_PROG_MKDIR_P
AC_PROG_SED

AC_CHECK_PROG([PDFLATEX],[pdflatex],[pdflatex])
AC_CHECK_PROG([BIBTEX],[bibtex],[bibtex])


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
dnl Generate files
dnl ================================================================
AC_CONFIG_FILES([
        Makefile
        proposal/Makefile
])

AC_OUTPUT

echo ""
echo "You can now run \`make'."
