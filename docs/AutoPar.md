# Notes

Approaches to automatic parallelisation:

* Loop nest parallelisation:
  * Transforms and restructures nest loop code for parallel execution.
  * Relies on accurate dependence analysis to inform the compiler of
    the legal code transformations. This is a complex problem.
    * In many cases, dependencies cannot be accurately determined
      until run-time, leading to over-conservative dependence
      relations. Approaches to this problem include:
      * Performing dependence analysis dynamically at runtime,
        e.g. through speculative parallelisation.
      * Requiring the programmer to give "hints" to the compiler by
        annotating their source programs. Bad due to extra effort for
        the programmer and bit-rot of annotations. Straddles the
        border of "automatic" parallelisation, in the case of
        something like OpenMP.
      * Designing languages and frameworks which minimise static
        uncertainty about dependencies. E.g. stricter programming
        languages, Algorithmic Skeletons. See next section.
    * Assuming accurate dependence information, the next problem is
      identifying from the set of possible transformations those that
      will provide legal parallelised code. Approaches to this problem
      include:
      * Using a transformation framework (e.g. unimodular, polyhedral)
        to reason about legal transformations in a purely mathematical
        way. XXX: What are these frameworks, and how do they work?
    * Similar to the previous problem, there is also the challenge of
      identifying the set of transformations provide the greatest
      performance profit. Another tough problem, as the space of
      transformations is often huge, and the best sequence of
      transformations may not be in any way obvious. Approaches to
      this problem include:
      * Selecting transformations based on empirical performance data
        (e.g. iterative compilation, auto-tuning). Big topic, with
        many approaches to searching the space, reducing
        dimensionality, using ML to build predictive models, etc.
      * Using analytic models to calculate cost/benefit ratio for
        transformations. Often hard to do for arbitrary input
        programs. Can be combined with empirical data + ML (see
        above).
* Speculative parallelisation
  * XXX:
* Skeletal programming
  * XXX:

# Reading list

##### Pre 1980

###### > 300 citations

1. D. Heller, “A Survey of Parallel Algorithms in Numerical Linear
   Algebra,” Siam Rev., vol. 20, no. 4, pp. 740–777, 1978.

###### > 150 citations

1. D. J. Kuck, “A Survey of Parallel Machine Organization and
   Programming,” ACM Comput. Surv., vol. 9, no. 1, pp. 29–59, 1977.

##### 1980 - 2000

###### > 1000 citations

1. D. F. Bacon, S. L. Graham, and O. J. Sharp, “Compiler
   transformations for high-performance computing,” ACM Comput. Surv.,
   vol. 26, no. 4, pp. 345–420, 1994.

1. R. D. Blumofe, C. F. Joerg, B. C. Kuszmaul, C. E. Leiserson,
   K. H. Randall, and Y. Zhou, “Cilk: an efficient multithreaded
   runtime system,” Proc. ACM SIGPLAN Symp. Princ. Pract. Prallel
   Program., vol. 30, no. 1, pp. 207–216, 1995.

###### > 500 citations

1. W.-M. W. Hwu, S. A. Mahlke, W. Y. Chen, P. P. Chang, N. J. Warter,
   R. A. Bringmann, R. G. Ouellette, R. E. Hank, T. Kiyohara,
   G. E. Haab, J. G. Holm, and D. M. Lavery, “The superblock: An
   effective technique for VLIW and superscalar compilation,”
   J. Supercomput., vol. 7, no. 1–2, pp. 229–248, May 1993.

1. D. B. Skillicorn and D. Talia, “Models and languages for parallel
   computation,” ACM Comput. Surv., vol. 30, no. 2, pp. 123–169, 1998.

Survey papers:

1. D. a. Padua and M. J. Wolfe, “Advanced compiler optimizations for
   supercomputers,” Commun. ACM, vol. 29, no. 12, pp. 1184–1201, 1986.

1. D. J. Kuck, D. J. Kuck, R. H. Kuhn, R. H. Kuhn, D. a. Padua,
   D. a. Padua, B. Leasure, B. Leasure, M. Wolfe, and M. Wolfe,
   “Dependence graphs and compiler optimizations,” Proc. 8th ACM
   SIGPLAN-SIGACT Symp. Princ. Program. Lang. - POPL ’81, pp. 207–218,
   1981.

1. D. H. Bailey, “Automatic Translation of Fortran Programs to
   Multiprecision,” vol. 9, no. 4, pp. 491–542, 1991.

###### > 300 citations

A survey paper of auto-parallelization in the early 90's, structured
as follows: 2 - Dependence analysis theory, 3 - parallelizing code
transformations, 4 - a review of the effectiveness of implementations:

1. U. Banerjee, R. Eigenmann, and A. Nicolau, “Automatic program
   parallelization,” Proc. IEEE, vol. 81, pp. 211–243, 1993.

1. A. Aiken and A. Nicolau, “Optimal Loop Parallelization,” in
   Proceedings of the SIGPLAN ’88 Conference on Programming Language
   Design and Implementation, 1988, vol. 23, no. 7, pp. 308–317.

1. S. Gupta and N. Dutt, “SPARK: A high-level synthesis framework for
   applying parallelizing compiler transformations,” in VLSI
   Design, 2003. Proceedings. 16th International Conference on, 2003,
   pp. 461–466.

1. R. Duncan, “Survey of parallel computer architectures,” Computer
   (Long. Beach. Calif)., vol. 23, pp. 5–16, 1990.

###### > 150 citations

1. M. W. Hall, S. P. Amarasinghe, B. R. Murphy, S.-W. Liao, and
   M. S. Lam, “Detecting coarse-grain parallelism using an
   interprocedural parallelizing compiler,” in
   Supercomputing, 1995. Proceedings of the IEEE/ACM SC95 Conference,
   1995, p. 49.

1. G. Gupta, E. Pontelli, K. a. M. Ali, M. Carlsson, and
   M. V. Hermenegildo, “Parallel execution of prolog programs: a
   survey,” ACM Trans. Program. Lang. Syst., vol. 23, no. 4,
   pp. 472–602, 2001.

###### > 20 citations

1. P. Walsh and C. Ryan, “Paragen: A Novel Technique for the
   Autoparallelisation of Sequential Programs using Genetic
   Programming,” in Proceedings of the 1st Annual Conference on
   Genetic Programming, 1996, pp. 406–409.

1. P. M. Petersen and D. a Padua, “Machine-Independent Evaluation of
   Parallelizing Compilers,” Computer (Long. Beach. Calif)., vol. 559,
   1992.


-------

#### 2000 - 2015


###### > 1500 citations

1. K. Compton and S. Hauck, “Reconfigurable computing: a survey of
   systems and software,” ACM Comput. Surv., vol. 34, no. 2,
   pp. 171–210, 2002.

###### > 500 citations

1. E. a Lee, “The Problem with Threads,” Computer
   (Long. Beach. Calif)., vol. 39, no. 5, pp. 33–42, 2006.

###### > 150 citations

1. H. Gonz, “A survey of algorithmic skeleton frameworks: high-level
   structured parallel programming enablers,” pp. 1135–1160, 2010.

-----

#### Books:

[Scheduling and Automatic Parallelization](https://books.google.co.uk/books?id=PEnln_iwipgC&lpg=PR9&ots=JPjwPt-aQf&lr&pg=PP1#v=onepage&q&f=false)
