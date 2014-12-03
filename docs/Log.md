# September 2014

## Friday 12th

### Notes for next supervisor meeting:

 * Weekly (?) meetings with Pavlos
 * Need to know convenient times
 * Logbook / meeting notes:
   * Same as undergrad project: [Markdown log hosted on GitHub](https://github.com/ChrisCummins/pip-db/blob/master/Documentation/Log.md)?
 * MSc project
   * LLVM good, GCC bad
   * [Algorithmic Skeletons: Structured Management of Parallel Computation](https://www.google.co.uk/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&ved=0CCkQFjAB&url=http%3A%2F%2Fhomepages.inf.ed.ac.uk%2Fmic%2FPubs%2Fskeletonbook.ps.gz&ei=2f0WVNfmF47saNyigIAN&usg=AFQjCNGcrR6rY75gPqfz7y4Dr_Ztj48_dg&sig2=SNZO8I1r1oor05uXABraUA&bvm=bv.75097201,d.d2s&cad=rja)
     * Extended version of PhD thesis
     * "Increase raw performance by replication as well as by
       miniaturisation"
   * Similarities to HOF in functional languages
   * C++ Templates
     * What advantages over function pointers in C?
   * Compiled languages vs runtimes
     * There are some interesting lesser used languages with features
       that would simplify the task of auto parallelising.
     * E.g. Clojure: "For the concurrent programming future,
       pervasive, unmoderated mutation simply has to go".
     * Supposedly Haskell is even more "pure" and so capable of
       implicit parallelisation.

### Courses

Deadline 22/9/14, although choices are "not binding at this stage".

 * [Compiler Optimisation](http://www.inf.ed.ac.uk/teaching/courses/copt/)
   * Michael O'Boyle / Hugh Leather ?
   * 10 credits
   * Semester 2

 * [Probabilistic Modelling and Reasoning](http://www.inf.ed.ac.uk/teaching/courses/pmr/)
   * Amos Storkey
   * 10 credits
   * Semester 2

 * [Advances in Programming Languages](http://www.inf.ed.ac.uk/teaching/courses/apl/2014-2015/)
   * Ian Stark
   * 10 credits
   * Semester 1

 * [Machine Learning and Pattern Recognition](http://www.inf.ed.ac.uk/teaching/courses/mlpr/)
   * Chris Williams and Iain Murray
   * 10 credits
   * Semester 1
   * Tuesdays and Fridays, 10-10:50 am

 * [Parallel Design Patterns](http://www.epcc.ed.ac.uk/education-training/training/course-portfolio/parallel-design-patterns)

 * [Advanced Parallel Programming](http://www.epcc.ed.ac.uk/education-training/training/course-portfolio/advanced-parallel-programming)


## Monday 15th

Could we perform static analysis in JIT runtime?
[Relevant discussion](https://www.usenix.org/legacy/event/jvm02/yu/yu_html/node3.html).

**Persistent data structure:** a data structure that always preserves
the previous version of itself when modified. Thus effectively
immutable, as operations do not update the structure in-place, but
yield a new updated structure.

### TODO:

1. Decide on courses
1. Investigate MSc project options
1. Read [Hacker's guide to LLVM](http://www.aosabook.org/en/llvm.html)
1. Work through machine learning maths preliminaries
1. Work through a practical MPI tutorial
1. Read up on C++ templating
1. Concurrent programming pattern: Actor
1. Become familiar with Haskell, Scala, and Erlang
1. Take out relevant library books:
   * Mathematics for Engineers - Croft and Davison
   * Data Mining: Practical Machine Learning Tools and Techniques

### Notes from meeting with Pavlos & Hugh

Google hangout 9pm

 * The HOF component of Skeletons becomes less interesting/relevant as
   the complexity of skeletons grow. So something simple like divide
   and conquer could be implemented in a small amount of code, but
   complexity comes from the mechanics of parallelising, adding
   robustness, coordinating heterogeneous devices etc.

 * It may be more difficult to get work published for "non-standard"
   languages:
   * E.g. If the majority of published work is in C++, then an
     implementation in Clojure would be compare unfavourably in terms
     of performance.

 * Get organised about making notes on papers when reading them. There
   will be a *lot* of reading, so a good database of notes is
   essential. There are applications to help manage these.

 * It's important to get an understanding of the conferences and
   competitors in your field, so as to know *who* and *what* to keep
   an eye on.

 * **TODO:** In 2 months, create a spreadsheet of relevant competitors
   and conferences. Include paper submission deadlines, dates for
   publishing proceedings, etc.

 * **TODO:** Re-implement the memory read/write instrumentation from
   the programming assessment for LLVM, or implement something of
   comparable interest.

 * **TODO:** Implement a simple sequential divide and conquer skeleton
   in C++.

 * The reading club will restart at some point. Non-mandatory since
   MSc but relevant/useful.

 * **TODO:** At some point, I should pick a paper to critically
   analyse for Pavlos and Hugh. Same as reading club but on one-to-one
   basis.


## Tuesday 16th

There are weekly lectures for Pervasive Parallelism module on Tuesdays
and Thursdays, 15:10-16:00 in the Faculty Room South, David Hume
Tower.

Reference management software to test:

 * Mendeley 7
 * Zotero 6
 * Jabref 4
 * BibDesk 3 (only runs on Mac)
 * Endnote 1
 * Refworks 1

TODO:

 * Read The Researcher's Bible - DONE
 * Find one relevant conference / competitor and note details - DONE
 * Follow a practical tutorial on MPI
 * Find an up-to-date paper on Skeletons - DONE
 * Begin working through MLPR background maths sheet - DONE

### Notes on "[Learning from Data: Supplementary Mathematics](http://www.inf.ed.ac.uk/teaching/courses/mlpr/notes/mlpr-supplementary-maths.pdf)" (D Barber)

Scalar - a quantitity which "has magnitude but which is not related to
any definite direction in space."

Vectors - "It has magnitude and is also related to a definite
direction in space." When representing vectors as letters, they are
often underlined. In print, the are represented using either boldface
or a directional arrow above the letter.

Norm - another term for the length/magnitude of a vector.

Vector addition - using parallelogram law. Commutative and
associative.

Coplanar - in the same plane.

Position vectors - "The point *P* can be represented by the position
vector **p**, which is a vector whose initial point is the origin of
the coordinate system."

Unit vectors along the direction of the coodinate axes *X, Y, Z* are
commonly denoted as **i** = (1, 0, 0), **j** = (0, 1, 0), **k** = (0,
0, 1).

There's a bunch of mathematical symbols (esp. regarding set theory)
which I'm unfamiliar with. Should be sure to work through
[the list](http://en.wikipedia.org/wiki/List_of_mathematical_symbols).


### Notes on [The Researcher's Bible](http://homepages.inf.ed.ac.uk/bundy/how-tos/resbible.html)

 * The thesis + viva is to prove:

> that the student is capable of pursuing original research making a
> significant contribution to knowledge or understanding in the field
> of study, relating particular research projects to the general body
> of knowledge in the field, and presenting the results of the
> research in a critical and scholarly way.
> ([regulation 40](http://www.drps.ed.ac.uk/14-15/regulations/postgrad.php#Grounds))

 * American PhD students spend 5/6 years studying.
 * Read the literature and talk to fellow workers to find out what the
   state of the art is. In particular, check the further work section
   of papers for ideas for research.
 * After choosing a topic:
   * Read the literature.
   * Talk to people.
   * Tackle a simplified version of the problem.
   * Write down ideas in a working paper.
   * Give a talk to a small group.
 * "Your program must be explainable at a higher level than code, for
   it to make a real contribution to knowledge. Try to plan your
   program theoretically before going to the terminal."
 * Set aside a part of the week for reading reviews and abstracts and
   skimming papers.
 * Pick a "non-threatening, attractive task" to start each day with.
 * Do not underestimate the importance of being enthusiastic about
   your research topic.
 * "Beginning students tend to underestimate the worthiness and
   overestimate the doability of projects".
 * Do not focus on implementation and performance at the expense of
   vauge assumptions and principles.
 * Make writing a regular part of your life. Document:
   * Ideas you have
   * Documentation of programs
   * Lecture notes
   * Notes on papers
 * [Edinburgh publications](http://www.ed.ac.uk/schools-departments/informatics/research/publications/)
 * Advice for writing:
   * A paper should have a clear message that you understand and is
     repeated in the title, abstract, introduction, conclusion, and
     structure.
   * Think of your audience as intelligent, but (a) ignorant and (b)
     given to wilful misunderstanding.
   * A reader should get the main idea of the paper from the first
     page. Long rambling introductions should be pruned ruthlessly.
   * The basic framework for a scientific paper is: what
     claim/hypothesis am I making and what is the evidence for this
     claim.
   * Have a particular reader in mind to keep the technical standard
     of the paper uniform.
   * Clearly state what is new or better about what you have done.
   * A bi-directional combination of top down and bottom up writings
     styles is the most successful.
 * Advice for reading:
   * Think of the work of others as arranged in concentric circles
     around your own.
   * Outer circle - skim papers and abstracts to achieve appropriate
     level of familliarity with this circle. Set aside an hour each
     week to skim the latest arrivals.
   * Middle circle - Read some papers right through and discuss with
     others. Write a paragraph on each paper as you read it.
   * Inner circle - deep understanding, requiring several reads of a
     paper. Work through examples. Talk to the author about it. Write
     to them with a list of queries and/or criticisms. Teach it to
     others.
  * Have a succinct answer prepared for the question "what do you
    consider to be the key contributions in your thesis?"


### Notes from Ppar Lecture

3:10pm Murray Cole

Lecture Notes: http://www.inf.ed.ac.uk/teaching/courses/ppar/

 * Read ahead lecture material and course texts.
 * **14/10/14** - I give Davide Pinato's presentation about 2
     scientific papers.
 * **21/10/14** - Martin Ruefenacht presents my material about 2
     scientific papers.
 * You should aim to read 10-20 papers per week
   (abstract+conclusions).
 * **TODO:** Read through miscellaneous resources section of lecture
     slides.
 * **TODO:** Pick 2 interesting papers within the "inner circle" and
   discuss with Pavlos.
 * Read project proposals, in order to learn how to write better
   project proposals.

First presentation content:

 * Context
 * Key contribution
 * Impact
 * Critical reflection

Today's relevant conference was PLDI (see google doc).


## Wednesday 17th

Have set up Mendeley account to use for tracking notes on acadaemic
papers.

Skeletons:

> structured parallel programming abstraction that provide programmers
> with a predefined set of algorithmic templates that can be combined,
> nest and parameterized with sequential code to produce complex
> programs.
> ([Reference](http://homepages.inf.ed.ac.uk/hleather/publications/2012_autotuneparallelskeletons_ppl.pdf))

[Monte Carlo methods](http://en.wikipedia.org/wiki/Monte_Carlo) -
class of algorithms that rely on repeated random sampling to obtain
numerical results. Often used when it is difficult or impossible to
obtain a closed-form expression, or unfeasible to apply a
deterministic algorithm.

### Second Ppar lunch

Pavlos' advice on finding relevant literature:

 * Start with google scholar searches.
 * Search by keyword, rank by number of citations.
 * Once you find relevant papers, look at related reading.
 * If you have an important paper, look for other papers which cite
   it. E.g. for algorithmic skeletons, almost every paper cites
   Murray.

Pavlos is away from 2-12th October, so must organise around that.

For the machine learning courses, it may not be required to go into a
deep mathematical understanding for most uses, although it would
obviously help when things go wrong. A important field of maths is
statistics for researchers.

[Confidence interval](http://en.wikipedia.org/wiki/Confidence_interval)
- a measure of reliability of an estimate. If confidence intervals are
  constructed across many separate data analyses of repeated
  experiments, the proportion of such intervals that contain the true
  value of the parameter will match the confidence level.

### Notes on [What is a Literature Review](https://www.wiki.ed.ac.uk/pages/viewpage.action?title=What+is+a+Literature+Review&spaceKey=irrirpwiki)

> a summary and critical evaluation of all the knowledge pertaining to
> a specific research question, which has already been established by
> other researchers.

 * Identifies gaps in the field
 * Critically comments on the methods of the previous research
 * May group the research into themes or topics.
 * Start out broad, then become more specific (inverted pyramid).

```
-----------------------------
 \                         /      Broad issues
  \                       /
   \---------------------/
    \                   /         Studies which overlap
     \                 /          with your research
      \               /
       \-------------/
        \           /             Studies that are directly
         \         /              related to your investigation
          \       /
           \     /
            \   /
             \ /
              X
```

 * First briefly explain the broad issues.
 * Then narrow the focus to deal with the studies related to the
   subject area of your question.
 * Finally, hone in on research which is directly related to the
   specific research question.

[Further reading](http://www.ais.up.ac.za/med/tnm800/tnmwritingliteraturereviewlie.htm).


## Thursday 18th

 * Added skeleton LaTeX files for project proposal, based on pip-db
   report.
 * Started working through C++ template and MPI examples and programs,
   staring simple while gaining familiarity with syntax/paradigms
   (what happened to all the pointers??).

### Differences between C++ Templating engine and C pre-processor

 * There are superficial similarities between the goals of C++
   templates and some C libraries (E.g. GLib).
 * The major difference between the two is that templates are a
   *compile time* feature, whereas the pre-processor is purely textual
   substitution.
 * This means that Templates are type aware, and (to a greater or
   lesser extent) capable of performing more advanced optimisation and
   verification than would be possible purely with the
   pre-processor. E.g. "If a function is passed in as a template
   parameter, you can't tell how complex it is or transform it in any
   white box way."

Today's relevant conference was HiPEAC (see google doc).


## Friday 19th

## Notes on [Statistics Tutorial](https://explorable.com/statistics-tutorial)

```
statistical_methods(raw_data) = data_set
```

Calculations of central tendency (mean - arithmetic or geometric,
media, or mode) assume normal distribution of data.

Statistics for research involves understanding:

1. the relationship between probability and statistics.
1. the two major branches of statistics: descriptive statistics and
   inferential statistics.
1. how statistics relates to the scientific method.

[Reentrant](http://en.wikipedia.org/wiki/Reentrancy_(computing)) - A
subroutine is reentrant if it can be interrupted during its execution
and then safely called again before its previous invocations complete
execution. E.g.

[Simulated annealing](http://en.wikipedia.org/wiki/Simulated_annealing) -
A generic probabilistic metaheuristic for the global optimization
problem of locating a good approximation to the global optimum of a
given function in a large search space.


### Notes on Divide and Conquer skeleton

Some early thoughts on how I think a generic divide and conquer
template would behave:

```
User defined functions:

   bool  Dac::isIndivisble(T)  - Determine whether to split "T" or solve
   T ... DaC::split(T)         - Split a "T" into 2 or more "T" subcomponents
   T     DaC::process(T)       - The "solution" function (e.g. in merge sort,
                                 return the single element sorted list)
   T     DaC::merge(T, T, ...) - Combine 2 or more "T"s into a single T
```

The constructor for the skeleton would accept these functions as
arguments, as well as a dataset. There would need to be a getter
function for returning the processed (divided and conquered) data:

```
   DaC(isIndivisible, split, process, merge, data)
   Dac::get()
```

The getter could behave in one of two ways, depending on when we start
to process the data:

 * If the skeleton were to begin processing the data *immediately*
   upon construction, then the getter would simply return the
   processed data (if available), else block.
 * If the skeleton deferred processing of the data until the data is
   requested, then invoking the getter method would begin this
   process, and block until completion.

The former method seems much more intuitive, although it would be
simple enough to add a flag to the constructor which enabled the lazy
evaluation of data if the user desired.


### Notes from Ppar Lecture

3:10pm Hugh - How to give a conference presentation

 * Conference presentation comes *after* paper is published, but
   shouldn't be derivative of.
 * Presentation is advert for paper.
 * complicated != smart.
 * Memorise speech for first few slides and stick to a slow tempo.
 * Lead with important message, but intro sets context and should get
   audience thinking about problem.
 * Ask Murray about joining CArD mailing list.


I've implemented a simple Divide and Conquer template skeleton in C++,
will make a couple of specialisations for it on Monday (I've only got
MergeSort atm) and have a play around with it.


## Monday 22nd

I spent a little bit of time at the weekend (too eager) adapting the
divide and conquer template to a fixed depth (FDDC) design.

I need to critically consider memory allocation before going any
further, since the hacked-together implementation from Friday no-doubt
leaks memory like a sieve. Should the muscle functions be responsible
for allocating data? Possibilities include:

 * Muscle functions are responsible for allocating new memory when
   processing data, and releasing old data.
 * The skeleton is responsible for allocating *all* memory, and the
   muscle functions accept out parameters and write to them (not
   possible for `split()` unless the skeleton knows the size of the
   split data ahead of time).
 * Muscle functions operating *in-place* where possible (not possible
   for something like `split()`.

### First attempt at designing a DC skeleton

Here's the basic skeleton template (just the muscle function
declarations):

```
DC<T> {
    bool isIndivisible(T)
    T[]  split(T)
    T    solve(T)
    T    merge(T[])
}
```

And here's pseudo-code for the `divide_and_conquer()` function, which
does the actual work:

```
divide_and_conquer(T) {
    if isIndivisible(T):
        return solve(T)
    else:
        return merge(map(divide_and_conquer, split(T)))
}
```

Note that we can use the C++
[transform](http://www.cplusplus.com/reference/algorithm/transform/)
function to perform the `map(divide_and_conquer, split(T))`
transformation.


### Questions and thoughts on DaC

 * Could a `split()` operation ever produce *more* data than goes into
   it? For a merge sort, it simply splits a list in two.
 * Could a divide and conquer algorithm ever transform the *type* of
   data? For merge sort, it's `merge_sort(list<A>) : list<A>`. Could
   there be a DaC algorithm where the `solve()` function had a
   different data type for its input and output?
 * Is there a divide and conquer algorithm which *can't* be expressed
   using a fixed depth? E.g. for merge sort, `split()` always returns
   2 lists. Could there be a `split()` which returns *n* lists, where
   the value of *n* is dependent on the input data?
 * Similarly, is there any divide and conquer algorithm for which the
   `split()` function *doesn't* perform an even split? E.g. in merge
   sort, a split operation on a list of 100 items will return two
   lists, each containing 50 items. Are there split operations that
   would return lists of different length? (excluding of course when
   the input size is odd).
 * Is there a divide and conquer algorithm for which the input *T*
   *can't* be expressed using vectors? E.g. for merge sort, we use
   list<A>. Does any DaC algorithm *not* perform on
   lists/arrays/vectors?
 * Would it be useful to augment the `divide_and_conquer()` function
   with a parameter which indicates it's *depth*? E.g. adding a
   integer `depth` parameter:

```
divide_and_conquer(T, depth) {
    if isIndivisible(T):
        return solve(T)
    else:
        return merge(map(divide_and_conquer, split(T), depth++))
}
```

```
struct data {
    void  *data;
    size_t element_size;
    int    length;
};

FDDC<T> {
    bool isIndivisible(T*)
    T**  split(T*)
    T*   solve(T*)
    T*   merge(T**)
}
```

To see examine the size of individual functions in a C++ executable:

```
$ nm -CSr --size-sort a.out
0000000000400c35 00000000000001c8 T merge(vector<int>*, vector<int>*, vector<int>*)
0000000000400dfd 0000000000000130 T divide_and_conquer(vector<int>*, vector<int>*, int)
0000000000400a9f 0000000000000115 T split(vector<int>*, vector<int>*, vector<int>*)
00000000006016c0 0000000000000110 B std::cout@@GLIBCXX_3.4
0000000000400f2d 00000000000000ef T main
...
```

Testing the new dac implementation, sorting 1,000,000 integers:

```
$ valgrind --max-stackframe=4000040 ./dac
...
==544== HEAP SUMMARY:
==544==     in use at exit: 16 bytes in 1 blocks
==544==   total heap usage: 4,999,998 allocs, 4,999,997 frees, 235,611,352 bytes allocated
==544==
==544== LEAK SUMMARY:
==544==    definitely lost: 16 bytes in 1 blocks
==544==    indirectly lost: 0 bytes in 0 blocks
==544==      possibly lost: 0 bytes in 0 blocks
==544==    still reachable: 0 bytes in 0 blocks
==544==         suppressed: 0 bytes in 0 blocks
```

And the DC template implementation:

```
$ valgrind --max-stackframe=4000040 ./dc-merge-sort
...
==599== HEAP SUMMARY:
==599==     in use at exit: 0 bytes in 0 blocks
==599==   total heap usage: 2,131,044 allocs, 2,131,044 frees, 83,934,244 bytes allocated
==599==
==599== All heap blocks were freed -- no leaks are possible
```

Time comparison:

```
./dac
Time to sort  100000 integers:   34 ms
./dc-merge-sort
Time to sort  100000 integers:  606 ms
./fddc-merge-sort
Time to sort  100000 integers:  636 ms
```

DC merge sort after optimising the vector usage in `merge()`:

```
$ valgrind --max-stackframe=4000040 ./dc-merge-sort
...
==1690== HEAP SUMMARY:
==1690==     in use at exit: 0 bytes in 0 blocks
==1690==   total heap usage: 1,899,990 allocs, 1,899,990 frees, 73,281,312 bytes allocated
==1690==
==1690== All heap blocks were freed -- no leaks are possible
```

Comparison of timings as of commit 03f2b01:

```
./dac
Time to sort  200000 integers:   58 ms
Time to sort  400000 integers:   97 ms
Time to sort  600000 integers:  145 ms
Time to sort  800000 integers:  197 ms
Time to sort 1000000 integers:  268 ms
Time to sort 1200000 integers:  297 ms
Time to sort 1400000 integers:  363 ms
Time to sort 1600000 integers:  401 ms
Time to sort 1800000 integers:  505 ms
Time to sort 2000000 integers:  515 ms
./dc-merge-sort
Time to sort  200000 integers:  144 ms
Time to sort  400000 integers:  284 ms
Time to sort  600000 integers:  421 ms
Time to sort  800000 integers:  555 ms
Time to sort 1000000 integers:  706 ms
Time to sort 1200000 integers:  855 ms
Time to sort 1400000 integers:  989 ms
Time to sort 1600000 integers: 1197 ms
Time to sort 1800000 integers: 1308 ms
Time to sort 2000000 integers: 1443 ms
./fddc-merge-sort
Time to sort   25000 integers:   12 ms
Time to sort   50000 integers:   31 ms
Time to sort   75000 integers:   42 ms
Time to sort  100000 integers:   57 ms
Time to sort  125000 integers:   70 ms
Time to sort  150000 integers:   84 ms
Time to sort  175000 integers:   90 ms
Time to sort  200000 integers:  105 ms
./lib-sort
std::sort:
Time to sort  200000 integers:    8 ms
Time to sort  400000 integers:   19 ms
Time to sort  600000 integers:   30 ms
Time to sort  800000 integers:   45 ms
Time to sort 1000000 integers:   54 ms
Time to sort 1200000 integers:   61 ms
Time to sort 1400000 integers:   74 ms
Time to sort 1600000 integers:   86 ms
Time to sort 1800000 integers:   97 ms
Time to sort 2000000 integers:  107 ms
std::sort_stable:
Time to sort  200000 integers:   11 ms
Time to sort  400000 integers:   22 ms
Time to sort  600000 integers:   47 ms
Time to sort  800000 integers:   57 ms
Time to sort 1000000 integers:   63 ms
Time to sort 1200000 integers:   75 ms
Time to sort 1400000 integers:   89 ms
Time to sort 1600000 integers:  103 ms
Time to sort 1800000 integers:  117 ms
Time to sort 2000000 integers:  136 ms
```


## Tuesday 23rd

Comparing profiles of `dac` with `dc-merge-sort`:

```
$ gprof dac gmon.out | less
...
 %   cumulative   self              self     total
 time   seconds   seconds    calls  ms/call  ms/call  name
 79.23      0.98     0.98 10999990     0.00     0.00  merge(vector<int> const&, vector<int> const&, vector<int>*)
  8.89      1.09     0.11                             test_dac_sort(unsigned long)
  6.47      1.17     0.08       10     8.02   112.78  divide_and_conquer(vector<int>*, vector<int>*, int)
  4.45      1.23     0.06 10999990     0.00     0.00  split(vector<int>*, vector<int>*, vector<int>*)
  0.81      1.24     0.01 11000000     0.00     0.00  solve(vector<int>*, vector<int>*)
  0.40      1.24     0.01                             isIndivisible(vector<int>*)
...
$ gprof dc-merge-sort gmon.out | less
...
  %   cumulative   self              self     total
 time   seconds   seconds    calls  ms/call  ms/call  name
 38.59      1.33     1.33       10   133.14   325.35  DC<std::vector<int, std::allocator<int> > >::_dac(std::vector<int, std::allocator<int> >)
 34.82      2.53     1.20 10999990     0.00     0.00  DC<std::vector<int, std::allocator<int> > >::merge(std::vector<std::vector<int, std::allocator<int> >, std::allocator<std::vector<int, std::allocator<int> > >
 20.89      3.25     0.72 10999990     0.00     0.00  DC<std::vector<int, std::allocator<int> > >::split(std::vector<int, std::allocator<int> >)
```

Wheres the `dac` implementation spends most of it's time in the
(understandably) expensive `merge()` operation, the DC skeleton
implementation has a much more even distribution of execution times
across the merge, split, and recursive functions. Possible reasons for
this:

 * The DC skeleton uses `sdt::vector<int>` objects instead of a custom
   vector class, and copying operations are more expensive.
 * Values are passed and returned by value, rather than out
   parameters. This could be cause for lot of additional memory
   copying.

Interestingly, a refactoring the `dac` implementation into a
skeleton-style template class provided a modest speedup:

```
dac
Time to sort  200000 integers:   62 ms
Time to sort  400000 integers:  111 ms
Time to sort  600000 integers:  163 ms
Time to sort  800000 integers:  220 ms
Time to sort 1000000 integers:  278 ms
Time to sort 1200000 integers:  332 ms
Time to sort 1400000 integers:  391 ms
Time to sort 1600000 integers:  441 ms
Time to sort 1800000 integers:  499 ms
Time to sort 2000000 integers:  569 ms
DC<int>
Time to sort  200000 integers:   58 ms
Time to sort  400000 integers:   96 ms
Time to sort  600000 integers:  150 ms
Time to sort  800000 integers:  217 ms
Time to sort 1000000 integers:  259 ms
Time to sort 1200000 integers:  306 ms
Time to sort 1400000 integers:  352 ms
Time to sort 1600000 integers:  411 ms
Time to sort 1800000 integers:  532 ms
Time to sort 2000000 integers:  514 ms
```

To recurse using a separate thread (C++ thread lib):

```
std::thread left(&DC<T>::divide_and_conquer, this, in_left, out_left, next_depth);
```

Number of nodes (*N*) in tree of depth *d*:

```
N = 1 + 1^2 + 2^2 ... d^2
```


## Wednesday 24th

Advice from Pavlos on finding relevant conferences:

 * There are websites which collate conferences and let you perform
   key word searches on them.
 * See where relevant researchers are publishing their work (e.g. look
   up Hugh's publication list).
 * Getting a reasonable list of relevant conferences shouldn't take
   more than 3-4 hours.

Weekly meetings with Pavlos organised for Friday 12pm.

### Notes on dac parallelisation

The number of threads spawned for each parallelisation depth:

```
parallelisation_depth  thread_count  (thread_count(d)-thread_count(d-1))
0                      1
1                      3             2
2                      7             4
3                      15            8
4                      31            15  <---  (!? not a power of 2)
5                      63            32
6                      127           64
7                      255           128
8                      511           256
```

So the series is:

```
n = 1 + 1^2 + 2^2 + 3^2 + ... + d^2
```

Which means that for a 4 core machine, we'd expect the optimum
parallelisation depth to be 2.


## Thursday 25th

### Notes for meeting with Pavlos

 * Skeleton implementation:
   * Fixed degree divide and conquer skeleton using C++ templates.
   * Muscle functions operate in-place (using out functions).
      * Some memory allocated by "worker" function (lots of heap
        thrashing).
      * split() and merge() function actually do majority of the heavy
        lifting, since we don't assume to know the size of the split /
        merged data. (could we predict this?)
   * Naive multi-threading using depth limit.
      * Lock free since we allocate memory in the parent thread.
   * Example merge sort implementation.
      * Testing on example data set.
      * Results of tests:
         * Non-linear performance (why?).
 * Next week: LLVM, and picking papers for Ppar presentation
 * Is there a budget for text books?


## Friday 26th

Cooley and Tukey's 1964 paper "An Algorithm for the Machine
Calculation of Complex Fourier Series" introduced the Fast Fourier
Transform. Improved discrete Fourier transform performance from O(n^2)
to O(n log n).

### Notes from meeting with Pavlos

 * DaC skeleton:
   * The skeleton provides an *interface* for the programmer to write
     solutions, so it should allow enough flexibility for the user to
     "do what they want" within the muscle functions, which could
     include:
     * Transforming data types within `solve()`.
     * Performing uneven `split()` operations, or even returning more
       data than went in.
     * Producing variable sized results for `split()` and `merge()`.
   * "Optimising" a skeleton doesn't mean improving memory allocation,
     so much as dividing the job into work for different devices,
     servers, cores, etc (perhaps using MPI).
   * I should look at
     [eSkel](http://homepages.inf.ed.ac.uk/mic/eSkel/) for a possible
     "real" skeleton implementation to examine.
   * We can assume that all DaC algorithms operate on vectors.
   * For this simple implementation exercise, the important thing
     isn't to produce something *better* than the existing solutions,
     but to understand the faults and be able to explain what
     does/doesn't work from my own attempt, and *why*.
   * gprof provides a good starting point for profiling. The three
     factors that will affect performance are:
     1. The size of the input problem.
     1. Allocating memory.
     1. The cost of creating new threads.
   * I should investigate the non-linear performance of my merge sort,
     looking particularly at the points in which the gradient of the
     execution time changes. This is a lot of heap thrashing, so we
     can expect shifts in performance to occur around points at which
     the system spends more time allocating memory (e.g. it may the
     same amount of time to allocate 7Kb as 4Kb, but twice as long to
     allocate 8Kb).
 * There is budget for "small research related purchases". Pavlos is
   going to ask Murray if that covers textbooks. I should make a list
   of the books I need.
 * Writing an LLVM phase should not take too long, and isn't beyond
   the scope of people without compiler knowledge.
 * Pavlos will be out of contact from next Thursday, so I should chase
   up any questions before then.


### Notes from meeting with Hugh and Pavlos

 * In C++, virtual functions are stored in v tables, so when a class
   is instantiated and a virtual function invoked, the program must
   first dereference the vtable, look up the function pointer, then
   dereference this in order to execute the function. C++ templates
   support passing in functions as parameters, so it may be possible
   to optimise out these layers of redirection and get rid of the need
   for virtual muscle functions.
 * When designing a skeleton, some things to consider:
   1. How much can the compiler optimise.
   1. Ease of use for the user/caller.
   1. What do other skeleton implementations do? How does yours
      compare against those?
 * I should consider how to representation the skeleton abstraction to
   the user. For ease of use, it would be better if the DaC skeleton
   was a function call, so that the merge sort would have the same
   usage as the stdlib sort functions.
 * Merge sort should be in-place.

TODO:

 * Modify the backend of the skeleton so that it performs in-place.
 * Redesign the front-end of the skeleton so that it is a simple
   function call.

Memory usage during test program for `std::stable_sort`:

```
==6177== HEAP SUMMARY:
==6177==     in use at exit: 7,804 bytes in 1 blocks
==6177==   total heap usage: 121 allocs, 120 frees, 328,008,444 bytes allocated
```

And the same test using my merge sort:

```
==6110== HEAP SUMMARY:
==6110==     in use at exit: 6,740 bytes in 1 blocks
==6110==   total heap usage: 168,500,601 allocs, 168,500,600 frees, 8,659,562,260 bytes allocated
```

So the bottleneck is pretty clear. Make the skeleton perform in-place.


## Sunday 28th

I've revisited "Introduction to Algorithms" sections 2.3.1 (merge
sort):

 * Terminology: divide, conquer, combine (instead of split, solve,
   merge).
 * Merge sort using sentinels and array indices `left`, `mid`,
   `right`:

```
Merge(A, left, mid, right):
    n1 = mid - left + 1   // Length of left subarray
    n2 = right - mid      // Length of right subarray

    // Copy left subarray and append sentinel
    L[] = new array(n1+1)
    for (i = 0; i < n1; i++):
        L[i] = A[left + i]
    L[n1] = sentinel()

    // Copy right subarray and append sentinel
    R[] = new array(n2+1)
    for (i = 0; i < n2; i++):
        R[i] = A[right + i]
    R[n2] = sentinel()

    l = 0, r = 0
    for (i = mid; i < right; i++):
        if L[l] <= R[r]:
            A[i] = L[l++]
        else:
            A[i] = R[r++]
```

 * Terminology: *recursion case* - when the problem can be divided and
   conquered recursively, *base case* - when the problem can be solved
   directly.
 * It is often profitable to "bottom out" before the problem size
   reaches 1, resorting to insertion sort for small enough
   vectors. This could be an optimisation worth exploring in the merge
   sort skeleton.


## Monday 29th

### Merge Sort Skeleton execution times

#### 14.09.26-10.00.00 and 14.09.26-15.47.20

 * Merge sort with lots of heap thrashing, no insertion sort
   optimisation.

![14.09.26-10.00.00](assets/14.09.26-10.00.00.png)
![14.09.26-15.47.20](assets/14.09.26-15.47.20.png)

#### 14.09.29-11.53.00

 * In-place merge sort, no insertion sort optimisation.

![14.09.29-11.53.00](assets/14.09.29-11.53.00.png)

#### 14.09.29-13.07.59

 * In-place merge sort, with insertion sort being used to sort lists
   of 100 elements or less.

![14.09.29-13.07.59](assets/14.09.29-13.07.59.png)

### 14.09.29-16.24.00

 * In-place merge sort using a *function template* skeleton, instead
   of an object orientated class based skeleton.

![14.09.29-16.24.00](assets/14.09.29-16.24.00.png)


## Tuesday 30th

Task classification of skeletons:

 * Data-parallel skeletons
 * Task-parallel skeletons
 * Resolution skeletons

Functional classification of skeletons:

 * Coordination
 * Functional
 * Object-oriented
 * Imperative

TODO:

 * Optimise `merge()` to use only n/2 memory allocations (only copy
   left list to temporary memory).
 * Look up ITA's parallel merge algorithm.


# October 2014

## Wednesday 1st

Adam sent me a link to a
[parallel in-place merge sort](http://www.drdobbs.com/parallel/parallel-in-place-merge-sort/240169094)
implementation in C++ which could be useful for comparing against my
merge sort skeleton.

Papers *about* Skeletons:

 * Algorithmic Skeletons: Structured Management of Parallel Computation (Cole)
 * Bringing skeletons out of the closet (Cole)
 * A survey of algorithmic frameworks (Gonz)

Papers *using* Skeletons:

 * Auto-Tuning Parallel Skeletons (Collins)
 * Parallel Programming Using Skeleton Functions (Darlington)

Papers about Skeleton *implementations*:

 * Skil: an imperative language with algorithmic skeletons (Botorog)
 * Skandium: Multi-core Programming with Algorithmic Skeletons (Leyton)
 * Prototyping a Library of Algorithmic Skeletons (Javed)
 * HDC: a higher-order language for divide-and-conquer (Herrmann)
 * Flexible Skeletal Programming with eSkel (Benoit)
 * Shared Memory Multiprocessor Support for Functional Array Processing in SAC (Grelck)


### Notes from meeting with Pavlos

PPar presentation:
 * A good portion of the 15 minutes will be spent providing background
   to skeletons.
 * Start by explaining the horrors of parallel programming on modern
   hardware (heterogeneous, the requirement for both application and
   systems programming skills, etc). Present some existing solutions
   for this problem, and then introduce skeletons.
 * A good choice of papers would be one which defines the theory of
   skeletons, and one which describes a concrete
   implementation/framework.

Example usage of the STL sort function:

```
#define NUM_ELEMS 100000
int *a = get_big_array(NUM_ELEMS);

std::stable_sort(a, a + NUM_ELEMS);
```

Example usage of the Class based skeleton:

```
#define NUM_ELEMS 100000
int *a = get_big_array(NUM_ELEMS);

MergeSort<int> sort();

sort.set_data(a, a + NUM_ELEMS);
sort.set_parallelisation_depth(2);
sort.set_split_threshold(100);

sort.run();
```

Example usage of the skeleton function:

```
#define NUM_ELEMS 100000
int *a = get_big_array(NUM_ELEMS);

merge_sort(a, a + NUM_ELEMS);
```


## Thursday 2nd

The number of nodes *N* in a balanced binary tree of depth *d* can be
calculated as: `N = 2^d - 1`. So, to re-arrange with respect to *d*:

```
N = 2^d - 1
2^d = N + 1
log2(2^d) = log2(N + 1)
dlog2(2) = log2(N + 1)
d = log2(N + 1)
```


## Friday 3rd

Now that I have a working divide and conquer skeleton implemented, I'm
going to build Skandium and compare usage + execution times against
their Java skeleton framework.

 * Checked out `https://github.com/mleyton/Skandium.git` repo.
 * Imported sources into Eclipse (I couldn't figure out the build
   system that they were using, lots of strange errors about
   conflicting versions of dependencies).

### In-place Divide and Conquer

Divide and conquer algorithms tend to perform one of two tasks: either
they *solve* a problem in order to return a solution value, or they
*transform* the input data in some manner. For example, the max
subarray problem is a *solution* orientated DaC algorithm, in which
the input is an array and the output is an integer value for the
maximum value obtained in a subarray. The merge sort algorithm is an
example of a *transformation* DaC, in which the return value is a
sorted permutation of the input data. While it is possible to describe
transformation algorithms using the same terminology as solution
algorithms (in which the return data is a transformed permutation of
the input data), it is possible to optimise the transformation
algorithms by instead removing the return values and considering them
in-place.

### Notes on Max Subarray DaC

Psuedocode:

```
def max_subarray(A) {

    # is_indivisible:
    if len(A) == 1:
        # conquer:
        return A

    # divide:
    left = A[0:len(A/2)]
    right = A[len(A/2)+1:len(A)]

    # recurse:
    l = self(left)
    m = max_crossing_subarray(left, right)
    r = self(right)

    # combine:
    return max(l, m, r)
}

def max_crossing_subarray(L, R) {
    sum = 0, l = 0
    for i in reverse(L):
        sum += i
        l = max(l, sum)

    sum = 0, r = 0
    for i in R:
        sum += i
        r = max(r, sum)

    return l + r
}
```

And our divide and conquer skeleton:

```
def divide_and_conquer(T):
    if is_indivisible(T):
        return solve(T)
    else:
        return combine(map(divide_and_conquer, split(T)))
```

So mapping this to skeleton functions:

```
class Problem:
    this.left = T[]
    this.right = T[]

def is_indivisible(T):
    return len(T.right) or len(T.left) == 1

def solve(T):
    if len(T.right):
        sum = 0, l = 0
        for i in reverse(T.left):
            sum += i
            l = max(l, sum)

        sum = 0, r = 0
        for i in T.right:
            sum += i
            r = max(r, sum)

        return l + r
    else:
        return T.left[0]

def split(T):
    A = [3]
    mid = len(T.left) / 2
    A[0] = T.left[0:mid]
    A[1] = T.left[mid:]
    A[2] = {A[0], A[1]}
    return A

def combine(T[]):
    return max(T)
```


## Tuesday 7th

There are two possible areas for research in algorithmic skeletons
which appear to lacking from the existing literature:

 * **Completely *transparently parallelising* skeletons**. The amount
   of explicit parallelisation support required from the user of
   Skeletons varies from framework to framework. For example, Skandium
   offers fairly a abstracted view of computation, but still requires
   the user to set the number of threads (i.e. declare the available
   resources), and to wrap shared memory access in muscle functions
   with `synchronized` constructs. By contrast, eSkel requires
   absolute explicit control of parallelisation by building on top of
   MPI, and so would have a relatively high cost-of-entry for anyone
   who is looking to parallelise a sequential program (at a minimum,
   they would need to refactor their code to use MPI/eSkel data
   structures, and add in coordination logic for the master/slave
   processes). This leaves room for the development of a skeleton
   framework which performs the parallelisation entirely
   transparently. This would allow users to simply replace existing
   sequential constructs with their skeleton equivalent *without*
   having to concern themselves with any of the additional
   complexities such as coordinating available resources, or even
   being aware that the skeleton operates in parallel. For example, my
   simple `merge_sort()` skeleton could serve as a direct replacement
   for any call to the STL stable sort function, and would "magically"
   offer the user a ~x2.8 speedup. This ideal library of "drop-in"
   replacements would satisfy two of Cole's pragmatic skeleton
   manifesto:

     * Propagate the concept with minimal conceptual disruption.
     * Show the pay-back.

 * **Intelligent skeletons which determine *when* to
   parallelise**. Skeletons provide the necessary coordination logic
   to take a selection of muscle functions and to parallelise
   them. However, there appears to be no skeleton implementations
   which attempt to determine *when* to parallelise a muscle function
   by analysing the muscle function itself. For example, in the case
   of a merge sort skeleton, you would expect full parallelisation
   when dealing with large datasets. However, if only sorting 10
   elements, then the cost of coordinating separate threads and shared
   memory access may be greater than simply executing
   sequentially. Similarly, a divide and conquer skeleton may apply
   heuristics to the individual divide, conquer, and combine muscles
   to determine which are the most computationally expensive, and so
   which would benefit most from parallelisation. In the case of merge
   sort, the `conquer()` muscle simply returns a single element array,
   but for a max-subarray skeleton, the `conquer()` muscle can perform
   many more operations, totalling up the maximum value of two
   subarrays. In this case, an "intelligent" skeleton may decide
   (either at compile or execution time) to execute merge sort conquer
   muscle sequentially, but to parallelise the equivalent muscle for
   max-subarray.


## Thursday 9th

### Building LLVM:

1. Clone https://github.com/llvm-mirror/llvm.git.
1. Clone https://github.com/llvm-mirror/clang.git into `llvm/tools`.
1. Clone https://github.com/llvm-mirror/clang-tools-extra.git into
   `llvm/tools/clang/tools`.
1. Clone https://github.com/llvm-mirror/compiler-rt.git into
   `llvm/projects`.
1. Clone https://github.com/llvm-mirror/test-suite.git into `llvm/projects`.
1. Out of tree build: `mkdir build && cd build && ../configure &&
   make`

### Execute LLVM bitcode natively

Load kernel module:

```
mount binfmt_misc -t binfmt_misc /proc/sys/fs/binfmt_misc
```

Persistent mount with `/etc/fstab`:

```
# Execute arbitrary file formats as executables
none  /proc/sys/fs/binfmt_misc binfmt_misc defaults 0 0
```

See the Arch Wiki
[binfmt_misc](https://wiki.archlinux.org/index.php/Binfmt_misc_for_Java#Registering_the_file_type_with_binfmt_misc)
page for instructions on Java Wrappers. For native execution of LLVM
bitcode (NOTE: `/bin/lli` is system specific):

```
% echo ':llvm:M::BC::/bin/lli:' > /proc/sys/fs/binfmt_misc/register
```

To emit bitcode:

```
$ clang -emit-llvm hello.c -c -o hello.bc
$ chmod +x hello.bc
$ ./hello.bc
```

To emit bytecode:

```
$ clang -emit-llvm hello.c -S -o hello.ll
```

To compile bitcode to native assembly:

```
$ llc hello.bc -o hello.s
```

To disassemble bitcode to LLVM IR:

```
$ llvm-dis < hello.bc
```

A totally manual compilation process:

```
# Compile source code to bytecode:
$ clang -emit-llvm hello.c -S -o hello.ll
# Compile bytecode to bitcode:
$ clang -emit-llvm hello.ll -c -o hello.bc
# Compile bitcode to native assembly:
$ llc hello.bc -o hello.s
# Compile native assembly to DSO:
$ clang hello.s -c -o hello.o
# Link DSO to executable:
$ clang hello.o -o hello
```

And for multiple files:

```
# C source code to bytecode:
$ clang -emit-llvm hello.c -S -o hello.ll
$ clang -emit-llvm foo.c -S -o foo.ll
# Link:
$ clang foo.ll hello.ll -o hello
```


## Saturday 11th

The
[Microsoft Academic Search tool](http://academic.research.microsoft.com/Keyword/1231/algorithmic-skeletons?query=algorithmic%20skeletons)
has some really nice meta properties about searches, such as lists of
relevant conferences and journals, visualisations of citations, etc.


## Sunday 12th

Example Hello World LLVM pass invocation:

```
$ Debug+Asserts/bin/opt -load Debug+Asserts/lib/LLVMHello.so -stats -hello < ~/hello.bc > hello.bc
```


## Monday 13th

Pavlos' tips for critically reviewing papers:

1. The most common criticism for technical papers is errors in the
   experimental method. Was their evaluation sound? Did they compare
   their results with existing solutions?
1. For theoretical papers, the criticism usually centres around
   finding errors in the author's fundamental reasoning or argument.


## Tuesday 14th

The library has a copy of Structured Parallel Programming: Patterns
for Efficient Computation (2008) in
[PDF format](http://www.sciencedirect.com.ezproxy.is.ed.ac.uk/science/book/9780124159938). This
should be high up on my reading list.

TODO: Install `intel-tbb` on `cec`.


## Wednesday 15th

Useful reading in clang's C++ template handling:

 * [The Dreaded Two-Phase Name Lookup](http://blog.llvm.org/2009/12/dreaded-two-phase-name-lookup.html)

An experiment to see if C++ template information is carried into LLVM
bytecode. Take the following C++ program:

```
int add(int a, int b) {
  return a + b;
}

double fadd(double a, double b) {
  return a + b;
}

int main() {
  double x = add(0.0, 0.0);                // Double
  return     add(static_cast<int>(x), 0);  // Integer
}
```

When compiled with clang++ -O0, this generates the following bytecode:

```
; ModuleID = 'test.cc'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define i32 @_Z3addii(i32 %a, i32 %b) #0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  call void @mcount() #1
  store i32 %a, i32* %1, align 4
  store i32 %b, i32* %2, align 4
  %3 = load i32* %1, align 4
  %4 = load i32* %2, align 4
  %5 = add nsw i32 %3, %4
  ret i32 %5
}

declare void @mcount()

; Function Attrs: nounwind uwtable
define double @_Z4fadddd(double %a, double %b) #0 {
  %1 = alloca double, align 8
  %2 = alloca double, align 8
  call void @mcount() #1
  store double %a, double* %1, align 8
  store double %b, double* %2, align 8
  %3 = load double* %1, align 8
  %4 = load double* %2, align 8
  %5 = fadd double %3, %4
  ret double %5
}

; Function Attrs: nounwind uwtable
define i32 @main() #0 {
  %1 = alloca i32, align 4
  %x = alloca double, align 8
  call void @mcount() #1
  store i32 0, i32* %1
  %2 = call i32 @_Z3addii(i32 0, i32 0)
  %3 = sitofp i32 %2 to double
  store double %3, double* %x, align 8
  %4 = load double* %x, align 8
  %5 = fptosi double %4 to i32
  %6 = call i32 @_Z3addii(i32 %5, i32 0)
  ret i32 %6
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }

!llvm.ident = !{!0}

!0 = metadata !{metadata !"clang version 3.5.0 (tags/RELEASE_350/final)"}
```

We then refactor the two distinct `add` functions into a single
template function:

```
template<typename T>
T add(T a, T b) {
  return a + b;
}

int main() {
  float x = add(0.0, 0.0);                // Float
  return    add(static_cast<int>(x), 0);  // Integer
}
```

And compile this in the same way, generating the bytecode:

```
; ModuleID = 'test.cc'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: uwtable
define i32 @main() #0 {
  %1 = alloca i32, align 4
  %x = alloca double, align 8
  call void @mcount() #2
  store i32 0, i32* %1
  %2 = call double @_Z3addIdET_S0_S0_(double 0.000000e+00, double 0.000000e+00)
  store double %2, double* %x, align 8
  %3 = load double* %x, align 8
  %4 = fptosi double %3 to i32
  %5 = call i32 @_Z3addIiET_S0_S0_(i32 %4, i32 0)
  ret i32 %5
}

declare void @mcount()

; Function Attrs: nounwind uwtable
define linkonce_odr double @_Z3addIdET_S0_S0_(double %a, double %b) #1 {
  %1 = alloca double, align 8
  %2 = alloca double, align 8
  call void @mcount() #2
  store double %a, double* %1, align 8
  store double %b, double* %2, align 8
  %3 = load double* %1, align 8
  %4 = load double* %2, align 8
  %5 = fadd double %3, %4
  ret double %5
}

; Function Attrs: nounwind uwtable
define linkonce_odr i32 @_Z3addIiET_S0_S0_(i32 %a, i32 %b) #1 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  call void @mcount() #2
  store i32 %a, i32* %1, align 4
  store i32 %b, i32* %2, align 4
  %3 = load i32* %1, align 4
  %4 = load i32* %2, align 4
  %5 = add nsw i32 %3, %4
  ret i32 %5
}

attributes #0 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }

!llvm.ident = !{!0}

!0 = metadata !{metadata !"clang version 3.5.0 (tags/RELEASE_350/final)"}
```

A comparison of the two sets of bytecodes:

 * In the template version, the functions appear in the order they are
instantiated; in the explicit version, the functions appear in the
order they are declared.

 * The functions instantiated from templates have a `linkonce_odr`
keyword annotation, explained
[here](http://llvm.org/docs/LangRef.html#linkage).


### Notes from meeting with Hugh and Pavlos

 * The first draft of a presentation *will* fail.
 * Typically, I'll need to supply a first draft to supervisors 3-5
   weeks before giving the presentation, in order to have adequate
   time to make modifications.

PPar slides:
 * There's no clear message. The presentation tries to cover too many
   topics and so doesn't cover any adequately.
 * Process for writing a presentation:
   * Decide on topics
   * Decide on the message
   * Sketch ideas on paper
   * Put sketches into computer
 * There's too much text. Don't use text where a diagram would
   suffice.
 * The psuedocode for DaC *doesn't* actually show how writing a
   parallelised skeleton would be hard.
 * The claim that parallel programming is hard is unsubstantiated.
 * Examples of skeletons in the "wild" include MapReduce and Intel
   TBB. Neither identify themselves as skeletons.
 * The presentation doesn't emphasise the *necessity* of skeletons.

Sloccount for intel-tbb:

```
Totals grouped by language (dominant language first):
cpp:         104704 (90.81%)
ansic:         9421 (8.17%)
asm:            441 (0.38%)
sh:             342 (0.30%)
java:           226 (0.20%)
objc:            98 (0.08%)
pascal:          68 (0.06%)

Total Physical Source Lines of Code (SLOC)                = 115,300
```

Sloccount for hadoop-common:

```
Totals grouped by language (dominant language first):
java:        827667 (96.19%)
ansic:        26658 (3.10%)
sh:            3733 (0.43%)
cpp:           1903 (0.22%)
python:         276 (0.03%)
perl:           211 (0.02%)

Total Physical Source Lines of Code (SLOC)                = 860,448
```

Sloccount for OpenMPI:

```
ansic:       664595 (87.32%)
cpp:          44461 (5.84%)
f90:          17980 (2.36%)
sh:           15848 (2.08%)
java:          7071 (0.93%)
perl:          5298 (0.70%)
python:        3202 (0.42%)
asm:           1291 (0.17%)
lex:            860 (0.11%)
fortran:        237 (0.03%)
csh:            172 (0.02%)
tcl:             39 (0.01%)
sed:             24 (0.00%)

Total Physical Source Lines of Code (SLOC)                = 761,078
```

The
[Parallel programming languages and systems](http://www.inf.ed.ac.uk/teaching/courses/ppls/)
course contains relevant notes and lecture material.


## Thursday 16th


### PPar Presentation

Topic: The direction of skeletons research

Message: Algorithmic Skeletons *are* the future of parallel programming.

Structure:

 * Writing robust parallel software is impossibly time
   consuming. (why? Provide evidence)
 * Algorithmic skeletons solve this time sink. (how?)
 * Algorithmic skeletons are *everywhere*.
 * This is how skeletons used to be designed: [1]
 * Why didn't [1] succeed?
 * This is how skeletons are designed now: [2]
 * Why does [2] succeed?
 * What pattern are we seeing?


## Notes from meeting with Hugh and Pavlos

Revised slides:

 * The opening description of the *purpose* of skeletons is a lot
   stronger, although now it's missing a high-level description of
   what skeletons actually *are*.
 * The topic of "future of skeleton research" is the weakest bit,
   since I'm not confident on it.
 * Instead, perhaps focus more on the differences between the two
   papers. What did I like about each? What didn't I like?
 * The skeleton manifesto can be dropped.


## Sunday 19th

### Reasoning about Algorithmic Skeletons

#### Claims
1. All computer hardware is parallel.
1. In order to gain continued performance improvements, programmers
   must write software for parallel hardware.
1. Writing parallel software requires prohibitive effort.
1. Common programming problems can be phrased in terms of reusable
   patterns of activities.
1. Algorithmic Skeletons are implementations of reusable patterns.
1. Algorithmic Skeletons can be parallelised.

#### Hypothesis

1. Algorithmic Skeletons can address the problem of prohibitive effort
   for parallel programming.

In order to justify this hypothesis, we require algorithmic skeletons
to:

1. Encapsulate a number of common patterns which are in *regular use*.
1. Be demonstrably *quicker* than serial or parallel implementations
   with equivalent effort.
1. Be demonstrably *easier to use* than parallel implementations with
   equivalent effort.
1. Be demonstrably *safer* to use (i.e. more robust) than parallel
   implementations with equivalent effort.

Problems with current Skeletons research:

1. They do not compare speed against *equivalent effort* parallel
   implementations. They either compare only against other skeletons
   implementations, or tuned parallelised implementations.
1. They compare themselves against different example problems, and
   often only one or two per publication.

Problems that Skeletons must address:

1. Nesting. Skeletons must be nest-able to arbitrary depths without
   problems.
1. Scalability. If Skeletons are to be widely adopted, then they must
   prove to be able to scale to realistically sized hardware.
1. Sequential performance. Similar to the scalability problem, we must
   be able to demonstrate that Skeletons will not significantly hamper
   performance when operating on only single core architectures.
1. Coordination overhead. The overhead required to orchestrate
   parallel execution must not be significant compared to the
   performance benefits of the parallelisation.
1. Error handling. Known behaviour for error conditions,
   e.g. exception handling.
1. Pattern abstraction. The appropriate granularity of abstraction
   must be chosen. For example, do we enforce all divide and conquer
   Skeletons to return a value, or do we have separate Skeletons to
   support in-place manipulation of data?

Potential innovations for Skeletons:

1. Runtime optimisations. Using instrumentation and profiling at
   runtime to further optimise execution. This could be as simple as
   determining what are the "hot" code paths and optimising for them,
   or there could be a mechanism for assigning weights to each muscle
   function and parallelising them appropriately.
1. Compiler optimisations. By using predefined patterns, it may be
   possible to further optimise performance by exploiting compile time
   (i.e. static) optimisations.

Possible MSc project idea:

Algorithmic skeletons offer flexible, generic patterns of computation,
yet their coordination logic is *inflexible*. This means that it could
be possible to invoke a skeleton in such a way that it would execute
in significantly *sub-optimal* time by simply supplying muscle
functions that do not fit with this inflexible model.

For example, in a divide and conquer skeleton, it may be assumed that
the `merge` muscle is a computationally intensive task, whereas the
`solve` muscle is comparatively cheap. The skeleton author may then
choose to parallelise execution of `merge` muscles, leaving the
`solve` muscles sequential. By instantiating this skeleton with a
computationally intensive `solve` muscle but a lightweight `merge`
muscle, the benefits of parallelising the `merge` muscle would be
negated.

The solution:

* For each *unit* of parallelism (e.g. muscle function, recursion,
  etc), assign a *weight*. The purpose of the weight is to provide an
  indicator of the amount of *work done* by a unit. This could either
  be a fixed constant value, or could depend on the size of input
  parameters.
* Use these assigned weights to determine *if* if it is profitable to
  parallelise a unit, or invoke it sequentially. A static heuristic
  could compare the weight of the unit against the cost of
  parallelistation (e.g. creating a new thread, or distributing the
  task).
* It would be easy to evaluate at compile time whether a unit will
  operate in fixed time or not.
* If the work done by a unit is constant, then the decision on whether
  to parallelise or not could be performed statically at compile time.
* If the work done depends on the size of the input parameters, then
  it could be expressed using the standard notation for time
  complexity. Decisions on whether or not to parallelise would have to
  be made at run-time.
* An alternative method of assigning weights to a function could be by
  actually measuring the system clock during run-time. This could give
  a reasonable approximation for units which are executed repeatedly,
  allowing the skeleton to build up a profile of the unit weights over
  the course of a program's lifetime.


## Tuesday 21st

Read an
[interesting blog post](http://blogs.msdn.com/b/vcblog/archive/2014/04/16/parallel-stl-democratizing-parallelism-in-c.aspx)
about Intel, NVIDIA, and Microsoft's collaborative effort to introduce
a parallelised version of the C++ STL, by each implementing their own
versions. Of particular interest is this excerpt:

> As is always the case with parallelization, not every program will
> benefit from using the Parallel STL, so don't just go sprinkling
> your STL code with par willy-nilly. You still need to find a
> bottleneck in your program that's worth parallelizing. In some
> cases, your program will need to be rewritten tobecome amenable to
> parallelism.

What we would need from a *truly* successful parallel STL is logic to
determine *when* to parallelise, so that the programmer would not have
to identify the performance bottlenecks themselves.


## Friday 24th

### Analysing performance of memory instrumentation

Sequential:
```
---------- INSTRUMENTATION ----------
Number of STORE instructions executed: 1382005059
Number of LOAD instructions executed:  1709966643
```

Parallelisation depth 2:
```
---------- INSTRUMENTATION ----------
Number of STORE instructions executed: 448304099
Number of LOAD instructions executed:  585520151
```

Why does the parallelised version require an order of magnitude fewer
stores/load? Perhaps it's a race condition in the counter increment
logic?


### 24.10.2014 meeting with Hugh and Pavlos:

 * MSc project question: furthering the existing work on auto-tuning
   skeletons by optimising static attributes, could we produce better
   results by considering dynamic attributes? Things to consider:
   * Iterative compilation is a crowded field, so the best chance of
     getting published is by performing these optimisations during
     run-time.
   * I should read some of the existing papers on run-time adaption
     (not just wrt. skeletons) to see what has already been done, and
     how + to what extent.
   * A likely approach:
     1. Instrument a set of representative skeleton programs in order
        to analyse run time behaviour.
     2. Discover a set of attributes which have the greatest effect on
        performance.
     3. Get a feel for the effect of the attributes by manually
        adjusting the benchmarks.
     4. Find the simplest way to adjust these attributes "on the fly".
 * I've completed a first attempt at instrumenting instructions in
   LLVM bitcode so that programs will print the number of load and
   store instructions executed.
   * The current implementation is not thread safe. Race conditions
     during the incrementing of counters can be negated either by
     using atomics or thread local counter. Thread local counters
     would reduce the amount of locking required.
   * LLVM has exit hooks which can replace the need to instrument
     every return instruction from `main` function.


## Sunday 26th

Control flow graphs can be visualised using
`opt -view-cfg program.bc >/dev/null`.


## Monday 27th

Useful high-level topics to become familiar with:

 * [Adaptive optimisation](http://en.wikipedia.org/wiki/Adaptive_optimization)
 * [Profile-guided optimisation](http://en.wikipedia.org/wiki/Profile-guided_optimization)
 * [Hot spot](http://en.wikipedia.org/wiki/Hot_spot_(computer_programming))
 * [Profiling](http://en.wikipedia.org/wiki/Profiling_(computer_programming))
 * [Hypervisor](http://en.wikipedia.org/wiki/Hypervisor)


## Tuesday 28th

Date for MSc research proposal presentation: **21st November**. Adam
will be presenting my slides.


## Wednesday 29th

List of
[Profile-Guided Optimization](http://llvm.org/OpenProjects.html#profileguided)
projects in LLVM.


## Friday 31st

#### Reading list from this week:

* A. Collins, C. Fensch, H. Leather, and M. Cole, *“MaSiF: Machine
  learning guided auto-tuning of parallel skeletons,”* 20th
  Annu. Int. Conf. High Perform. Comput., pp. 186–195, Dec. 2013.
  * Auto-tuning tool that selects best static parameters for
    skeletons.
  * Reduced search space with PCA.
  * Offline kNN ML training.
* Z. Wang and M. F. P. O. Boyle, *“Mapping Parallelism to Multi-cores:
  A Machine Learning Based Approach,”* in ACM Sigplan Notices, 2009,
  no. 15, pp. 75–84.
  * Auto-tuning tool that maps MPI programs to hardware.
  * Uses profiling runs to gather dynamic features.
  * Offline kNN ML training.
  * Results have been bested by [1].
* G. Contreras and M. Martonosi, *“Characterizing and improving the
  performance of Intel Threading Building Blocks,”* in Workload
  Characterization, 2008. IISWC 2008. IEEE International Symposium on,
  2008, pp. 57–66.
  * Profiling TBB on benchmarks using profiling runs.
* U. Dastgeer, J. Enmyren, and C. W. Kessler, *“Auto-tuning SkePU: a
  multi-backend skeleton programming framework for multi-GPU
  systems,”* in Proceedings of the 4th International Workshop on
  Multicore Software Engineering, 2011, pp. 25–32.
  * Generating execution plans SkePU, which selects static features
    such as CPU / GPU, num of workers etc.
  * Offline ML training (no method given).
* G. Fursin, C. Miranda, O. Temam, E. Yom-tov, E. Bonilla, J. Thomson,
  H. Leather, C. Williams, and M. O. Boyle, *“MILEPOST GCC: machine
  learning based research compiler,”* in GCC Summit, 2008.
  * Setting compiler heuristics based on profiling runs.
  * Offline ML training.
* K. D. Cooper, A. Grosul, T. J. Harvey, S. Reeves, D. Subramanian,
  L. Torczon, and T. Waterman, *“ACME: adaptive compilation made
  efficient,”* ACM SIGPLAN Not., vol. 40, no. 7, pp. 69–77, 2005.
  * Setting compilation flags for a program using hill climbing,
  genetic algos + 2 others.
  * Simulates execution in order to reduce number of profiling runs.
  * Front-end GUI for twiddling parameters.
* P. Prabhu, *“Safe Programmable Speculative Parallelism,”* in
  Proceedings of Programming Language Design and Implementation
  (PLDI), 2010.
  * C# library which adds language constructs for programmer level
    speculative parallelism.
* Z. Wang and M. F. P. O. Boyle, *“Partitioning Streaming Parallelism
  for Multi-cores: A Machine Learning Based Approach,”* in Proceedings
  of the 19th international conference on Parallel architectures and
  compilation techniques, 2010, pp. 307–318.
  * Auto-tuning partitions for StreamIt programs.
  * Generates 3k random partitions and then tries to find closest to
    "ideal".
  * Offline kNN ML training.
* K. Asanovic, J. Wawrzynek, D. Wessel, K. Yelick, R. Bodik,
  J. Demmel, T. Keaveny, K. Keutzer, J. Kubiatowicz, N. Morgan,
  D. Patterson, and K. Sen, *“A view of the parallel computing
  landscape,”* Commun. ACM, vol. 52, no. 10, p. 56, Oct. 2009.


#### Notes from meeting with Hugh and Pavlos 31.10.2014

 * TODO: By next week, have a firm idea of research proposal, and a
   set of proposal presentation slides; OR back-out and have a
   contingency plan.
 * TODO: Have a look through relevant publications by:
   * Alexander Collins
   * Grigori Fursin (run-time adaption)
   * Chris Williams (ML)
 * A reasonable metric is to aim for 5 papers per hour. This means
   being very selective about which sections
 * There was talk of developing run-time adaption for PetaBricks.


# November 2014

## Sunday 2nd

A list of the tunable parameters and program features used in:

> A. Collins, C. Fensch, H. Leather, and M. Cole, “MaSiF: Machine
> learning guided auto-tuning of parallel skeletons,” 20th
> Annu. Int. Conf. High Perform. Comput., pp. 186–195, Dec. 2013.

For TBB:

```
Tunable parameters:
threads                  1 - (1.5 x core-count)
grainsize                1 - 2^16
partitioner              {auto,affinit,simple}_partitioner
allocator                {std::,tbb_,zero_,cache_aligned_,scalable_}allocator,

Features:
skeleton_used            parallel_for, parallel_reduce
data_structure_read      blocked_range,shared_array,concurrent_vector
data_structure_written   atomic_variable,concurrent_vector,shared_{variable,array}
time complexity*
```

For FastFlow:

```
Tunable parameters:
numworkers               1 - (1.5 x core-count)
buffertype               bounded or undounded
buffersize               1 - 2^20
batchsize                1 - 2^20
cachealign               64,128,256

Features:
skeleton_used            farm,farm-with-feedback
collector_thread         yes,no
number_of_tasks_created*
size_of_task*
```

A list of the tunable parameters and program features used in:

> U. Dastgeer, J. Enmyren, and C. W. Kessler, “Auto-tuning SkePU: a
> multi-backend skeleton programming framework for multi-GPU systems,”
> in Proceedings of the 4th International Workshop on Multicore
> Software Engineering, 2011, pp. 25–32.

```
Tunable parameters:
problem_size_range       1 - infinity
backend                  CPU,OMP,GPU_backend
no_of_threads            only available for OMP backend
grid_size                only available for GPU backends
block_size               only available for GPU backends

Features:
problem_size*
copy_up_time             only available for GPU backends
copy_down_time           only available for GPU backends
kernel_execution_time    only available for GPU backends
total_time               only available for OMP backend
overhead_time            only available for OMP backend
```

## Monday 3rd

#### Notes from meeting with Alex

* They did look into applying MaSiF at runtime by modifying FastFlow
  to add a separate monitoring thread and using sampling
  profiling. The performance benefits they experienced **did not**
  outweigh the overhead of performing the dynamic parameter
  tuning. Possible reasons for this include:
  * Benchmarks that didn't suit the benefits of dynamic optimisation.
    * Small compute kernels.
    * Short-, not long-running programs.
  * Optimising parameters at run-time was expensive. The overhead of
    determining whether or not a parameter set was optimal or not was
    relatively low - performance overhead was negligible when the
    runtime decided not to make any changes.
* Don't underestimate the importance of good, widely accepted
  benchmarks when performing experiments.
* It could be worth talking to Michel Steuwer and Christoph Dubach
  about the work they've been doing on SkelCL.


SkelCL is an interesting algorithmic skeleton library which targets
hetergeous systems by compiling skeletons for execution on multiple
GPUs using OpenCL. Since OpenCL programs are compiled at runtime from
strings, it is by its very nature dynamically compiled. If OpenCL
compilation exposes options for optimising compiled kernels, then
perhaps this could be an interesting optimisation space to search at
runtime? Advantages that make SkelCL a viable platform for
experimenting include that it is open source (GPL), actively
developed, and the primary author Michel is at Edinburgh.


# Tuesday 4th

A list of the benchmarks, targets, and datasets for
[Parboil](http://impact.crhc.illinois.edu/Parboil/parboil.aspx):

```
BENCHMARK, TARGETS, DATASETS
bfs, base  cuda  cuda_base  omp_base  opencl_base  opencl_nvidia, 1M  NY  SF  UT
cutcp, base  cuda  cuda_base  omp_base  opencl_base  opencl_nvidia, large  small
histo, base  cuda  cuda_base  cuda-base  omp_base  opencl_base  opencl_nvidia  threaded, default  large
lbm, cpu  cuda  cuda_base  omp_cpu  opencl_base  opencl_nvidia, long  short
mri-gridding, base  cuda  cuda-base  omp_base  opencl_base  opencl_nvidia, mrig_small.tgz  small
mri-q, cpu  cuda  omp_base  opencl  opencl_nvidia, large  small
sad, base  cpu  cuda  cuda_base  opencl_base  opencl_nvidia, default  large
sgemm, base  cuda  cuda_base  omp_base  opencl_base  opencl_nvidia, medium  small
spmv, cpu   cuda_base      cuda_tex  opencl_ati      opencl_base    opencl_tex cuda  cuda_base_tex  omp_base  opencl_ati_vec  opencl_nvidia  opencl_tex_nvidia, large  medium  small
stencil, cpu  cuda  cuda_base  cuda_fermi  omp_base  opencl_base  opencl_fermi  opencl_nvidia, default  small
tpacf, base  cuda  cuda_base  omp_base  opencl_base  opencl_nvidia, large  medium  small
```

The benchmarks can be ran with the command:

```
./parboil <benchmark> <target> <dataset>
```

Considerations for proposing a SkelCL dynamic auto-tuner:

1. What features and attributes can I select, and at what level
   (i.e. individual muscles, whole program, etc.)?
   * Compilation space: Optimisation flags and levels.
   * Device space: Number of threads to execute, distribution of work.
   * Skeleton space: What kind of muscle function is it.
1. How am I going to exploit the *structure* provided by skeletons?
1. Why is a *dynamic* approach better than static?
   * No offline training phase.
   * Can use runtime features (i.e. the contents and behaviour of
     muscle functions and input data).
1. What is the current state of the art? How is my solution novel?


#### Notes from meeting with Hugh and Pavlos 4.11.2014

* Notes from my meeting with Alex:
  * They did look doing autotuning FastFlow at runtime by adding a
    separate monitoring thread which used sampling profiling, and did
    the re-optimising heavy-lifting.
  * The results weren't positive. The overhead of performing the
    re-optimising was greater than the benefits of the optimisations.
  * The actual overhead of profiling and comparing current
    configuration against training data was relatively small.
  * They attributed the poor performance to short-, not long-running
    benchmarks, which don't offer adequate time to converge on optimal
    configurations.
  * TODO: Find out exactly *what* optimisations they did, and how. Get
    a better understanding of why they weren't successful, and what I
    can do differently.
* SkelCL:
  * Written by Michel in 2011, and actively developed.
  * Targets heterogeneous devices, and compiles OpenCL kernels at
    runtime from source strings.
  * There's a compilation space associated with OpenCL, in additional
    to the usual skel-specific params.
    * I should contact Alberto Magni. He has been working on PTX,
      which is a bytecode IR for GPU programming with an LLVM backend.
  * I should re-read Grigori's paper on continuous compilation which
    suggests that optimisation parameters are insensitive to datasets.
  * The general idea isn't novel (it's a re-hash of iterative
    compilation). In order to generate 10 months work and not 2 weeks
    work, I'll need to come up with some extra challenges and
    opportunities for *innovation*. These could include:
    * Extracting features from datasets. This isn't possible using
      offline training. Note this isn't necessarily a trivial task -
      even something as simple as the number of elements in a dataset
      may be expensive to compute if dealing with arbitrarily nested
      structures.
    * Splitting long-running skeletons into epochs and changing the
      behaviour of the compiled kernel between them. This would be a
      more interesting engineering challenge than simply recompiling
      the same kernel with different params between invocations.
    * Using the additional structure of skeletons as features for a ML
      model. This could be a simple as having different models for
      different skeletons.
    * Supporting arbitrary nesting of skeletons. How do we distribute
      nested skeletons amongst available resources?


## Wednesday 5th

Goals for today:
* Make two graphs plotting the effects of two distinct optimisation
  parameters on a skeleton's runtime, and show how they interact.
* Meet with Michel to discuss SkelCL autotuning.
* Create rough sketches for PPar slides.

Notes from meeting with Michel and Adam:
* Currently, SkelCL maps work items to threads directly.
* NVIDIA and ATI have different memory hierarchies, which means memory
  access patterns need to be specialised for the underlying hardware.
* Relevant related work is Alberto's Thread coarsening paper, which
  takes a compiler-level approach to mapping work items to
  threads. Performing this optimisation at the library level should
  provide advantages since we have a higher-level overview of the
  code.
* Some input data and kernels produce irregular loads,
  e.g. calculating Mandelbrot sets (The pixels which aren't in the set
  can be calculated quickly, the pixels which are in the set require
  expensive computations).
* Build script for Arch Linux.


## Saturday 8th

The brilliantly named "Stats Make Me Cry" blog has a
[good post](http://www.statsmakemecry.com/smmctheblog/the-time-for-reproducible-research-is-now)
about the importance of reproducible research, and some related
reading resources. It's nice to see "Literate Programming" crop up
again. The blog has a bunch of other good posts as well, including one
about the importance of
[submitting early drafts](http://www.statsmakemecry.com/smmctheblog/the-worst-mistake-made-on-a-dissertation-is.html).


## Tuesday 11th

A list of papers read since Oct 31, organised by amount of time spent
reading them.

< 5 min

1. K. D. Cooper, M. W. Hall, and K. Kennedy, **“Procedure Cloning,”**
   in Computer Languages, 1992., Proceedings of the 1992 International
   Conference on, 1992, pp. 96–105.
1. W. F. Ogilvie, P. Petoumenos, Z. Wang, and H. Leather, **“Active
   learning accelerated automatic heuristic construction for parallel
   program mapping,”** Proc. 23rd Int. Conf. Parallel
   Archit. Compil. - PACT ’14, pp. 481–482, 2014.
1. P. Diniz and M. Rinard, **“Dynamic Feedback: An Effective Technique
   for Adaptive Computing,”** in ACM SIGPLAN Notices, 1997, pp. 71–84.
1. F. Chang and V. Karamcheti, **“A Framework for Automatic Adaptation
   of Tunable Distributed Applications,”** Cluster Comput., vol. 4,
   no. 1, pp. 49–62, 2001.
1. I. Chung, J. K. Hollingsworth, and H. Server, **“Active Harmony:
   Towards Automated Performance Tuning,”** in Proceedings of the 2002
   ACM/IEEE conference on Supercomputing, 2002, pp. 1–11.
1. T. J. Edler von Koch and B. Franke, **“Limits of Region-Based
   Dynamic Binary Parallelization,”** ACM SIGPLAN Not., vol. 48,
   no. 7, pp. 13–22, 2013.
1. G. Karsai, A. Ledeczi, J. Sztipanovits, G. Peceli, G. Simon, and
   T. Kovacshazy, **“An Approach to Self-adaptive Software Based on
   Supervisory Control,”** Self-adaptive Softw. Appl., pp. 24–38,
   2003.
1. V. Bhat, M. Parashar, M. Khandekar, N. Kandasamy, and
   S. Abdelwahed, **“Enabling Self-Managing Applications using
   Model-based Online Control Strategies,”** in 2006 IEEE
   International Conference on Autonomic Computing, 2006, pp. 15–24.
1. J. Enmyren and C. Kessler, **“SkePU: a multi-backend skeleton
   programming library for multi-GPU systems,”** in Proceedings of the
   fourth international workshop on High-level parallel programming
   and applications, 2010, pp. 5–14.
1. J. A. Stratton, C. Rodrigues, I. Sung, N. Obeid, L. Chang,
   N. Anssari, G. D. Liu, and W. W. Hwu, **“Parboil: A Revised
   Benchmark Suite for Scientific and Commercial Throughput
   Computing,”** Cent. Reliab. High-Performance Comput., 2012.
1. H. Hoffmann, S. Sidiroglou, S. Misailovic, A. Agarwal, M. Rinard,
   and M. Carbin, **“Dynamic Knobs for Power-Aware Computing,”** 2010.

5-12 min

1. G. Fursin, R. Miceli, A. Lokhmotov, M. Gerndt, A. Malony,
   Z. Chamski, D. Novillo, D. Del Vento, and M. Baboulin,
   **“Collective Mind: towards practical and collaborative
   auto-tuning,”** Sci. Program., vol. 22, no. 4, pp. 209–329, 2014.
1. A. W. Memon and G. Fursin, **“Crowdtuning: systematizing
   auto-tuning using predictive modeling and crowdsourcing,”**. in
   PARCO mini-symposium on’Application Autotuning for HPC
   (Architectures)', 2013.
1. M. Namolaru, A. Cohen, G. Fursin, A. Zaks, and A. Freund,
   **“Practical aggregation of semantical program properties for machine
   learning based optimization,”** Proc. 2010
   Int. Conf. Compil. Archit. Synth. Embed. Syst. - CASES ’10, p. 197,
   2010.
1. Y. Chen, S. Fang, Y. Huang, L. Eeckhout, G. Fursin, O. Temam, and
   C. Wu, **“Deconstructing iterative optimization,”** ACM
   Trans. Archit. Code Optim., vol. 9, no. 3, pp. 1–30, Sep. 2012.
1. A. Collins, D. Grewe, S. Lee, and A. Susnea, **“NOVA: A Functional
   Language for Data Parallelism,”** 2013.
1. J. Jim and G. Fursin, **“Predictive Runtime Code Scheduling for
   Heterogeneous Architectures,”** in High Performance Embedded
   Architectures and Compilers, 2009, pp. 19–33.
1. G. Fursin, A. Cohen, M. O. Boyle, O. Temam, and A. P. Method, **“A
   Practical Method For Quickly Evaluating Program Optimizations,”**
   High Perform. Embed. Archit. Compil., pp. 29–46, 2005.
1. J. Auslander, M. Philipose, C. Chambers, S. J. Eggers, and
   B. N. Bershad, **“Fast, effective dynamic compilation,”** in ACM
   SIGPLAN Notices, 1996, pp. 149–159.
1. V. Bala and E. Duesterwald, **“Dynamo: A Transparent Dynamic
   Optimization System,”** in ACM SIGPLAN Notices, 2000, pp. 1–12.
1. J. Ansel and C. Chan, **“PetaBricks,”** XRDS Crossroads, ACM
   Mag. Students, vol. 17, no. 1, p. 32, Sep. 2010.
1. J. Ansel, C. Chan, Y. L. Wong, M. Olszewski, Q. Zhao, A. Edelman,
   and S. Amarasinghe, **“PetaBricks: A Language and Compiler for
   Algorithmic Choice,”** in ACM SIGPLAN Conference on Programming
   Language Design and Implementation, 2009, vol. 44, no. 6,
   pp. 38–49.
1. M. J. Voss and R. Eigenmann, **“High-Level Adaptive Program
   Optimization with ADAPT,”** in ACM SIGPLAN Notices, 2001,
   pp. 93–102.
1. W. F. Ogilvie, P. Petoumenos, Z. Wang, and H. Leather, **“Fast
   Automatic Heuristic Construction Using Active Learning,”** in
   LCPC’14, 2014.
1. C.-K. Luk, S. Hong, and H. Kim, **“Qilin: Exploiting Parallelism on
   Heterogeneous Multiprocessors with Adaptive Mapping Categories and
   Subject Descriptors,”** in Microarchitecture, 2009. MICRO-42. 42nd
   Annual IEEE/ACM International Symposium on, 2009, pp. 45–55.
1. C. Chan, J. Ansel, Y. L. Wong, S. Amarasinghe, and A. Edelman,
   **“Autotuning multigrid with PetaBricks,”** in ACM/IEEE Conference on
   Supercomputing, 2009.
1. C. Dubach, **“Using Machine-Learning to Efficiently Explore the
   Architecture / Compiler Co-Design Space,”** Univeristy of
   Edinburgh, 2009.
1. S. Rul, H. Vandierendonck, J. D. Haene, and K. De Bosschere, **“An
   Experimental Study on Performance Portability of OpenCL Kernels,”**
   in 2010 Symposium on Application Accelerators in High Performance
   Computing (SAAHPC’10), 2010, pp. 4–6.
1. Y. Dotsenko, S. S. Baghsorkhi, B. Lloyd, and N. K. Govindaraju,
   **“Auto-tuning of fast fourier transform on graphics processors,”**
   ACM SIGPLAN Not., vol. 46, no. 8, pp. 257–266, 2011.
1. J. Fang, A. L. Varbanescu, and H. Sips, **“An Auto-tuning Solution
   to Data Streams Clustering in OpenCL,”** in Computational Science
   and Engineering (CSE), 2011 IEEE 14th International Conference on,
   2011, pp. 587–594.
1. M. Steuwer and S. Gorlatch, **“SkelCL: Enhancing OpenCL for
   High-Level Programming of Multi-GPU Systems,”** Parallel
   Comput. Technol., pp. 258–272, 2013.

12-30 min

1. G. Fursin, C. Miranda, S. Pop, A. Cohen, and O. Temam, **“Practical
   Run-time Adaptation with Procedure Cloning to Enable Continuous
   Collective Compilation,”** in Proceedings of the GCC Developers’
   Summit, 2007, pp. 39–46.
1. M. Tartara and S. Crespi Reghizzi, **“Continuous learning of
   compiler heuristics,”** ACM Trans. Archit. Code Optim., vol. 9,
   no. 4, pp. 1–25, Jan. 2013.
1. J. Ansel and U. O. Reilly, **“SiblingRivalry: Online Autotuning
   Through Local Competitions,”** in International Conference on
   Compilers Architecture and Synthesis for Embedded Systems, 2012.
1. M. Steuwer and S. Gorlatch, **“High-level Programming for Medical
   Imaging on Multi-GPU Systems Using the SkelCL Library,”** Procedia
   Comput. Sci., vol. 18, pp. 749–758, Jan. 2013.

> 30 min

1. D. Nuzman, R. Eres, S. Dyshel, and M. Zalmanovici, **“JIT
   Technology with C / C ++ : Feedback-Directed Dynamic Recompilation
   for Statically Compiled Languages,”** ACM Trans. Arch. Code
   Optim.}, vol. 10, no. 4, p. 59, 2013.

## Wednesday 12th

Goal for today:
* Create slides for methodology / evaluation. 4 minutes of content ~=
  8 slides.

Methodology:
* When presenting methodology, skip over the stuff which has been done
  before, and focus on how I am going to do things differently.
* Data gathering: identifying a set of parameters and features that
  influence the performance of SkelCL skeletons.
* Develop models that relates tunable parameters to performance.
* Implement a dynamic optimiser which creates and uses these model
  online.

Evaluation:
* Find, or rewrite, a set of benchmarks that my solution can be
  compared against which target the same hardware.
* Perhaps compare results against *hand tuned* solution.

Hugh's suggestions for presentation slides:
1. Be exact and clear about the things that I am proposing to do which
   **have not been done before**.
1. Why are the challenges challenging? The audience should **want** to
   see the results of my research.

## Friday 14th

Goals for today:
* Re-read Fursins' paper on Continuous Optimisation.
* Write a statistically rigorous implementation of the `time` program,
  which will execute a given command a number of times and report
  variance between runs and confidence intervals.


## Saturday 15th

Adam sent me a link to an interesting blog post about
[Linear vs Binary Search](https://schani.wordpress.com/2010/04/30/linear-vs-binary-search/).


## Monday 17th

I spent the weekend hacking on a small tool
[srtime](https://github.com/ChrisCummins/srtime) for performing
statistically rigorous program performance evaluation. Perhaps the
next step would be a "performance evaluation tool (PET)" project which
provides an API which wraps this timer and allows users to specify
variants which can be iterated over and timed?


## Tuesday 18th

I have been given the opportunity to write a survey paper on automatic
parallelisation, due March / April time. I should let my supervisors
know this week whether I will take up the offer or not.

[Thrust](http://thrust.github.io/) is a parallel algorithms library in
the style of the C++ STL. It's intended for high performance
heterogeneous programming. It provides host and device container
types, and requires the user to manually copy to and from devices.


## Wednesday 19th

I have discovered [plot.ly](https://plot.ly), a web interface for
creating nice simple 2D plots. This will come in handy for quickly
creating nice "modern" looking plots.


## Friday 21st

The proposal presentation went well, and there was some good questions
at the end. Murray raised a very interesting point - clarifying
whether the "persistent data" part of my dynamic optimiser proposal
meant storing data at a per-program or per-skeleton level. I should be
clear in the proposal to emphasise that the data collected captures a
bit of context about the skeleton call and is applicable to all future
skeleton calls, irrespective of the program. This is a key advantage
over current dynamic optimisers, which tailor their optimisations to
the host procress.


## Monday 24th

Notes from meeting with Pavlos:

* Important deadlines: 1000 word summary of research (due Wednesday);
  6 page research proposal (due next Friday).
* All effort this week (and last) has been focused on the research
  proposal. Currently it's just a ~2000 word brain dump.
* Method section for research proposal doesn't have to be
  ultra-specific. Pose various ideas for how you will approach the
  problem.
* Reference counts for research proposals is generally pretty low. ~12
  ref for 6 pages, almost always < 20 ref for an 8 page paper.
* Yes to the offer of writing an auto parallelisation survey
  paper. The biggest time sink will be the background reading. Pavlos,
  Hugh and Zheng will help with draft reviews.
* My reading over the past 10 days has been quite low (averaging about
  3 papers / day). I will try to keep my
  [reading list](https://github.com/ChrisCummins/msc-thesis/blob/master/docs/ReadingLog.md)
  up to date, but sometimes forget (I'm keeping all my papers & notes
  tracked with Mendeley). If you're reading for a specific purpose
  (e.g. finding relevant citations for a paper), you can generally
  decide whether a paper is worth reading in the first 20 seconds.
* TODO: Send Pavlos & Hugh my research summary *tomorrow*.
* TODO: Send Pavlos & Hugh my first draft research proposal by
  *Friday* (earlier is better).
* TODO: Send Pavlos 3-4 papers that I thought were interesting.
* Pavlos is going to send me a list of sub-topics and important papers
  for the auto parallelisation survey in the next ~2.5 weeks.

New term: **register spilling**. Moving a variable from a register to
main memory is called *spilling*. The reverse operation is called
*filling*. Compilers use multiple factors to decide which variables to
spill: execution time, code space, and data space.

There is a very interesting look book about parallel programming,
available freely online:
[Is Parallel Programming Hard, And, If So, What Can You Do About It?](https://www.kernel.org/pub/linux/kernel/people/paulmck/perfbook/perfbook-1c-e1.pdf).


## Tuesday 25th

I've been looking for quantifiable examples of the decreasing rate of
improvement of single-core performance in modern CPUs. The best I
could find was this blog post:
[A Look Back at Single-Threaded CPU Performance](http://preshing.com/20120208/a-look-back-at-single-threaded-cpu-performance/).


## Wednesday 26th

Pavlos has reviewed my 1000 word research summary. Apart from some
grammatical errors, the biggest criticism was a weak ending. The
conclusion needs to end with a "this research will change the world"
statement.

I notice that textlint reports extensive use of the passive voice in
my research summary. Many of the sentences seem to flow quite
naturally, for example:

"Effective searching of these spaces can only be performed by
collecting empirical data rather than building predictive models."

Rephrasing into the active voice puts an unnatural emphasis on the
actor (in this case: "we"):

"We can effectively search these spaces only by collecting empirical
data rather than building predictive models."

I found a good blog post on the use of passive voice in scientific
writing,
[The Value of the Passive Voice](http://www.biomedicaleditor.com/passive-voice.html).
It identifies three cases where the passive voice may be preferred to
the active:

1. The performer is unknown, irrelevant, or obvious
1. The performer is less important than the action
1. The recipient is the main topic

They have a related article,
["Hedging" in Scientific Writing](http://www.biomedicaleditor.com/hedging.html),
which describes the overuse of cautious language in scientific
writing.


## Thursday 27th

I have attempted to articulate two scenarios in which I believe the
performance of a dynamic autotuner for Algorithmic Skeletons will
exceed the current state of the art:

1. A dynamic autotuner will provide improved performance over a
   "general case" implementation. This is especially important for
   Algorithmic Skeletons since they present a generic library
   interface where the author cannot infer many properties about the
   type of input data *or* input algorithms (muscle functions).
1. A dynamic autotuner will provide improved performance over a
   statically tuned piece of code for *varying* data inputs.

----

I have been trying to decompose my title "Dynamic autotuning of
Algorithmic Skeletons" in terms of where the two topics "Dynamic
autotuning" and "Algorithmic Skeletons" fit within the big picture. I
think it's interesting that I place both topics at the 5th tier down,
and that the third topic in both hierarchies shares a common theme of
automation:

Pervasive Parallelism -> Parallel software -> Automatic
parallelisation -> Library level -> Algorithmic Skeletons

Computational performance -> Program optimisation -> Automatic
optimisation -> Parameter tuning -> Dynamic autotuning

----

A short note on language use: general advice is that while "optimum"
and "optimal" may be used interchangeably when used as an adjective,
only "optimum" may be used as a noun. As a result, it may be more
clear to use only "optimum" as a noun, and "optimal" as an adjective.


## Friday 28th

The goal of today is to get a first draft of my project proposal
completed and sent to Pavlos and Hugh for an initial critique.


# December 2014


## Monday 1st

The [PPar course page](http://www.inf.ed.ac.uk/teaching/courses/ppar/)
lists 12 criteria that will be used to evaluate the project
proposal. Based on my first draft, the following six points need
further addressing before submission:

* Sound approach: the proposed solution is well-founded and has a
  reasonable chance of success.
* Originality: it is clear which concepts and ideas originate from the
  student and which are borrowed from the literature and from the
  discussions with the supervisor.
* Capable of execution: the student has the necessary skills
  (programming or otherwise) to successfully complete the project in
  the allocated timeframe.
* Well-defined tasks and deliverables: each of the tasks in the plan
  is well defined with clear inputs and deliverables with a clear role
  to plan in delivering the final dissertation. The decomposition of
  the proposed problem/issue into subtasks is sound in the sense that
  it constitutes a plan that has a high chance of succeeding and
  producing deliverables that can be combined to support a passable
  MSc dissertation. It is also clear that there are no missing tasks -
  ensure adequate planning for writing up. Clear milestones are
  identified for each task.
* Resources are attainable: the sources for any required equipment,
  funding and human effort are identified. It should be clear that the
  student will be capable of providing the necessary human effort.
* Evaluation: The proposal contains a concrete plan for evaluating the
  proposed work, including description of appropriate benchmark
  datasets, evaluation measures, gold-standard judgements and
  baselines to compare against.

Challenges:
* Overhead: dynamic compilation, machine learning.
* Convergence time: focused search, dimensionality reduction, search
  space bounding.
* Evaluation: selecting representative benchmarks, stochastic systems.

----

Notes on first draft review with Pavlos:
* The abstract and conclusion need to sell the work better.
* The methodology is a little vague, and the evaluation a little too
  specific.
* The background section should cover a piece of related work
  paragraph by paragraph:
  * "[author] did [USP] in [paper], but did not cover [use case]"
  * "[author] achieved [USP] in [paper], but only for [use case]"
  * "The [USP] from [paper] can be applied successfully to [use case]"
* Once the proposal is done, I should focus my effort on background
  reading. I should aim to get through 40 - 50 papers before
  Christmas.


## Wednesday 3rd

Hugh's comments on my proposal draft is that it's too dense and needs
pictures and graphs. A paper has two narratives: the potentially
boring words, and the pictures. The pictures narrative must stand up
without requiring the words (the captions must be
self-contained). Also, the Gantt chart item labels are not specific.
