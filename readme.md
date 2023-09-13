<H1>RegEMO: Regularized Evolutionary Multi-objective Optimization</H1>

<H2>Introduction</H2>
Multi-objective optimization problems give rise to a set of Pareto-optimal solutions, each of which makes a trade-off among the objectives. When multiple Pareto-optimal solutions are to be implemented for different applications as platform-based solutions, a solution principle common to them is highly desired for easier understanding, implementation, and management purposes. RegEMO is a systematic search methodology that deviates from finding Pareto-optimal solutions but finds a set of near Pareto-optimal solutions sharing common principles of a desired structure and still having a trade-off of objectives in them. RegEMO has been demonstrated over a number of constrained and unconstrained multi-objective test problems. Thereafter, we demonstrate the practical significance of the proposed approach to a number of engineering design problems. Searching for a set of solutions with common principles of desire, rather than Pareto-optimal solutions without any common structure, is a practically meaningful task and this paper should encourage more such developments in the near future.

<H2>Code</H2>
Bi-level optimization: [Regularity Driver](https://github.com/Ritam-Guha/regemo/tree/tevc/regemo/algorithm/regularity_driver.py)

Lower level optimization: [Regularity Search](https://github.com/Ritam-Guha/regemo/tree/tevc/regemo/algorithm/regularity_search.py)
