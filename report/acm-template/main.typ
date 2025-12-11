#import "@preview/clean-acmart:0.0.1": acmart, acmart-ccs, acmart-keywords, acmart-ref, to-string

#let cuhk = super(sym.suit.spade)

#let title = [Information Retrieval From "Scratch" project]

#let authors = (
  // You can use grouped affiliations with mark
  (
    name: [Luuk van der Heijden Hu],
    email: [jlhu\@cse.cuhk.edu.hk],
    mark: cuhk,
  ),
  (
    // Should I use string or content? It doesn't matter
    name: "Jacoppo Surname2",
    email: "email2@email.com",
    mark: super(sym.suit.diamond),
  ),
  // Or you can put affiliations directly in the author list
  // (
  //   name: [FirstName Surname],
  //   email: [email\@email.com],
  //   // You can put any thing here, and they will automatically be appended below the author name
  //   department: [Department of Computer Science and Engineering],
  //   institute: [The Chinese University of Hong Kong],
  //   city: [Hong Kong],
  // ),
)
#let affiliations = (
  (
    name: [Radoub University],
    mark: cuhk,
    department: [Department of Computer Science and Engineering],
    // You can put any thing here, and they will automatically be appended below
    // city: [Hong Kong],
  ),
  (
    name: [Radoud University],
    mark: super(sym.suit.diamond),
    department: [Department of Computer Science and Engineering],
  ),
  // More affiliations
)
#let conference = (
  name:  [ACM SIGOPS 31th Symposium on Operating Systems Principles],
  short: [SOSP ’25],
  year:  [2025],
  date:  [October 13–16],
  venue: [Seoul, Republic of Korea],
)
#let doi = "https://doi.org/10.1145/0000000000"
#let ccs = (
  (
    generic: [Software and its engineering],
    specific: ([Virtual machines], [Virtual memory], ),
  ),
  (
    generic: [Computer systems organization],
    specific: ([Heterogeneous (hybrid) systems], ),
  ),
)
#let keywords = ("Virtual machine", "Virtual memory", "Operating system", )

#show: acmart.with(
  title: title,
  authors: authors,
  affiliations: affiliations,
  conference: conference,
  doi: "",
  copyright: "none",
  // Set review to submission ID for the review process or to "none" for the final version.
  // review: [\#001],
)




= Abstract
We have done some things
= Todo

Experiment with different bm25 parameters.

Discuss that we tried hybrid search, but that the index size is prohibitive for the entire OWI document set. This might be fixeable if you had enough time, or got stuff working on the cluster, but we do not have that luxury.

Re-rank different amounts of documents, right now we are just doing 

Maybe try pagerank, but this is tricky 

Discuss t

#acmart-ccs(ccs)
#acmart-keywords(keywords)
#acmart-ref(to-string(title), authors, conference, doi)

= Introduction
Here we talk about information retrieval.
== Paper overview
something
= Related Work
Here we need to discuss the MS Marco dataset that the cross encoder we use was trained on. We might have to mention the fact that our dataset does or does not resemble that set very well, or it does. 
= Methods <sec:methods>
#set math.equation(numbering: "1.")
Explain the evaluation methods we use, MRR\@k, precision\@k, recall\@k, NDCG\@k 

= Results

#show table.cell.where(y: 0): strong

#table(
  columns: 5,
  table.header[Method][P10][P20][P50][P100],
  [bm25], [1],[1],[1],[1]
)
#table(
  columns: 5,
  table.header[Method][MRR][R20][NDCG10][NDCG50],
  [RR10], [1],[1],[1],[1],
  [RR20], [1],[1],[1],[1],
)
= Discussion
#lorem(300)

#bibliography("refs.bib", title: "References", style: "royal-society-of-chemistry")

#colbreak(weak: true)
#set heading(numbering: "A.a.a")

= Artifact Appendix
In this section we show how to reproduce our findings.

