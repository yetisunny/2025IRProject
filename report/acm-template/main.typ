#import "@preview/clean-acmart:0.0.1": acmart, acmart-ccs, acmart-keywords, acmart-ref, to-string

#let cuhk = super(sym.suit.spade)

#let title = [
  Information Retrieval Project
]
#let authors = (
  (
    // Should I use string or content? It doesn't matter
  name: "luuk van der heijden",
    email: "email1@email.com",
    mark: super(sym.suit.diamond),
  ),
  (
    // Should I use string or content? It doesn't matter
    name: "Jacoppo van der heijden Surname2",
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
    name: [Radoud],
    mark: cuhk,
    department: [],
    // You can put any thing here, and they will automatically be appended below
    // city: [Hong Kong],
  ),
  (
    name: [Institution/University Name],
    mark: super(sym.suit.diamond),
    department: [Department Name],
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
)
#let keywords = ()

#show: acmart.with(
  title: title,
  copyright: none,
  
  // Set review to submission ID for the review process or to "none" for the final version.
  // review: [\#001],
)



= Abstract
this is something I must do hahah
= Todo

Experiment with different bm25 parameters.

Try different hybrid search fusion parameters

Re-rank different amounts of documents, right now we are just doing 

Maybe try pagerank, but this is tricky 

Discuss t

#acmart-ccs(ccs)
#acmart-keywords(keywords)
#acmart-ref(to-string(title), authors, conference, doi)

= Introduction
Information retrieval systems come in all shapes and sizes.
== Paper overview
something
= Related Work
Here we need to discuss the MS Marco dataset that the cross encoder we use was trained on. We might have to mention the fact that our dataset does or does not resemble that set very well, or it does. 
= Methods <sec:methods>
#set math.equation(numbering: "1.")
In order to to effeciient retrieval on the document set we have built an lexical index.
We have also used a = Acknowledgement

#bibliography("refs.bib", title: "References", style: "royal-society-of-chemistry")

#colbreak(weak: true)
#set heading(numbering: "A.a.a")

= Artifact Appendix
In this section we show how to reproduce our findings.

