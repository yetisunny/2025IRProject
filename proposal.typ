= Information Retrieval Project Proposal  
== Group 4
==== Luuk van der heijden S1020340
==== Jacopo Levati S1174645


= Motivation
For the project we intend to do the creation of an IR system from scratch. The motivation for this mostly stems from the fact that we will learn the most from this and can see the impact of different ranking methods.

= Research Goal
Our goal is to try and create the best possible IR system we can, while learning the most about the different parts of how the system are implemented. 

= Resources

For our document collection we intend to use the ROBUST dataset from 2005, that uses the "AQUAINT Corpus of English News Text" , this dataset contains news articles from three different sources. It also comes with 50 hard queries, and it also comes with relevant document pairs so we can evaluate how well our system actualy works.

We want to use python for the project as it has a lot of libraries that can help us implement key parts of the system. 

We intend to use GTE-ModernColBERT model for a neural based approach, and use the similarity of embeddings of the queries and the documents to create a ranking.

= Experimenal design

We will have to create the key parts of an IR system like the index, and then we intend to create an initial ranking using something like BM25, followed with an LLM  embedding based approach using the GTE-ModernColBERT model, we think this will retrieve more contextually relevant documents.

We intend to evaluate our system using the relevant document pairs provided by the ROBUST dataset by computing the mean average precision

After we have this initial system working we can try improve it based on the queries that we will be provided by TAS.
= Outcome 
An implementation in python of an information retrieval system that can provide you with relevant documents using a neural based approach in combination with lexical retrieval.

== Footnote
We do not have a third member listed on this document because we could not get in contact with them. We have tried e-mailing them, and looked in the discord, but nog heard anythhing. We hand this in with the knowledge that we are not a full team yet, but would like somebody that also wants to work on the first project.

