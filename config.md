<!--
Add here global page variables to use throughout your website.
-->
+++
author = "Shuyi Jia"
mintoclevel = 2

# Add here files or directories that should be ignored by Franklin, otherwise
# these files might be copied and, if markdown, processed by Franklin which
# you might not want. Indicate directories by ending the name with a `/`.
# Base files such as LICENSE.md and README.md are ignored by default.
ignore = ["node_modules/"]

# RSS (the website_{title, descr, url} must be defined to get RSS)
generate_rss = true
website_title = "Franklin Template"
website_descr = "Example website using Franklin"
website_url   = "https://tlienart.github.io/FranklinTemplates.jl/"
+++

<!--
Add here global latex commands to use throughout your pages.
-->
\newcommand{\R}{\mathbb R}
\newcommand{\scal}[1]{\langle #1 \rangle}
\newcommand{\vect}[1]{\boldsymbol{#1}}
\newcommand{\bexam}[1]{\boldsymbol{#1}^{(n)}}
\newcommand{\exam}[1]{{#1}^{(n)}}

\newcommand{\figenv}[3]{
~~~
<figure style="text-align:center;">
<img src="!#2" style="padding:0;#3" alt="#1"/>
<figcaption style="font-style: italic; color: black; font-size: 15px;">#1</figcaption>
</figure>
~~~
}