---
layout: publication
permalink: /publications/
title: publications
description: publications
years: [2022, 2021, 2020, 2019, 2017]
nav: true
nav_order: 1
---
<!-- _pages/publications.md -->

<p>An up-to-date list is available on <a href="https://scholar.google.com/citations?user=qNk6tgcAAAAJ" target="_blank" rel="noopener noreferrer">Google Scholar</a>.</p>

<div class="publications">

{%- for y in page.years %}
  <h2 class="year">{{y}}</h2>
  {% bibliography -f papers -q @*[year={{y}}]* %}
{% endfor %}

</div>
