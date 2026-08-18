---
layout: page
permalink: /publications/
title: publications
description: publications in reverse chronological order.
nav: true
nav_order: 2
lang: en
page_id: publications
---

{% assign locale = site.data.i18n[site.active_lang] %}
{% assign publications = site.publications | sort: "date" | reverse %}

<div class="legacy-publications">
  {% for publication in publications %}
    <article class="legacy-publication">
      <h2>
        <a href="{{ publication.url | relative_url }}">{{ publication.title }}</a>
      </h2>

      {% if publication.authors %}
        {% assign highlighted_authors = publication.authors | replace_first: "Yuchu Jiang", "<strong>Yuchu Jiang</strong>" %}
        <p class="legacy-publication__authors">{{ highlighted_authors }}</p>
      {% endif %}

      {% if publication.venue and publication.date %}
        <p class="legacy-publication__venue">{{ locale.published_in }} <em>{{ publication.venue }}</em>, {{ publication.date | date: "%Y" }}</p>
      {% endif %}

      {% if publication.excerpt %}
        <p class="legacy-publication__excerpt">{{ publication.excerpt }}</p>
      {% endif %}

      {% if publication.citation %}
        <p class="legacy-publication__citation">{{ locale.recommended_citation }}: {{ publication.citation }}</p>
      {% endif %}

      {% if publication.paperurl %}
        <p class="legacy-publication__paper"><a href="{{ publication.paperurl }}">{{ locale.download_paper }}</a></p>
      {% endif %}
    </article>
  {% endfor %}
</div>
