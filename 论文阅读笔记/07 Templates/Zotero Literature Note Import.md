---
type: paper-note
title: "{{title | replace('\"', '\\\"')}}"
authors:
{% for c in creators %}{% if c.creatorType == "author" %}
  - "{{c.lastName or c.name}}{% if c.firstName %}, {{c.firstName}}{% endif %}"
{% endif %}{% endfor %}
year: {% if date %}{{date | format("YYYY")}}{% endif %}
venue: "{{publicationTitle or proceedingsTitle or conferenceName or publisher or ""}}"
status: unread
tags:
  - paper
{% for t in tags %}
  - "{{t.tag}}"
{% endfor %}
topics: []
zotero_key: "{{citekey}}"
doi: "{{DOI or ""}}"
url: "{{url or ""}}"
pdf: "{{pdfZoteroLink or localLibrary or ""}}"
created: "{{importDate | format("YYYY-MM-DD")}}"
updated: "{{importDate | format("YYYY-MM-DD")}}"
---

# {{title}}

## Citation

- Citekey: `{{citekey}}`
- Zotero: {{zoteroSelectURI}}
- PDF: {{pdfZoteroLink or localLibrary or ""}}

## Bibliography

{{bibliography}}

{% if abstractNote %}
## Abstract

{{abstractNote}}
{% endif %}

## Why This Paper

- Problem it solves:
- Why it matters:
- What makes it worth reading:

## Core Idea

- One-sentence summary:
- Method intuition:
- Key design choices:
- Trade-offs:

## Method Breakdown

- Input / output:
- Training signal:
- Objective / loss:
- Inference or rollout flow:
- Important implementation details:

## Evidence

- Main experiments:
- What the ablations really prove:
- Baselines that matter:
- Failure cases / limits:

## My Notes

{% persist "notes" %}
{% if isFirstImport %}
- [ ] Read full paper
- [ ] Fill authors / topics
- [ ] Decide whether it is worth deep reading
{% endif %}
{% endpersist %}

## Imported Annotations

{% persist "annotations" %}
{% set newAnnotations = annotations | filterby("date", "dateafter", lastImportDate) %}
{% if newAnnotations.length > 0 %}
### Import {{importDate | format("YYYY-MM-DD HH:mm")}}
{% for a in newAnnotations %}
#### Page {{a.page}}
{% if a.annotatedText %}
> {{a.annotatedText}}
{% endif %}
{% if a.comment %}

Comment: {{a.comment}}
{% endif %}
{% if a.imageRelativePath %}

![]({{a.imageRelativePath}})
{% endif %}

{% endfor %}
{% endif %}
{% endpersist %}

