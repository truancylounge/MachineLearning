Both are formats used by **Common Crawl** (the web-scraping project that underlies most LLM pretraining data, including the FineWeb pipeline you linked earlier). WARC is the raw archived page; WET is the plain-text extraction of that same page. Here's a side-by-side example for one crawled URL.

## WARC (Web ARChive) — the raw capture

Contains the **full HTTP response**, including headers *and* the complete original HTML — exactly as the web server sent it.

```
WARC/1.0
WARC-Type: response
WARC-Target-URI: http://example.com/movies/review
WARC-Date: 2024-01-15T10:23:45Z
WARC-Record-ID: <urn:uuid:3d4f8a2e-...>
Content-Type: application/http; msgtype=response
Content-Length: 1847

HTTP/1.1 200 OK
Content-Type: text/html; charset=UTF-8
Server: nginx

<!DOCTYPE html>
<html>
<head><title>Movie Review: Teddy Bear Adventures</title></head>
<body>
<nav>Home | Reviews | Contact</nav>
<div class="ad-banner">Advertisement: Buy tickets now!</div>
<article>
  <h1>A Cute Teddy Bear Adventure — Review</h1>
  <p>I loved this movie's plot!</p>
</article>
<footer>© 2024 MovieSite. All rights reserved.</footer>
</body>
</html>
```

## WET (WARC Encapsulated Text) — extracted plain text

Same underlying page, but the HTML has been stripped down to just the **rendered text content** — no tags, no HTTP headers, no server metadata.

```
WARC/1.0
WARC-Type: conversion
WARC-Target-URI: http://example.com/movies/review
WARC-Date: 2024-01-15T10:23:45Z
WARC-Record-ID: <urn:uuid:9b7c1f0a-...>
WARC-Refers-To: <urn:uuid:3d4f8a2e-...>
Content-Type: text/plain
Content-Length: 178

Movie Review: Teddy Bear Adventures

Home | Reviews | Contact

Advertisement: Buy tickets now!

A Cute Teddy Bear Adventure — Review

I loved this movie's plot!

© 2024 MovieSite. All rights reserved.
```

## Key differences

| | WARC | WET |
|---|---|---|
| **Contains** | Full HTML + HTTP headers + server metadata | Plain extracted text only |
| **File size** | Large (raw markup, images referenced, scripts) | Much smaller (text-only) |
| **`WARC-Type`** | `response` (or `request`, `warcinfo`) | `conversion` |
| **`WARC-Refers-To`** | N/A | Points back to the original WARC record it was derived from |
| **Used for** | Full page reconstruction, HTML-level analysis | Direct input to text-based NLP/LLM pretraining pipelines |

## Why this matters for the ablation/FineWeb context you were reading about

Notice the WET output above still contains **nav links, an ad banner, and a copyright footer** — mixed in right alongside the actual article text. WET extraction only strips *HTML tags*; it does **not** remove boilerplate/junk text. This is exactly the gap that pipelines like FineWeb address: they take this raw WET text and apply additional filtering steps (removing boilerplate, deduplicating, language-filtering, quality-scoring) *on top of* the WET extraction — which ties directly back to the ablation study concept from earlier: each of those filtering steps is itself typically justified by an ablation showing it measurably improves downstream model performance.