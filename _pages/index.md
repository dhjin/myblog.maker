---
permalink: /
title: "About me"
excerpt: "About me"
author_profile: true
---

- husbund, violin player, linguist, christian
## Recent Posts..
<ul>
  {% for post in site.posts %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a>
    </li>
  {% endfor %}
</ul>
