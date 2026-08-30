/* Make the header title navigate home, matching the logo's behavior. */
document$.subscribe(function () {
  var title = document.querySelector(".md-header__title");
  if (title && !title.dataset.stHome) {
    title.dataset.stHome = "1";
    title.style.cursor = "pointer";
    title.addEventListener("click", function () {
      var logo = document.querySelector(".md-header__button.md-logo");
      if (logo && logo.href) {
        window.location.href = logo.href;
      }
    });
  }
});

/* Scroll-reveal: fade+rise sections as they enter the viewport.
   Elements already in view on load are shown immediately (no animation). */
document$.subscribe(function () {
  if (typeof IntersectionObserver === "undefined") return;

  var targets = document.querySelectorAll(
    ".md-typeset h2, .md-typeset .grid, .md-typeset .tabbed-set"
  );

  var obs = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      if (e.isIntersecting) {
        e.target.classList.add("st-in");
        obs.unobserve(e.target);
      }
    });
  }, { threshold: 0.07, rootMargin: "0px 0px -36px 0px" });

  targets.forEach(function (el) {
    var rect = el.getBoundingClientRect();
    var alreadyVisible = rect.top < window.innerHeight && rect.bottom > 0;
    if (alreadyVisible) {
      /* skip animation — already in viewport */
      return;
    }
    el.classList.add("st-reveal");
    obs.observe(el);
  });
});
