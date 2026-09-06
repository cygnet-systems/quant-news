/**
 * Fold state for the Pipeline Activity log's "Details" disclosure.
 *
 * The panel body is rewritten on every poll tick that changes the feed,
 * and dash-renderer recreates the nodes wholesale (see
 * feed_scroll_anchor.js). A freshly built <details> has no `open`
 * attribute, so a log the reader unfolded snapped shut on the next update
 * — mid-run, that was every 1.5 s. The browser owns the toggle and Dash
 * never sees it (html.Details has an `open` prop, but a native toggle does
 * not write it back), so the state is kept here: each toggle is recorded,
 * and each rewrite re-applies it before the next paint. MutationObserver
 * callbacks run before paint, so the fold never flickers.
 *
 * Remembered for the tab's session (sessionStorage, when allowed), so a
 * reload keeps the reader's choice; a new tab starts folded, as before.
 *
 * Expanding the panel (the arrows control; app.py sets the
 * progress-panel-expanded class) unfolds the log as well: the expanded
 * panel is the "read the whole run" view, and a wide panel over a folded
 * two-line log was the complaint. Restoring the panel puts the fold back
 * the way it was before the expand, so a reader who keeps the log folded
 * in the small panel is not left with it open.
 *
 * While expanded, the log's height is set here too: the panel has a
 * fixed height, but the lines sit inside a <details>, whose content the
 * browser wraps in a pseudo-element that neither percentage heights nor
 * flex can size through. So after every rewrite, toggle, class change
 * and window resize, the lines get whatever the stepper and the Details
 * summary leave of the feed, as an inline max-height; the small panel
 * clears it and keeps the stylesheet's cap.
 */
(function () {
  var KEY = "qn_progress_details_open";
  var EXPANDED = "progress-panel-expanded";
  var open = false;
  var beforeExpand = null;
  try {
    open = sessionStorage.getItem(KEY) === "1";
  } catch (e) { /* storage blocked: in-memory for the page's life */ }

  function details() {
    var host = document.getElementById("progress-feed-scroll");
    return host && host.querySelector("details.progress-details");
  }

  function remember(value) {
    open = value;
    try {
      sessionStorage.setItem(KEY, value ? "1" : "0");
    } catch (e) { /* ignored */ }
  }

  // `toggle` does not bubble; capture it at the document instead so the
  // listener survives the node being replaced.
  document.addEventListener(
    "toggle",
    function (ev) {
      var el = ev.target;
      if (el && el.classList && el.classList.contains("progress-details")) {
        remember(el.open);
        size();
      }
    },
    true
  );

  window.addEventListener("resize", size);

  function size() {
    var panel = document.getElementById("progress-panel");
    var host = document.getElementById("progress-feed-scroll");
    var lines = host && host.querySelector(".progress-feed-lines");
    if (!panel || !lines) return;
    if (!panel.classList.contains(EXPANDED)) {
      if (lines.style.maxHeight) lines.style.maxHeight = "";
      return;
    }
    var el = details();
    var used = 0;
    for (var n = host.firstElementChild; n; n = n.nextElementSibling) {
      if (n !== el && n !== lines) used += n.getBoundingClientRect().height;
    }
    var summary = el && el.querySelector(".progress-details-summary");
    if (summary) used += summary.getBoundingClientRect().height;
    var avail = Math.floor(host.clientHeight - used - 2);
    if (avail > 60) lines.style.maxHeight = avail + "px";
  }

  function apply() {
    var el = details();
    if (el && el.open !== open) el.open = open;
    size();
  }

  function watch(host) {
    if (host.dataset.detailsStateWatched) return;
    host.dataset.detailsStateWatched = "1";
    new MutationObserver(apply).observe(host, { childList: true });
    apply();
  }

  function onPanelClass(panel) {
    var expanded = panel.classList.contains(EXPANDED);
    if (expanded === (panel.dataset.detailsExpanded === "1")) return;
    panel.dataset.detailsExpanded = expanded ? "1" : "0";
    if (expanded) {
      beforeExpand = open;
      remember(true);
    } else if (beforeExpand !== null) {
      remember(beforeExpand);
      beforeExpand = null;
    }
    apply();
  }

  function watchPanel(panel) {
    if (panel.dataset.detailsPanelWatched) return;
    panel.dataset.detailsPanelWatched = "1";
    panel.dataset.detailsExpanded = panel.classList.contains(EXPANDED) ? "1" : "0";
    new MutationObserver(function () {
      onPanelClass(panel);
      // The class lands while the panel's width/height transition is at
      // its start, so the room measured now is the small panel's; measure
      // again when the transition ends (and once more on a timer, for a
      // browser that runs no transition).
      setTimeout(size, 250);
    }).observe(panel, { attributes: true, attributeFilter: ["class"] });
    panel.addEventListener("transitionend", size);
  }

  function scan() {
    var host = document.getElementById("progress-feed-scroll");
    if (host) watch(host);
    var panel = document.getElementById("progress-panel");
    if (panel) watchPanel(panel);
  }

  new MutationObserver(scan).observe(document.body, {
    childList: true,
    subtree: true,
  });
  scan();
})();
