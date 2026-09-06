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
 */
(function () {
  var KEY = "qn_progress_details_open";
  var open = false;
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
      }
    },
    true
  );

  function apply() {
    var el = details();
    if (el && el.open !== open) el.open = open;
  }

  function watch(host) {
    if (host.dataset.detailsStateWatched) return;
    host.dataset.detailsStateWatched = "1";
    new MutationObserver(apply).observe(host, { childList: true });
    apply();
  }

  function scan() {
    var host = document.getElementById("progress-feed-scroll");
    if (host) watch(host);
  }

  new MutationObserver(scan).observe(document.body, {
    childList: true,
    subtree: true,
  });
  scan();
})();
