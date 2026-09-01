/**
 * Reading-position anchor for the Pipeline Activity feed.
 *
 * dash-renderer keys child wrappers by their tree path, not by the `key`
 * prop, so any tick that rewrites the feed's children recreates every row
 * node — and recreating the nodes of the column-reverse scroller throws the
 * scroll position away. No server output can fix that, so the position is
 * preserved here: before each rewrite the row under the viewport edge is
 * remembered (content, not pixels — the 45-row window slides), and after the
 * rewrite that same row is put back where it was. MutationObserver callbacks
 * run before the next paint, so the restore never flickers.
 *
 * Three reading states, three behaviours:
 *   at rest (scrollTop 0)      -> leave alone, the reset position IS rest;
 *   pinned to the newest edge  -> keep following the newest lines;
 *   parked mid-feed            -> re-anchor to the remembered row.
 * A tick that renders a NEW run boundary skips restoration entirely: the
 * snap-to-newest clientside callback in app.py owns that scroll.
 */
(function () {
  var state = { top: 0, pinned: false, anchor: null, delta: 0, lastRun: null };

  function lastRunText(feed) {
    var runs = feed.querySelectorAll(".progress-icon-run");
    if (!runs.length) return null;
    var row = runs[runs.length - 1].parentNode;
    return row ? row.textContent : null;
  }

  function capture(feed) {
    state.top = feed.scrollTop;
    state.pinned = feed.scrollTop <= -(feed.scrollHeight - feed.clientHeight) + 4;
    state.anchor = null;
    if (!feed.scrollTop) return; // at rest: nothing to preserve
    var fr = feed.getBoundingClientRect();
    var kids = feed.children;
    for (var i = 0; i < kids.length; i++) {
      var b = kids[i].getBoundingClientRect();
      if (b.bottom > fr.top && b.top < fr.bottom) {
        state.anchor = kids[i].textContent;
        state.delta = b.top - fr.top;
        return;
      }
    }
  }

  function restore(feed) {
    if (!state.top) return; // was at rest: stay at rest
    if (state.pinned && feed.lastElementChild) {
      feed.lastElementChild.scrollIntoView({ block: "nearest" });
      return;
    }
    if (state.anchor !== null) {
      var fr = feed.getBoundingClientRect();
      var kids = feed.children;
      for (var i = 0; i < kids.length; i++) {
        if (kids[i].textContent === state.anchor) {
          feed.scrollTop +=
            kids[i].getBoundingClientRect().top - fr.top - state.delta;
          return;
        }
      }
    }
    // Anchor row trimmed out of the window (or the feed was swapped by
    // another publisher): the old offset is the best approximation left.
    feed.scrollTop = state.top;
  }

  function attach(feed) {
    if (feed.dataset.scrollAnchored) return;
    feed.dataset.scrollAnchored = "1";
    state.lastRun = lastRunText(feed);
    capture(feed);
    feed.addEventListener(
      "scroll",
      function () {
        capture(feed);
      },
      { passive: true }
    );
    new MutationObserver(function () {
      var run = lastRunText(feed);
      if (run !== state.lastRun) {
        // A new run boundary just rendered — the snap callback scrolls.
        state.lastRun = run;
      } else {
        restore(feed);
      }
      capture(feed);
    }).observe(feed, { childList: true });
  }

  function scan() {
    var feed = document.getElementById("progress-feed-scroll");
    if (feed) attach(feed);
  }

  new MutationObserver(scan).observe(document.body, {
    childList: true,
    subtree: true,
  });
  scan();
})();
