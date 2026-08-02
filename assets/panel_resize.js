/**
 * Adds a left-edge drag handle to the context panel for resizing.
 * Retries until the panel element exists in the DOM (Dash renders async).
 */
(function () {
  function attach(panel) {
    if (panel.querySelector(".panel-resize-handle")) return;

    var handle = document.createElement("div");
    handle.className = "panel-resize-handle";
    panel.prepend(handle);

    var startX, startW;

    handle.addEventListener("mousedown", function (e) {
      e.preventDefault();
      startX = e.clientX;
      startW = panel.offsetWidth;
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";
      document.addEventListener("mousemove", onDrag);
      document.addEventListener("mouseup", onStop);
    });

    function onDrag(e) {
      var delta = startX - e.clientX;
      var newW = Math.min(Math.max(startW + delta, 300), 700);
      panel.style.width = newW + "px";
      panel.style.minWidth = newW + "px";
      var grid = panel.parentElement;
      if (grid && grid.classList.contains("dashboard-grid")) {
        grid.style.gridTemplateColumns =
          "var(--sidebar-width) 1fr " + newW + "px";
      }
    }

    function onStop() {
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      document.removeEventListener("mousemove", onDrag);
      document.removeEventListener("mouseup", onStop);
    }
  }

  var observer = new MutationObserver(function () {
    var panel = document.getElementById("context-panel");
    if (panel) {
      observer.disconnect();
      attach(panel);
    }
  });

  observer.observe(document.body, { childList: true, subtree: true });

  // Also try immediately in case panel already exists
  var panel = document.getElementById("context-panel");
  if (panel) {
    observer.disconnect();
    attach(panel);
  }
})();
