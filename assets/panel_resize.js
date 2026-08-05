/**
 * Left-edge drag handle for the Analyze context panel.
 *
 * The panel is mounted and unmounted by the router every time you enter or
 * leave /analyze, so the observer stays connected for the life of the page
 * rather than disconnecting after the first attach. Disconnecting would leave
 * the handle missing on every visit after the first.
 */
(function () {
  var MIN_W = 300;
  var MAX_W = 700;

  function attach(panel) {
    if (panel.querySelector(".panel-resize-handle")) return;

    var handle = document.createElement("div");
    handle.className = "panel-resize-handle";
    panel.prepend(handle);

    var startX, startW;

    function onDrag(e) {
      var delta = startX - e.clientX;
      var newW = Math.min(Math.max(startW + delta, MIN_W), MAX_W);
      panel.style.width = newW + "px";
      panel.style.minWidth = newW + "px";
      var grid = panel.parentElement;
      if (grid && grid.classList.contains("analyze-grid")) {
        grid.style.gridTemplateColumns = "minmax(0, 1fr) " + newW + "px";
      }
    }

    function onStop() {
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      document.removeEventListener("mousemove", onDrag);
      document.removeEventListener("mouseup", onStop);
    }

    handle.addEventListener("mousedown", function (e) {
      e.preventDefault();
      startX = e.clientX;
      startW = panel.offsetWidth;
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";
      document.addEventListener("mousemove", onDrag);
      document.addEventListener("mouseup", onStop);
    });
  }

  function scan() {
    var panel = document.getElementById("context-panel");
    if (panel) attach(panel);
  }

  new MutationObserver(scan).observe(document.body, {
    childList: true,
    subtree: true,
  });
  scan();
})();
