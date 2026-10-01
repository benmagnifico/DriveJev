import { loadingFailed, nextPaint } from "/third_party/jevpilot/src/loading-screen.js";

// Paint the lightweight HTML loader before downloading/building the 3D world.
nextPaint()
  .then(() => import("./app.js"))
  .catch(loadingFailed);
