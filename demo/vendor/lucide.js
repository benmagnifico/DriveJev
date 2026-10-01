/** Minimal ESM entry for the vendored lucide 0.577.0 (ISC): only the icons this demo uses. */
import replaceElement from './lucide/dist/esm/replaceElement.js';
export { default as Braces } from './lucide/dist/esm/icons/braces.js';
export { default as RotateCw } from './lucide/dist/esm/icons/rotate-cw.js';
export { default as Video } from './lucide/dist/esm/icons/video.js';
export { default as Pause } from './lucide/dist/esm/icons/pause.js';
export { default as Play } from './lucide/dist/esm/icons/play.js';
export { default as Map } from './lucide/dist/esm/icons/map.js';
export { default as Maximize } from './lucide/dist/esm/icons/maximize.js';
export { default as Minimize } from './lucide/dist/esm/icons/minimize.js';
export { default as Sparkles } from './lucide/dist/esm/icons/sparkles.js';
export { default as ArrowUp } from './lucide/dist/esm/icons/arrow-up.js';
export { default as CornerUpLeft } from './lucide/dist/esm/icons/corner-up-left.js';
export { default as CornerUpRight } from './lucide/dist/esm/icons/corner-up-right.js';
export { default as Flag } from './lucide/dist/esm/icons/flag.js';
export { default as ArrowUpRight } from './lucide/dist/esm/icons/arrow-up-right.js';
export { default as X } from './lucide/dist/esm/icons/x.js';
export { default as Copy } from './lucide/dist/esm/icons/copy.js';
export { default as Download } from './lucide/dist/esm/icons/download.js';
export { default as RotateCcw } from './lucide/dist/esm/icons/rotate-ccw.js';
export { default as Plus } from './lucide/dist/esm/icons/plus.js';
export { default as Minus } from './lucide/dist/esm/icons/minus.js';
export { default as Grip } from './lucide/dist/esm/icons/grip.js';
export { default as ChevronDown } from './lucide/dist/esm/icons/chevron-down.js';
export { default as ChevronsUpDown } from './lucide/dist/esm/icons/chevrons-up-down.js';
export { default as Cloud } from './lucide/dist/esm/icons/cloud.js';
export { default as Server } from './lucide/dist/esm/icons/server.js';
export { default as PanelRight } from './lucide/dist/esm/icons/panel-right.js';
export { default as ScanEye } from './lucide/dist/esm/icons/scan-eye.js';
export { default as Route } from './lucide/dist/esm/icons/route.js';
export function createIcons({ icons = {}, nameAttr = 'data-lucide', attrs = {}, root = document } = {}) {
  for (const element of root.querySelectorAll(`[${nameAttr}]`)) replaceElement(element, { nameAttr, icons, attrs });
}
