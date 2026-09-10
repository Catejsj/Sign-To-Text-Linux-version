# Vendored three.js

`three@0.169.0`, three files, fetched from unpkg:

| file | from |
|---|---|
| `three.module.js` | `build/three.module.js` |
| `GLTFLoader.js` | `examples/jsm/loaders/GLTFLoader.js` |
| `BufferGeometryUtils.js` | `examples/jsm/utils/BufferGeometryUtils.js` |

**One local edit.** `GLTFLoader.js` imports `../utils/BufferGeometryUtils.js`,
which assumes the upstream directory layout. All three files sit side by side
here, so that import was rewritten to `./BufferGeometryUtils.js`. Re-apply it
if you ever refresh these files.

`index.html` maps the bare specifier `three` to `three.module.js` with an
import map, which is what lets `GLTFLoader.js` load unmodified otherwise.

## Why vendored rather than a CDN

The control panel is served by Flask on localhost and has to work with no
internet — recording sessions happen wherever the signer is. There is no
build step, no npm and no Node on this machine; the browser loads these as
ES modules directly.

MIT licensed, © three.js authors.
