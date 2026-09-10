/**
 * The browser 3D view: a rigged character driven by the live capture.
 *
 * This is a port of src/avatar_pose.py, and the port is deliberate rather
 * than incidental. Skinning a 200k-vertex character in numpy costs ~40 ms a
 * frame and runs on the same thread as the camera, which is why the desktop
 * viewer decimates to 40k and bakes colours into vertices. Here the GPU does
 * the skinning for free, so the model is drawn whole, at full detail, with
 * its real textures and alpha.
 *
 * Outline shells are skipped in both viewers; see the note in `load()` for
 * why drawing them was tried here and did not work.
 *
 * What crosses the wire is only the landmarks — seven joints and two hands,
 * a few hundred bytes a frame. The retargeting happens here.
 *
 * Keeping the two implementations honest
 * --------------------------------------
 * Both consume the SAME scene coordinates (engine.scene_from_pose) and the
 * SAME rig description (`<model>.rig.json`), and both implement the same
 * matrix convention:
 *
 *     bone.matrixWorld = S · T(p_new) · R · T(-p_rest) · rest_world
 *
 * where S maps avatar space onto scene space. three.js binds skins with an
 * identity bindMatrix (GLTFLoader: `mesh.bind(skeleton, _identityMatrix)`),
 * so its shader computes Σ w · (matrixWorld · boneInverse) · position —
 * exactly the Python formula. Folding S into each bone matrix works because
 * skinning is linear in the bone matrices.
 *
 * If you change the pose maths, change it in both, and re-run
 * `scripts/check_avatar.py --selftest` for the Python side.
 */
import * as THREE from './vendor/three/three.module.js';
import { GLTFLoader } from './vendor/three/GLTFLoader.js';

/** MediaPipe hand topology — the same 21 edges the desktop viewer draws. */
const HAND_BONES = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

const HAND_COLOR = 0x33c7ff;

/** Which bone aims at which, and the joints giving the target direction. */
const ARM_CHAIN = [
  ['leftupperarm', 'leftlowerarm', 'l_shoulder', 'l_elbow'],
  ['leftlowerarm', 'lefthand', 'l_elbow', 'l_wrist'],
  ['rightupperarm', 'rightlowerarm', 'r_shoulder', 'r_elbow'],
  ['rightlowerarm', 'righthand', 'r_elbow', 'r_wrist'],
];

const REQUIRED_JOINTS = ['l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow',
                         'l_wrist', 'r_wrist', 'nose'];

const BONE_ALIASES = {
  hips: ['hips', 'pelvis'],
  spine: ['spine'],
  chest: ['chest', 'upperchest'],
  neck: ['neck'],
  head: ['head'],
  leftshoulder: ['leftshoulder', 'shoulderl'],
  rightshoulder: ['rightshoulder', 'shoulderr'],
  leftupperarm: ['leftupperarm', 'leftarm', 'upperarml'],
  rightupperarm: ['rightupperarm', 'rightarm', 'upperarmr'],
  leftlowerarm: ['leftlowerarm', 'leftforearm', 'forearml'],
  rightlowerarm: ['rightlowerarm', 'rightforearm', 'forearmr'],
  lefthand: ['lefthand', 'handl'],
  righthand: ['righthand', 'handr'],
};

const normaliseName = (s) => (s || '').toLowerCase().replace(/[^a-z0-9]/g, '');

/** Is this material an inverted-hull outline shell? Mirrors gltf_min.py. */
function isOutlineMaterial(name) {
  const n = (name || '').toLowerCase();
  return n.endsWith('_line') || n.includes('outline');
}


/**
 * Both hands as real geometry rather than lines.
 *
 * WebGL ignores `linewidth` on almost every platform, so a LineSegments rig
 * would be 1px wide — the exact problem that made the hands invisible in the
 * desktop viewer. Instanced spheres and capsules cost nothing and read at any
 * size.
 */
class HandRig {
  constructor(scene, color = HAND_COLOR) {
    const jointGeo = new THREE.SphereGeometry(1, 10, 8);
    const boneGeo = new THREE.CylinderGeometry(1, 1, 1, 6, 1);
    const mat = new THREE.MeshBasicMaterial({ color });

    this.joints = new THREE.InstancedMesh(jointGeo, mat, 21);
    this.bones = new THREE.InstancedMesh(boneGeo, mat, HAND_BONES.length);
    this.joints.frustumCulled = false;
    this.bones.frustumCulled = false;
    this.visible = false;
    this.setVisible(false);
    scene.add(this.joints, this.bones);

    this._m = new THREE.Matrix4();
    this._q = new THREE.Quaternion();
    this._up = new THREE.Vector3(0, 1, 0);
    this._a = new THREE.Vector3();
    this._b = new THREE.Vector3();
    this._d = new THREE.Vector3();
    this._s = new THREE.Vector3();
  }

  setVisible(v) {
    this.visible = v;
    this.joints.visible = v;
    this.bones.visible = v;
  }

  /** `points` is 21 × [x,y,z] in scene coordinates, or null to hide. */
  update(points, radius) {
    if (!points || points.length !== 21) { this.setVisible(false); return; }
    this.setVisible(true);

    for (let i = 0; i < 21; i++) {
      const p = points[i];
      this._m.makeScale(radius, radius, radius);
      this._m.setPosition(p[0], p[1], p[2]);
      this.joints.setMatrixAt(i, this._m);
    }
    this.joints.instanceMatrix.needsUpdate = true;

    for (let i = 0; i < HAND_BONES.length; i++) {
      const [a, b] = HAND_BONES[i];
      this._a.fromArray(points[a]);
      this._b.fromArray(points[b]);
      this._d.subVectors(this._b, this._a);
      const len = this._d.length();
      if (len < 1e-9) {
        // Degenerate bone: scale it to nothing rather than emit a NaN basis.
        this._m.makeScale(0, 0, 0);
        this.bones.setMatrixAt(i, this._m);
        continue;
      }
      this._q.setFromUnitVectors(this._up, this._d.divideScalar(len));
      this._s.set(radius * 0.55, len, radius * 0.55);
      this._m.compose(
        this._a.lerp(this._b, 0.5), this._q, this._s);
      this.bones.setMatrixAt(i, this._m);
    }
    this.bones.instanceMatrix.needsUpdate = true;
  }
}


/** Orthonormal frame (columns side, up, forward) from two rough axes. */
function makeBasis(side, up) {
  const s = side.clone().normalize();
  const u = up.clone().addScaledVector(s, -up.dot(s)).normalize();
  const f = new THREE.Vector3().crossVectors(s, u).normalize();
  return new THREE.Matrix4().makeBasis(s, u, f);
}


/**
 * The rig: bone lookup, hierarchy repair, and the per-frame solve.
 */
class Rig {
  constructor(root, spec) {
    this.spec = spec || {};
    this.mirror = this.spec.mirror !== false;
    this.smoothing = typeof this.spec.smoothing === 'number'
      ? this.spec.smoothing : 0.45;
    this.handScale = 0.0;

    // Flatten the scene graph into an indexed node list, mirroring how the
    // Python side walks glTF nodes.
    this.nodes = [];
    const index = new Map();
    root.updateMatrixWorld(true);
    root.traverse((o) => { index.set(o, this.nodes.length); this.nodes.push(o); });
    this.index = index;

    this.parents = this.nodes.map((o) =>
      (o.parent && index.has(o.parent)) ? index.get(o.parent) : -1);

    this.restWorld = this.nodes.map((o) => o.matrixWorld.clone());
    this.restPos = this.restWorld.map((m) =>
      new THREE.Vector3().setFromMatrixPosition(m));

    this.byName = new Map();
    this.nodes.forEach((o, i) => { if (o.name) this.byName.set(o.name, i); });

    this.bones = this._resolveBones();

    this._applyChainParents();
    this.order = this._topologicalOrder();
    this.collapse = this._handDescendants();

    const l = this.restPos[this.bones.leftupperarm];
    const r = this.restPos[this.bones.rightupperarm];
    this.restShoulderWidth = l.distanceTo(r);
    if (!(this.restShoulderWidth > 1e-9)) {
      throw new Error("the rig's shoulders sit on top of each other");
    }
    this.restNeck = l.clone().add(r).multiplyScalar(0.5);

    const hips = this.bones.hips;
    const up = hips !== undefined
      ? this.restNeck.clone().sub(this.restPos[hips])
      : new THREE.Vector3(0, 1, 0);
    if (up.lengthSq() < 1e-18) up.set(0, 1, 0);
    this.avatarBasisT = makeBasis(l.clone().sub(r), up).transpose();

    // Scratch, so the per-frame solve allocates nothing.
    this.accum = this.nodes.map(() => new THREE.Matrix4());
    this.newPos = this.restPos.map((v) => v.clone());
    this._ema = new Map();
    this._emaScale = null;
    this._tmpA = new THREE.Vector3();
    this._tmpB = new THREE.Vector3();
    this._q = new THREE.Quaternion();
    this._rot = new THREE.Matrix4();
    this._m = new THREE.Matrix4();
    this._t = new THREE.Matrix4();
    this._S = new THREE.Matrix4();
    this._handVec = new THREE.Vector3();
  }

  /**
   * A node index from a name written in the glTF file.
   *
   * GLTFLoader renames nodes on import — `PropertyBinding.sanitizeNodeName`
   * strips `. [ ] : /` and turns whitespace into underscores, so an Auto-Rig
   * Pro bone called `c_arm_twist_offset.l_0116` arrives as
   * `c_arm_twist_offsetl_0116`. The sidecar is written against the names in
   * the file (which is what check_avatar.py prints and what Blender shows),
   * so every lookup has to try the sanitised form too. Using three's own
   * function rather than reimplementing the rule keeps this correct if the
   * rule changes.
   */
  _lookup(name) {
    if (this.byName.has(name)) return this.byName.get(name);
    const clean = THREE.PropertyBinding.sanitizeNodeName(name);
    return this.byName.has(clean) ? this.byName.get(clean) : undefined;
  }

  _resolveBones() {
    const out = {};
    // Name matching first, so a partial sidecar still gets the rest.
    const normalised = this.nodes.map((o) => normaliseName(o.name));
    for (const [bone, keys] of Object.entries(BONE_ALIASES)) {
      for (const key of keys) {
        const k = normaliseName(key);
        let best = -1;
        for (let i = 0; i < normalised.length; i++) {
          if (!normalised[i].includes(k)) continue;
          if (best === -1 || this.nodes[i].name.length < this.nodes[best].name.length) {
            best = i;
          }
        }
        if (best !== -1) { out[bone] = best; break; }
      }
    }
    // The sidecar overrides, because a person looked at this file.
    for (const [bone, name] of Object.entries(this.spec.bones || {})) {
      const i = this._lookup(name);
      if (i !== undefined) out[bone.toLowerCase()] = i;
    }

    const missing = ['leftupperarm', 'rightupperarm', 'leftlowerarm',
                     'rightlowerarm', 'lefthand', 'righthand']
      .filter((b) => out[b] === undefined);
    if (missing.length) {
      throw new Error('the rig is missing arm bones: ' + missing.join(', ')
        + '. Describe them in the .rig.json sidecar.');
    }
    return out;
  }

  /** Rejoin a chain the exporter flattened. See avatar_pose.py. */
  _applyChainParents() {
    for (const [child, parent] of Object.entries(this.spec.chain_parents || {})) {
      const c = this._lookup(child), p = this._lookup(parent);
      if (c === undefined || p === undefined) continue;
      this.parents[c] = p;
    }
    for (let start = 0; start < this.parents.length; start++) {
      const seen = new Set();
      let j = start;
      while (j !== -1) {
        if (seen.has(j)) {
          throw new Error('chain_parents creates a cycle through node '
            + (this.nodes[j].name || j));
        }
        seen.add(j);
        j = this.parents[j];
      }
    }
  }

  _topologicalOrder() {
    const depth = this.parents.map((_, i) => {
      let d = 0, j = this.parents[i], guard = 0;
      while (j !== -1 && guard++ < this.parents.length) { d++; j = this.parents[j]; }
      return d;
    });
    return this.nodes.map((_, i) => i).sort((a, b) => depth[a] - depth[b]);
  }

  _handDescendants() {
    const roots = ['lefthand', 'righthand']
      .map((b) => this.bones[b]).filter((i) => i !== undefined);
    const children = new Map();
    this.parents.forEach((p, i) => {
      if (p === -1) return;
      if (!children.has(p)) children.set(p, []);
      children.get(p).push(i);
    });
    const out = new Set();
    const stack = [...roots];
    while (stack.length) {
      const n = stack.pop();
      if (out.has(n)) continue;
      out.add(n);
      for (const c of (children.get(n) || [])) stack.push(c);
    }
    return out;
  }

  /** Avatar space → scene space, or null if the frame cannot place it. */
  sceneTransform(joints) {
    const ls = joints.l_shoulder, rs = joints.r_shoulder;
    if (!ls || !rs || !joints.nose) return null;
    const l = this._tmpA.fromArray(ls);
    const r = this._tmpB.fromArray(rs);
    const width = l.distanceTo(r);
    if (!(width > 1e-9)) return null;

    const neck = l.clone().add(r).multiplyScalar(0.5);
    // World up, never the neck→nose direction: one noisy facial landmark must
    // not be able to roll — or invert — the whole body. See avatar_pose.py.
    const side = this.mirror ? r.clone().sub(l) : l.clone().sub(r);
    const rotation = makeBasis(side, new THREE.Vector3(0, 1, 0))
      .multiply(this.avatarBasisT);

    let scale = width / this.restShoulderWidth;
    if (this.smoothing < 1.0) {
      if (this._emaScale !== null) {
        scale = this.smoothing * scale + (1 - this.smoothing) * this._emaScale;
      }
      this._emaScale = scale;
    }
    return { rotation, scale, avatarOrigin: this.restNeck, sceneOrigin: neck };
  }

  _smooth(key, dir) {
    if (this.smoothing >= 1.0) return dir;
    const prev = this._ema.get(key);
    if (!prev) { this._ema.set(key, dir.clone()); return dir; }
    const blended = dir.clone().multiplyScalar(this.smoothing)
      .addScaledVector(prev, 1 - this.smoothing);
    if (blended.lengthSq() < 1e-18) { this._ema.set(key, dir.clone()); return dir; }
    blended.normalize();
    this._ema.set(key, blended.clone());
    return blended;
  }

  _aimDirections(targets) {
    const out = new Map();
    for (const [bone, childBone, a, b] of ARM_CHAIN) {
      const node = this.bones[bone], child = this.bones[childBone];
      if (node === undefined || child === undefined) continue;
      if (!targets[a] || !targets[b]) continue;
      const d = targets[b].clone().sub(targets[a]);
      if (d.lengthSq() < 1e-18) continue;
      out.set(node, { child, dir: this._smooth(node, d.normalize()) });
    }

    // Head, bounded to 60° off rest so a stray nose cannot invert it.
    const neck = this.bones.neck, head = this.bones.head;
    if (neck !== undefined && head !== undefined && targets.nose
        && targets.l_shoulder && targets.r_shoulder) {
      const restDir = this.restPos[head].clone().sub(this.restPos[neck]);
      const mid = targets.l_shoulder.clone().add(targets.r_shoulder)
        .multiplyScalar(0.5);
      const d = targets.nose.clone().sub(mid);
      if (d.lengthSq() > 1e-18 && restDir.lengthSq() > 1e-18) {
        d.normalize();
        if (d.dot(restDir.normalize()) > 0.5) {
          out.set(neck, { child: head, dir: this._smooth(neck, d) });
        }
      }
    }
    return out;
  }

  /**
   * Pose every bone for this frame. Returns false if the frame was unusable,
   * in which case the previous pose is left standing.
   */
  solve(joints) {
    const frame = this.sceneTransform(joints);
    if (!frame) return false;
    const { rotation, scale, avatarOrigin, sceneOrigin } = frame;

    // S = T(sceneOrigin) · rotation · scale · T(-avatarOrigin)
    const S = this._S.identity()
      .makeTranslation(sceneOrigin.x, sceneOrigin.y, sceneOrigin.z)
      .multiply(rotation)
      .multiply(new THREE.Matrix4().makeScale(scale, scale, scale))
      .multiply(new THREE.Matrix4().makeTranslation(
        -avatarOrigin.x, -avatarOrigin.y, -avatarOrigin.z));

    // Targets into avatar space, so the aiming never mixes coordinate frames.
    const inv = new THREE.Matrix4().copy(S).invert();
    const targets = {};
    for (const [k, v] of Object.entries(joints)) {
      targets[k] = new THREE.Vector3().fromArray(v).applyMatrix4(inv);
    }
    const aims = this._aimDirections(targets);

    for (const i of this.order) {
      const p = this.parents[i];
      this.accum[i].identity();
      if (p !== -1) {
        this.accum[i].copy(this.accum[p]);
        this._tmpA.subVectors(this.restPos[i], this.restPos[p])
          .applyMatrix4(this.accum[p]);
        this.newPos[i].copy(this.newPos[p]).add(this._tmpA);
      } else {
        this.newPos[i].copy(this.restPos[i]);
      }
      const aim = aims.get(i);
      if (aim) {
        this._tmpB.subVectors(this.restPos[aim.child], this.restPos[i]);
        if (this._tmpB.lengthSq() > 1e-18) {
          this._tmpB.normalize().applyMatrix4(this.accum[i]).normalize();
          this._q.setFromUnitVectors(this._tmpB, aim.dir);
          this._rot.makeRotationFromQuaternion(this._q);
          this.accum[i].premultiply(this._rot);
        }
      }
      if (this.collapse.has(i)) {
        this._handVec.set(this.handScale, this.handScale, this.handScale);
        this.accum[i].scale(this._handVec);
      }
    }

    // bone.matrixWorld = S · T(p_new) · accum · T(-p_rest) · rest_world
    for (let i = 0; i < this.nodes.length; i++) {
      const node = this.nodes[i];
      if (!node.isBone) continue;
      const m = this._m;
      m.makeTranslation(-this.restPos[i].x, -this.restPos[i].y,
                        -this.restPos[i].z);
      m.multiply(this.restWorld[i]);
      m.premultiply(this.accum[i]);
      m.premultiply(this._t.makeTranslation(
        this.newPos[i].x, this.newPos[i].y, this.newPos[i].z));
      m.premultiply(S);
      node.matrixWorld.copy(m);
    }
    return true;
  }
}


export class AvatarView {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({
      canvas, antialias: true, alpha: true,
    });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(32, 1, 0.01, 100);
    this.camera.position.set(0, 0.95, 3.1);
    this.camera.lookAt(0, 0.95, 0);

    // Flat, even light. These are toon models: the artwork is in the texture,
    // and dramatic shading fights it.
    this.scene.add(new THREE.AmbientLight(0xffffff, 2.2));
    const key = new THREE.DirectionalLight(0xffffff, 1.1);
    key.position.set(0.4, 1.2, 2.0);
    this.scene.add(key);

    this.handL = new HandRig(this.scene);
    this.handR = new HandRig(this.scene);

    this.rig = null;
    this.root = null;
    this.skeletons = [];
    this.running = false;
    this.lastSeq = -1;
    this.showHands = true;
    this._frame = null;
    this._onResize = () => this.resize();
    addEventListener('resize', this._onResize);
  }

  resize() {
    const w = this.canvas.clientWidth || 1;
    const h = this.canvas.clientHeight || 1;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    // three's fov is vertical, so a narrow panel would crop the signer's arms
    // sideways — exactly the part worth seeing. Widen the vertical fov on tall
    // canvases to hold the horizontal extent fixed instead.
    const base = 32;
    this.camera.fov = this.camera.aspect < 1
      ? THREE.MathUtils.radToDeg(2 * Math.atan(
          Math.tan(THREE.MathUtils.degToRad(base) / 2) / this.camera.aspect))
      : base;
    this.camera.updateProjectionMatrix();
  }

  async load(modelUrl, rigUrl) {
    let spec = {};
    try {
      const r = await fetch(rigUrl);
      if (r.ok) spec = await r.json();
    } catch (_) { /* a missing sidecar is normal */ }

    const gltf = await new Promise((resolve, reject) => {
      new GLTFLoader().load(modelUrl, resolve, undefined, reject);
    });

    if (this.root) this.dispose_model();
    this.root = gltf.scene;
    this.scene.add(this.root);
    this.root.updateMatrixWorld(true);

    this.rig = new Rig(this.root, spec);

    let vertices = 0, outlines = 0;
    const skeletons = new Set();
    this.root.traverse((o) => {
      if (o.isSkinnedMesh) {
        skeletons.add(o.skeleton);
        const mats0 = Array.isArray(o.material) ? o.material : (o.material ? [o.material] : []);
        if (!mats0.some((m) => isOutlineMaterial(m.name))) {
          vertices += o.geometry.attributes.position.count;
        }
        // glTF says a skinned mesh's own transform is ignored; GLTFLoader
        // binds with an identity bindMatrix, so the skinned result is already
        // in the frame we are writing bone matrices in. Leaving a node
        // transform on the mesh would apply it a second time.
        o.matrixAutoUpdate = false;
        o.matrixWorldAutoUpdate = false;
        o.matrix.identity();
        o.matrixWorld.identity();
        o.frustumCulled = false;
      }
      const mats = Array.isArray(o.material) ? o.material : (o.material ? [o.material] : []);
      if (mats.some((m) => isOutlineMaterial(m.name))) {
        // Skipped, same as the desktop viewer — but for a better reason than
        // "Open3D cannot cull back faces", which was the original one.
        //
        // Drawing them back-faces-only, the textbook inverted hull, was tried
        // here and produced solid black slabs over the character. Two reasons,
        // both measurable in this model: the hull is inflated by 0.0075 units
        // on a 4.7-unit figure (0.16%), too little to clear the surface
        // reliably; and much of a character like this is thin cloth — sleeves,
        // hat brim, cape — which is an open surface with no inside for a hull
        // to hide in, so its "back" faces land in front of the body.
        //
        // A real toon outline needs the vertex shader to widen the hull along
        // the normal, which is what MToon does and what this export dropped.
        // Until that is implemented, not drawing them is correct.
        o.visible = false;
        outlines++;
      }
    });

    // We write bone.matrixWorld ourselves; three must not recompute it from
    // the parent chain, which would undo the flattened-rig repair.
    this.rig.nodes.forEach((o) => {
      if (o.isBone) { o.matrixAutoUpdate = false; o.matrixWorldAutoUpdate = false; }
    });

    this.skeletons = [...skeletons];
    this.handRadius = this.rig.restShoulderWidth * 0.055;
    return {
      vertices,
      outlines,
      bones: Object.keys(this.rig.bones).length,
      sidecar: Object.keys(spec).length > 0,
    };
  }

  setLandmarks(payload) {
    if (!payload || payload.seq === this.lastSeq) return;
    this.lastSeq = payload.seq;
    this._frame = payload;
  }

  _apply() {
    const f = this._frame;
    if (!f || !this.rig) return;
    const joints = f.joints || {};
    if (!REQUIRED_JOINTS.every((k) => joints[k])) return;

    if (this.rig.solve(joints)) {
      for (const s of this.skeletons) s.update();
    }
    if (this.showHands) {
      this.handL.update(f.lhand, this.handRadius);
      this.handR.update(f.rhand, this.handRadius);
    } else {
      this.handL.setVisible(false);
      this.handR.setVisible(false);
    }
  }

  start() {
    if (this.running) return;
    this.running = true;
    this.resize();
    const loop = () => {
      if (!this.running) return;
      this._apply();
      this.renderer.render(this.scene, this.camera);
      if (this._capture) {
        // Read inside the same frame: without preserveDrawingBuffer the
        // drawing buffer is cleared once the tick ends, and paying that cost
        // permanently for an occasional screenshot is not worth it.
        const done = this._capture;
        this._capture = null;
        done(this.canvas.toDataURL('image/png'));
      }
      this._raf = requestAnimationFrame(loop);
    };
    loop();
  }

  /** A PNG data URL of the next rendered frame. For screenshots and tests. */
  capture() {
    return new Promise((resolve, reject) => {
      if (!this.running) { reject(new Error('viewer is not running')); return; }
      this._capture = resolve;
    });
  }

  stop() {
    this.running = false;
    if (this._raf) cancelAnimationFrame(this._raf);
  }

  dispose_model() {
    if (!this.root) return;
    this.scene.remove(this.root);
    this.root.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      const mats = Array.isArray(o.material) ? o.material : (o.material ? [o.material] : []);
      for (const m of mats) {
        for (const k of Object.keys(m)) {
          if (m[k] && m[k].isTexture) m[k].dispose();
        }
        m.dispose();
      }
    });
    this.root = null;
    this.rig = null;
    this.skeletons = [];
  }

  dispose() {
    this.stop();
    removeEventListener('resize', this._onResize);
    this.dispose_model();
    this.renderer.dispose();
  }
}
